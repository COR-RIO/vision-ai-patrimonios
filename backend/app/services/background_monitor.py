"""
Monitoramento contínuo em background: varre TODAS as câmeras dos
patrimônios (não só as que estão selecionadas no frontend), sequencialmente
— uma câmera por vez, para não sobrecarregar a máquina que roda YOLO/HF.

Assim o sistema fica de olho o tempo todo, mesmo com ninguém olhando a
página de Monitoramento; qualquer risk_alert (YOLO/preditivo) ou alert
(SSIM/mudança física) vira um registro em alert_service, consultável via
GET /api/alerts.
"""

import asyncio
import logging
import time

from app.config import (
    TRANSIENT_CLASSES,
    PATRIMONIOS,
    BACKGROUND_MONITOR_ENABLED,
    BACKGROUND_MONITOR_INTERVAL_SECONDS,
)
from app.api import monitor as monitor_api
from app.models.change_detector import covered_fraction

logger = logging.getLogger(__name__)

# Último resultado de captura por câmera — permite ao frontend saber se uma
# câmera está entregando vídeo de verdade agora, sem precisar de uma captura
# nova (reusa o que esse loop já faz a cada 20s) nem de acessar o iframe
# (que é de outra origem, então JS não consegue inspecionar o conteúdo dele).
_camera_status: dict[str, dict] = {}


def get_camera_status() -> dict[str, dict]:
    """Cópia do último resultado de captura de cada câmera monitorada."""
    return {code: dict(status) for code, status in _camera_status.items()}


def _all_camera_codes() -> list[str]:
    codes: list[str] = []
    for p in PATRIMONIOS:
        for code in p.get("camera_codes", []):
            if code not in codes:
                codes.append(code)
    return codes


def _process_camera(code: str) -> None:
    """Captura + detecta + registra alertas de uma câmera (síncrono, roda em thread)."""
    camera_service = monitor_api.camera_service
    detection_service = monitor_api.detection_service
    change_detector = monitor_api.change_detector

    camera = camera_service.get_camera_by_code(code)
    camera_name = camera.get("name", f"Câmera {code}") if camera else f"Câmera {code}"
    stream_url = (camera.get("stream_url") if camera else None) or camera_service.get_stream_url(code)
    if not stream_url:
        logger.debug(f"[background_monitor] Sem stream_url pra câmera {code}, pulando")
        return

    frame = monitor_api.get_frame(stream_url, code, 15.0)
    _camera_status[code] = {"has_video": frame is not None, "checked_at": time.time()}
    if frame is None:
        logger.debug(f"[background_monitor] Falha ao capturar frame da câmera {code}")
        return

    monitor_api.save_camera_snapshot(frame, code, camera_name, "monitoramento")

    # Câmera com análise contínua (live_analysis): YOLO/pose/alertas já
    # rodam lá, com rastreamento — aqui só a comparação SSIM, usando as
    # detecções de lá pra saber onde há gente/veículo
    from app.services import live_analysis
    live = live_analysis.latest_result(code)
    if live:
        result = {"yolo_detection": live["yolo_detection"]}
    else:
        result = detection_service.detect_full(frame, camera_id=code)
        monitor_api.record_risk_alert(code, camera_name, result.get("risk_alert"), image=frame)
        monitor_api.record_interaction_alert(code, camera_name, result.get("interaction_alert"), image=frame)

    # Pessoas/veículos no quadro — excluídos do diff SSIM (senão alguém
    # sentado ao lado da estátua vira "X% do monumento alterado").
    transient_boxes = [
        d["bbox"] for d in result.get("yolo_detection", {}).get("objects", [])
        if d["class_name"] in TRANSIENT_CLASSES and not d.get("is_statue")
    ]

    # Garante que toda câmera tenha uma referência SSIM — sem isso, mudança
    # física (pichação, dano, peça removida) nunca gera alerta pra essa câmera.
    # Só cria com a estátua livre: uma referência com turista na frente
    # faria todo print seguinte parecer "alterado".
    if code not in change_detector.monitored:
        if covered_fraction(frame, code, transient_boxes) < 0.05:
            change_detector.set_reference(code, frame, detection_service,
                                          objects=result.get("yolo_detection", {}).get("objects", []))
            logger.info(f"[background_monitor] Referência SSIM criada automaticamente para câmera {code}")
    else:
        change_result = change_detector.check(code, frame, detection_service, ignore_boxes=transient_boxes)
        if change_result.get("success"):
            monitor_api.record_ssim_alert(code, camera_name, change_result.get("alert"), image=frame)


async def run_forever() -> None:
    """Loop infinito: varre todas as câmeras, uma por vez, com pausa entre elas."""
    codes = _all_camera_codes()
    if not codes:
        logger.warning("[background_monitor] Nenhuma câmera configurada em PATRIMONIOS, loop não iniciado")
        return

    logger.info(f"[background_monitor] Iniciado — {len(codes)} câmeras, intervalo {BACKGROUND_MONITOR_INTERVAL_SECONDS}s")
    loop = asyncio.get_event_loop()

    while True:
        for code in codes:
            try:
                await loop.run_in_executor(None, _process_camera, code)
            except Exception as e:
                logger.warning(f"[background_monitor] Erro processando câmera {code}: {e}")
            await asyncio.sleep(BACKGROUND_MONITOR_INTERVAL_SECONDS)


def start() -> "asyncio.Task | None":
    """Cria a task de monitoramento contínuo, se habilitado."""
    if not BACKGROUND_MONITOR_ENABLED:
        logger.info("[background_monitor] Desabilitado (BACKGROUND_MONITOR_ENABLED=false)")
        return None
    return asyncio.create_task(run_forever())
