"""
Store de alertas em memória — histórico consultável via API (`/api/alerts`)
para que outros sistemas já em produção possam buscar os alertas gerados
pelo monitoramento (preditivo/YOLO ou por mudança/SSIM).
"""

import itertools
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Union

import cv2
import numpy as np

from app.config import ASSETS_DIR, NOTIFY_COOLDOWN_SECONDS, PATRIMONIOS

logger = logging.getLogger(__name__)

MAX_ALERTS = 500
# Imagem da câmera no momento de cada alerta (aberta ao clicar no alerta)
ALERT_IMAGES_DIR = ASSETS_DIR / "alerts"
# O mesmo alerta (câmera + origem + nível) não entra de novo no histórico
# antes disto — a análise contínua avalia 2x/s e registraria o mesmo
# "pessoa em cima da estátua" dezenas de vezes seguidas.
DEDUP_SECONDS = 120

# (camera, source, level) → último alerta registrado
_last_stored: dict[tuple[str, str, str], dict] = {}

_LEVEL_RANK = {"MODERADO": 1, "ALTO": 2, "CRÍTICO": 3}

# Quais alertas viram notificação na tela (som + mensagem). O resto fica
# só no histórico/painel — senão o operador se acostuma a ignorar o som.
#   interaction: ALTO+ (pessoa em cima confirmada, mão na área sensível por muito tempo)
#   risk:        objeto de risco encostado na estátua, ALTO+ (faca/tesoura, ou permanência)
#   ssim:        mudança física na estátua confirmada, ALTO+
#   surface:     superfície protegida alterada com pessoa suspeita junto, ALTO+
#   loitering:   nunca (informativo)
_NOTIFY_MIN_LEVEL = {"interaction": 2, "risk": 2, "ssim": 2, "surface": 2}

# câmera → patrimônio: câmeras do mesmo monumento compartilham o cooldown
_CAMERA_PATRIMONIO = {code: str(p["id"]) for p in PATRIMONIOS for code in p.get("camera_codes", [])}

# patrimônio (ou câmera avulsa) → (timestamp, nível) da última notificação
_last_notified: dict[str, tuple[float, int]] = {}


def incident_key(camera_code: str) -> str:
    """Chave do episódio: o patrimônio da câmera, ou a própria câmera se avulsa."""
    pid = _CAMERA_PATRIMONIO.get(camera_code)
    return f"patrimonio:{pid}" if pid else f"camera:{camera_code}"


def _should_notify(camera_code: str, source: str, level: str) -> bool:
    rank = _LEVEL_RANK.get(level, 0)
    if rank < _NOTIFY_MIN_LEVEL.get(source, 99):
        return False
    key = incident_key(camera_code)
    last = _last_notified.get(key)
    now = time.time()
    # Mesmo monumento notificado há pouco (qualquer câmera, qualquer tipo):
    # o operador já está olhando — fica só no histórico. Nível mais alto
    # que o já notificado (ALTO → CRÍTICO) notifica na hora.
    if last and now - last[0] < NOTIFY_COOLDOWN_SECONDS and rank <= last[1]:
        return False
    _last_notified[key] = (now, rank)
    return True

_alerts: list[dict] = []
_lock = threading.Lock()
_id_counter = itertools.count(1)


def _save_image(alert_id: int, camera_code: str, image: Union[bytes, np.ndarray]) -> Optional[str]:
    """Grava o frame do alerta (JPEG pronto ou array RGB) e devolve a URL em /assets."""
    try:
        if isinstance(image, np.ndarray):
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            image = buf.tobytes()
        ALERT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        # Os ids recomeçam a cada reinício do backend: o horário evita sobrescrever
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{alert_id}_{camera_code}.jpg"
        (ALERT_IMAGES_DIR / name).write_bytes(image)
        return f"/assets/alerts/{name}"
    except Exception as e:
        logger.warning(f"Erro ao salvar imagem do alerta {alert_id}: {e}")
        return None


def _delete_image(entry: dict) -> None:
    url = entry.get("image") or ""
    if url.startswith("/assets/alerts/"):
        try:
            (ALERT_IMAGES_DIR / url.rsplit("/", 1)[1]).unlink(missing_ok=True)
        except Exception:
            pass


def add_alert(camera_code: str, camera_name: str, level: str, message: str,
              source: str, objects: Optional[list[str]] = None,
              evidence: Optional[dict] = None,
              image: Optional[Union[bytes, np.ndarray]] = None) -> dict:
    """Registra um novo alerta e retorna o registro criado.

    `image`: frame da câmera no momento do alerta (JPEG ou array RGB) —
    só é gravado em disco se o alerta for de fato registrado (não duplicado).
    """
    key = (camera_code, source, level)
    with _lock:
        prev = _last_stored.get(key)
        if prev and time.time() - prev["timestamp"] < DEDUP_SECONDS:
            return prev
        notify = _should_notify(camera_code, source, level)
        alert_id = next(_id_counter)
    # Evento de superfície já tem o "durante" gravado nas evidências
    image_url = ((evidence or {}).get("images") or {}).get("durante")
    if not image_url and image is not None:
        image_url = _save_image(alert_id, camera_code, image)
    entry = {
        "id": alert_id,
        "timestamp": time.time(),
        "camera_code": camera_code,
        "camera_name": camera_name,
        "level": level,
        "message": message,
        # "risk" (objeto de risco), "interaction" (pose), "loitering"
        # (presença contínua), "ssim" (mudança física no print) ou
        # "surface" (superfície alterada no vídeo, com pessoa suspeita)
        "source": source,
        "objects": objects or [],
        # Eventos de superfície: imagens antes/durante/depois + clipe
        "evidence": evidence,
        # URL do frame da câmera no momento do alerta (ou None)
        "image": image_url,
        "notify": notify,  # true = frontend toca som + mostra mensagem
    }
    with _lock:
        _last_stored[key] = entry
        _alerts.append(entry)
        if len(_alerts) > MAX_ALERTS:
            for old in _alerts[: len(_alerts) - MAX_ALERTS]:
                _delete_image(old)
            del _alerts[: len(_alerts) - MAX_ALERTS]
    logger.info(f"🚨 Alerta registrado: {camera_code} [{level}] {message}")
    return entry


def update_evidence(alert_id: int, evidence: dict) -> None:
    """Completa as evidências de um alerta já registrado (depois.jpg + clipe)."""
    with _lock:
        for a in reversed(_alerts):
            if a["id"] == alert_id:
                a["evidence"] = evidence
                return


def get_alerts(since: Optional[float] = None, camera_code: Optional[str] = None,
               level: Optional[str] = None, limit: int = 100,
               after_id: Optional[int] = None, notify_only: bool = False) -> list[dict]:
    """Consulta alertas, mais recentes primeiro, com filtros opcionais."""
    with _lock:
        results = list(_alerts)

    if after_id is not None:
        results = [a for a in results if a["id"] > after_id]
    if notify_only:
        results = [a for a in results if a["notify"]]
    if since is not None:
        results = [a for a in results if a["timestamp"] >= since]
    if camera_code:
        results = [a for a in results if a["camera_code"] == camera_code]
    if level:
        results = [a for a in results if a["level"] == level]

    results.sort(key=lambda a: a["timestamp"], reverse=True)
    return results[:limit]


def last_id() -> int:
    with _lock:
        return _alerts[-1]["id"] if _alerts else 0
