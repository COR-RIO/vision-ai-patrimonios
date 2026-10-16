"""
Rotas de monitoramento ao vivo com detecção em tempo real.
Captura frames do stream Tixxi, processa com YOLO + HF e retorna análise.
"""

import base64
import logging
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.services.camera_service import CameraService
from app.services.detection_service import DetectionService
from app.services import zone_service, alert_service
from app.models.change_detector import ChangeDetector, covered_fraction
from app.config import IMAGES_DIR, TRANSIENT_CLASSES
from pydantic import BaseModel
from app.schemas.schemas import ZoneInput, StatueInput

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitor", tags=["Monitoramento ao Vivo"])

# Teto de segurança pra requisições em lote (/multi, /streams) — não é mais
# 4, já que o pedido é acompanhar todas as câmeras dos patrimônios ao vivo.
MAX_BATCH_CAMERAS = 16

camera_service: CameraService = None
detection_service: DetectionService = None
change_detector = ChangeDetector()


def init_routes(cam_svc: CameraService, det_svc: DetectionService):
    global camera_service, detection_service
    camera_service = cam_svc
    detection_service = det_svc


def _parse_mjpeg_frame(stream_url: str, timeout_sec: float = 5.0) -> Optional[np.ndarray]:
    """
    Lê o stream MJPEG via HTTP e extrai o primeiro frame JPEG.
    Funciona com streams multipart/x-mixed-replace (padrão de câmeras IP).
    
    Usa headers de navegador para evitar bloqueios.
    
    Args:
        stream_url: URL do stream MJPEG
        timeout_sec: Tempo máximo para capturar
        
    Returns:
        np.ndarray: Frame em RGB ou None se falhar
    """
    import io
    import warnings
    import requests as req
    
    # Suprime aviso de SSL verify=False (stream interno)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": stream_url.split("?")[0],
    }
    
    buffer = b""
    start = time.time()
    
    try:
        with req.get(stream_url, headers=headers, stream=True,
                     timeout=timeout_sec, verify=False) as resp:
            if resp.status_code != 200:
                logger.warning(f"MJPEG HTTP {resp.status_code} para {stream_url[:80]}")
                return None
            
            ct = resp.headers.get("content-type", "")
            
            # Se for image/jpeg direto, é um frame único
            if "image/jpeg" in ct or "image/jpg" in ct:
                img = Image.open(io.BytesIO(resp.content))
                return np.array(img)
            
            for chunk in resp.iter_content(chunk_size=16384):
                if time.time() - start > timeout_sec:
                    break
                if not chunk:
                    continue
                buffer += chunk
                
                # Procura marcadores JPEG no buffer
                start_jpeg = buffer.find(b'\xff\xd8')
                end_jpeg = buffer.find(b'\xff\xd9')
                
                if start_jpeg >= 0 and end_jpeg >= 0 and end_jpeg > start_jpeg:
                    jpeg_data = buffer[start_jpeg:end_jpeg + 2]
                    try:
                        img = Image.open(io.BytesIO(jpeg_data))
                        img.verify()  # valida integridade
                        img = Image.open(io.BytesIO(jpeg_data))
                        return np.array(img)
                    except Exception:
                        # JPEG corrompido, continua procurando
                        buffer = buffer[end_jpeg + 2:]
                        continue
                
                # Se buffer crescer demais, limpa (mantém últimos 64KB)
                if len(buffer) > 1024 * 512:  # 512KB
                    if start_jpeg >= 0:
                        buffer = buffer[start_jpeg:]
                    else:
                        buffer = buffer[-65536:]
    except Exception as e:
        logger.warning(f"MJPEG parser error: {e}")
    
    return None


# Cor de fundo (BGR) da tela de "Reconectando..." exibida pelo player
# quando o stream cai — se a maior parte do frame capturado for essa cor,
# não é vídeo real, é a UI de erro do player renderizada pelo Playwright.
_STREAM_ERROR_BG_BGR = np.array([26, 20, 14])


def _is_stream_error_frame(frame_rgb: np.ndarray) -> bool:
    """Detecta se o frame capturado é a tela de erro do player, não vídeo real.

    Cobre dois casos: (1) a tela de "Reconectando..." (cor de fundo conhecida)
    e (2) qualquer outra tela de erro do player sobre fundo preto — ex.: a
    mensagem "codecs not supported by client" (erro de negociação WebRTC) —
    que renderiza como texto branco/cinza sobre um fundo quase todo preto.
    Câmeras de rua monitoradas aqui são todas externas/iluminadas, então
    vídeo real nunca fica com >90% dos pixels quase pretos (brilho médio
    ~4-5, como confirmado nos frames de erro reais salvos em produção).
    """
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    dist = np.linalg.norm(bgr.astype(int) - _STREAM_ERROR_BG_BGR, axis=2)
    if (dist < 15).mean() > 0.5:
        return True
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return (gray < 20).mean() > 0.9


def capture_frame(stream_url: str, timeout_sec: float = 10.0,
                   camera_code: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Captura um frame REAL do stream da câmera.
    
    O stream retorna HTML/JS renderizado no navegador (iframe).
    A estratégia principal é usar Playwright (Chromium headless)
    para abrir a página e fazer screenshot do que aparece.
    
    Args:
        stream_url: URL do stream
        timeout_sec: Tempo máximo para capturar o frame
        
    Returns:
        np.ndarray: Frame em RGB ou None se falhar
    """
    import io
    from PIL import Image

    # ─── Tentativa 0: HLS direto (sem navegador) ───────────────────
    # OpenCV/ffmpeg decodificam HLS nativamente (inclusive H.265), sem as
    # limitações de codec do WebRTC no navegador. Bem mais rápido e confiável
    # que a captura via Playwright quando disponível.
    if camera_code and camera_service:
        try:
            hls_url = camera_service.get_stream_url(camera_code)
        except Exception:
            hls_url = None
        if hls_url:
            try:
                cap = cv2.VideoCapture(hls_url)
                start = time.time()
                frame = None
                while time.time() - start < timeout_sec:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.1)
                cap.release()
                if frame is not None:
                    logger.info("✅ HLS: frame capturado com sucesso (sem navegador)!")
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"HLS direto falhou: {e}")

    # ─── Tentativa 1: Playwright (Chromium headless) ──────────────
    # O stream é HTML/JS (exibido em iframe no frontend).
    # Playwright renderiza a página e tira screenshot do vídeo/canvas.
    try:
        from playwright.sync_api import sync_playwright
        
        # Tenta até 2x: se falhar com 401, renova stream_url e tenta de novo
        for tentativa in range(2):
            current_url = stream_url
            
            with sync_playwright() as p:
                try:
                    # Google Chrome de verdade — confirmado que negocia e decodifica
                    # H.265 no WebRTC (codec real das câmeras da Tixxi). O Chromium
                    # padrão do Playwright e o Edge NÃO oferecem H.265, e o servidor
                    # rejeita a negociação com "codecs not supported by client".
                    # SEM --disable-gpu: decodificação de H.265 depende de aceleração
                    # de hardware — desabilitar a GPU faz a negociação passar mas o
                    # vídeo nunca renderiza (trava na tela de "Reconectando").
                    browser = p.chromium.launch(
                        channel="chrome",
                        headless=True,
                        args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
                    )
                except Exception:
                    logger.warning("Chrome não encontrado, usando Chromium padrão do Playwright (sem H.265)")
                    browser = p.chromium.launch(
                        headless=True,
                        args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
                    )
                context = browser.new_context(
                    viewport={"width": 800, "height": 600},
                    locale="pt-BR",
                    bypass_csp=True,
                )
                page = context.new_page()
                
                # Monitora erros de rede (401 = token expirado)
                token_expired = False
                def handle_response(resp):
                    nonlocal token_expired
                    if resp.status == 401:
                        token_expired = True
                page.on('response', handle_response)
                
                # Navega até o stream
                logger.info(f"Playwright: tentativa {tentativa+1}, abrindo stream...")
                page.goto(current_url, timeout=timeout_sec * 1000,
                          wait_until="domcontentloaded")
                
                # Aguarda React carregar e procurar video
                page.wait_for_timeout(3000)
                
                # Se detectou 401 e é a primeira tentativa, renova token e tenta de novo
                if token_expired and tentativa == 0:
                    logger.info("Token expirado, renovando stream_url...")
                    browser.close()
                    # Tenta renovar via camera_service
                    try:
                        if camera_service and camera_service.refresh():
                            cam = camera_service.get_camera_by_code(
                                stream_url.split("CODE=")[1].split("&")[0]
                                if "CODE=" in stream_url else ""
                            )
                            if cam and cam.get("stream_url"):
                                stream_url = cam["stream_url"]
                                logger.info("Stream URL renovada!")
                    except Exception:
                        pass
                    continue  # tenta de novo com nova URL
                
                # Tenta dar play no video via JS
                page.evaluate("""() => {
                    const v = document.querySelector("video");
                    if (v) {
                        v.muted = true;
                        v.play().catch(() => {});
                    }
                }""")
                # H.265 (codec real das câmeras) tem tempo de negociação/decode do
                # primeiro frame bem mais variável que VP8/H264 — uma espera fixa
                # falha de forma inconsistente (às vezes 3s bastam, às vezes não).
                # Faz polling ativo até o <video> ter dimensões reais (frame decodificado)
                # em vez de uma pausa fixa, com um teto maior pra dar tempo à negociação.
                has_video_data = page.evaluate("""async () => {
                    for (let i = 0; i < 40; i++) {
                        const v = document.querySelector("video");
                        if (v && v.videoWidth > 0 && v.videoHeight > 0) return true;
                        await new Promise(r => setTimeout(r, 500));
                    }
                    return false;
                }""")
                if not has_video_data:
                    logger.warning(f"Playwright: vídeo não decodificou a tempo (tentativa {tentativa+1})")

                # Frame direto do <video> (16:9, resolução do vídeo) — MESMO
                # espaço de coordenadas da captura contínua (live_capture).
                # O screenshot da página (800x600) mostrava o vídeo cortado
                # nas laterais, e as zonas calibradas não batiam entre os dois.
                from app.services.live_capture import _GRAB_JS, GRAB_MAX_WIDTH, _decode
                grabbed = page.evaluate(_GRAB_JS, [0.9, GRAB_MAX_WIDTH])
                frame_rgb = _decode(grabbed["data"]) if grabbed else None
                if frame_rgb is None:
                    screenshot = page.screenshot(type="jpeg", quality=90)
                    frame_rgb = np.array(Image.open(io.BytesIO(screenshot)).convert("RGB"))
                browser.close()

            if _is_stream_error_frame(frame_rgb):
                logger.warning(f"Playwright: capturou tela de 'Reconectando' (tentativa {tentativa+1}), não é vídeo real")
                continue  # tenta de novo (nova sessão) antes de cair pros outros métodos

            logger.info("✅ Playwright: frame capturado com sucesso!")
            return frame_rgb

    except Exception as e:
        logger.warning(f"Playwright falhou: {e}")
    
    # ─── Tentativa 2: MJPEG HTTP streaming ──────────────────────
    frame = _parse_mjpeg_frame(stream_url, timeout_sec)
    if frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ─── Tentativa 3: OpenCV (MJPEG/RTSP) ───────────────────────
    try:
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        start = time.time()
        frame = None
        while time.time() - start < timeout_sec:
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.1)
        
        cap.release()
        
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning(f"OpenCV não conseguiu capturar: {e}")
    
    # ─── Tentativa 4: requests direto ───────────────────────────
    try:
        import requests as req
        resp = req.get(stream_url, timeout=timeout_sec)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"Requests direto falhou: {e}")
    
    return None


# ─── Captura compartilhada (cache + lock por câmera) ─────────────
# Uma captura via Playwright leva de 5 a 60s e abre uma sessão WebRTC nova
# na Tixxi. Sem coordenação, o loop do frontend (/multi), a comparação
# (/change) e o background_monitor capturavam a MESMA câmera em paralelo —
# sessões simultâneas derrubam umas às outras (tela de "Reconectando") e
# cada ciclo pagava a captura 2-3x. Com o lock, quem chega enquanto uma
# captura está em andamento espera e reaproveita o frame recém-capturado.
FRAME_CACHE_SECONDS = 15.0

_frame_cache: dict[str, tuple[float, np.ndarray]] = {}
_frame_locks: dict[str, threading.Lock] = {}
_frame_locks_guard = threading.Lock()


def get_frame(stream_url: str, camera_code: str, timeout_sec: float = 15.0,
              max_age: float = FRAME_CACHE_SECONDS) -> Optional[np.ndarray]:
    """capture_frame com cache curto e no máximo uma captura por câmera por vez.

    Bloqueante — em rotas async, chamar via run_in_threadpool.
    max_age=0 força captura nova (ex.: print manual).
    """
    # Câmera com captura contínua (live_capture): frame de agora, sem abrir
    # navegador. max_age=0 (print manual) também aceita — o frame é de <5s.
    from app.services import live_capture
    live = live_capture.latest_frame(camera_code, max_age=5.0)
    if live is not None:
        return live

    with _frame_locks_guard:
        lock = _frame_locks.setdefault(camera_code, threading.Lock())
    with lock:
        cached = _frame_cache.get(camera_code)
        if cached and time.time() - cached[0] <= max_age:
            return cached[1].copy()
        frame = capture_frame(stream_url, timeout_sec, camera_code=camera_code)
        if frame is not None:
            _frame_cache[camera_code] = (time.time(), frame)
            return frame.copy()
        return None


# Labels de captura automática/repetitiva — sobrescrevem sempre o mesmo
# arquivo (1 por câmera+label) em vez de acumular um arquivo novo a cada
# captura. "print" (snapshot manual, botão explícito) fica de fora e
# continua gerando um arquivo por captura, já que é uma ação deliberada
# do usuário e não um loop automático.
_OVERWRITE_LABELS = {"monitoramento", "comparacao", "referencia"}


def save_camera_snapshot(frame: np.ndarray, camera_code: str,
                          camera_name: str = "",
                          label: str = "captura") -> Optional[str]:
    """
    Salva o frame capturado da câmera em assets/images/{camera_code}/.

    Para labels em _OVERWRITE_LABELS (monitoramento/comparacao/referencia),
    usa nome fixo: {camera_name}_{camera_code}_{label}.jpg — cada nova
    captura sobrescreve a anterior, evitando acumular imagens durante o
    monitoramento automático (que roda a cada poucos segundos).
    Para os demais labels (ex.: "print"), mantém o timestamp no nome.

    Args:
        frame: Imagem numpy (RGB)
        camera_code: Código da câmera
        camera_name: Nome do monumento/câmera
        label: Tipo de captura (ex: "referencia", "comparacao", "monitoramento")

    Returns:
        str: Caminho relativo salvo ou None se falhar
    """
    try:
        # Cria pasta específica da câmera
        cam_dir = IMAGES_DIR / camera_code
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Sanitiza nome
        safe_name = (camera_name or f"Camera_{camera_code}")
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in safe_name)
        safe_name = safe_name.strip().replace(" ", "_")

        if label in _OVERWRITE_LABELS:
            filename = f"{safe_name}_{camera_code}_{label}.jpg"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{camera_code}_{label}_{timestamp}.jpg"

        filepath = cam_dir / filename

        # Salva como JPEG
        cv2.imwrite(str(filepath), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        logger.info(f"📸 Snapshot salvo: {filepath}")
        return str(filepath.relative_to(IMAGES_DIR.parent.parent))  # relativo à raiz do projeto
    except Exception as e:
        logger.warning(f"Erro ao salvar snapshot: {e}")
        return None


def record_risk_alert(camera_code: str, camera_name: str, risk_alert: Optional[dict],
                      image=None) -> None:
    """Registra no alert_service um risk_alert (YOLO/preditivo), se houver."""
    if risk_alert:
        alert_service.add_alert(
            camera_code=camera_code,
            camera_name=camera_name,
            level=risk_alert["level"],
            message=risk_alert["message"],
            source="risk",
            image=image,
            objects=risk_alert.get("objects"),
        )


def record_loitering_alert(camera_code: str, camera_name: str, loitering_alert: Optional[dict]) -> None:
    """Permanência de pessoas junto ao monumento NÃO vira alerta: sentar no
    banco do Drummond por muito tempo é o uso normal do lugar. Continua só
    como informação no painel da câmera (loitering_alert na resposta)."""
    return None


def record_interaction_alert(camera_code: str, camera_name: str, interaction_alert: Optional[dict],
                             image=None) -> None:
    """Registra no alert_service um alerta de interação com a estátua (pose), se houver."""
    if interaction_alert:
        alert_service.add_alert(
            camera_code=camera_code,
            camera_name=camera_name,
            level=interaction_alert["level"],
            message=interaction_alert["message"],
            source="interaction",
            image=image,
            objects=interaction_alert.get("events"),
        )


def record_ssim_alert(camera_code: str, camera_name: str, alert: Optional[dict],
                      image=None) -> None:
    """Registra no alert_service um alert de mudança física (SSIM), se houver."""
    if alert:
        alert_service.add_alert(
            camera_code=camera_code,
            camera_name=camera_name,
            level=alert["level"],
            message=alert["message"],
            source="ssim",
            image=image,
        )


@router.get("/live/{camera_code}")
async def monitor_live(
    camera_code: str,
    confidence: Optional[float] = Query(None),
    include_image: bool = Query(True),
):
    """
    Monitora uma câmera ao vivo.
    
    Tenta capturar frame do stream para processar com YOLO.
    Se falhar (stream HTML/JS), retorna apenas a URL para o frontend
    exibir em iframe + análise baseada no último frame bem-sucedido.
    
    Retorna:
    - stream_url: URL para exibir no iframe
    - Detecção YOLO (se captura OK)
    - Score de risco de vandalismo (se captura OK)
    - Predição HF (se captura OK)
    """
    if not detection_service:
        raise HTTPException(status_code=500, detail="Serviço não inicializado")
    
    start_time = time.time()
    
    # 1. Obtém dados da câmera
    camera = camera_service.get_camera_by_code(camera_code)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Câmera {camera_code} não encontrada")
    
    camera_name = camera.get("name", f"Câmera {camera_code}")
    stream_url = camera.get("stream_url") or camera_service.get_stream_url(camera_code)
    
    from app.services import live_analysis
    live = live_analysis.latest_result(camera_code)
    if live:
        return _live_to_response(live, camera_code, include_image)

    # 2. Monta resposta base
    response = {
        "success": True,
        "camera_code": camera_code,
        "camera_name": camera_name,
        "stream_url": stream_url or "",
        "timestamp": time.time(),
        "yolo_detection": {"objects": [], "counts": {}, "total_objects": 0},
        "risk_alert": None,
        "loitering_alert": None,
        "interaction_alert": None,
        "hf_prediction": None,
        "image_base64": None,
        "frame_captured": False,
    }
    
    # 3. Tenta capturar frame (timeout curto — 3s)
    if stream_url:
        try:
            frame = await run_in_threadpool(get_frame, stream_url, camera_code)
        except Exception:
            frame = None
        
        if frame is not None:
            response["frame_captured"] = True
            
            # Pipeline de detecção
            result = detection_service.detect_full(frame, camera_id=camera_code, confidence=confidence)
            
            response["yolo_detection"] = {
                "objects": result.get("yolo_detection", {}).get("objects", []),
                "counts": result.get("yolo_detection", {}).get("counts", {}),
                "total_objects": result.get("yolo_detection", {}).get("total_objects", 0),
            }
            response["risk_alert"] = result.get("risk_alert")
            response["loitering_alert"] = result.get("loitering_alert")
            response["interaction_alert"] = result.get("interaction_alert")
            response["hf_prediction"] = result.get("hf_prediction")
            record_risk_alert(camera_code, camera_name, response["risk_alert"], image=frame)
            record_loitering_alert(camera_code, camera_name, response["loitering_alert"])
            record_interaction_alert(camera_code, camera_name, response["interaction_alert"], image=frame)
            response["snapshot_path"] = save_camera_snapshot(frame, camera_code, camera_name, "monitoramento")
            
            # Imagem anotada
            if include_image:
                annotated = result.get("yolo_detection", {}).get("annotated_image", frame)
                if annotated is not None:
                    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                                             [cv2.IMWRITE_JPEG_QUALITY, 85])
                    response["image_base64"] = base64.b64encode(buffer).decode("utf-8")
    
    elapsed = (time.time() - start_time) * 1000
    response["processing_time_ms"] = round(elapsed, 2)
    
    return response


@router.get("/multi")
async def monitor_multi(
    codes: str = Query(..., description="Códigos das câmeras separados por vírgula"),
    confidence: Optional[float] = Query(None),
    include_image: bool = Query(True, description="Inclui o frame anotado (YOLO) em base64"),
):
    """
    Monitora múltiplas câmeras em quadrante.
    Ex: /api/monitor/multi?codes=001175,000194,001963,003352
    
    Retorna análise de todas as câmeras em uma única chamada.
    """
    if not detection_service:
        raise HTTPException(status_code=500, detail="Serviço não inicializado")
    
    camera_codes = [c.strip() for c in codes.split(",") if c.strip()]
    if not camera_codes:
        raise HTTPException(status_code=400, detail="Nenhum código de câmera informado")
    
    from app.services import live_analysis

    results = []
    for code in camera_codes[:MAX_BATCH_CAMERAS]:
        # Câmera com vídeo contínuo: a análise (YOLO + rastreamento) já roda
        # sozinha a cada ~0,5s — devolve o resultado pronto em vez de
        # capturar e detectar de novo (e sem registrar alertas duplicados)
        live = live_analysis.latest_result(code)
        if live:
            results.append(_live_to_response(live, code, include_image))
            continue

        cam_result = {
            "camera_code": code,
            "success": False,
            "yolo_detection": {"objects": [], "counts": {}, "total_objects": 0},
            "risk_alert": None,
            "loitering_alert": None,
            "interaction_alert": None,
            "hf_prediction": None,
            "frame_captured": False,
        }
        
        try:
            camera = camera_service.get_camera_by_code(code)
            cam_result["camera_name"] = camera.get("name", f"Câmera {code}") if camera else f"Câmera {code}"
            
            stream_url = camera.get("stream_url") if camera else None
            if not stream_url and camera:
                stream_url = camera_service.get_stream_url(code)
            cam_result["stream_url"] = stream_url or ""
            
            if stream_url:
                try:
                    frame = await run_in_threadpool(get_frame, stream_url, code)
                except Exception:
                    frame = None
                
                if frame is not None:
                    cam_result["frame_captured"] = True
                    # 📸 Salva snapshot do frame em assets/images/
                    cam_name = cam_result.get("camera_name", f"Câmera {code}")
                    cam_result["snapshot_path"] = save_camera_snapshot(frame, code, cam_name, "monitoramento")
                    
                    result = detection_service.detect_full(frame, camera_id=code, confidence=confidence)
                    cam_result["yolo_detection"] = {
                        "objects": result.get("yolo_detection", {}).get("objects", []),
                        "counts": result.get("yolo_detection", {}).get("counts", {}),
                        "total_objects": result.get("yolo_detection", {}).get("total_objects", 0),
                    }
                    cam_result["risk_alert"] = result.get("risk_alert")
                    cam_result["loitering_alert"] = result.get("loitering_alert")
                    cam_result["interaction_alert"] = result.get("interaction_alert")
                    cam_result["hf_prediction"] = result.get("hf_prediction")
                    record_risk_alert(code, cam_name, cam_result["risk_alert"], image=frame)
                    record_loitering_alert(code, cam_name, cam_result["loitering_alert"])
                    record_interaction_alert(code, cam_name, cam_result["interaction_alert"], image=frame)

                    # Frame anotado — o frontend exibe esse print no lugar do
                    # vídeo ao vivo (modo "Snapshots"), sem abrir conexão
                    # contínua com a câmera no navegador.
                    if include_image:
                        annotated = result.get("yolo_detection", {}).get("annotated_image", frame)
                        if annotated is not None:
                            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
                            cam_result["image_base64"] = base64.b64encode(buffer).decode("utf-8")
                    cam_result["timestamp"] = time.time()
            
            cam_result["success"] = True
            
        except Exception as e:
            logger.error(f"Erro ao processar câmera {code}: {e}")
            cam_result["error"] = str(e)
        
        results.append(cam_result)
    
    return {
        "success": True,
        "timestamp": time.time(),
        "cameras": results,
        "total": len(results),
    }


@router.get("/demo")
async def monitor_demo(
    cenario: str = Query("depredacao", description="Cenário: furto, depredacao, multidao, veiculo_suspeito"),
    camera_code: str = Query("001175", description="Código da câmera para exibir no resultado"),
):
    """
    Modo demonstração: gera uma cena sintética de vandalismo e executa
    todo o pipeline de detecção (YOLO + Vandalismo + HF).
    
    Retorna imagem anotada com bounding boxes + análise completa.
    Útil para testar e visualizar o funcionamento do sistema.
    """
    from app.models.demo_scene import gerar_cena_demo
    
    # Gera cena sintética
    frame = gerar_cena_demo(cenario)
    
    # Executa pipeline completo
    result = detection_service.detect_full(frame, camera_id=camera_code, confidence=0.25)
    
    # Anotação
    import base64
    annotated = result.get("yolo_detection", {}).get("annotated_image", frame)
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_b64 = base64.b64encode(buffer).decode("utf-8")
    
    camera = camera_service.get_camera_by_code(camera_code)
    camera_name = camera.get("name", f"Câmera {camera_code}") if camera else f"Câmera {camera_code}"
    
    return {
        "success": True,
        "camera_code": camera_code,
        "camera_name": camera_name,
        "cenario": cenario,
        "descricao": gerar_cena_demo.__doc__.strip() if hasattr(gerar_cena_demo, "__doc__") else "",
        "frame_captured": True,
        "image_base64": image_b64,
        "yolo_detection": {
            "objects": result.get("yolo_detection", {}).get("objects", []),
            "counts": result.get("yolo_detection", {}).get("counts", {}),
            "total_objects": result.get("yolo_detection", {}).get("total_objects", 0),
        },
        "risk_alert": result.get("risk_alert"),
        "hf_prediction": result.get("hf_prediction"),
        "processing_time_ms": result.get("processing_time_ms", 0),
    }


@router.get("/streams")
async def get_streams(
    codes: str = Query(..., description="Códigos separados por vírgula"),
):
    """
    Rápido: retorna apenas as URLs de stream para exibir nos iframes.
    Sem captura de frame, sem YOLO — apenas as URLs.
    """
    if not camera_service:
        raise HTTPException(status_code=500, detail="Serviço não inicializado")

    camera_codes = [c.strip() for c in codes.split(",") if c.strip()]
    result = []

    for code in camera_codes[:MAX_BATCH_CAMERAS]:
        camera = camera_service.get_camera_by_code(code)
        name = camera.get("name", f"Câmera {code}") if camera else f"Câmera {code}"
        stream_url = camera.get("stream_url") if camera else None
        if not stream_url:
            stream_url = camera_service.get_stream_url(code)

        # URL HLS (protocolo real da nossa conta na Tixxi) — usada pelo player
        # <video>+hls.js do frontend. stream_url (WebRTC) fica só como fallback
        # pro iframe antigo enquanto o HLS não estiver 100% liberado pela Tixxi.
        try:
            hls_url = camera_service.get_stream_url(code)
        except Exception:
            hls_url = None

        result.append({
            "camera_code": code,
            "camera_name": name,
            "stream_url": stream_url or "",
            "hls_url": hls_url or "",
            # "html" (página com player, precisa de <iframe>) ou "raw" (MJPEG
            # puro, precisa de <img>) — vem direto da Tixxi, indica como
            # renderizar stream_url no frontend.
            "stream_type": (camera.get("stream_type") if camera else None) or "html",
        })

    return {"success": True, "cameras": result}


@router.get("/camera-status")
async def get_camera_status(
    codes: str = Query(..., description="Códigos separados por vírgula"),
):
    """
    Status de vídeo por câmera, baseado no último resultado do monitoramento
    em background (não dispara captura nova) — usado pelo frontend pra
    decidir se recarrega o iframe (que é de outra origem, então JS não
    consegue inspecionar o conteúdo dele diretamente).
    """
    from app.services import background_monitor

    status = background_monitor.get_camera_status()
    camera_codes = [c.strip() for c in codes.split(",") if c.strip()]

    result = {}
    for code in camera_codes:
        info = status.get(code)
        result[code] = {
            "has_video": info["has_video"] if info else None,  # None = ainda não checado
            "checked_at": info["checked_at"] if info else None,
        }

    return {"success": True, "cameras": result}


# ─── Snapshot Manual ────────────────────────────────────────────

@router.post("/snapshot/{camera_code}")
async def capture_snapshot(
    camera_code: str,
):
    """
    CAPTURA um print do que a câmera está mostrando AGORA
    e salva em assets/images/ com o nome do monumento.
    
    Retorna o caminho do arquivo salvo para visualização.
    """
    if not camera_service:
        raise HTTPException(status_code=500, detail="Serviço não inicializado")
    
    camera = camera_service.get_camera_by_code(camera_code)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Câmera {camera_code} não encontrada")
    
    camera_name = camera.get("name", f"Câmera {camera_code}")
    stream_url = camera.get("stream_url") or camera_service.get_stream_url(camera_code)
    
    if not stream_url:
        raise HTTPException(status_code=400, detail="Câmera não possui stream URL")
    
    # Captura frame real (sem fallback sintético)
    frame = await run_in_threadpool(get_frame, stream_url, camera_code, 20.0, 0)
    
    if frame is None:
        return {
            "success": False,
            "error": "Não foi possível capturar frame da câmera. "
                     "Verifique se o stream está acessível.",
            "camera_code": camera_code,
            "camera_name": camera_name,
        }
    
    # Salva snapshot
    snapshot_path = save_camera_snapshot(frame, camera_code, camera_name, "print")
    
    # Codifica como base64 pra exibição imediata
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, 92])
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return {
        "success": True,
        "camera_code": camera_code,
        "camera_name": camera_name,
        "snapshot_path": snapshot_path,
        "image_base64": image_base64,
        "timestamp": time.time(),
        "message": f"📸 Print salvo: {snapshot_path}",
    }


# ─── Detecção de Mudanças no Monumento (SSIM + HF) ─────────────

@router.post("/change/{camera_code}/reference")
async def set_reference(
    camera_code: str,
    usar_demo: bool = Query(False, description="Forçar demo se captura falhar"),
):
    """
    CAPTURA frame da câmera e define como REFERÊNCIA do monumento.
    
    Extrai automaticamente o ROI (região da estátua) e armazena.
    Comparações futuras usarão esta imagem como base para detectar
    pichação, danos, peças removidas, etc.
    """
    if not detection_service or not camera_service:
        raise HTTPException(status_code=500, detail="Serviço não inicializado")

    # Carrega HF model em background
    if not detection_service.hf_vandalism.model_loaded:
        try:
            detection_service.hf_vandalism.load_model()
        except Exception:
            pass

    # Tenta capturar frame real da câmera
    frame = None
    camera = camera_service.get_camera_by_code(camera_code)
    if camera:
        stream_url = camera.get("stream_url") or camera_service.get_stream_url(camera_code)
        if stream_url:
            try:
                frame = await run_in_threadpool(get_frame, stream_url, camera_code)
            except Exception:
                pass

    if frame is None:
        # Sem fallback pra cena de demonstração: referência sintética faria
        # todo print real seguinte parecer "monumento alterado"
        logger.warning(f"Não foi possível capturar frame da câmera {camera_code} — referência não alterada")
        return {"success": False, "error": "Não foi possível capturar a câmera — referência não alterada"}

    # 📸 Salva snapshot da referência em assets/images/
    camera_name_clean = camera.get("name", f"Câmera {camera_code}") if camera else f"Câmera {camera_code}"
    save_camera_snapshot(frame, camera_code, camera_name_clean, "referencia")

    result = change_detector.set_reference(camera_code, frame, detection_service)

    # Codifica frame de referência pra exibição
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, 85])
    result["reference_image_base64"] = base64.b64encode(buffer).decode("utf-8")

    return result


@router.get("/change/{camera_code}")
async def detect_changes(
    camera_code: str,
    usar_demo: bool = Query(False, description="Forçar demo se captura falhar"),
):
    """
    VERIFICA se houve mudança no monumento.
    
    1. Captura frame atual da câmera
    2. Extrai ROI da estátua
    3. Compara com referência via SSIM
    4. Se mudança detectada, executa HF model no ROI
    
    Retorna:
    - similarity_score: (1=idêntico)
    - change_percentage: % do ROI alterado
    - highlight_image: imagem com diferenças em vermelho
    - hf_prediction: probabilidades do HF model
    - alert: alerta se mudança significativa ou roubo/vandalismo
    """
    if not detection_service or not camera_service:
        raise HTTPException(status_code=500, detail="Serviço não inicializado")

    # Captura frame atual pra comparação
    current_frame = None
    camera = camera_service.get_camera_by_code(camera_code)
    if camera:
        stream_url = camera.get("stream_url") or camera_service.get_stream_url(camera_code)
        if stream_url:
            try:
                current_frame = await run_in_threadpool(get_frame, stream_url, camera_code)
            except Exception:
                pass

    # Sem frame real não compara: a cena de demonstração contra a referência
    # real (ou o contrário) dava "monumento X% alterado" falso
    if current_frame is None:
        return {"success": False, "error": "Não foi possível capturar a câmera — comparação não realizada"}

    # 📸 Salva snapshot da comparação em assets/images/
    camera_name_comp = camera.get("name", f"Câmera {camera_code}") if camera else f"Câmera {camera_code}"
    save_camera_snapshot(current_frame, camera_code, camera_name_comp, "comparacao")

    # Detecta elementos passageiros (pessoas/veículos, YOLO) no frame atual
    # pra excluir essas áreas do cálculo de % alterado — alguém passando ou
    # um carro na rua não pode virar "dano" no monumento
    ignore_boxes = []
    yolo_objects = None
    try:
        yolo_result = detection_service.yolo_detector.detect(current_frame, camera_code=camera_code)
        yolo_objects = yolo_result.get("objects", [])
        ignore_boxes = [
            d["bbox"] for d in yolo_result.get("objects", [])
            if d["class_name"] in TRANSIENT_CLASSES and not d.get("is_statue")
        ]
    except Exception as e:
        logger.warning(f"Erro YOLO (elementos passageiros) no change detection: {e}")

    # Sem referência: cria com este frame, mas só com a estátua livre — uma
    # referência com turista na frente faria todo print seguinte parecer alterado
    if camera_code not in change_detector.monitored:
        if covered_fraction(current_frame, camera_code, ignore_boxes) >= 0.05:
            return {"success": False, "error": "Aguardando o monumento ficar livre de pessoas para criar a referência"}
        change_detector.set_reference(camera_code, current_frame, detection_service, objects=yolo_objects)
        return {"success": False, "error": "Referência criada — a comparação começa no próximo ciclo"}

    result = change_detector.check(camera_code, current_frame, detection_service, ignore_boxes=ignore_boxes)
    if not result.get("success"):
        return result

    camera_name_alert = camera.get("name", f"Câmera {camera_code}") if camera else f"Câmera {camera_code}"
    record_ssim_alert(camera_code, camera_name_alert, result["alert"], image=current_frame)

    return {
        "success": True,
        "camera_code": camera_code,
        "timestamp": result["timestamp"],
        "similarity_score": result["similarity_score"],
        "change_percentage": result["change_percentage"],
        "change_regions": result["change_regions"],
        "significant_changes": result["significant_changes"],
        "visible_percentage": result.get("visible_percentage"),
        "ssim_alert_level": result["ssim_alert_level"],
        "confirmation": result["confirmation"],
        "hf_prediction": result["hf_prediction"],
        "alert": result["alert"],
        "alert_level": result["alert_level"],
        "ignored_objects_count": result["ignored_objects_count"],
        "ignored_objects_alert": result["ignored_objects_alert"],
        "highlight_image_base64": result["highlight_image_base64"],
        "reference_roi_base64": result.get("reference_roi_base64"),
        "current_roi_base64": result.get("current_roi_base64"),
        "compared_area": result.get("compared_area"),
        "reference_time": result.get("reference_time"),
    }


# ─── Calibração de Zona (quadrante por câmera) ─────────────────

@router.get("/zone/{camera_code}")
async def get_zone(camera_code: str):
    """Retorna a zona (quadrante) calibrada da câmera, ou o padrão."""
    return {
        "success": True,
        "camera_code": camera_code,
        "zone": zone_service.get_zone(camera_code),
        "is_custom": zone_service.is_custom(camera_code),
    }


@router.put("/zone/{camera_code}")
async def put_zone(camera_code: str, zone: ZoneInput):
    """Salva a zona (quadrante) calibrada da câmera."""
    try:
        saved = zone_service.set_zone(camera_code, zone.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "camera_code": camera_code, "zone": saved}


@router.delete("/zone/{camera_code}")
async def delete_zone(camera_code: str):
    """Remove a calibração custom da câmera, voltando ao quadrante padrão."""
    default = zone_service.reset_zone(camera_code)
    return {"success": True, "camera_code": camera_code, "zone": default}


# ─── Captura contínua (fase 1) + análise contínua (fase 2) ──────

def _live_to_response(live: dict, code: str, include_image: bool) -> dict:
    """Resultado da análise contínua no mesmo formato de /live e /multi."""
    from app.services import live_analysis
    resp = {
        "success": True,
        "camera_code": code,
        "camera_name": live["camera_name"],
        "stream_url": "",
        "timestamp": live["timestamp"],
        "frame_captured": True,
        "live": True,  # veio da análise contínua (vídeo), não de print
        "yolo_detection": live["yolo_detection"],
        "risk_alert": live["risk_alert"],
        "interaction_alert": live["interaction_alert"],
        "loitering_alert": live["loitering_alert"],
        # Superfície protegida: alerta do instante, último evento (com
        # evidências), estado da comparação e IDs suspeitos recentes
        "surface_alert": live["surface_alert"],
        "surface_event": live["surface_event"],
        "surface": live["surface"],
        "suspects": live["suspects"],
        "people_near_statue": live["people_near_statue"],
        "hf_prediction": None,
        "processing_time_ms": live["processing_time_ms"],
    }
    if include_image:
        jpg = live_analysis.latest_jpeg(code)
        resp["image_base64"] = base64.b64encode(jpg).decode("utf-8") if jpg else None
    return resp


@router.get("/live-capture/status")
async def live_capture_status():
    """FPS real, reconexões, idade do último frame e RAM/CPU do Chrome por câmera."""
    from app.services import live_capture
    from app.services import live_analysis
    return {"success": True, **live_capture.status(), "analysis": live_analysis.status()}


@router.get("/live-analysis/{camera_code}/frame.jpg")
async def live_analysis_frame(camera_code: str):
    """Último frame ANALISADO (caixas do YOLO + IDs do rastreamento + contorno)."""
    from fastapi.responses import Response
    from app.services import live_analysis
    jpg = live_analysis.latest_jpeg(camera_code)
    if jpg is None:
        raise HTTPException(status_code=404, detail="Sem análise recente para esta câmera")
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/live-analysis/{camera_code}/stream.mjpg")
async def live_analysis_stream(camera_code: str, request: Request):
    """
    Frames ANALISADOS em MJPEG (multipart/x-mixed-replace): cada análise nova
    é empurrada assim que fica pronta. Caixas e vídeo são o MESMO frame —
    nada de sobrepor detecção velha no vídeo ao vivo (o "fantasma").
    """
    import asyncio
    from fastapi.responses import StreamingResponse
    from app.services import live_analysis

    if live_analysis.latest_jpeg_seq(camera_code) is None:
        raise HTTPException(status_code=404, detail="Sem análise recente para esta câmera")

    async def frames():
        last_seq, idle_since = -1, time.time()
        while not await request.is_disconnected():
            item = live_analysis.latest_jpeg_seq(camera_code)
            if item and item[0] != last_seq:
                last_seq, idle_since = item[0], time.time()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(item[1])).encode() + b"\r\n\r\n" + item[1] + b"\r\n")
            elif time.time() - idle_since > 30:
                return  # análise parou (câmera caiu): encerra; o <img> dispara onError
            await asyncio.sleep(0.05)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@router.get("/surface-events")
async def surface_events(camera_code: Optional[str] = Query(None), limit: int = Query(20, ge=1, le=200)):
    """Eventos de superfície (pessoa suspeita + mudança) com URLs das evidências."""
    from app.services import evidence_service
    events = await run_in_threadpool(evidence_service.list_events, camera_code, limit)
    return {"success": True, "total": len(events), "events": events}


@router.get("/live-capture/{camera_code}/frame.jpg")
async def live_capture_frame(camera_code: str):
    """Último frame da captura contínua (404 se a câmera não tem worker ou está sem vídeo)."""
    from fastapi.responses import Response
    from app.services import live_capture
    frame = live_capture.latest_frame(camera_code, max_age=10.0)
    if frame is None:
        raise HTTPException(status_code=404, detail="Sem frame recente para esta câmera")
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/simulate/scenarios")
async def simulate_scenarios():
    from app.services import damage_sim
    return {"success": True, "scenarios": [{"id": k, "label": v} for k, v in damage_sim.SCENARIOS.items()]}


@router.post("/simulate/{camera_code}")
async def simulate_damage(camera_code: str, cenario: str = Query(..., description="pichacao, tinta, quebra_cabeca, ...")):
    """
    TESTE da detecção: aplica um dano simulado na estátua sobre a imagem
    ATUAL da câmera e roda a mesma comparação que decide o alerta — numa
    CÓPIA do estado da câmera. Não registra alerta, não troca referência e
    não interfere no monitoramento real.
    """
    import copy
    from app.models import change_detector as cdm
    from app.models import statue_compare
    from app.services import damage_sim

    if cenario not in damage_sim.SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Cenário inválido: {cenario}")
    state = change_detector.monitored.get(camera_code)
    if not state or "reference_frame" not in state:
        raise HTTPException(status_code=409, detail="Esta câmera ainda não tem referência da estátua. "
                                                    "Aguarde o monitoramento criar (com a estátua livre) e tente de novo.")
    frame = await _fresh_frame(camera_code)
    if frame is None:
        raise HTTPException(status_code=503, detail="Não foi possível capturar a imagem atual da câmera")

    def run():
        ref_gray = state["reference_gray"]
        h, w = ref_gray.shape[:2]
        cur = cv2.resize(frame, (w, h)) if frame.shape[:2] != (h, w) else frame
        cfg = zone_service.get_statue(camera_code)
        sens = cfg.get("sensitive")
        sens_px = (int(sens["x_start"] * w), int(sens["y_start"] * h), int(sens["x_end"] * w),
                   int(sens["y_end"] * h)) if sens else None

        def measure(img):
            sim = cdm.ChangeDetector()
            st = copy.deepcopy(state)
            st["streak"] = []
            learning = st.get("learn_n", 0) < cdm.LEARN_CHECKS
            st["learn_n"] = cdm.LEARN_CHECKS  # mede direto, sem a fase de aprendizado
            st["reference_time"] = time.time()  # não renova a referência na cópia
            sim.monitored[camera_code] = st
            objs = detection_service.yolo_detector.detect(img, camera_code=camera_code, draw=False)["objects"]
            ign = [d["bbox"] for d in objs if d["class_name"] in TRANSIENT_CLASSES and not d.get("is_statue")]
            r = sim.check(camera_code, img, None, ignore_boxes=ign)
            r["still_learning"] = learning
            return r

        # Base do teste: a imagem ATUAL. Se agora não dá pra comparar (gente
        # na frente da estátua, câmera fora de posição), usa a imagem de
        # REFERÊNCIA (estátua livre) — o teste sempre mostra o dano.
        base, source, why = None, "atual", None
        M, info = statue_compare.align(ref_gray, cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY), state["area"])
        if M is None:
            why = f"a imagem atual não alinhou com a referência ({info.get('reason')})"
        else:
            aligned = cv2.warpAffine(cur, M, (w, h))
            a = measure(aligned)
            if a.get("success"):
                base = aligned
            else:
                why = (a.get("error") or "comparação indisponível agora").split(" — ")[0]
        if base is None:
            base, source = state["reference_frame"], "referencia"
            a = measure(base)
        damaged = damage_sim.apply(base, state["silhouette"], cenario, sens_px)
        b = measure(damaged)

        # Antes/depois com a estátua e o entorno (a imagem da câmera como
        # ela é, com o dano aplicado), contorno da área comparada em amarelo
        x1, y1, x2, y2 = state["area"]
        px, py = int((x2 - x1) * 0.6), int((y2 - y1) * 0.25)
        cx1, cy1, cx2, cy2 = max(x1 - px, 0), max(y1 - py, 0), min(x2 + px, w), min(y2 + py, h)
        edge = cv2.morphologyEx(state["silhouette"].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

        def view(img):
            v = img.copy()
            v[edge] = (255, 215, 0)
            v = v[cy1:cy2, cx1:cx2]
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(v, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode("utf-8")

        return {"sem_dano": a, "com_dano": b, "source": source, "why": why,
                "before": view(base), "after": view(damaged)}

    res = await run_in_threadpool(run)
    if "error" in res:
        raise HTTPException(status_code=422, detail=res["error"])
    a, b = res["sem_dano"], res["com_dano"]
    if not b.get("success"):
        raise HTTPException(status_code=422, detail=b.get("error") or "Comparação não pôde ser feita agora")
    level = b["ssim_alert_level"]
    would_alert = cdm._LEVEL_RANK.get(level, 0) >= cdm._LEVEL_RANK["ALTO"]
    before = a.get("change_percentage") if a.get("success") else None
    return {
        "success": True,
        "camera_code": camera_code,
        "cenario": cenario,
        "cenario_label": damage_sim.SCENARIOS[cenario],
        "sem_dano_pct": before,
        "com_dano_pct": b["change_percentage"],
        "increase_pct": b["confirmation"]["increase_pct"],
        "level": level,
        "would_alert": would_alert,
        "still_learning": b.get("still_learning"),
        # "atual" = dano aplicado na imagem de agora; "referencia" = agora não
        # dava (motivo em source_reason), usou a imagem de referência
        "source": res["source"],
        "source_reason": res["why"],
        "before_base64": res["before"],
        "after_base64": res["after"],
        "verdict": (f"DETECTADO: {damage_sim.SCENARIOS[cenario]} gera nível {level} "
                    f"({b['change_percentage']:.1f}% da estátua alterada). Em operação, o alerta sai quando a "
                    f"alteração se mantém por {cdm.SSIM_CONFIRM_CHECKS} comparações seguidas "
                    f"(≥{int(cdm.SSIM_CONFIRM_SECONDS // 60)} min)." if would_alert else
                    f"NÃO detectado como alerta: {damage_sim.SCENARIOS[cenario]} deu nível {level} "
                    f"({b['change_percentage']:.1f}% alterada) — abaixo do limite de ALTO."),
        "reference_roi_base64": b.get("reference_roi_base64"),
        "current_roi_base64": b.get("current_roi_base64"),
        "highlight_image_base64": b.get("highlight_image_base64"),
        "compared_area": b.get("compared_area"),
    }


class AiPrintInput(BaseModel):
    # Imagem anotada que o operador está vendo (print + IA). Vazio = usa o
    # último frame analisado da câmera (vídeo contínuo)
    image_base64: Optional[str] = None


@router.post("/save-ai-print/{camera_code}")
async def save_ai_print(camera_code: str, payload: AiPrintInput):
    """
    Salva a imagem da câmera COM as marcações da IA (caixas, classe e
    confiança de cada detecção, estátua, zona) em assets/images/<código>/,
    com data e hora no nome — pra registro/evidência do monitoramento.
    """
    jpg = None
    if payload.image_base64:
        try:
            jpg = base64.b64decode(payload.image_base64.split(",")[-1])
        except Exception:
            raise HTTPException(status_code=400, detail="Imagem inválida")
    else:
        from app.services import live_analysis
        jpg = live_analysis.latest_jpeg(camera_code, max_age=30)
    if not jpg:
        raise HTTPException(status_code=404, detail="Sem imagem analisada pela IA para esta câmera")
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Imagem inválida")
    camera = camera_service.get_camera_by_code(camera_code) if camera_service else None
    camera_name = camera.get("name", f"Câmera {camera_code}") if camera else f"Câmera {camera_code}"
    path = save_camera_snapshot(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), camera_code, camera_name, "print_ia")
    if not path:
        raise HTTPException(status_code=500, detail="Não foi possível salvar a imagem")
    return {"success": True, "camera_code": camera_code, "path": path,
            "url": "/" + path.replace("\\", "/").split("backend/", 1)[-1]}


@router.get("/statue/{camera_code}")
async def get_statue(camera_code: str):
    """Contorno da estátua + área sensível da câmera (null = não calibrado)."""
    return {"success": True, "camera_code": camera_code, **zone_service.get_statue(camera_code)}


@router.put("/statue/{camera_code}")
async def set_statue(camera_code: str, payload: StatueInput):
    try:
        entry = zone_service.set_statue(
            camera_code,
            payload.statue.model_dump(),
            payload.sensitive.model_dump() if payload.sensitive else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # A referência do SSIM foi recortada pela área antiga — descarta pra
    # ser recriada com o contorno novo no próximo ciclo.
    change_detector.monitored.pop(camera_code, None)

    # O que está dentro do contorno É o monumento: grava a aparência da
    # estátua a partir da MESMA imagem em que o operador desenhou (a da
    # calibração); sem ela, captura uma atual. O YOLO não tem classe
    # "estátua" — sem isso a figura de bronze sai como "pessoa".
    from app.services import statue_memory
    cached = _calibration_frames.get(camera_code)
    frame = cached[1] if cached and time.time() - cached[0] <= CALIBRATION_FRAME_MAX_AGE else None
    if frame is None:
        frame = await _fresh_frame(camera_code)
    memory = statue_memory.save(camera_code, frame, payload.statue.model_dump()) if frame is not None         else {"saved": False, "reason": "não foi possível capturar a câmera"}
    return {"success": True, "camera_code": camera_code, **entry, "statue_appearance": memory}


@router.delete("/statue/{camera_code}")
async def reset_statue(camera_code: str):
    from app.services import statue_memory
    change_detector.monitored.pop(camera_code, None)
    statue_memory.clear(camera_code)
    return {"success": True, "camera_code": camera_code, **zone_service.reset_statue(camera_code)}


# Frame mostrado na calibração (código → (ts, frame RGB)). Ao salvar o
# contorno da estátua, a aparência é gravada a partir DESTA imagem — a
# mesma em que o operador desenhou.
_calibration_frames: dict[str, tuple[float, np.ndarray]] = {}
CALIBRATION_FRAME_MAX_AGE = 1800


async def _fresh_frame(camera_code: str) -> Optional[np.ndarray]:
    """Captura AGORA (sem cache); None se a câmera não responder."""
    camera = camera_service.get_camera_by_code(camera_code)
    if not camera:
        return None
    stream_url = camera.get("stream_url") or camera_service.get_stream_url(camera_code)
    if not stream_url:
        return None
    try:
        return await run_in_threadpool(get_frame, stream_url, camera_code, 15.0, 0)
    except Exception:
        return None


@router.get("/zone/{camera_code}/frame")
async def get_zone_frame(camera_code: str):
    """
    Frame ATUAL da câmera (captura nova, sem cache e sem imagem de
    demonstração) para desenhar a calibração. Se a câmera não responder,
    devolve erro — calibrar em cima de imagem antiga ou sintética deixaria
    o contorno no lugar errado.
    """
    if not camera_service:
        raise HTTPException(status_code=500, detail="Serviço não inicializado")

    frame = await _fresh_frame(camera_code)
    if frame is None:
        raise HTTPException(status_code=503, detail="Não foi possível capturar a imagem atual da câmera. Tente novamente.")
    captured_at = time.time()
    _calibration_frames[camera_code] = (captured_at, frame)

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {
        "success": True,
        "camera_code": camera_code,
        "frame_captured": True,
        "captured_at": captured_at,
        "image_base64": base64.b64encode(buffer).decode("utf-8"),
    }


@router.get("/change/{camera_code}/history")
async def change_history(camera_code: str):
    """Histórico de verificações de mudanças no monumento"""
    return {
        "success": True,
        "camera_code": camera_code,
        "history": change_detector.get_history(camera_code),
    }


# ─── Utilitário ─────────────────────────────────────────────────

def _create_monument_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Cria frame sintético de monumento para fallback"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (0, 0), (width, height // 3), (200, 210, 220), -1)
    cv2.rectangle(img, (0, height * 2 // 3), (width, height), (160, 160, 150), -1)
    # Monumento no centro
    cx, cy = width // 2, height // 2
    cv2.rectangle(img, (cx - 60, cy - 60), (cx + 60, cy + 60), (140, 130, 120), -1)
    cv2.circle(img, (cx, cy - 90), 35, (160, 150, 140), -1)
    return img

