"""
Ação contra a estátua, via YOLO-pose (keypoints COCO) no vídeo contínuo.

Tocar, abraçar, segurar a mão do Drummond pra foto, sentar ou subir no
pedestal é o uso normal de um monumento turístico — NÃO é alerta. O que
se procura é o GOLPE: o punho (ou o pé) se move bruscamente
(STRIKE_SPEED, em alturas do corpo por segundo), na direção da figura, e
termina SOBRE ela — bater, chutar, raspar com objeto.

  - golpes da MESMA pessoa (ID do ByteTrack) dentro de
    STRIKE_WINDOW_SECONDS: a partir de STRIKE_MIN_COUNT → ALTO;
  - com objeto na mão (garrafa, taco, faca...) um golpe já basta;
  - antes disso, MODERADO só no vídeo (não vai pro histórico).

A prova do dano continua sendo a comparação da estátua com a referência
(surface_monitor / change_detector): uma alteração física confirmada até
INTERACTION_MEMORY_SECONDS depois de um golpe confirmado sobe pra CRÍTICO.

Nos prints (câmeras sem vídeo contínuo) não há como medir movimento —
só a comparação com a referência vale.
"""

import logging
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import (
    POSE_MODEL,
    POSE_KEYPOINT_CONF,
    STRIKE_SPEED,
    STRIKE_MAX_JUMP,
    STRIKE_WINDOW_SECONDS,
    STRIKE_MIN_COUNT,
    INTERACTION_GRACE_SECONDS,
    STATUE_SELF_IOU,
)

logger = logging.getLogger(__name__)

# Índices dos keypoints COCO-17
WRISTS = (9, 10)
ANKLES = (15, 16)
# Nos prints (sem vídeo) a mesma situação reaparece minutos depois: tolera
# esse intervalo sem zerar a contagem
PRINT_GRACE_SECONDS = 300
# Intervalo máximo entre duas poses da mesma pessoa pra medir velocidade
MAX_STEP_SECONDS = 1.5

# Último golpe CONFIRMADO por câmera (timestamp) — lido pela change_detector
# e pela análise contínua (alteração física logo depois → CRÍTICO)
_last_interaction: dict[str, float] = {}
# (câmera, pessoa, keypoint) → (ts, x, y) da última posição do punho/pé
_limbs: dict[tuple[str, str, int], tuple[float, float, float]] = {}
# (câmera, pessoa) → deque[(ts, com_objeto)] dos golpes recentes
_strikes: dict[tuple[str, str], deque] = {}
_lock = threading.Lock()


def last_interaction(camera_code: str) -> Optional[float]:
    with _lock:
        return _last_interaction.get(camera_code)


def _expire(camera_code: str, now: float, grace: float) -> None:
    with _lock:
        for key in [k for k, v in _limbs.items() if k[0] == camera_code and now - v[0] > grace]:
            del _limbs[key]
        for key in [k for k in _strikes if k[0] == camera_code]:
            q = _strikes[key]
            while q and now - q[0][0] > STRIKE_WINDOW_SECONDS:
                q.popleft()
            if not q:
                del _strikes[key]


def clear(camera_code: str, grace: float = INTERACTION_GRACE_SECONDS) -> None:
    """Ninguém junto da estátua neste frame/print: expira as contagens."""
    _expire(camera_code, time.time(), grace)


def _inside(pt, box, margin: float = 0.0) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return x1 - mx <= x <= x2 + mx and y1 - my <= y <= y2 + my


def _overlap_frac(a, b) -> float:
    """Fração da caixa `a` que está dentro da caixa `b`."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1)
    return (ix2 - ix1) * (iy2 - iy1) / area_a


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def is_statue_itself(bbox, statue_px) -> bool:
    """Detecção de 'pessoa' que é a própria estátua (figura humana).

    IoU alto com o contorno, ou caixa inteira dentro do contorno ocupando
    boa parte dele (estátuas pequenas no quadro, onde a caixa do YOLO sai
    mais justa que o contorno calibrado e o IoU fica baixo).
    """
    if not statue_px:
        return False
    if iou(bbox, statue_px) >= STATUE_SELF_IOU:
        return True
    area_box = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1)
    area_statue = max((statue_px[2] - statue_px[0]) * (statue_px[3] - statue_px[1]), 1)
    return _overlap_frac(bbox, statue_px) >= 0.9 and area_box / area_statue >= 0.3


class InteractionAnalyzer:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_failed = False
        # Modelos ultralytics não são thread-safe — a análise contínua
        # (live_analysis) e as rotas/background usam o mesmo analisador
        self._infer_lock = threading.Lock()

    def load_model(self) -> bool:
        if self.model_loaded or self._load_failed:
            return self.model_loaded
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(Path(POSE_MODEL)))  # ultralytics baixa se não existir
            self.model_loaded = True
            logger.info(f"Modelo YOLO-pose carregado: {POSE_MODEL}")
        except Exception as e:
            # Não tenta de novo a cada print (download bloqueado etc.)
            self._load_failed = True
            logger.error(f"YOLO-pose indisponível — análise de interação desativada: {e}")
        return self.model_loaded

    def analyze(self, image: np.ndarray, camera_code: str,
                statue_px: tuple,
                annotated: Optional[np.ndarray] = None,
                statue_boxes: Optional[list] = None,
                people: Optional[list[tuple[list, Optional[int]]]] = None,
                objects: Optional[list] = None,
                continuous: bool = False) -> Optional[dict]:
        """
        Roda pose num recorte ao redor da estátua (pessoas pequenas no
        quadro ganham resolução) e procura golpes contra a figura.

        statue_boxes: caixas da própria estátua (figura) — o golpe precisa
            terminar sobre ela; sem elas, vale o contorno calibrado.
        people: [(bbox, track_id)] das pessoas rastreadas — casa cada pose
            com o ID pra contar golpes por pessoa.
        objects: caixas de objetos que servem pra bater (RISK_CLASSES) —
            golpe com o punho dentro de uma delas = com objeto.
        continuous: só no vídeo há movimento pra medir; nos prints, None.

        Desenha no `annotated` (se dado) os punhos/pés que golpearam. O
        retorno traz `progress` (golpes por pessoa), mesmo antes de confirmar.
        """
        if not continuous or not self.load_model():
            return None

        h, w = image.shape[:2]
        sx1, sy1, sx2, sy2 = statue_px
        pad_x, pad_y = (sx2 - sx1) * 0.6, (sy2 - sy1) * 0.4
        cx1, cy1 = max(int(sx1 - pad_x), 0), max(int(sy1 - pad_y), 0)
        cx2, cy2 = min(int(sx2 + pad_x), w), min(int(sy2 + pad_y), h)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        try:
            with self._infer_lock:
                results = self.model(crop, conf=0.3, verbose=False)
        except Exception as e:
            logger.warning(f"Erro YOLO-pose: {e}")
            return None

        # Alvo do golpe: a FIGURA (não o banco/pedestal inteiro do contorno)
        figure = statue_px
        if statue_boxes:
            xs1, ys1, xs2, ys2 = zip(*statue_boxes)
            figure = (min(xs1), min(ys1), max(xs2), max(ys2))
        fcx, fcy = (figure[0] + figure[2]) / 2, (figure[1] + figure[3]) / 2

        now = time.time()
        hits: list[tuple] = []  # (x, y, cor) pra desenhar
        res = results[0] if results else None
        kps = getattr(res, "keypoints", None)
        if kps is not None and kps.data is not None and len(kps.data) > 0:
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
            for i, person in enumerate(kps.data.cpu().numpy()):  # (17, 3): x, y, conf
                if i >= len(boxes):
                    continue
                bx = boxes[i]
                pbox = (bx[0] + cx1, bx[1] + cy1, bx[2] + cx1, bx[3] + cy1)
                # Só quem está junto da estátua interessa — e a própria
                # estátua (figura humana) também sai pelo pose
                if (_overlap_frac(pbox, statue_px) < 0.1 or is_statue_itself(pbox, statue_px)
                        or any(iou(pbox, sb) >= 0.5 for sb in (statue_boxes or []))):
                    continue
                # Velocidade só faz sentido seguindo a MESMA pessoa (ID)
                best = max(((iou(pbox, pb), tid) for pb, tid in (people or []) if tid is not None),
                           default=(0.0, None))
                if best[0] < 0.3:
                    continue
                who = str(best[1])
                body = max(pbox[3] - pbox[1], 1.0)
                pts = [(x + cx1, y + cy1, c) for x, y, c in person]

                struck, with_obj = None, False
                for k in WRISTS + ANKLES:
                    x, y, c = pts[k]
                    if c < POSE_KEYPOINT_CONF:
                        continue
                    with _lock:
                        prev = _limbs.get((camera_code, who, k))
                        _limbs[(camera_code, who, k)] = (now, x, y)
                    if not prev or not 0 < now - prev[0] <= MAX_STEP_SECONDS:
                        continue
                    # Deslocamento em alturas do corpo: rápido, mas sem "pulo"
                    # de keypoint, indo NA DIREÇÃO da figura e terminando nela
                    step = math.hypot(x - prev[1], y - prev[2]) / body
                    toward = math.hypot(x - fcx, y - fcy) < math.hypot(prev[1] - fcx, prev[2] - fcy)
                    if (step / (now - prev[0]) >= STRIKE_SPEED and step <= STRIKE_MAX_JUMP
                            and toward and _inside((x, y), figure, margin=0.05)):
                        struck = (x, y)
                        with_obj = with_obj or (k in WRISTS and any(
                            _inside((x, y), ob, margin=0.3) for ob in (objects or [])))
                if struck:  # no máximo um golpe por pessoa por frame
                    with _lock:
                        _strikes.setdefault((camera_code, who), deque()).append((now, with_obj))
                    hits.append((*struck, (255, 0, 0) if with_obj else (255, 0, 255)))

        _expire(camera_code, now, INTERACTION_GRACE_SECONDS)
        with _lock:
            counts = {who: (len(q), any(o for _, o in q)) for (cam, who), q in _strikes.items()
                      if cam == camera_code}
        if not counts:
            return None

        if annotated is not None:
            for x, y, color in hits:
                cv2.circle(annotated, (int(x), int(y)), 11, color, 3)

        progress = [
            {"event": "golpe_na_estatua", "label": "golpe na estatua" + (" com objeto" if obj else ""),
             "who": who, "count": n, "needed": 1 if obj else STRIKE_MIN_COUNT}
            for who, (n, obj) in sorted(counts.items(), key=lambda kv: -kv[1][0])
        ]
        top = max(progress, key=lambda p: p["count"] / p["needed"])
        objeto = " com objeto na mão" if top["needed"] == 1 else ""
        if top["count"] >= top["needed"]:
            with _lock:
                _last_interaction[camera_code] = now
            level = "ALTO"
            msg = (f"🚨 ALTO: Pessoa #{top['who']} golpeando a estátua{objeto} "
                   f"({top['count']} movimento(s) brusco(s) em {int(STRIKE_WINDOW_SECONDS)}s) — "
                   f"possível vandalismo, requer validação humana")
        else:
            level = "MODERADO"
            msg = (f"⚠️ MODERADO: Movimento brusco da pessoa #{top['who']} contra a estátua{objeto} "
                   f"({top['count']}/{top['needed']})")

        return {
            "level": level,
            "message": msg,
            "events": ["golpe_na_estatua"],
            "strikes": top["count"],
            "progress": progress,
        }


# Instância única compartilhada por todos os detectores (um só modelo de
# pose na memória, mesmo com um detector por câmera na análise contínua)
shared_analyzer = InteractionAnalyzer()
