"""
Análise CONTÍNUA sobre o vídeo ao vivo (fase 2 da cascata).

Para cada câmera com captura contínua (live_capture), roda o YOLO com
rastreamento (ByteTrack) a LIVE_ANALYSIS_FPS sobre o frame mais recente:

  existe pessoa? → perto da estátua? → há quanto tempo ESTA pessoa (ID)?
      → pose (mão na área sensível / em cima) → alertas

Diferença pro monitoramento por prints: com ID por pessoa, a permanência
é real ("pessoa #12 encostada na estátua há 4 min"), não "sempre tem
alguém" — em monumento turístico isso era sempre verdade.

Detecção com pessoa identificada (superfície protegida):
  - a superfície protegida é o contorno calibrado da estátua (vermelho) e
    a zona monitorada é o quadrante da câmera (amarelo);
  - cada pessoa rastreada tem um estado: "passando" (fora da zona),
    "na_zona" ou "suspeito" (ficou junto à superfície por
    SURFACE_SUSPECT_SECONDS);
  - SurfaceMonitor compara a superfície visível com um fundo que se
    atualiza sozinho; mudança sustentada com um suspeito presente (ou
    visto há pouco) → alerta CRÍTICO em poucos segundos, com imagens
    antes/durante/depois e um trecho do vídeo (evidence_service).
  O alerta é sempre um POSSÍVEL evento pra validação humana: não há
  reconhecimento facial nem identificação de pessoas — o "#ID" é só o
  número do rastreamento enquanto a pessoa está em cena.

O resultado (frame anotado + detecções + alertas) fica em latest_result()
e é o que as rotas /live, /multi e o background_monitor usam para essas
câmeras, em vez de rodar a detecção de novo (evita estado de rastreamento
duplicado e alertas repetidos).
"""

import logging
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from app.config import (
    LIVE_ANALYSIS_FPS,
    PERSON_LOITERING_ALERT_SECONDS,
    SURFACE_SUSPECT_SECONDS,
    SURFACE_SUSPECT_MEMORY_SECONDS,
    SURFACE_CHANGE_MIN_FRAC,
    SURFACE_CRITICAL_FRAC,
    SURFACE_CHANGE_CONFIRM_SECONDS,
    SURFACE_CLEAR_CONFIRM_SECONDS,
    SURFACE_CHANGE_GAP_SECONDS,
    SURFACE_UNATTRIBUTED_ABSORB_SECONDS,
    SURFACE_EVENT_COOLDOWN_SECONDS,
    EVIDENCE_PRE_SECONDS,
    EVIDENCE_POST_SECONDS,
    INTERACTION_MEMORY_SECONDS,
    SURFACE_VISIBLE_MIN_NCC,
)
from app.models import interaction
from app.models.detector import PatrimonyDetector, _zone_box, _frac_box
from app.models.surface_monitor import SurfaceMonitor
from app.services import alert_service, evidence_service, live_capture, statue_memory, zone_service

logger = logging.getLogger(__name__)

# ID que some por mais que isto é descartado (saiu de cena)
TRACK_FORGET_SECONDS = 10
# "Pessoa" sobre o contorno que não se move (centro desloca menos que
# STATIC_MAX_SHIFT da largura do frame) por STATIC_MIN_SECONDS = a estátua.
# 45 s: turista posando pra foto não fica tão parado por tanto tempo.
STATIC_MIN_SECONDS = 45
# Parado = 85% das posições a menos de STATIC_MAX_SHIFT da LARGURA DA
# CAIXA do centro mediano. Relativo à caixa e robusto a frames ruins: o
# stream corrompido faz a caixa da estátua tremer, e um frame isolado
# fora do lugar não pode impedir o reconhecimento.
STATIC_MAX_SHIFT = 0.15
STATIC_INLIER_FRAC = 0.85
# Aparência da estátua (tons de cinza normalizados) usada pra decidir, a
# cada frame, se a caixa "pessoa" no lugar da estátua É a estátua (imagem
# igual à aprendida) ou alguém parado na frente dela (imagem diferente)
STATUE_TEMPLATE_SIZE = (48, 96)
STATUE_MATCH_IOU = 0.6
STATUE_MATCH_NCC = 0.55
# Referência não reconhecida por este tempo é descartada (reaprende)
STATUE_REF_TTL = 600
# Falha de transmissão: fração da imagem (fora das pessoas) que mudou mais
# que GLITCH_PIXEL_DIFF níveis de cinza entre duas análises
GLITCH_SIZE = (160, 90)
GLITCH_PIXEL_DIFF = 35
GLITCH_FRAC = 0.30
# Clipe de evidência no máximo este tanto antes da confirmação (a janela de
# confirmação é de minutos; o buffer de frames fica limitado na memória)
MAX_CLIP_SECONDS = 90
# Superfície "desobstruída" pra servir de imagem ANTES do evento
BEFORE_MIN_VISIBLE = 0.9
# Faixa vermelha "possível pichação" no vídeo por este tempo após o alerta;
# depois, só a linha "alerta registrado às ..." até EVENT_NOTE_SECONDS
EVENT_BANNER_SECONDS = 20
EVENT_NOTE_SECONDS = 600

# Cores (RGB) da sobreposição
_GREEN = (70, 200, 70)
_ORANGE = (255, 160, 0)
_RED = (230, 30, 30)
_YELLOW = (255, 215, 0)
_MAGENTA = (255, 0, 200)
_STATE_COLOR = {"passando": _GREEN, "na_zona": _ORANGE, "suspeito": _RED}
# Hershey (cv2.putText) não desenha acento nem "·" — rótulos em ASCII
_STATE_TEXT = {"na_zona": "junto a estatua", "suspeito": "junto ha tempo"}
_STATUE_COLOR = (120, 170, 255)
_OTHER_COLOR = (230, 230, 230)


def _within_reach(bbox: list, surface_px: tuple) -> bool:
    """Pessoa ao alcance da superfície: caixa a até ~um braço (0,35 da altura
    da pessoa) do contorno. Sobreposição com o contorno não basta —
    superfície estreita (poste, pedestal) quase não fica "dentro" da caixa
    de quem está ao lado. Altura em vez de largura: varia menos com a pose."""
    x1, y1, x2, y2 = bbox
    sx1, sy1, sx2, sy2 = surface_px
    reach = (y2 - y1) * 0.35
    dx = max(sx1 - x2, x1 - sx2, 0)
    dy = max(sy1 - y2, y1 - sy2, 0)
    return dx <= reach and dy <= reach


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _shift_score(centers: list, box_w: float) -> float:
    """Deslocamento (em larguras da caixa) abaixo do qual estão STATIC_INLIER_FRAC
    das posições, medido a partir do centro mediano."""
    c = np.asarray(centers, np.float32)
    med = np.median(c, axis=0)
    d = np.abs(c - med).max(axis=1) / box_w
    return float(np.quantile(d, STATIC_INLIER_FRAC))


def _normalize(t: np.ndarray) -> np.ndarray:
    return (t - t.mean()) / (t.std() + 1e-6)


def _statue_tpl(image: np.ndarray, bbox) -> np.ndarray:
    """Aparência do recorte (cinza, tamanho fixo, média 0 / desvio 1) — o
    produto médio de dois desses é a correlação normalizada (NCC)."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(STATUE_TEMPLATE_SIZE[::-1], np.float32)
    g = cv2.cvtColor(cv2.resize(crop, STATUE_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return _normalize(g.astype(np.float32))


def _figure_box(statue_refs: dict, contour_px: Optional[tuple], w: int, h: int) -> Optional[tuple]:
    """Superfície protegida = a figura da estátua (caixa do YOLO reconhecida
    como estátua, com folga de 6%), recortada pelo contorno calibrado. Sem a
    estátua reconhecida ainda, usa o contorno inteiro."""
    if not statue_refs:
        return contour_px
    xs1, ys1, xs2, ys2 = zip(*(r["bbox"] for r in statue_refs.values()))
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    mx, my = (x2 - x1) * 0.06, (y2 - y1) * 0.06
    x1, y1, x2, y2 = int(max(x1 - mx, 0)), int(max(y1 - my, 0)), int(min(x2 + mx, w)), int(min(y2 + my, h))
    if contour_px:
        x1, y1 = max(x1, contour_px[0]), max(y1, contour_px[1])
        x2, y2 = min(x2, contour_px[2]), min(y2, contour_px[3])
    return (x1, y1, x2, y2) if x2 - x1 >= 8 and y2 - y1 >= 8 else contour_px


def _ascii(text: str) -> str:
    """cv2.putText (Hershey) não desenha acento."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _hud_lines(r: dict, near: list[dict], surface: Optional[dict], surface_status: str,
               statue_status: str = "") -> list[tuple[str, tuple]]:
    """Painel no vídeo: o que o sistema está vendo e por que (não) alerta."""
    people = sum(1 for d in r["objects"] if d["class_name"] == "pessoa" and not d.get("is_statue"))
    lines = [(f"Pessoas: {people} | junto a estatua: {len(near)}", (255, 255, 255))]
    if statue_status:
        lines.append((f"Estatua: {statue_status}", _STATUE_COLOR))
    if surface_status:
        color = _RED if surface_status.startswith("ALERTA") else (
            _YELLOW if "aguardando" in surface_status or "verificando" in surface_status else (200, 200, 200))
        lines.append((f"Superficie: {surface_status}", color))
    ia = r.get("interaction_alert")
    for p in (ia or {}).get("progress", [])[:3]:
        done = p["count"] >= p["needed"]
        lines.append((f"#{p['who']} {p['label']}: {p['count']}/{p['needed']}"
                      + (" - ALERTA" if done else " (movimento brusco)"),
                      _RED if done else _ORANGE))
    if r.get("risk_objects"):
        nomes = ", ".join(sorted({d["class_name"] for d in r["risk_objects"]}))
        lines.append((f"Objeto de risco encostado na estatua: {nomes}", _RED))
    return [(_ascii(t), c) for t, c in lines]


class CameraAnalysis:
    def __init__(self, code: str):
        self.code = code
        # Um detector por câmera: o estado do ByteTrack fica no modelo
        self.detector = PatrimonyDetector()
        self.last_seq = -1
        self.last_run = 0.0
        # track_id → {"first": ts, "last": ts, "near_since": ts|None,
        #             "origin": (cx, cy), "max_shift": float, "suspect": bool}
        self.tracks: dict[int, dict] = {}
        # IDs reconhecidos como a própria estátua (parados sobre o contorno)
        self.statue_ids: set[int] = set()
        # Estátua aprendida: id → {"bbox", "tpl" (aparência), "seen", "ncc"}.
        # O detector consulta statue_check() pra cada caixa "pessoa"
        self.statue_refs: dict[int, dict] = {}
        self.statue_status: str = "aprendendo (precisa ficar visivel e parada)"
        self.calib_version = 0
        # Superfície alterada sem ninguém por perto desde (ts) — confirmação
        # Segundos SOMADOS em que a alteração seguiu visível com a estátua
        # livre (ninguém ao alcance). Somado, não seguido: em monumento
        # movimentado quase nunca há 1 min inteiro sem ninguém junto.
        self.clear_accum = 0.0
        self.prev_eval_ts: Optional[float] = None
        self.below_since: Optional[float] = None  # alteração abaixo do limite desde (ts)
        # Situação da superfície pra sobreposição/painel (texto explicativo)
        self.surface_status: str = ""
        self.confirm: Optional[dict] = None     # evento decidido, aguardando o frame anotado
        self.prev_small: Optional[np.ndarray] = None  # frame anterior reduzido (falha de transmissão)
        self.glitch_frac = 0.0
        # Suspeitos recentes (track_id → último ts visto), sobrevivem ao track
        self.recent_suspects: dict[int, float] = {}
        self.surface = SurfaceMonitor()
        # Evidências: frames anotados recentes (ts, jpeg) pro clipe
        self.evidence_buf: deque = deque()
        self.before_jpg: Optional[bytes] = None
        self.change_since: Optional[float] = None
        self.pending: Optional[dict] = None     # evento aguardando o "depois"
        self.cooldown_until = 0.0
        self.last_event: Optional[dict] = None  # meta do último evento (frontend)
        self.result: Optional[dict] = None
        self.result_seq = 0                     # incrementa a cada análise (MJPEG)
        self.name: Optional[str] = None
        self._lock = threading.Lock()
        self._times: list[float] = []  # ts das últimas análises (FPS real)

    def glitch_check(self, frame: np.ndarray, person_boxes: list) -> bool:
        """Frame corrompido/salto de câmera: fração grande da imagem (fora das
        pessoas) mudou de uma análise pra outra (~0,5 s). Na orla normal —
        ondas, guarda-sóis, gente passando — isso fica bem abaixo do limite."""
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), GLITCH_SIZE,
                           interpolation=cv2.INTER_AREA).astype(np.int16)
        prev, self.prev_small = self.prev_small, small
        if prev is None or prev.shape != small.shape:
            return False
        h, w = frame.shape[:2]
        sx, sy = GLITCH_SIZE[0] / w, GLITCH_SIZE[1] / h
        keep = np.ones(small.shape, bool)
        for x1, y1, x2, y2 in person_boxes:
            keep[int(y1 * sy):int(y2 * sy) + 1, int(x1 * sx):int(x2 * sx) + 1] = False
        if keep.mean() < 0.2:
            return False
        frac = float((np.abs(small - prev) > GLITCH_PIXEL_DIFF)[keep].mean())
        self.glitch_frac = round(frac, 3)
        return frac >= GLITCH_FRAC

    def statue_check(self, bbox: list, image: np.ndarray) -> bool:
        """A caixa é a estátua? Posição parecida com a aprendida E a imagem no
        lugar da estátua igual à aparência aprendida. Alguém parado na frente
        da estátua tem a mesma caixa, mas outra imagem → é pessoa."""
        now = time.time()
        # Estátua calibrada: compara a caixa com a mesma região da imagem
        # da calibração (statue_memory) — vale sobre o aprendizado automático
        if "calib" in self.statue_refs:
            sim = statue_memory.similarity(self.code, bbox, image)
            if sim is None:
                return False
            ok = sim >= statue_memory.MATCH_NCC
            self.statue_status = (f"calibrada - reconhecida (semelhanca {sim:.2f})" if ok else
                                  f"calibrada - pessoa na frente da estatua (semelhanca {sim:.2f})")
            if ok:
                self.statue_refs["calib"]["seen"] = now
            return ok
        for ref in self.statue_refs.values():
            if _iou(bbox, ref["bbox"]) < ref.get("iou_min", STATUE_MATCH_IOU):
                continue
            cur = _statue_tpl(image, ref["bbox"])
            ncc = float((cur * ref["tpl"]).mean())
            ref["ncc"] = ncc
            if ncc >= STATUE_MATCH_NCC:
                ref["seen"] = now
                if ncc >= 0.8:  # acompanha devagar a luz do dia
                    ref["tpl"] = _normalize(0.95 * ref["tpl"] + 0.05 * cur)
                self.statue_status = f"reconhecida (semelhanca {ncc:.2f})"
                return True
            self.statue_status = f"encoberta por pessoa na frente (semelhanca {ncc:.2f})"
        return False

    def statue_visibility(self, frame: np.ndarray, surface_px: tuple) -> Optional[float]:
        """Semelhança (NCC) da figura INTEIRA com a aparência conhecida da
        estátua: a da calibração, ou a aprendida (melhor das referências).
        None se ainda não há aparência conhecida. Pichação/peça faltando
        mudam parte da figura; ônibus ou guarda-sol parado na frente (que o
        YOLO não detectou) a tiram de vista — a semelhança despenca."""
        if "calib" in self.statue_refs:
            return statue_memory.similarity(self.code, surface_px, frame)
        sims = [float((_statue_tpl(frame, r["bbox"]) * r["tpl"]).mean())
                for r in self.statue_refs.values() if "tpl" in r]
        return max(sims) if sims else None

    def sync_calibration(self, frame: np.ndarray) -> None:
        """Aparência da estátua gravada na calibração vira referência fixa
        (não precisa esperar a estátua ficar 45 s parada pra reconhecer)."""
        m = statue_memory.get(self.code)
        if not m:
            if self.statue_refs.pop("calib", None) is not None:
                self.calib_version = 0
            return
        if m["version"] == self.calib_version and "calib" in self.statue_refs:
            return
        h, w = frame.shape[:2]
        b = m["box"]
        self.statue_refs["calib"] = {
            "bbox": [int(b["x_start"] * w), int(b["y_start"] * h), int(b["x_end"] * w), int(b["y_end"] * h)],
            "seen": time.time(), "ncc": 1.0, "calibrated": True,
        }
        self.calib_version = m["version"]
        self.statue_status = "calibrada (aparencia gravada na calibracao)"

    def update_tracks(self, objects: list[dict], ts: float, frame_w: int,
                      surface_px: Optional[tuple] = None, frame: Optional[np.ndarray] = None) -> list[dict]:
        """Atualiza permanência e estado por ID (grava d["state"]); devolve as pessoas junto à estátua."""
        near_now = []
        for d in objects:
            tid = d.get("track_id")
            if d["class_name"] != "pessoa" or tid is None:
                continue
            x1, y1, x2, y2 = d["bbox"]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            t = self.tracks.setdefault(tid, {"first": ts, "last": ts, "near_since": None,
                                             "origin": center, "max_shift": 0.0, "suspect": False})
            t["last"] = ts
            t.setdefault("centers", []).append(center)
            del t["centers"][:-400]
            t["max_shift"] = _shift_score(t["centers"], max(x2 - x1, 1))
            # Estátua: parada sobre o contorno tempo suficiente
            if (tid not in self.statue_ids and frame is not None and (d["statue_overlap"] or 0) >= 0.8
                    and ts - t["first"] >= STATIC_MIN_SECONDS and t["max_shift"] < STATIC_MAX_SHIFT
                    and not any(_iou(d["bbox"], r["bbox"]) >= STATUE_MATCH_IOU for r in self.statue_refs.values())):
                self.statue_ids.add(tid)
                self.statue_refs[tid] = {"bbox": list(d["bbox"]), "tpl": _statue_tpl(frame, d["bbox"]),
                                         "seen": time.time(), "ncc": 1.0}
                self.recent_suspects.pop(tid, None)
                t["suspect"] = False
                d["is_statue"] = True
                self.statue_status = "reconhecida (aprendida agora)"
                logger.info(f"[análise {self.code}] ID #{tid} reconhecido como a própria estátua "
                            f"(parado {STATIC_MIN_SECONDS}s)")
            # Alguém na frente da estátua chega aqui com is_statue=False
            # (aparência não bate) e conta como gente perto
            if d.get("is_statue"):
                t["suspect"] = False
                self.recent_suspects.pop(tid, None)
                continue
            # Com contorno calibrado: ao alcance da superfície; sem: dentro da zona
            near = _within_reach(d["bbox"], surface_px) if surface_px else d["in_zone"]
            if near:
                t["near_since"] = t["near_since"] or ts
                near_now.append({"track_id": tid, "seconds": round(ts - t["near_since"], 1)})
                if ts - t["near_since"] >= SURFACE_SUSPECT_SECONDS:
                    t["suspect"] = True
            else:
                t["near_since"] = None
            # Suspeito continua suspeito enquanto está em cena (mesmo se afastar)
            if t["suspect"]:
                self.recent_suspects[tid] = ts
                d["state"] = "suspeito"
            else:
                d["state"] = "na_zona" if d["in_zone"] or near else "passando"
        for tid in [k for k, v in self.tracks.items() if ts - v["last"] > TRACK_FORGET_SECONDS]:
            del self.tracks[tid]
            self.statue_ids.discard(tid)
        now = time.time()
        for key in [k for k, r in self.statue_refs.items()
                    if not r.get("calibrated") and now - r["seen"] > STATUE_REF_TTL]:
            del self.statue_refs[key]
            logger.info(f"[análise {self.code}] referência da estátua #{key} expirada — reaprendendo")
        for tid in [k for k, v in self.recent_suspects.items() if ts - v > SURFACE_SUSPECT_MEMORY_SECONDS]:
            del self.recent_suspects[tid]
        return sorted(near_now, key=lambda x: -x["seconds"])

    def analysis_fps(self) -> float:
        now = time.time()
        recent = [t for t in self._times if now - t <= 10]
        return round((len(recent) - 1) / (recent[-1] - recent[0]), 2) if len(recent) > 1 else 0.0


# ─── Sobreposição (desenhada sobre o frame limpo) ───────────────

def _label(img: np.ndarray, text: str, org: tuple, color: tuple, scale: float):
    """Texto colorido sobre fundo escuro, acima de `org` (canto sup. esq. da caixa)."""
    th = max(int(scale * 2), 1)
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org[0], max(org[1] - 4, h + base + 2)
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 6, y + 2), (25, 25, 25), -1)
    cv2.putText(img, text, (x + 3, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def _banner(img: np.ndarray, text: str, scale: float, strong: bool):
    h, w = img.shape[:2]
    th = max(int(scale * 2), 1)
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    bar = tht + base + 12
    if strong:
        cv2.rectangle(img, (0, h - bar), (w, h), _RED, -1)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), _RED, max(int(4 * scale), 2))
        cv2.putText(img, text, ((w - tw) // 2, h - base - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), th, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (0, h - bar), (tw + 12, h), (160, 20, 20), -1)
        cv2.putText(img, text, (6, h - base - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), th, cv2.LINE_AA)


def draw_overlay(img: np.ndarray, objects: list[dict], zone_px: tuple, contour_px: Optional[tuple],
                 surface_px: Optional[tuple], surface: Optional[dict], risk_objects: list[dict],
                 last_event: Optional[dict], ts: float, hud: Optional[list] = None):
    """
    Zona (amarelo), contorno calibrado (laranja fino), figura protegida
    (vermelho), mudança (magenta), TODAS as detecções do YOLO com classe e
    confiança (%), pessoas por estado/ID e o painel explicando o que o
    sistema está vendo e por que (não) alerta.
    """
    s = max(img.shape[1] / 1280, 0.4)
    lw = max(int(2 * s), 1)
    zx1, zy1, zx2, zy2 = zone_px
    cv2.rectangle(img, (zx1, zy1), (zx2, zy2), _YELLOW, lw)
    _label(img, "Zona monitorada", (zx1, zy2 + int(24 * s)), _YELLOW, 0.5 * s)

    if contour_px and contour_px != surface_px:
        cx1, cy1, cx2, cy2 = contour_px
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), _ORANGE, 1)
    if surface_px:
        sx1, sy1, sx2, sy2 = surface_px
        if surface and surface.get("mask") is not None:
            roi = img[sy1:sy2, sx1:sx2]
            m = surface["mask"][:roi.shape[0], :roi.shape[1]]
            roi[m] = (0.45 * roi[m] + 0.55 * np.array(_MAGENTA)).astype(np.uint8)
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), _RED, lw + 1)
        _label(img, "Estatua protegida", (sx1, sy2 + int(24 * s)), _RED, 0.5 * s)

    risk_ids = {id(d) for d in risk_objects}
    for d in objects:
        x1, y1, x2, y2 = d["bbox"]
        conf = f"{d['confidence'] * 100:.0f}%"
        if d["class_name"] == "pessoa" and d.get("is_statue"):
            color, text, thick = _STATUE_COLOR, f"estatua (ignorada) {conf}", lw
        elif d["class_name"] == "pessoa":
            state = d.get("state") or "passando"
            color = _STATE_COLOR[state]
            tid = f" #{d['track_id']}" if d.get("track_id") is not None else ""
            text = f"pessoa {conf}{tid}" + (f" - {_STATE_TEXT[state]}" if state in _STATE_TEXT else "")
            thick = lw if state == "passando" else lw + 1
        elif id(d) in risk_ids:
            color, text, thick = _RED, f"{d['class_name']} {conf} - RISCO", lw + 1
        else:
            color, text, thick = _OTHER_COLOR, f"{d['class_name']} {conf}", 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        _label(img, _ascii(text), (x1, y1), color, 0.45 * s)

    # Painel (canto superior esquerdo abaixo do logo do COR, que o player já tem)
    y = int(150 * s)
    for text, color in hud or []:
        _label(img, text, (int(10 * s), y), color, 0.5 * s)
        y += int(26 * s)

    if last_event:
        age = ts - last_event["timestamp"]
        if age <= EVENT_BANNER_SECONDS:
            _banner(img, "Estatua alterada apos as pessoas sairem - requer validacao humana", 0.7 * s, True)
        elif age <= EVENT_NOTE_SECONDS:
            hhmm = datetime.fromtimestamp(last_event["timestamp"]).strftime("%H:%M:%S")
            _banner(img, f"Alerta registrado as {hhmm}", 0.5 * s, False)


class LiveAnalyzer:
    def __init__(self, fps: float):
        self.interval = 1.0 / fps
        self.cameras: dict[str, CameraAnalysis] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-analysis", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        from app.api import monitor as monitor_api  # import tardio (evita ciclo)

        # Uma thread, câmeras em rodízio: inferência sequencial (sem disputar
        # CPU entre modelos) e cada câmera limitada a LIVE_ANALYSIS_FPS
        while not self._stop.is_set():
            did_work = False
            for code in live_capture.codes():
                cam = self.cameras.get(code) or self.cameras.setdefault(code, CameraAnalysis(code))
                if time.time() - cam.last_run < self.interval:
                    continue
                stream = live_capture.get_stream(code)
                item = stream.latest_with_ts(max_age=3.0) if stream else None
                if not item or item[2] == cam.last_seq:
                    continue  # sem frame novo
                ts, frame, seq = item
                cam.last_seq, cam.last_run = seq, time.time()
                try:
                    self._analyze(cam, frame, ts, monitor_api)
                    did_work = True
                except Exception as e:
                    logger.warning(f"[análise {code}] erro: {e}")
            if not did_work:
                self._stop.wait(0.05)

    def _analyze(self, cam: CameraAnalysis, frame: np.ndarray, ts: float, monitor_api):
        t0 = time.time()
        cam.sync_calibration(frame)
        # Com a estátua já reconhecida, o detector recebe as caixas de
        # referência dela (ver detector: caixa que cresceu = turista fundido)
        r = cam.detector.detect(frame, camera_code=cam.code, track=True, draw=False,
                                statue_check=cam.statue_check)
        h, w = frame.shape[:2]
        zone_px = _zone_box(w, h, cam.code)
        contour_px = _frac_box(zone_service.get_statue(cam.code)["statue"], w, h)
        # Superfície protegida = a FIGURA da estátua (caixa do YOLO parada
        # sobre o contorno), não o retângulo calibrado inteiro — o banco do
        # Drummond é pra sentar; o que se protege é a estátua
        surface_px = _figure_box(cam.statue_refs, contour_px, w, h)
        near = cam.update_tracks(r["objects"], ts, w, surface_px, frame)

        if cam.name is None and monitor_api.camera_service:
            c = monitor_api.camera_service.get_camera_by_code(cam.code)
            cam.name = c.get("name", cam.code) if c else cam.code
        name = cam.name or cam.code

        # ─── Falha de transmissão ────────────────────────────────────
        # Stream com blocos borrados/congelados (perda de pacote) muda boa
        # parte da imagem de uma vez — a superfície "altera" sem nada ter
        # acontecido. Nesses frames a análise da superfície fica em pausa.
        glitch = cam.glitch_check(frame, [d["bbox"] for d in r["objects"]])

        # ─── Estátua à vista? ────────────────────────────────────────
        # Ônibus/caminhão parado na frente nem sempre é detectado pelo YOLO;
        # sem isso a figura "inteira mudou" com ninguém junto → falso alarme.
        visibility = cam.statue_visibility(frame, surface_px) if surface_px else None
        hidden = visibility is not None and visibility < SURFACE_VISIBLE_MIN_NCC

        # ─── Superfície protegida ────────────────────────────────────
        surface = None
        if surface_px and glitch:
            # Só pausa: não zera as contagens (frame ruim no meio de uma
            # alteração real não pode reiniciar a janela de minutos)
            cam.surface_status = "PAUSADA - imagem com falha de transmissao"
        elif surface_px and hidden:
            # Também só pausa — e não atualiza o fundo (não aprende o ônibus)
            cam.surface_status = f"PAUSADA - estatua encoberta (semelhanca {visibility:.2f})"
        elif surface_px:
            # Tudo que o YOLO achou na frente da superfície (menos a própria
            # estátua) é oclusão, não mudança
            occluders = [d["bbox"] for d in r["objects"] if not d.get("is_statue")]
            surface = cam.surface.update(frame, surface_px, occluders)

        # Permanência junto ao monumento: só informativa (painel/vídeo).
        # Sentar no banco do Drummond por muito tempo é o uso normal do lugar
        # — não vira alerta nem entra no histórico.
        loitering = None
        if near and near[0]["seconds"] >= PERSON_LOITERING_ALERT_SECONDS:
            top = near[0]
            loitering = {
                "level": "INFO",
                "message": f"👥 Pessoa #{top['track_id']} junto ao monumento há {int(top['seconds'] // 60)} min (informativo)",
                "dwell_seconds": top["seconds"],
            }

        surface_alert = None if glitch or hidden else self._surface_events(cam, name, ts, surface, near)

        annotated = r["annotated_image"]
        hud = _hud_lines(r, near, surface, cam.surface_status, cam.statue_status)
        draw_overlay(annotated, r["objects"], zone_px, contour_px, surface_px, surface,
                     r["risk_objects"], cam.last_event, ts, hud)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg = buf.tobytes()
        # Depois de desenhar: o alerta guarda o frame já anotado (caixas/zonas)
        monitor_api.record_risk_alert(cam.code, name, r["risk_alert"], image=jpg)
        monitor_api.record_interaction_alert(cam.code, name, r["interaction_alert"], image=jpg)
        self._evidence(cam, ts, jpg, surface)

        objects = [{k: v for k, v in d.items()} for d in r["objects"]]
        with cam._lock:
            cam._times = [t for t in cam._times if time.time() - t <= 10] + [time.time()]
            cam.result_seq += 1
            cam.result = {
                "timestamp": ts,
                "camera_code": cam.code,
                "camera_name": name,
                "jpeg": jpg,
                "yolo_detection": {
                    "objects": objects,
                    "counts": r["counts"],
                    "total_objects": r["total_objects"],
                },
                "risk_alert": r["risk_alert"],
                "interaction_alert": r["interaction_alert"],
                "loitering_alert": loitering,
                "surface_alert": surface_alert,
                "surface_event": cam.last_event,
                "surface": {
                    "calibrated": surface_px is not None,
                    "change_frac": surface["change_frac"] if surface else None,
                    "clear_change_frac": surface.get("clear_change_frac") if surface else None,
                    "stable_change_frac": surface.get("stable_change_frac") if surface else None,
                    "status": cam.surface_status,
                    "transmission_glitch": glitch,
                    "visible_frac": surface["visible_frac"] if surface else None,
                },
                "suspects": sorted(cam.recent_suspects),
                "people_near_statue": near,
                "processing_time_ms": round((time.time() - t0) * 1000, 1),
            }

    def _surface_events(self, cam: CameraAnalysis, name: str, ts: float,
                        surface: Optional[dict], near: list[dict]) -> Optional[dict]:
        """
        Decide se a superfície da estátua foi alterada de verdade.

        Turista abraçando/encostando/tirando foto muda os pixels da estátua
        enquanto está lá — e a mudança some quando ele sai. Pichação, peça
        arrancada ou dano CONTINUAM depois. Por isso o evento só confirma
        quando a alteração se mantém com a estátua LIVRE (ninguém ao alcance)
        por SURFACE_CLEAR_CONFIRM_SECONDS, e alguém esteve junto antes
        (suspeito recente). Atualiza cam.surface_status (texto pro vídeo).
        """
        if surface is None:
            cam.surface_status = ""
            return None

        # Só conta a mudança fora do halo de quem está na frente E parada
        # (sem movimento): pessoa não detectada e falha de transmissão mexem
        clear_frac = surface.get("stable_change_frac", surface.get("clear_change_frac", surface["change_frac"]))
        pct = round(clear_frac * 100, 1)
        if clear_frac < SURFACE_CHANGE_MIN_FRAC:
            # Queda curta (alguém passou na frente, frame ruim) não zera: só
            # "some" se ficar abaixo do limite por SURFACE_CHANGE_GAP_SECONDS
            cam.below_since = cam.below_since or ts
            if cam.change_since is None or ts - cam.below_since >= SURFACE_CHANGE_GAP_SECONDS:
                cam.change_since = None
                cam.clear_accum = 0.0
                cam.surface_status = f"Estatua sem alteracao ({pct}%)"
            return None
        cam.below_since = None
        # Tempo desde a última avaliação (limitado: lacuna grande não conta)
        dt = min(ts - cam.prev_eval_ts, 5.0) if cam.prev_eval_ts else 0.0
        cam.prev_eval_ts = ts

        cam.change_since = cam.change_since or ts
        sustained = ts - cam.change_since
        # A própria estátua (quando um frame ruim a fez parecer pessoa) não conta
        suspects = sorted(i for i in cam.recent_suspects if i not in cam.statue_ids and i not in cam.statue_refs)
        if not suspects:
            if sustained >= SURFACE_UNATTRIBUTED_ABSORB_SECONDS:
                # Mudou sem ninguém junto (luz, objeto deixado, câmera mexeu):
                # vira fundo — não dispara quando alguém chegar depois
                logger.info(f"[análise {cam.code}] mudança na superfície sem suspeito — absorvida")
                cam.surface.absorb()
                cam.change_since = None
            cam.surface_status = f"Alteracao {pct}% sem ninguem junto antes - ignorando (luz/sombra)"
            return None

        # Alguém ao alcance da estátua agora: a "alteração" pode ser a própria
        # pessoa/sombra — espera ela se afastar pra ver se a mudança fica
        if near:
            cam.surface_status = (f"Alteracao {pct}% ha {sustained:.0f}s com pessoas junto - "
                                  f"aguardando a estatua ficar livre")
            return None

        cam.clear_accum += dt
        clear_for = cam.clear_accum
        if clear_for < SURFACE_CLEAR_CONFIRM_SECONDS or sustained < SURFACE_CHANGE_CONFIRM_SECONDS:
            cam.surface_status = (f"Alteracao {pct}% ha {sustained:.0f}/{SURFACE_CHANGE_CONFIRM_SECONDS:.0f}s, "
                                  f"vista com a estatua livre por {clear_for:.0f}/{SURFACE_CLEAR_CONFIRM_SECONDS:.0f}s (somado) - verificando")
            return None
        if cam.pending or ts < cam.cooldown_until:
            cam.surface_status = f"Alteracao {pct}% - alerta ja registrado"
            return None

        # ─── Confirma o evento ───────────────────────────────────────
        ids = ", ".join(f"#{i}" for i in suspects)
        level = "CRÍTICO" if clear_frac >= SURFACE_CRITICAL_FRAC else "ALTO"
        # Golpe confirmado há pouco + alteração que ficou = dano: CRÍTICO
        struck = interaction.last_interaction(cam.code)
        golpe = ""
        if struck and ts - struck <= INTERACTION_MEMORY_SECONDS:
            level = "CRÍTICO"
            golpe = f" Houve movimento brusco contra a estátua há ~{max(int((ts - struck) // 60), 1)} min."
        alert = {
            "level": level,
            "message": (f"🎨 {level}: Estátua alterada ({pct}% da figura) e a alteração CONTINUOU depois que "
                        f"as pessoas se afastaram ({int(clear_for)}s com a estátua livre, somados). Pessoa(s) {ids} "
                        f"estiveram junto antes.{golpe} Possível pichação/dano — requer validação humana"),
        }
        cam.confirm = {"alert": alert, "pct": pct, "suspects": suspects, "ids": ids}
        cam.surface_status = f"ALERTA: alteracao {pct}% permaneceu com a estatua livre"
        return alert

    def _evidence(self, cam: CameraAnalysis, ts: float, jpg: bytes, surface: Optional[dict]):
        """Buffer do clipe, imagem 'antes', registro do evento confirmado e fechamento do clipe."""
        keep = EVIDENCE_PRE_SECONDS + MAX_CLIP_SECONDS + EVIDENCE_POST_SECONDS
        cam.evidence_buf.append((ts, jpg))
        while cam.evidence_buf and ts - cam.evidence_buf[0][0] > keep:
            cam.evidence_buf.popleft()
        if surface and surface["visible_frac"] >= BEFORE_MIN_VISIBLE and surface["change_frac"] == 0:
            cam.before_jpg = jpg

        # Evento confirmado agora: registra com a imagem deste frame ("durante")
        if cam.confirm:
            c, cam.confirm = cam.confirm, None
            name = cam.name or cam.code
            event_id = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
            meta = evidence_service.start_event(cam.code, event_id, {
                "timestamp": ts,
                "camera_name": name,
                "change_started_at": cam.change_since,
                "change_percent": c["pct"],
                "suspect_track_ids": c["suspects"],
                "message": c["alert"]["message"],
            }, cam.before_jpg, jpg)
            entry = alert_service.add_alert(cam.code, name, c["alert"]["level"], c["alert"]["message"],
                                            source="surface",
                                            objects=[f"pessoa {i}" for i in c["ids"].split(", ")], evidence=meta)
            cam.pending = {"meta": meta, "alert_id": entry["id"], "finalize_at": ts + EVIDENCE_POST_SECONDS,
                           # Do início da alteração (limitado a MAX_CLIP_SECONDS antes de agora)
                           "clip_from": max((cam.change_since or ts) - EVIDENCE_PRE_SECONDS, ts - MAX_CLIP_SECONDS)}
            cam.cooldown_until = ts + SURFACE_EVENT_COOLDOWN_SECONDS
            cam.last_event = meta
            logger.warning(f"[análise {cam.code}] 🎨 evento de superfície {event_id}: {c['pct']}% alterado, "
                           f"suspeitos {c['ids']}")

        # Evento aguardando o "depois": fecha e grava o clipe
        if cam.pending and ts >= cam.pending["finalize_at"]:
            p, cam.pending = cam.pending, None
            frames = [(t, j) for t, j in cam.evidence_buf if t >= p["clip_from"]]
            alert_id = p["alert_id"]

            def _done(meta, cam=cam, alert_id=alert_id):
                alert_service.update_evidence(alert_id, meta)
                cam.last_event = meta

            evidence_service.finish_event(p["meta"], jpg, frames, on_done=_done)
            cam.surface.absorb()  # a mudança já virou evento: passa a ser o novo normal
            cam.change_since = None
            cam.clear_accum = 0.0


# ─── Registro global ─────────────────────────────────────────────
_analyzer: Optional[LiveAnalyzer] = None


def start() -> None:
    global _analyzer
    if _analyzer or not live_capture.codes():
        return
    _analyzer = LiveAnalyzer(LIVE_ANALYSIS_FPS)
    _analyzer.start()
    logger.info(f"🧠 Análise contínua iniciada @ {LIVE_ANALYSIS_FPS} FPS por câmera")


def stop() -> None:
    if _analyzer:
        _analyzer.stop()


def latest_result(camera_code: str, max_age: float = 5.0) -> Optional[dict]:
    """Última análise da câmera (sem o JPEG), se recente."""
    cam = _analyzer.cameras.get(camera_code) if _analyzer else None
    if not cam:
        return None
    with cam._lock:
        r = cam.result
    if not r or time.time() - r["timestamp"] > max_age:
        return None
    return {k: v for k, v in r.items() if k != "jpeg"}


def latest_jpeg(camera_code: str, max_age: float = 10.0) -> Optional[bytes]:
    item = latest_jpeg_seq(camera_code, max_age)
    return item[1] if item else None


def latest_jpeg_seq(camera_code: str, max_age: float = 10.0) -> Optional[tuple[int, bytes]]:
    """(seq, jpeg) do último frame analisado — seq muda a cada análise nova."""
    cam = _analyzer.cameras.get(camera_code) if _analyzer else None
    if not cam:
        return None
    with cam._lock:
        r, seq = cam.result, cam.result_seq
    return (seq, r["jpeg"]) if r and time.time() - r["timestamp"] <= max_age else None


def status() -> list[dict]:
    if not _analyzer:
        return []
    out = []
    for code, cam in _analyzer.cameras.items():
        with cam._lock:
            r = cam.result
        out.append({
            "camera_code": code,
            "analysis_fps": cam.analysis_fps(),
            "last_analysis_age_s": round(time.time() - r["timestamp"], 1) if r else None,
            "processing_time_ms": r["processing_time_ms"] if r else None,
            "tracked_people": len(cam.tracks) - len(cam.statue_ids),
            "statue_track_ids": sorted(cam.statue_ids),
            "people_near_statue": r["people_near_statue"] if r else [],
            "suspects": r["suspects"] if r else [],
            "surface": r["surface"] if r else None,
        })
    return out
