"""
Detector de Mudanças em Patrimônios.
Combina:
  1. SSIM no ROI da estátua → detecta pichação, danos, peças removidas
  2. HF Model (KzRyan/Burglary_and_Vandalism) → confirma se é roubo/vandalismo

O ROI (Região de Interesse) é extraído do centro do frame,
onde a estátua/monumento está posicionada.
"""

import base64
import logging
import time
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.config import (
    INTERACTION_MEMORY_SECONDS,
    STATUE_OCCLUSION_SKIP,
    SURFACE_VISIBLE_MIN_NCC,
    SSIM_INCREASE_MODERADO,
    SSIM_INCREASE_ALTO,
    SSIM_INCREASE_CRITICO,
    SSIM_CONFIRM_CHECKS,
    SSIM_CONFIRM_SECONDS,
    SSIM_BASELINE_WINDOW,
)
from app.models import interaction, statue_compare
from app.services import zone_service

logger = logging.getLogger(__name__)

_LEVEL_RANK = {"NORMAL": 0, "MODERADO": 1, "ALTO": 2, "CRÍTICO": 3}
# Folga em volta da figura da estátua (fração da caixa)
FIGURE_PAD = 0.04
# Pixel alterado: SSIM local abaixo de 1 - isto (estrutura) ou distância de
# cor Lab acima disto (tinta), já descontada a variação global de luz
SSIM_PIXEL_DIFF = 0.45
COLOR_PIXEL_DIFF = 28.0
# Borda tirada da silhueta da estátua (fração da altura dela): o alinhamento
# nunca é perfeito ao pixel e a borda contra o fundo viraria diferença
SILHOUETTE_DILATE = 0.06
# Folga em volta da silhueta de quem está na frente (cabelo, braço, sombra)
PERSON_MASK_PAD = 0.04
# Comparações seguidas sem conseguir alinhar → a câmera mudou de posição:
# a referência é descartada e refeita (com a estátua livre)
REALIGN_RESET_CHECKS = 5
# Menor mancha de alteração que conta (fração da silhueta visível)
MIN_CHANGE_BLOB = 0.004
# Silhueta no lugar da estátua só é a estátua se a imagem ali bater com a
# referência (correlação); abaixo disso é pessoa parada na frente
STATUE_SAME_NCC = 0.5
# Pixel com brilho a partir disto (0-255) = reflexo do sol no bronze
GLARE_LEVEL = 235
# Referência renovada a cada X s enquanto tudo está normal e a estátua
# está pelo menos Y visível (acompanha a luz do dia)
REF_REFRESH_SECONDS = 1800
REF_REFRESH_MIN_VISIBLE = 0.9
# Fundo que se mexe: média móvel de "mudou" por pixel; acima do nível sai da conta
DYNAMIC_RATE = 0.2
DYNAMIC_LEVEL = 0.3
# Comparações de aprendizado (sem alerta) logo após criar a referência
LEARN_CHECKS = 4
# Uma única mancha contínua a partir disto (% da estátua visível) já é ALTO
BLOB_ALTO_PCT = 1.5
# Comparação com mais que isto alterado não entra no aprendizado (frame ruim)
LEARN_MAX_CHANGE = 40.0


def _b64(img_rgb: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _zone_rect_px(frame: np.ndarray, camera_code: Optional[str] = None) -> tuple:
    """Retângulo comparado pelo SSIM (em pixels do frame original).

    Com o contorno da estátua calibrado, compara SÓ a estátua — mar, areia,
    guarda-sóis e gente passando ao redor deixam de contar como "alteração
    no monumento". Sem contorno, usa a zona (comportamento antigo).
    """
    zone = zone_service.get_statue(camera_code)["statue"] or zone_service.get_zone(camera_code)
    h, w = frame.shape[:2]
    x1 = int(w * zone["x_start"])
    x2 = int(w * zone["x_end"])
    y1 = int(h * zone["y_start"])
    y2 = int(h * zone["y_end"])
    return x1, y1, x2, y2


def figure_box_frac(frame: np.ndarray, camera_code: Optional[str], objects: Optional[list]) -> Optional[dict]:
    """Caixa da FIGURA da estátua (em frações do frame): a detecção que o
    detector marcou como a própria estátua (is_statue), recortada pelo
    contorno calibrado e com uma folga pequena. É o que a comparação usa —
    o contorno calibrado costuma incluir banco, chão e fundo."""
    statue = [d["bbox"] for d in (objects or []) if d.get("is_statue")]
    if not statue:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(statue, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    cfg = zone_service.get_statue(camera_code)["statue"]
    if cfg:  # não sai do contorno calibrado
        x1, y1 = max(x1, int(cfg["x_start"] * w)), max(y1, int(cfg["y_start"] * h))
        x2, y2 = min(x2, int(cfg["x_end"] * w)), min(y2, int(cfg["y_end"] * h))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    px, py = (x2 - x1) * FIGURE_PAD, (y2 - y1) * FIGURE_PAD
    return {"x_start": max(x1 - px, 0) / w, "y_start": max(y1 - py, 0) / h,
            "x_end": min(x2 + px, w) / w, "y_end": min(y2 + py, h) / h}


def _frac_px(frac: dict, frame: np.ndarray) -> tuple:
    h, w = frame.shape[:2]
    return (int(w * frac["x_start"]), int(h * frac["y_start"]), int(w * frac["x_end"]), int(h * frac["y_end"]))


def covered_fraction(frame: np.ndarray, camera_code: Optional[str], boxes: Optional[list],
                     area: Optional[tuple] = None) -> float:
    """Fração (0-1) da área comparada pelo SSIM coberta pelas caixas dadas."""
    zx1, zy1, zx2, zy2 = area or _zone_rect_px(frame, camera_code)
    if not boxes or zx2 <= zx1 or zy2 <= zy1:
        return 0.0
    cover = np.zeros((zy2 - zy1, zx2 - zx1), dtype=np.uint8)
    for (px1, py1, px2, py2) in boxes:
        ix1, iy1 = max(px1, zx1) - zx1, max(py1, zy1) - zy1
        ix2, iy2 = min(px2, zx2) - zx1, min(py2, zy2) - zy1
        if ix2 > ix1 and iy2 > iy1:
            cover[iy1:iy2, ix1:ix2] = 1
    return float(cover.mean())


def extrair_roi(frame: np.ndarray, camera_code: Optional[str] = None) -> np.ndarray:
    """Extrai a região de interesse (onde está o monumento) do frame.

    Usa a zona calibrada da câmera (zone_service), se houver;
    senão cai no quadrante padrão.
    """
    x1, y1, x2, y2 = _zone_rect_px(frame, camera_code)
    return frame[y1:y2, x1:x2]


class ChangeDetector:
    """
    Detecta mudanças no monumento comparando frames ao longo do tempo.
    
    Fluxo:
    1. set_reference() → captura frame, extrai ROI da estátua, armazena
    2. check() → captura novo frame, extrai ROI, compara com referência via SSIM
    3. Se SSIM mostrar mudança significativa → executa HF model no ROI
    4. Alerta combina: gravidade da mudança SSIM + confirmação HF
    """

    def __init__(self):
        # {camera_code: {
        #     "reference_image": np.ndarray (ROI),
        #     "reference_time": float,
        #     "last_check": float,
        #     "last_hf": dict,
        #     "history": list
        # }}
        self.monitored: dict[str, dict] = {}
        self.change_history: dict[str, list] = {}

    def set_reference(self, camera_code: str, frame: np.ndarray,
                      det_service=None, objects: Optional[list] = None) -> dict:
        """
        Define a imagem de referência do monumento.

        Guarda o frame inteiro (pra alinhar as próximas imagens) e a
        SILHUETA da estátua (segmentação) — a comparação acontece só dentro
        dela. Sem a estátua reconhecida, usa o contorno calibrado inteiro.
        objects: detecções do YOLO deste frame; se None e houver det_service,
        roda a detecção aqui.
        """
        if objects is None and det_service is not None:
            try:
                objects = det_service.yolo_detector.detect(frame, camera_code=camera_code, draw=False)["objects"]
            except Exception as e:
                logger.warning(f"YOLO na referência: {e}")
        figure = figure_box_frac(frame, camera_code, objects)
        area = _frac_px(figure, frame) if figure else _zone_rect_px(frame, camera_code)

        # Silhueta da estátua (a figura de bronze, sem banco/chão/fundo)
        silhouette = None
        statue_boxes = [d["bbox"] for d in (objects or []) if d.get("is_statue")]
        if statue_boxes:
            box = max(statue_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            silhouette = statue_compare.statue_silhouette(frame, box)
        if silhouette is None:
            silhouette = np.zeros(frame.shape[:2], bool)
            silhouette[area[1]:area[3], area[0]:area[2]] = True
            mode = "contorno calibrado"
        else:
            # A segmentação (treinada com gente, não com bronze) costuma
            # perder mãos/pernas da estátua: dilata a silhueta e limita à
            # caixa da estátua. O fundo que entrar junto (mar, guarda-sol) é
            # aprendido como "fundo que se mexe" nas primeiras comparações
            # (ver "dynamic" em check) e sai da conta.
            bh = box[3] - box[1]
            k = max(int(bh * SILHOUETTE_DILATE), 1)
            silhouette = cv2.dilate(silhouette.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
            px, py = int((box[2] - box[0]) * 0.03), int(bh * 0.03)
            in_box = np.zeros_like(silhouette)
            in_box[max(box[1] - py, 0):box[3] + py, max(box[0] - px, 0):box[2] + px] = True
            silhouette &= in_box
            ys, xs = np.nonzero(silhouette)
            area = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            mode = "silhueta da estátua"

        roi = frame[area[1]:area[3], area[0]:area[2]]
        hf_result = None
        if det_service and det_service.hf_vandalism.model_loaded:
            try:
                hf_result = det_service.hf_vandalism.predict_image(roi)
            except Exception as e:
                logger.warning(f"HF na referência: {e}")

        self.monitored[camera_code] = {
            "reference_image": roi,
            "reference_frame": frame.copy(),
            "reference_gray": cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            "silhouette": silhouette,
            "area": area,
            "compared_area": mode,
            "figure_frac": figure,
            "reference_time": time.time(),
            "last_check": time.time(),
            "last_hf": hf_result,
            # % alterado das últimas comparações normais (linha de base)
            "baseline": [],
            # comparações seguidas com salto ALTO+: [(timestamp, nível)]
            "streak": [],
            # comparações seguidas sem conseguir alinhar (câmera mudou de posição)
            "misaligned": 0,
            # Por pixel da área: quanto ele costuma mudar em comparações
            # normais (0-1). Alto = fundo que se mexe (mar, guarda-sol,
            # areia vista pelas frestas) → fora da comparação
            "dynamic": np.zeros((area[3] - area[1], area[2] - area[0]), np.float32),
        }
        logger.info(f"Referência definida para câmera {camera_code} ({mode})")
        return {
            "success": True,
            "camera_code": camera_code,
            "timestamp": time.time(),
            "reference_hf": hf_result,
            "compared_area": mode,
        }

    def check(self, camera_code: str, current_frame: np.ndarray,
              det_service=None, ignore_boxes: Optional[list] = None) -> dict:
        """
        Compara o frame atual com a referência — SÓ na silhueta da estátua.

          1. alinha a imagem atual ao enquadramento da referência (a câmera
             mexe posição/zoom); se não der, pula (e depois de várias vezes
             seguidas descarta a referência pra ser refeita);
          2. tira da conta tudo que está NA FRENTE da estátua: pessoas
             (silhueta segmentada, com folga) e as caixas em ignore_boxes
             (veículos etc.);
          3. mede, dentro da silhueta visível, a área com estrutura
             diferente (peça faltando, quebra) ou cor diferente (tinta).
        """
        if camera_code not in self.monitored:
            return {"success": False, "error": "Câmera não monitorada. Defina referência primeiro."}

        ref_data = self.monitored[camera_code]
        if "reference_frame" not in ref_data:  # referência antiga (antes da silhueta): refaz
            self.monitored.pop(camera_code, None)
            return {"success": False, "error": "Referência antiga descartada — será recriada"}
        ref_img = ref_data["reference_image"]
        ref_frame = ref_data["reference_frame"]
        silhouette = ref_data["silhouette"]
        figure = ref_data.get("figure_frac")
        zx1, zy1, zx2, zy2 = ref_data["area"]
        h, w = ref_frame.shape[:2]
        if current_frame.shape[:2] != (h, w):
            current_frame = cv2.resize(current_frame, (w, h))

        # ─── 1. Alinhamento com a referência ───────────────────────
        M, align_info = statue_compare.align(ref_data["reference_gray"],
                                             cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY),
                                             (zx1, zy1, zx2, zy2))
        if M is None:
            ref_data["misaligned"] = ref_data.get("misaligned", 0) + 1
            if ref_data["misaligned"] >= REALIGN_RESET_CHECKS:
                logger.info(f"[{camera_code}] enquadramento mudou em {REALIGN_RESET_CHECKS} comparações "
                            f"seguidas — referência descartada, será recriada")
                self.monitored.pop(camera_code, None)
            return {"success": False, "skipped": True,
                    "error": f"Comparação pulada: {align_info.get('reason')} "
                             f"({ref_data['misaligned']}/{REALIGN_RESET_CHECKS} até refazer a referência)"}
        ref_data["misaligned"] = 0
        warped = cv2.warpAffine(current_frame, M, (w, h), flags=cv2.INTER_LINEAR)
        in_view = cv2.warpAffine(np.ones((h, w), np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0

        # ─── 2. O que está na frente da estátua (não conta) ─────────
        front = np.zeros((h, w), np.uint8)
        pad = max(int((zy2 - zy1) * PERSON_MASK_PAD), 3)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        for box, mk in statue_compare.people_masks(current_frame):
            mk_ref = cv2.warpAffine(mk.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
            # No lugar da estátua E com a imagem igual à da referência = a
            # própria estátua. Pessoa parada bem na frente dela ocupa o mesmo
            # lugar, mas a imagem é outra — é gente, sai da conta.
            if (statue_compare.mask_iou(mk_ref, silhouette) >= 0.6
                    and statue_compare.masked_ncc(ref_data["reference_gray"], warped_gray,
                                                  silhouette & mk_ref) >= STATUE_SAME_NCC):
                continue
            front |= cv2.dilate(mk_ref.astype(np.uint8), np.ones((pad, pad), np.uint8))
        # Caixas do YOLO (pessoas, veículos) também saem da conta, no
        # enquadramento da referência — só reduzem a área comparada
        ignored_count = 0
        for (px1, py1, px2, py2) in ignore_boxes or []:
            pts = cv2.transform(np.float32([[[px1, py1]], [[px2, py1]], [[px1, py2]], [[px2, py2]]]), M).reshape(-1, 2)
            bx1, by1 = np.clip(pts.min(axis=0).astype(int), 0, [w, h])
            bx2, by2 = np.clip(pts.max(axis=0).astype(int), 0, [w, h])
            if bx2 <= bx1 or by2 <= by1 or not silhouette[by1:by2, bx1:bx2].any():
                continue
            ignored_count += 1
            front[by1:by2, bx1:bx2] = 1
        front = front.astype(bool)

        sil = silhouette[zy1:zy2, zx1:zx2]
        visible = sil & ~front[zy1:zy2, zx1:zx2] & in_view[zy1:zy2, zx1:zx2]
        visible_frac = visible.sum() / max(sil.sum(), 1)
        visible_pct = visible_frac * 100
        if visible_frac < 1 - STATUE_OCCLUSION_SKIP:
            return {"success": False, "skipped": True,
                    "error": f"Estátua {100 - visible_pct:.0f}% encoberta por pessoas/veículos — comparação pulada neste print"}
        # Encoberta por algo que o YOLO não detectou (ônibus parado na frente):
        # a silhueta inteira deixa de parecer a da referência. Tinta/peça
        # faltando mudam parte dela, mas a estátua continua reconhecível.
        sil_ncc = statue_compare.masked_ncc(ref_data["reference_gray"], warped_gray, silhouette & in_view & ~front)
        if sil_ncc < SURFACE_VISIBLE_MIN_NCC:
            return {"success": False, "skipped": True,
                    "error": f"Estátua fora de vista (semelhança {sil_ncc:.2f} com a referência) — "
                             f"algo parado na frente; comparação pulada neste print"}

        current_roi = warped[zy1:zy2, zx1:zx2]

        # ─── 3. Diferença dentro da silhueta visível ─────────────────
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
        gray_cur = cv2.cvtColor(current_roi, cv2.COLOR_RGB2GRAY)
        # Reflexo do sol no bronze (pixel estourado em qualquer uma das duas
        # imagens) muda de lugar ao longo do dia: fica fora da comparação
        glare = cv2.dilate(((gray_ref >= GLARE_LEVEL) | (gray_cur >= GLARE_LEVEL)).astype(np.uint8),
                           np.ones((7, 7), np.uint8)).astype(bool)
        visible = visible & ~glare
        observed = visible.copy()  # sem o fundo aprendido — usado pra aprender
        dynamic = ref_data.get("dynamic")
        if dynamic is not None and dynamic.shape == visible.shape:
            visible = visible & (dynamic < DYNAMIC_LEVEL)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_ref = cv2.GaussianBlur(clahe.apply(gray_ref), (3, 3), 0)
        gray_cur = cv2.GaussianBlur(clahe.apply(gray_cur), (3, 3), 0)
        _, ssim_map = ssim(gray_ref, gray_cur, full=True, data_range=255)

        # Pixel alterado = ESTRUTURA diferente (peça faltando, quebra, risco)
        # OU COR diferente (tinta/pichação), descontada a variação de luz
        struct_changed = (1 - ssim_map) > SSIM_PIXEL_DIFF
        lab_ref = cv2.cvtColor(cv2.GaussianBlur(ref_img, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_cur = cv2.cvtColor(cv2.GaussianBlur(current_roi, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
        dlab = lab_cur - lab_ref
        dlab[..., 0] -= float(np.median(dlab[..., 0][visible])) if visible.any() else 0.0
        color_changed = np.sqrt((0.5 * dlab[..., 0]) ** 2 + dlab[..., 1] ** 2 + dlab[..., 2] ** 2) > COLOR_PIXEL_DIFF
        changed_any = struct_changed | color_changed
        changed = changed_any & visible
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(changed.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        thresh[~visible] = 0

        visible_pixels = max(int(visible.sum()), 1)
        score = float(ssim_map[visible].mean()) if visible.any() else 1.0

        # Só conta MANCHA de tamanho relevante: dano (tinta, peça faltando)
        # forma uma área contínua; pontinhos espalhados são ruído de
        # compressão e reflexo de sol no bronze
        min_blob = max(150, int(visible_pixels * MIN_CHANGE_BLOB))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_blob]
        kept = np.zeros_like(thresh)
        cv2.drawContours(kept, contours, -1, 255, -1)
        thresh = np.where(visible, kept & thresh, 0).astype(np.uint8)
        change_pct = int((thresh > 0).sum()) / visible_pixels * 100

        changes = []
        for cnt in contours:
            a = cv2.contourArea(cnt)
            if a >= min_blob:
                x, y, bw, bh = cv2.boundingRect(cnt)
                changes.append({"bbox": [int(x), int(y), int(x + bw), int(y + bh)], "area_pixels": int(a),
                                "area_percent": round(a / visible_pixels * 100, 3)})

        # Imagem das diferenças: fora da silhueta escurecido, encoberto em
        # azul, alterado em vermelho
        highlight = current_roi.copy()
        highlight[~sil] = (highlight[~sil] * 0.35).astype(np.uint8)
        hidden = sil & ~visible
        highlight[hidden] = (0.5 * highlight[hidden] + 0.5 * np.array([60, 120, 255])).astype(np.uint8)
        red = thresh > 0
        highlight[red] = (0.4 * highlight[red] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        for ch in changes:
            x1, y1, x2, y2 = ch["bbox"]
            cv2.rectangle(highlight, (x1, y1), (x2, y2), (255, 0, 0), 2)

        ignored_alert = None
        if hidden.any():
            ignored_alert = {
                "level": "INFO",
                "message": f"👤 {100 - visible_pct:.0f}% da estátua encoberta por pessoas/objetos (fora da comparação)",
                "count": ignored_count,
            }

        # ─── 2. Nível pelo AUMENTO sobre a linha de base ──────────
        # O % bruto sobe e desce com sol/sombra ao longo do dia; o que
        # interessa é um salto em relação ao normal recente dessa câmera.
        state = self.monitored[camera_code]
        baseline_hist = state.setdefault("baseline", [])
        streak = state.setdefault("streak", [])
        baseline = float(np.median(baseline_hist)) if baseline_hist else 0.0
        increase = max(change_pct - baseline, 0.0)
        ssim_alert_level = self._ssim_alert_level(increase)
        # Dano CONCENTRADO (peça arrancada, óculos do Drummond): uma mancha
        # contínua só é pequena perto da estátua inteira, mas é o alvo
        # clássico de furto — basta ela pra ser ALTO
        largest_blob = max((c["area_percent"] for c in changes), default=0.0)
        if (largest_blob >= BLOB_ALTO_PCT and increase >= BLOB_ALTO_PCT
                and _LEVEL_RANK[ssim_alert_level] < _LEVEL_RANK["ALTO"]):
            ssim_alert_level = "ALTO"

        # ─── Fundo que se mexe (mar, guarda-sol, areia pelas frestas) ──
        # Fase de APRENDIZADO: as primeiras LEARN_CHECKS comparações depois
        # da referência só observam quais pixels mudam — sem nível, sem
        # alerta. Depois, continua aprendendo devagar só com a situação
        # normal (dano persistente, ALTO+, nunca vira fundo).
        learning = False
        if dynamic is not None and dynamic.shape == observed.shape:
            n = ref_data.get("learn_n", 0)
            if n < LEARN_CHECKS and change_pct >= LEARN_MAX_CHANGE:
                # Frame corrompido/cena toda diferente: não entra no aprendizado
                learning = True
                ssim_alert_level, increase = "NORMAL", 0.0
            elif n < LEARN_CHECKS:
                learning = True
                ref_data.setdefault("learn_chg", np.zeros_like(dynamic))
                ref_data.setdefault("learn_obs", np.zeros_like(dynamic))
                ref_data["learn_chg"][observed] += changed_any[observed]
                ref_data["learn_obs"][observed] += 1
                ref_data["learn_n"] = n + 1
                if n + 1 == LEARN_CHECKS:
                    obs = ref_data.pop("learn_obs")
                    chg = ref_data.pop("learn_chg")
                    # nunca visto sem gente na frente: sem como saber → fora
                    dynamic[:] = np.where(obs > 0, chg / np.maximum(obs, 1), 1.0)
                    logger.info(f"[{camera_code}] fundo aprendido: {(dynamic >= DYNAMIC_LEVEL).mean() * 100:.0f}% "
                                f"da área sai da comparação")
                ssim_alert_level, increase = "NORMAL", 0.0
            elif _LEVEL_RANK[ssim_alert_level] < _LEVEL_RANK["ALTO"]:
                dynamic[observed] = (1 - DYNAMIC_RATE) * dynamic[observed] + DYNAMIC_RATE * changed_any[observed]

        # ─── 3. Confirmação: o salto precisa se manter ─────────────
        # Gente sentada ao lado, sombra de guarda-sol, reflexo: somem no
        # print seguinte. Peça faltando ou tinta: continuam.
        now = time.time()
        if _LEVEL_RANK[ssim_alert_level] >= _LEVEL_RANK["ALTO"]:
            streak.append((now, ssim_alert_level))
        else:
            streak.clear()
            if ssim_alert_level == "NORMAL" and not learning:
                baseline_hist.append(change_pct)
                del baseline_hist[:-SSIM_BASELINE_WINDOW]
        confirmed = (len(streak) >= SSIM_CONFIRM_CHECKS
                     and now - streak[0][0] >= SSIM_CONFIRM_SECONDS)
        # Nível confirmado = o menor entre as comparações da sequência
        confirmed_level = (min((lvl for _, lvl in streak[-SSIM_CONFIRM_CHECKS:]), key=_LEVEL_RANK.get)
                           if confirmed else "NORMAL")
        confirmation = {
            "baseline_pct": round(baseline, 2),
            "increase_pct": round(increase, 2),
            "checks": len(streak),
            "checks_needed": SSIM_CONFIRM_CHECKS,
            "learning": learning,
            "learn_progress": f"{ref_data.get('learn_n', LEARN_CHECKS)}/{LEARN_CHECKS}",
            "confirmed": confirmed,
        }

        # ─── 4. HF Model no ROI (só com mudança confirmada) ───────
        hf_result = None
        hf_alert = None
        if confirmed and det_service and det_service.hf_vandalism.model_loaded:
            try:
                hf_result = det_service.hf_vandalism.predict_image(current_roi)
                hf_alert, _ = self._hf_alert(hf_result)
            except Exception as e:
                logger.warning(f"Erro HF check: {e}")

        # ─── 5. Alerta combinado ─────────────────────────────────
        alert, final_level = self._combined_alert(
            confirmed_level, change_pct, changes, hf_result, hf_alert
        )
        if alert and alert.get("source") == "ssim":
            alert["message"] = (
                f"⚠️ {final_level}: Monumento alterado — {change_pct:.1f}% "
                f"(+{increase:.1f} p.p. sobre o normal de {baseline:.1f}%), "
                f"mantido em {len(streak)} comparações seguidas"
            )

        # ─── 6. Mudança confirmada logo depois de alguém mexer ────
        # Os prints são espaçados — o gesto de arrancar uma peça pode cair
        # entre dois deles. Se houve golpe CONFIRMADO contra a estátua no
        # vídeo há pouco e agora a mudança
        # se manteve, é o cenário de furto/dano: sobe pra CRÍTICO.
        last = interaction.last_interaction(camera_code)
        if alert and last and now - last <= INTERACTION_MEMORY_SECONDS:
            minutos = max(int((now - last) // 60), 1)
            final_level = "CRÍTICO"
            alert = {
                "level": "CRÍTICO",
                "message": (f"🚨 CRÍTICO: Estátua alterada ({change_pct:.1f}%, +{increase:.1f} p.p.) "
                            f"após interação há ~{minutos} min — possível retirada de peça/dano"),
                "source": "ssim+interacao",
            }

        # Alerta emitido: o estado atual vira o novo normal — só alerta de
        # novo se houver outro salto (não repete o mesmo dano a cada print)
        if alert:
            baseline_hist[:] = [change_pct]
            streak.clear()

        # Referência móvel: com tudo normal e a estátua inteira à vista, a
        # referência é renovada com a imagem atual (já alinhada) — acompanha
        # o sol mudando reflexo e sombra no bronze ao longo do dia. Nunca
        # renova com alteração em andamento (o dano não vira "normal").
        if (not alert and ssim_alert_level == "NORMAL" and not streak
                and visible_frac >= REF_REFRESH_MIN_VISIBLE
                and in_view[zy1:zy2, zx1:zx2].all()
                and now - ref_data["reference_time"] >= REF_REFRESH_SECONDS):
            ref_data.update(reference_frame=warped, reference_gray=cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY),
                            reference_image=current_roi.copy(), reference_time=now)
            baseline_hist.clear()
            logger.info(f"[{camera_code}] referência renovada (tudo normal, estátua livre)")

        # Atualiza estado
        self.monitored[camera_code]["last_check"] = time.time()
        self.monitored[camera_code]["last_hf"] = hf_result

        # Registra histórico
        timestamp = time.time()
        entry = {
            "timestamp": timestamp,
            "similarity_score": round(float(score), 4),
            "change_percentage": round(change_pct, 2),
            "change_regions": len(changes),
            "ssim_alert": ssim_alert_level,
            "hf_prediction": hf_result,
            "alert_level": final_level,
        }
        if camera_code not in self.change_history:
            self.change_history[camera_code] = []
        self.change_history[camera_code].append(entry)
        if len(self.change_history[camera_code]) > 100:
            self.change_history[camera_code].pop(0)

        # Codifica highlight como base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(highlight, cv2.COLOR_RGB2BGR),
                                 [cv2.IMWRITE_JPEG_QUALITY, 85])
        highlight_b64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "success": True,
            "camera_code": camera_code,
            "timestamp": timestamp,
            "similarity_score": round(float(score), 4),
            "change_percentage": round(change_pct, 2),
            # % da área calibrada da estátua visível nesta comparação
            "visible_percentage": round(visible_pct, 1),
            "change_regions": len(changes),
            "significant_changes": changes[:10],
            "ssim_alert_level": ssim_alert_level,
            "confirmation": confirmation,
            "hf_prediction": hf_result,
            "alert": alert,
            "alert_level": final_level,
            "ignored_objects_count": ignored_count,
            "ignored_objects_alert": ignored_alert,
            "highlight_image_base64": highlight_b64,
            # Lado a lado: a mesma área na referência e agora
            "reference_roi_base64": _b64(ref_img),
            "current_roi_base64": _b64(current_roi),
            "compared_area": ref_data.get("compared_area", "contorno calibrado"),
            "alignment": align_info,
            "reference_time": ref_data.get("reference_time"),
            "history_count": len(self.change_history[camera_code]),
        }

    def get_history(self, camera_code: str) -> list:
        """Retorna histórico de verificações"""
        return self.change_history.get(camera_code, [])

    def _ssim_alert_level(self, increase_pct: float) -> str:
        """Nível pelo aumento (pontos percentuais) sobre a linha de base.

        Número de regiões saiu do critério: textura de pedra/areia e ruído
        de compressão fragmentam o diff em muitas regiões pequenas sem que
        nada tenha mudado no monumento.
        """
        if increase_pct >= SSIM_INCREASE_CRITICO:
            return "CRÍTICO"
        if increase_pct >= SSIM_INCREASE_ALTO:
            return "ALTO"
        if increase_pct >= SSIM_INCREASE_MODERADO:
            return "MODERADO"
        return "NORMAL"

    def _hf_alert(self, prediction: dict) -> tuple:
        """Verifica alerta do HF model. Retorna (alert_dict, level)."""
        if not prediction or "error" in prediction:
            return None, None
        burglary = prediction.get("burglary", 0)
        vandalism_val = prediction.get("vandalism", 0)
        max_anormal = max(burglary, vandalism_val)
        if max_anormal < 0.3:
            return None, None
        if max_anormal >= 0.7:
            level = "CRÍTICO"
        elif max_anormal >= 0.5:
            level = "ALTO"
        else:
            level = "MODERADO"
        tipos = []
        if burglary >= 0.3:
            tipos.append("roubo/furto")
        if vandalism_val >= 0.3:
            tipos.append("vandalismo")
        msg = f"{level}: {', '.join(tipos).capitalize()} (confiança: {max_anormal*100:.0f}%)"
        alert = {
            "level": level,
            "max_probability": round(max_anormal, 3),
            "types": tipos,
            "burglary_prob": round(burglary, 3),
            "vandalism_prob": round(vandalism_val, 3),
            "message": msg,
        }
        return alert, level

    def _combined_alert(self, ssim_level: str, change_pct: float,
                        changes: list, hf_result: dict,
                        hf_alert: dict) -> tuple:
        """
        Combina alerta SSIM + HF.
        
        - SSIM sozinho com MODERADO+ já gera alerta de mudança física
        - HF confirma se é roubo/vandalismo
        - Se ambos alertam, nível sobe
        - Se NÃO há mudança física (SSIM ≈ 100%), SUPRIME alerta HF
          (modelo HF tende a dar falso-positivo em imagens estáticas
          pois foi treinado para vídeo e sempre produz ~62% em qualquer frame)
        """
        # Só SSIM já indica mudança física (pichação, dano, peça quebrada)
        has_ssim_change = ssim_level in ("MODERADO", "ALTO", "CRÍTICO")
        has_hf_alert = hf_alert is not None

        # ─── Se NÃO há mudança física, ignora HF completamente ─────
        # O modelo HF (KzRyan/Burglary_and_Vandalism) é de classificação
        # de vídeo e sempre produz ~60-62% em imagens estáticas, gerando
        # falso-positivo. Só consideramos HF se SSIM detectou alteração.
        if not has_ssim_change:
            if has_hf_alert:
                logger.debug(
                    f"HF alert suprimido (SSIM sem mudança): "
                    f"similarity=~{1 - change_pct/100:.4f}, "
                    f"hf={hf_alert.get('max_probability', 0):.3f}"
                )
            # Sem mudança SSIM → NORMAL, mesmo que HF alerte
            return None, "NORMAL"

        # O modelo HF NÃO decide nem sobe o nível: em imagem parada ele dá
        # 60-86% "vandalismo" pra quase qualquer cena (foi treinado com vídeo)
        # e transformava toda mudança pequena em CRÍTICO. O nível é só o da
        # alteração medida na estátua.
        if has_ssim_change:
            # Só mudança física (sem confirmação HF)
            level = ssim_level
            msg = (f"⚠️ {level}: {change_pct:.1f}% do monumento alterado "
                   f"({len(changes)} região(is))")
            alert = {"level": level, "message": msg, "source": "ssim"}
            return alert, level

        return None, "NORMAL"
