"""
Configurações do Sistema de Visão Computacional - Patrimônios do Rio
CO-RIO - Coordenadoria de Operações e Resiliência
Backend FastAPI
"""

import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# ─── Caminhos ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models_cache"
DATA_DIR = BASE_DIR / "data"

for d in [ASSETS_DIR, IMAGES_DIR, CACHE_DIR, MODELS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── API de Câmeras ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://10.50.3.96:8000")
API_KEY = os.getenv("API_KEY", "")
API_EMAIL = os.getenv("API_EMAIL", "")
API_PASSWORD = os.getenv("API_PASSWORD", "")

# ─── Stream ──────────────────────────────────────────────────────
STREAM_BASE_URL = os.getenv("STREAM_BASE_URL", "http://10.50.3.11:5002")
STREAM_ENDPOINT = "/stream"
STREAM_KEY = os.getenv("STREAM_KEY", "")

# Servidor da página de vídeo WebRTC/WHEP da Tixxi (stream_type="html").
# Usado pelo proxy em app/api/video_proxy.py — ver comentário lá do porquê.
VIDEO_PAGE_BASE_URL = os.getenv("VIDEO_PAGE_BASE_URL", "http://10.50.3.11:5001")

# O servidor de mídia só anuncia o IP público (187.111.99.23) nos candidatos
# ICE do SDP answer, que não é alcançável de dentro da rede interna (sem
# hairpin NAT) — a negociação WHEP dá 201 mas o ICE termina em "failed".
# O proxy de WHEP duplica cada candidato trocando o IP por este host interno
# (onde a porta de mídia está aberta). Vazio desativa. Padrão: o host de
# VIDEO_PAGE_BASE_URL.
WEBRTC_EXTRA_ICE_HOST = os.getenv(
    "WEBRTC_EXTRA_ICE_HOST", urlparse(VIDEO_PAGE_BASE_URL).hostname or ""
)

# Intervalo da renovação antecipada do token da API + KEYs de stream
# (que vencem em ~1h). 45min deixa folga antes do vencimento.
TOKEN_REFRESH_SECONDS = int(os.getenv("TOKEN_REFRESH_SECONDS", str(45 * 60)))

# ─── Webhook OctaVision ──────────────────────────────────────────
# Token exigido em ?token=... no endpoint /api/webhooks/octavision.
OCTAVISION_WEBHOOK_TOKEN = os.getenv("OCTAVISION_WEBHOOK_TOKEN", "")

# ─── Servidor FastAPI ────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# ─── YOLO ────────────────────────────────────────────────────────
YOLO_MODEL = os.getenv("YOLO_MODEL", str(BASE_DIR.parent / "yolo26n.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

# ─── Hugging Face ────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
VANDALISM_MODEL_REPO = os.getenv(
    "VANDALISM_MODEL_REPO", "KzRyan/Burglary_and_Vandalism"
)
VANDALISM_MODEL_FILE = os.getenv(
    "VANDALISM_MODEL_FILE", "HYBRID_CNN_TRANSFORMER_MODEL.pth"
)

# ─── Mapa ────────────────────────────────────────────────────────
DEFAULT_MAP_CENTER = {"lat": -22.9068, "lon": -43.1729}
DEFAULT_MAP_ZOOM = 11

# ─── Captura contínua de vídeo (fase 1 da cascata) ───────────────
# Câmeras com página de vídeo aberta permanentemente num Chrome headless,
# lendo frames a LIVE_CAPTURE_FPS (ver services/live_capture.py). Cada
# câmera = 1 Chrome (~200-400 MB de RAM) — começar com poucas e medir em
# GET /api/monitor/live-capture/status antes de adicionar mais.
LIVE_CAPTURE_CODES = [c.strip() for c in os.getenv("LIVE_CAPTURE_CODES", "000056").split(",") if c.strip()]
LIVE_CAPTURE_FPS = float(os.getenv("LIVE_CAPTURE_FPS", "4"))
LIVE_CAPTURE_BUFFER_SECONDS = float(os.getenv("LIVE_CAPTURE_BUFFER_SECONDS", "4"))
# Análise contínua (YOLO + rastreamento ByteTrack) sobre o vídeo ao vivo,
# por câmera — ver services/live_analysis.py
LIVE_ANALYSIS_FPS = float(os.getenv("LIVE_ANALYSIS_FPS", "2"))

# ─── Superfície protegida + pessoa suspeita (análise contínua) ───
# O que se procura é mudança na ESTRUTURA FÍSICA da estátua (pichação,
# peça arrancada, dano) — algo que FICA. Por isso as janelas são de
# minutos: pessoa, sombra, reflexo, vendedor passando e falha de
# transmissão duram segundos e não chegam a confirmar.
# Superfície = a figura da estátua (dentro do contorno calibrado).
# Pessoa junto à superfície por este tempo seguido → "esteve junto".
SURFACE_SUSPECT_SECONDS = float(os.getenv("SURFACE_SUSPECT_SECONDS", "4"))
# Uma mudança na superfície é associada a quem esteve junto até este tempo
# antes (cobre toda a janela de confirmação abaixo)
SURFACE_SUSPECT_MEMORY_SECONDS = float(os.getenv("SURFACE_SUSPECT_MEMORY_SECONDS", "900"))
# Fração da superfície visível alterada de forma persistente pra contar como
# mudança. Só conta a mudança LONGE de quem está na frente (ver
# SurfaceMonitor: sombra/mochila/roupa fora da caixa do YOLO ficam de fora).
# 1,5% gerava alerta com turista sentado ao lado do Drummond.
SURFACE_CHANGE_MIN_FRAC = float(os.getenv("SURFACE_CHANGE_MIN_FRAC", "0.03"))
# A partir desta fração a alteração é CRÍTICO; entre MIN e isto, ALTO
SURFACE_CRITICAL_FRAC = float(os.getenv("SURFACE_CRITICAL_FRAC", "0.08"))
# A alteração precisa estar presente (parada, fora das pessoas) há pelo
# menos este tempo...
SURFACE_CHANGE_CONFIRM_SECONDS = float(os.getenv("SURFACE_CHANGE_CONFIRM_SECONDS", "180"))
# ...e CONTINUAR com a estátua livre (ninguém ao alcance) por este tempo.
# Turista abraçando/encostando/sentado ao lado muda os pixels só enquanto
# está lá; pichação e dano ficam depois que todos saem.
SURFACE_CLEAR_CONFIRM_SECONDS = float(os.getenv("SURFACE_CLEAR_CONFIRM_SECONDS", "60"))
# Queda curta da alteração (alguém passou na frente, frame ruim) não zera
# a contagem — só some de verdade se ficar abaixo do limite por este tempo
SURFACE_CHANGE_GAP_SECONDS = float(os.getenv("SURFACE_CHANGE_GAP_SECONDS", "20"))
# Mudança sem ninguém junto antes, por este tempo, é absorvida (luz do dia,
# objeto deixado, câmera mexeu) — não dispara depois quando alguém chegar
SURFACE_UNATTRIBUTED_ABSORB_SECONDS = 300
# Após um evento, a mesma câmera não gera outro antes disto
SURFACE_EVENT_COOLDOWN_SECONDS = int(os.getenv("SURFACE_EVENT_COOLDOWN_SECONDS", "1800"))
# Pixel alterado precisa estar parado (sem movimento) por este tempo
SURFACE_STILL_SECONDS = float(os.getenv("SURFACE_STILL_SECONDS", "15"))
# Semelhança (NCC) da figura com a aparência gravada (calibração) abaixo
# disto = a estátua NÃO está à vista (ônibus/caminhão/guarda-sol parado na
# frente que o YOLO não detectou): a superfície fica em pausa. Pichação ou
# peça faltando mudam parte da figura, mas ela continua reconhecível.
SURFACE_VISIBLE_MIN_NCC = float(os.getenv("SURFACE_VISIBLE_MIN_NCC", "0.3"))
# Clipe de evidência: segundos antes do início da mudança e após a confirmação
EVIDENCE_PRE_SECONDS = float(os.getenv("EVIDENCE_PRE_SECONDS", "15"))
EVIDENCE_POST_SECONDS = float(os.getenv("EVIDENCE_POST_SECONDS", "5"))

# ─── Monitoramento contínuo em background ────────────────────────
# Varre todas as câmeras dos patrimônios sequencialmente (uma por vez,
# não em paralelo — evita sobrecarregar a máquina que roda o YOLO/HF).
BACKGROUND_MONITOR_ENABLED = os.getenv("BACKGROUND_MONITOR_ENABLED", "true").lower() == "true"
BACKGROUND_MONITOR_INTERVAL_SECONDS = int(os.getenv("BACKGROUND_MONITOR_INTERVAL_SECONDS", "20"))

# ─── Quadrante de foco no monumento (ROI) ────────────────────────
# Retângulo central do frame onde o monumento está posicionado.
# Usado tanto pelo diff SSIM (change_detector) quanto pelo filtro
# de zona do YOLO (detector) — mesma "zona do monumento" para os dois.
ROI_X_START = 0.20   # 20% da largura
ROI_X_END = 0.80     # 80% da largura
ROI_Y_START = 0.15   # 15% da altura
ROI_Y_END = 0.75     # 75% da altura

# Classes do YOLO (COCO) que representam objetos de risco ao patrimônio,
# com o nível de alerta BASE de cada uma (antes de considerar tempo de
# permanência — ver DWELL_ALERT_SECONDS / risk_tracker.py).
# faca/tesoura = perigo claro, alerta ALTO já na primeira detecção.
# Os demais são objetos comuns que também servem pra bater/quebrar —
# entram como MODERADO pra não gritar CRÍTICO por alguém só carregando
# uma garrafa d'água perto do monumento.
# Limitação: o COCO não tem classe pra spray de pichação, martelo,
# barra de metal etc. — isso exigiria um modelo customizado/
# open-vocabulary, fora de escopo aqui.
# guarda-chuva saiu da lista: nas câmeras de orla (Drummond) todo
# guarda-sol de praia atrás da estátua virava alerta falso.
RISK_CLASSES = {
    "faca": "ALTO",
    "tesoura": "ALTO",
    "taco_de_beisebol": "MODERADO",
    "garrafa": "MODERADO",
}

# ─── Contorno da estátua + interação (pose) ──────────────────────
# Com o contorno da estátua calibrado (zone_service.get_statue), um
# objeto de risco só conta se pelo menos esta fração da caixa dele
# estiver DENTRO do contorno — objeto só "atrás" na imagem (guarda-sol,
# garrafa na areia) não sobrepõe o suficiente.
STATUE_OBJECT_OVERLAP = 0.5

# Estátuas de figura humana são detectadas pelo YOLO como "pessoa". Uma
# detecção de pessoa com IoU >= isto em relação ao contorno calibrado é a
# PRÓPRIA estátua — sai das regras de pessoa (permanência, interação,
# área ignorada no SSIM).
STATUE_SELF_IOU = 0.5

# YOLO26-pose (keypoints: mãos, quadril, pés) — baixado automaticamente
# pelo ultralytics na primeira execução (~6 MB).
POSE_MODEL = os.getenv("POSE_MODEL", str(BASE_DIR.parent / "yolo26n-pose.pt"))
# YOLO26-seg (máscaras): silhueta exata da estátua e de quem está na frente
# na comparação com a referência — baixado automaticamente (~6 MB).
SEG_MODEL = os.getenv("SEG_MODEL", str(BASE_DIR.parent / "yolo26n-seg.pt"))
POSE_KEYPOINT_CONF = 0.35

# ─── Golpe na estátua (pose, só na análise contínua) ─────────────
# Tocar, abraçar, posar pra foto ou sentar/subir NÃO é alerta. O que se
# procura é AÇÃO contra a estátua: o punho se move bruscamente e termina
# SOBRE a figura (bater, chutar, raspar com objeto). Velocidade medida em
# alturas do corpo da pessoa por segundo (independe da distância da câmera).
# A 2 FPS o golpe (~0,3 s) cai entre dois frames: medido, um soco sai
# ~0,8-1,0; encostar a mão pra foto, ~0,2-0,3.
STRIKE_SPEED = float(os.getenv("STRIKE_SPEED", "0.6"))
# Deslocamento acima disto (alturas do corpo) entre dois frames = o
# keypoint "pulou" (pose trocou de pessoa/mão) — não é golpe
STRIKE_MAX_JUMP = 1.0
# Golpes da MESMA pessoa dentro desta janela...
STRIKE_WINDOW_SECONDS = float(os.getenv("STRIKE_WINDOW_SECONDS", "10"))
# ...a partir deste número → ALTO. Com objeto na mão (garrafa, taco,
# faca...), um golpe já basta.
STRIKE_MIN_COUNT = int(os.getenv("STRIKE_MIN_COUNT", "3"))
# Falha de detecção por até este tempo não zera a contagem (pose pisca)
INTERACTION_GRACE_SECONDS = 5

# Mudança física no contorno da estátua até este tempo depois de um
# golpe confirmado → CRÍTICO (possível dano).
INTERACTION_MEMORY_SECONDS = 600

# Se pessoas/veículos cobrem esta fração do contorno da estátua, a
# comparação SSIM desse print é pulada (estátua encoberta, resultado
# não confiável).
STATUE_OCCLUSION_SKIP = 0.4

# ─── Mudança física (SSIM) — critérios de alerta ─────────────────
# O nível sai do AUMENTO sobre a linha de base (mediana das últimas
# comparações normais), não do % bruto: sol/sombra mudam o % devagar ao
# longo do dia e a linha de base acompanha; dano/pichação é um salto.
SSIM_INCREASE_MODERADO = float(os.getenv("SSIM_INCREASE_MODERADO", "1.5"))   # pontos percentuais
SSIM_INCREASE_ALTO = float(os.getenv("SSIM_INCREASE_ALTO", "3"))
SSIM_INCREASE_CRITICO = float(os.getenv("SSIM_INCREASE_CRITICO", "10"))
# O salto precisa se repetir em N comparações seguidas cobrindo pelo menos
# X segundos: gente/sombra passa, peça faltando ou tinta ficam.
SSIM_CONFIRM_CHECKS = int(os.getenv("SSIM_CONFIRM_CHECKS", "4"))
SSIM_CONFIRM_SECONDS = float(os.getenv("SSIM_CONFIRM_SECONDS", "300"))
# Comparações "normais" usadas na linha de base
SSIM_BASELINE_WINDOW = 10

# Notificação na tela: o mesmo tipo de alerta (câmera + origem + nível)
# não notifica de novo antes deste intervalo. Nível maior notifica na hora.
NOTIFY_COOLDOWN_SECONDS = int(os.getenv("NOTIFY_COOLDOWN_SECONDS", "300"))
# O cooldown vale por PATRIMÔNIO (todas as câmeras dele e todos os tipos de
# alerta): um mesmo episódio no Drummond (câmeras 000056 e 001195) virava
# 3-4 notificações. Só nível mais alto que o já notificado fura o cooldown.

# Tempo (segundos) que um objeto de risco precisa permanecer dentro da
# zona do monumento para o alerta escalar para CRÍTICO, independente do
# nível base da classe.
DWELL_ALERT_SECONDS = 120

# Classes "passageiras" — pessoas e veículos que atravessam a zona do
# monumento (calçada, rua, rotatória) mas não fazem parte dele. O
# change_detector usa isso pra excluir essas áreas do cálculo de %
# alterado: um pedestre, ciclista ou carro passando não pode ser
# contado como pichação/dano físico no monumento.
TRANSIENT_CLASSES = {"pessoa", "bicicleta", "carro", "moto", "ônibus", "caminhão"}

# Tempo (segundos) que uma pessoa precisa permanecer continuamente dentro
# da zona do monumento pra gerar um alerta preventivo de permanência
# suspeita (risco de furto/vandalismo iminente, antes de qualquer dano
# acontecer). Mesma aproximação por presença de classe usada em
# RISK_CLASSES/risk_tracker — o YOLO não reidentifica indivíduos entre
# frames, então "a mesma pessoa" é aproximada por "sempre há uma pessoa
# presente na zona, sem sumir entre varreduras".
PERSON_LOITERING_ALERT_SECONDS = int(os.getenv("PERSON_LOITERING_ALERT_SECONDS", "600"))  # 10 min

# ─── Classes ─────────────────────────────────────────────────────
# Mapeamento 1:1 com a ordem real das 80 classes COCO usadas pelo YOLO
# (ver `model.names` do ultralytics) — traduzido para português.
PATRIMONY_CLASSES = {
    0: "pessoa", 1: "bicicleta", 2: "carro", 3: "moto", 4: "avião",
    5: "ônibus", 6: "trem", 7: "caminhão", 8: "barco", 9: "semáforo",
    10: "hidrante", 11: "placa_de_parada", 12: "parquímetro",
    13: "banco", 14: "pássaro", 15: "gato", 16: "cachorro",
    17: "cavalo", 18: "ovelha", 19: "vaca", 20: "elefante",
    21: "urso", 22: "zebra", 23: "girafa", 24: "mochila",
    25: "guarda-chuva", 26: "bolsa", 27: "gravata", 28: "mala",
    29: "frisbee", 30: "esquis", 31: "snowboard", 32: "bola_esportiva",
    33: "pipa", 34: "taco_de_beisebol", 35: "luva_de_beisebol",
    36: "skate", 37: "prancha_de_surf", 38: "raquete_de_tênis",
    39: "garrafa", 40: "taça_de_vinho", 41: "copo", 42: "garfo",
    43: "faca", 44: "colher", 45: "tigela", 46: "banana",
    47: "maçã", 48: "sanduíche", 49: "laranja", 50: "brócolis",
    51: "cenoura", 52: "cachorro_quente", 53: "pizza", 54: "rosquinha",
    55: "bolo", 56: "cadeira", 57: "sofá", 58: "vaso_de_planta",
    59: "cama", 60: "mesa_de_jantar", 61: "vaso_sanitário", 62: "monitor_tv",
    63: "notebook", 64: "mouse", 65: "controle_remoto", 66: "teclado",
    67: "celular", 68: "microondas", 69: "forno", 70: "torradeira",
    71: "pia", 72: "geladeira", 73: "livro", 74: "relógio",
    75: "vaso", 76: "tesoura", 77: "ursinho_de_pelúcia",
    78: "secador_de_cabelo", 79: "escova_de_dente",
}

INTEREST_CLASSES = {
    "pessoa": "Patrimônio Humano / Visitante",
    "carro": "Veículo", "moto": "Veículo",
    "ônibus": "Transporte Público", "caminhão": "Veículo de Carga",
    "bicicleta": "Mobilidade Urbana", "skate": "Mobilidade Urbana",
    "vaso": "Patrimônio Histórico / Ornamento",
    "vaso_de_planta": "Área Verde / Paisagismo",
    "cadeira": "Mobiliário Urbano",
    "livro": "Acervo / Documentação",
    "relógio": "Patrimônio Histórico",
    "semáforo": "Sinalização Urbana",
    "placa_de_parada": "Sinalização Urbana",
    "parquímetro": "Sinalização Urbana",
    "hidrante": "Equipamento Público",
    "cachorro": "Animal / Fauna Urbana",
    "gato": "Animal / Fauna Urbana",
    "pássaro": "Animal / Fauna Urbana",
    "cavalo": "Animal / Fauna Urbana",
}

# ─── Patrimônios Específicos do Rio ──────────────────────────────
# Lista oficial de patrimônios monitorados com seus códigos de câmera
PATRIMONIOS = [
    # Desativado (12/08) — nenhuma câmera disponível aponta pro monumento, ver histórico de testes no obs.
    # {
    #     "id": 1,
    #     "nome": "Betinho",
    #     "descricao": "Estátua de Herbert de Souza (Betinho) — Praia de Botafogo, altura da R. São Clemente",
    #     "bairro": "Botafogo",
    #     "endereco": "Mourisco - Praia de Botafogo - Botafogo, Rio de Janeiro - RJ, 22250-911",
    #     "categoria": "Personalidade / Direitos Humanos",
    #     "emoji": "🧓",
    #     "latitude": -22.949047,
    #     "longitude": -43.182428,
    #     "camera_codes": [],
    #     "obs": "CONFIRMADO (12/08): sem câmera exclusiva. Testadas e removidas: 003352 (parquinho, sem "
    #            "relação), 000300 e 000046 (avenida aberta, monumento encoberto pelas árvores do canteiro), "
    #            "000263 e 000299 (mesma avenida, sem enquadrar o monumento).",
    # },
    {
        "id": 2,
        "nome": "Marielle Franco",
        "descricao": "Memorial Marielle Franco — Terminal Menezes Cortes",
        "bairro": "Centro",
        "endereco": "Terminal Menezes Cortes - Centro, Rio de Janeiro - RJ, 20040-007",
        "categoria": "Personalidade / Direitos Humanos",
        "emoji": "✊🏿",
        "latitude": -22.906115,
        "longitude": -43.176134,
        "camera_codes": ["001963", "001962"],
    },
    # Desativado (12/08) — nenhuma câmera testada até agora aponta pro monumento, ver histórico no obs.
    # {
    #     "id": 7,
    #     "nome": "Clarice Lispector",
    #     "descricao": "Estátua de Clarice Lispector — Caminho dos Pescadores, Leme",
    #     "bairro": "Leme",
    #     "endereco": "Caminho dos Pescadores Ted Boy Marino - Leme",
    #     "categoria": "Personalidade / Literatura",
    #     "emoji": "📖",
    #     "latitude": -22.9627,
    #     "longitude": -43.1655,
    #     "camera_codes": [],
    #     "obs": "Sem câmera exclusiva. 000646, 001231 e 001232 já foram testadas e removidas — nenhuma "
    #            "aponta pro monumento. Candidatas ainda não testadas: 001219, 001220, 001229, 001230, "
    #            "001233, 001234, 006141 (mesmo cruzamento Av. Atlântica x Ponta do Leme).",
    # },
    # Desativado (12/08) — 000315 e 005534 testadas e removidas, mostram rua/trânsito, sem o relógio no quadro.
    # {
    #     "id": 8,
    #     "nome": "Relógio da Glória",
    #     "descricao": "Relógio da Glória — monumento histórico de 1905, Rua da Glória, 290",
    #     "bairro": "Glória",
    #     "endereco": "R. da Glória - Glória, Rio de Janeiro - RJ, 20241-190",
    #     "categoria": "Patrimônio Histórico",
    #     "emoji": "🕰️",
    #     "latitude": -22.920482,
    #     "longitude": -43.177031,
    #     "camera_codes": [],
    #     "obs": "Endereço corrigido (12/08) — cadastro antigo tinha bairro/endereço errados (Alto da Boa "
    #            "Vista não faz sentido pra esse monumento). 000315 e 005534 testadas e removidas (rua/ "
    #            "trânsito, sem o relógio no quadro). Candidatas ainda não testadas: 005531/005532 (R. da "
    #            "Glória, 58), 005533 (Chafariz da Glória, mesma rua).",
    # },
    {
        "id": 11,
        "nome": "Carlos Drummond de Andrade",
        "descricao": "Estátua de Carlos Drummond de Andrade — Orla de Copacabana (sentado no calçadão)",
        "bairro": "Copacabana",
        "endereco": "Av. Atlântica, 181",
        "categoria": "Personalidade / Literatura",
        "emoji": "🕶️",
        "latitude": -22.984141,
        "longitude": -43.18947,
        "camera_codes": ["001195", "000056"],
        "obs": "CONFIRMADO: câmeras 001195 ('AV. ATLÂNTICA 181') e 000056 ('MONUMENTO DRUMMOND') "
               "apontam corretamente para a estátua. 001021 ('DRUMMOND 02 - FIXA') testada e removida "
               "(12/08) — não ficou boa.",
    },
    {
        "id": 9,
        "nome": "Princesa Isabel",
        "descricao": "Monumento Princesa Isabel — Avenida Princesa Isabel, Copacabana",
        "bairro": "Copacabana",
        "endereco": "Av. Princesa Isabel - Copacabana",
        "categoria": "Personalidade / História",
        "emoji": "👑",
        "latitude": -22.9651,
        "longitude": -43.1736,
        "camera_codes": ["001175"],
    },
    {
        "id": 10,
        "nome": "Tom Jobim",
        "descricao": "Monumento Tom Jobim — Av. Francisco Bhering, Ipanema",
        "bairro": "Ipanema",
        "endereco": "Av. Francisco Bhering, S/N - Ipanema",
        "categoria": "Personalidade / Música",
        "emoji": "🎵",
        "latitude": -22.9879,
        "longitude": -43.1943,
        "camera_codes": ["000194"],
    },
]
