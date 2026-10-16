"""
Proxy transparente para a página de vídeo WebRTC (WHEP) da Tixxi.

A página servida em stream_url (stream_type="html") não é um vídeo estático:
ela roda JS próprio que negocia WebRTC contra o mesmo host, com chamadas
relativas (/auth/refresh, /app/session/start, /app/whep/<id>, ...). Quando
essa página é embutida num <iframe> de origem diferente da nossa, o
navegador bloqueia o cookie de sessão dessas chamadas (política de
terceiros/SameSite) e a negociação falha — mas funciona normal quando
aberta direto numa aba (mesma origem).

Servindo essa página e essas chamadas através do NOSSO backend (mesmo
host do frontend, ainda que porta diferente — SameSite considera "site",
não a porta), tudo passa a ser tratado como mesma origem pelo navegador e
o cookie de sessão para de ser bloqueado.

O vídeo/áudio em si trafega direto entre o navegador e o servidor de mídia
via WebRTC (ICE/SRTP) — não passa por aqui, só a sinalização HTTP.
"""

import logging
import re

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.config import VIDEO_PAGE_BASE_URL, WEBRTC_EXTRA_ICE_HOST

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Proxy de Vídeo (Tixxi)"])

_client = httpx.AsyncClient(timeout=15.0, verify=False)

# Headers hop-by-hop / específicos do host de origem — não repassar em
# nenhuma das duas direções (senão o Content-Length antigo derruba a
# resposta, ou o Host aponta pro lugar errado).
_STRIP_HEADERS = {
    "connection", "keep-alive", "transfer-encoding", "upgrade",
    "content-encoding", "content-length", "host",
}

# A página da Tixxi tem um auto-detector de DevTools: compara
# window.outerWidth/Height (janela INTEIRA do navegador) com innerWidth/
# Height (área da própria página) e, se a diferença passar de 160px,
# assume que o DevTools está aberto e se autodestrói (fecha a conexão
# WebRTC e apaga o <body>). Dentro do nosso <iframe> — bem menor que a
# janela do navegador — essa diferença é sempre enorme, então ele dispara
# a cada 1s achando (errado) que tem DevTools aberto, matando a conexão
# antes dela terminar de negociar. Removemos esse bloco na resposta que
# passa pelo nosso proxy (só afeta o embed dentro do nosso próprio
# dashboard autenticado — não altera o comportamento do link direto).
_STRIP_DEVTOOLS_GUARD = re.compile(
    r"\(function detectDevTools\(\)\s*\{.*?\}\)\(\);",
    re.DOTALL,
)


async def _proxy(request: Request, target_url: str) -> Response:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _STRIP_HEADERS}
    body = await request.body()

    try:
        upstream = await _client.request(
            request.method,
            target_url,
            headers=headers,
            content=body or None,
            params=request.query_params,
        )
    except httpx.HTTPError as e:
        logger.warning(f"Proxy de vídeo falhou ({target_url}): {e}")
        return Response(content=f"Erro ao contatar servidor de vídeo: {e}", status_code=502)

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _STRIP_HEADERS}
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=upstream.headers.get("content-type"),
    )


@router.get("/video")
async def proxy_video_page(request: Request):
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _STRIP_HEADERS}
    try:
        upstream = await _client.get(
            f"{VIDEO_PAGE_BASE_URL}/video", headers=headers, params=request.query_params,
        )
    except httpx.HTTPError as e:
        logger.warning(f"Proxy de vídeo falhou: {e}")
        return Response(content=f"Erro ao contatar servidor de vídeo: {e}", status_code=502)

    content_type = upstream.headers.get("content-type", "")
    content = upstream.content
    if "text/html" in content_type:
        html = content.decode("utf-8", errors="ignore")
        stripped, n = _STRIP_DEVTOOLS_GUARD.subn("", html)
        if n:
            logger.info(f"Removido guard anti-devtools da página de vídeo ({n}x)")
        content = stripped.encode("utf-8")

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _STRIP_HEADERS}
    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type=content_type,
    )


@router.get("/logo.jpg")
async def proxy_logo(request: Request):
    return await _proxy(request, f"{VIDEO_PAGE_BASE_URL}/logo.jpg")


@router.post("/auth/refresh")
async def proxy_auth_refresh(request: Request):
    return await _proxy(request, f"{VIDEO_PAGE_BASE_URL}/auth/refresh")


@router.post("/app/session/start")
async def proxy_session_start(request: Request):
    return await _proxy(request, f"{VIDEO_PAGE_BASE_URL}/app/session/start")


@router.post("/app/session/heartbeat")
async def proxy_session_heartbeat(request: Request):
    return await _proxy(request, f"{VIDEO_PAGE_BASE_URL}/app/session/heartbeat")


@router.post("/app/session/stop")
async def proxy_session_stop(request: Request):
    return await _proxy(request, f"{VIDEO_PAGE_BASE_URL}/app/session/stop")


# a=candidate:<foundation> <component> <transport> <priority> <ip> <port> ...
_CANDIDATE_LINE = re.compile(r"^a=candidate:(\S+) (\S+ \S+ \S+) (\S+) (.*)$")


def _add_internal_ice_candidates(sdp: str, host: str) -> str:
    """Duplica cada candidato ICE do SDP answer trocando o IP por `host`.

    Os candidatos originais (IP público) continuam lá, então quem acessa de
    fora da rede segue funcionando; de dentro, o navegador acha o caminho
    pelo IP interno. Ver WEBRTC_EXTRA_ICE_HOST em config.py.
    """
    out = []
    for line in sdp.splitlines():
        out.append(line)
        m = _CANDIDATE_LINE.match(line)
        if m and m.group(3) != host:
            foundation, middle, _ip, rest = m.groups()
            out.append(f"a=candidate:{foundation}9 {middle} {host} {rest}")
    return "\r\n".join(out) + "\r\n"


@router.post("/app/whep/{session_id}")
async def proxy_whep(request: Request, session_id: str):
    resp = await _proxy(request, f"{VIDEO_PAGE_BASE_URL}/app/whep/{session_id}")
    if WEBRTC_EXTRA_ICE_HOST and resp.status_code in (200, 201):
        sdp = resp.body.decode("utf-8", errors="ignore")
        if "a=candidate:" in sdp:
            headers = {k: v for k, v in resp.headers.items() if k.lower() not in _STRIP_HEADERS}
            return Response(
                content=_add_internal_ice_candidates(sdp, WEBRTC_EXTRA_ICE_HOST),
                status_code=resp.status_code,
                headers=headers,
                media_type=resp.media_type,
            )
    return resp
