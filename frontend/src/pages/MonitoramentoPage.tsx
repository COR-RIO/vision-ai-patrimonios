import { useState, useEffect, useCallback, useRef } from 'react'
import {
  ShieldAlert,
  Camera,
  Play,
  Pause,
  AlertTriangle,
  RefreshCw,
  WifiOff,
  Target,
  Users,
  X,
  FlaskConical,
} from 'lucide-react'
import { api } from '../api/client'
import { CameraClockOverlay, CameraStatusDot } from '../components/CameraClock'
import type { Patrimonio } from '../types'
import ZoneCalibrator from '../components/ZoneCalibrator'
import CameraStreamView from '../components/CameraStreamView'

// Evento de superfície (análise contínua): imagens antes/durante/depois +
// trecho do vídeo. `complete` = depois.jpg e clipe já gravados.
interface SurfaceEvent {
  event_id: string
  timestamp: number
  message: string
  change_percent: number
  suspect_track_ids: number[]
  images: { antes?: string; durante?: string; depois?: string }
  clip: string | null
  complete: boolean
}

// Evento de superfície fica em destaque no card por este tempo
const SURFACE_EVENT_VISIBLE_MS = 30 * 60 * 1000
// Acima disto, câmeras ao vivo voltam a atualizar por polling em vez de
// MJPEG — cada MJPEG prende uma das ~6 conexões HTTP/1.1 do navegador
// com o backend, e o resto da página (polling, alertas) ficaria na fila.
const MAX_MJPEG_STREAMS = 4

function SurfaceEventCard({ event }: { event: SurfaceEvent }) {
  const shots: [string, string | undefined][] = [
    ['Antes', event.images.antes],
    ['Durante', event.images.durante],
    ['Depois', event.images.depois],
  ]
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-2 space-y-2">
      <div className="flex items-start gap-2 text-red-800">
        <AlertTriangle size={16} className="shrink-0 mt-0.5" />
        <div>
          <p className="font-bold text-[10px]">{event.message}</p>
          <p className="text-[10px] opacity-70">
            {new Date(event.timestamp * 1000).toLocaleTimeString('pt-BR')} · possível evento para
            validação humana (sem reconhecimento facial)
          </p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-1">
        {shots.map(([label, src]) => (
          <figure key={label} className="text-center">
            {src ? (
              <a href={api.assetUrl(src)} target="_blank" rel="noreferrer">
                <img src={api.assetUrl(src)} alt={label} className="w-full rounded border border-red-200" />
              </a>
            ) : (
              <div className="aspect-video rounded border border-dashed border-red-200 flex items-center justify-center text-[9px] text-red-400">
                aguardando…
              </div>
            )}
            <figcaption className="text-[9px] text-red-700 mt-0.5">{label}</figcaption>
          </figure>
        ))}
      </div>
      {event.clip ? (
        <video src={api.assetUrl(event.clip)} controls muted className="w-full rounded border border-red-200" />
      ) : (
        <p className="text-[10px] text-red-600">{event.complete ? 'Clipe indisponível' : '🎞️ Gravando trecho do vídeo…'}</p>
      )}
    </div>
  )
}

interface DetectionFrame {
  camera_code: string
  camera_name: string
  stream_url?: string
  image_base64?: string
  snapshot_path?: string
  frame_captured?: boolean
  // true quando a última captura falhou e image_base64 é do scan anterior
  stale?: boolean
  processing_time_ms: number
  timestamp: number
  yolo_detection?: {
    objects: { class_name: string; confidence: number; patrimony_type: string; bbox: number[] }[]
    counts: Record<string, number>
    total_objects: number
  }
  // Novos campos do change detection (SSIM + HF)
  similarity_score?: number
  change_percentage?: number
  // % da área calibrada da estátua visível (sem pessoas na frente) na comparação
  visible_percentage?: number
  change_regions?: number
  significant_changes?: { bbox: number[]; area_percent: number }[]
  highlight_image_base64?: string
  // Comparação lado a lado: a mesma área (figura da estátua) na referência e agora
  reference_roi_base64?: string
  current_roi_base64?: string
  compared_area?: string
  reference_time?: number
  hf_prediction?: Record<string, number>
  alert?: {
    level: string
    message: string
    source?: string
  }
  alert_level?: string
  ssim_alert_level?: string
  // Mudança física: aumento sobre o normal da câmera e confirmação em
  // comparações seguidas (o alerta só sai com confirmed = true)
  confirmation?: {
    baseline_pct: number
    increase_pct: number
    checks: number
    checks_needed: number
    confirmed: boolean
    learning?: boolean
    learn_progress?: string
  }
  // Alerta preditivo do YOLO (objeto de risco dentro do quadrante do monumento)
  risk_alert?: {
    level: string
    message: string
    objects?: string[]
  }
  // Golpe na estátua (pose): movimento brusco contra a figura. MODERADO =
  // ainda confirmando (só no vídeo); ALTO = confirmado
  interaction_alert?: {
    level: string
    message: string
    events?: string[]
    strikes?: number
  }
  // Informativo: presença contínua de pessoas junto ao monumento
  loitering_alert?: {
    level: string
    message: string
    dwell_seconds?: number
  }
  // Análise contínua: superfície protegida alterada com pessoa suspeita junto
  surface_alert?: { level: string; message: string } | null
  surface_event?: SurfaceEvent | null
  suspects?: number[]
  // Pessoas/veículos detectados (YOLO) dentro da zona — não contam como alteração do monumento
  ignored_objects_count?: number
  ignored_objects_alert?: {
    level: string
    message: string
    count?: number
  }
}

function CameraQuadrant({
  patrimony,
  onSelect,
  isSelected,
}: {
  patrimony: Patrimonio
  onSelect: () => void
  isSelected: boolean
}) {
  return (
    <button
      onClick={onSelect}
      className={`p-2 rounded-lg border text-left transition-all ${
        isSelected
          ? 'border-red-400 bg-red-50 ring-1 ring-red-300'
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
    >
      <div className="flex items-center gap-2">
        <span className="text-lg">{patrimony.emoji}</span>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-gray-800 truncate">{patrimony.nome}</p>
          <p className="text-[10px] text-gray-400 truncate">
            {patrimony.camera_codes.length > 0
              ? patrimony.camera_codes.join(', ')
              : 'Sem câmera'}
          </p>
        </div>
        {patrimony.camera_codes.length > 0 && (
          <span className="status-dot online shrink-0" />
        )}
      </div>
    </button>
  )
}

// Patrimônio com mais de uma câmera — mostra cada uma separada, pra dar pra
// escolher qual(is) ver ao vivo em vez de ligar todas juntas de uma vez.
function CameraCodeGroup({
  patrimony,
  codeNames,
  selectedCodes,
  onToggleCode,
}: {
  patrimony: Patrimonio
  codeNames: Record<string, string>
  selectedCodes: string[]
  onToggleCode: (code: string) => void
}) {
  return (
    <div className="p-2 rounded-lg border border-gray-200 bg-white space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="text-lg">{patrimony.emoji}</span>
        <p className="text-xs font-medium text-gray-800 truncate flex-1">{patrimony.nome}</p>
      </div>
      <div className="pl-1 space-y-1">
        {patrimony.camera_codes.map((code) => {
          const isSelected = selectedCodes.includes(code)
          return (
            <button
              key={code}
              onClick={() => onToggleCode(code)}
              className={`w-full flex items-center gap-2 px-2 py-1 rounded-md text-left transition-all ${
                isSelected
                  ? 'border border-red-400 bg-red-50 ring-1 ring-red-300'
                  : 'border border-gray-100 bg-gray-50 hover:border-gray-300'
              }`}
            >
              <span className={`status-dot shrink-0 ${isSelected ? 'online' : ''}`} />
              <span className="text-[10px] text-gray-600 truncate flex-1">
                {codeNames[code] || `Câmera ${code}`}
              </span>
              <span className="text-[9px] text-gray-400 shrink-0">{code}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

// Danos simulados disponíveis no teste (mesmos ids de backend/app/services/damage_sim.py)
// Acima disto (vindo do Mapa), abre em Prints (IA) em vez de vídeo ao vivo
const MAX_LIVE_PLAYERS = 4

const SIM_SCENARIOS = [
  { id: 'pichacao', label: 'Pichação (spray)' },
  { id: 'tinta', label: 'Tinta jogada' },
  { id: 'quebra_cabeca', label: 'Cabeça quebrada' },
  { id: 'quebra_braco', label: 'Braço/mão quebrado' },
  { id: 'remocao_peca', label: 'Peça removida (óculos)' },
  { id: 'cobertura', label: 'Estátua coberta' },
  { id: 'derrubada', label: 'Estátua derrubada/levada' },
]

interface SimResult {
  camera_code: string
  cenario_label: string
  sem_dano_pct: number | null
  com_dano_pct: number
  increase_pct: number
  level: string
  would_alert: boolean
  still_learning?: boolean
  source?: 'atual' | 'referencia'
  source_reason?: string | null
  before_base64?: string
  after_base64?: string
  verdict: string
  reference_roi_base64?: string
  current_roi_base64?: string
  highlight_image_base64?: string
}

interface MonitoramentoProps {
  /** Câmeras pré-selecionadas (ex.: perímetro enviado pelo Mapa) */
  initialCodes?: string[]
  /** Já inicia o ciclo de monitoramento com as câmeras pré-selecionadas */
  autoStart?: boolean
}

export default function MonitoramentoPage({ initialCodes, autoStart = false }: MonitoramentoProps = {}) {
  const [patrimonios, setPatrimonios] = useState<Patrimonio[]>([])
  const [selectedCodes, setSelectedCodes] = useState<string[]>(() => initialCodes ?? [])
  const [streamUrls, setStreamUrls] = useState<Record<string, string>>({})
  // URL HLS por câmera (protocolo real da conta na Tixxi) — quando disponível,
  // usa o player <video>+hls.js em vez do iframe WebRTC.
  const [hlsUrls, setHlsUrls] = useState<Record<string, string>>({})
  // Tipo do stream_url por câmera, vindo direto da Tixxi: "html" (página com
  // player, precisa de <iframe>) ou "raw" (MJPEG puro, precisa de <img>).
  const [streamTypes, setStreamTypes] = useState<Record<string, string>>({})
  const [cameraNames, setCameraNames] = useState<Record<string, string>>({})
  // Nome de cada câmera individual (independe de seleção) — usado pra rotular
  // os chips quando um patrimônio tem mais de uma câmera disponível.
  const [codeNames, setCodeNames] = useState<Record<string, string>>({})
  const [frames, setFrames] = useState<Record<string, DetectionFrame>>({})
  const [running, setRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [streamErrors, setStreamErrors] = useState<Record<string, boolean>>({})
  const [calibratingCode, setCalibratingCode] = useState<string | null>(null)
  // O player da Tixxi (iframe) só reconecta sozinho em falha de rede/504 —
  // em erro 400 ("codecs not supported by client", o mais comum) ele desiste
  // e fica preto pra sempre. Esse contador força um reload completo do
  // iframe (nova tentativa de conexão) — só por clique manual no botão
  // "🔄 Recarregar" (ver reloadCamera), que também busca uma stream_url
  // nova antes de recarregar, cobrindo o caso da KEY ter expirado (~1h).
  const [manualReload, setManualReload] = useState<Record<string, number>>({})
  // "snapshot": mostra o último print analisado pela IA (YOLO + vandalismo),
  // capturado pelo backend a cada ciclo — o navegador não mantém conexão
  // aberta com a câmera, então não há player caindo/ficando preto.
  // "live": player ao vivo embutido (HLS/iframe), como antes.
  // Vindo do Mapa com várias câmeras: abre em Prints (IA) — 10 players de
  // vídeo ao mesmo tempo travam o navegador. Não grava como preferência.
  const openedFromMapMany = (initialCodes?.length ?? 0) > MAX_LIVE_PLAYERS
  const [viewMode, setViewMode] = useState<'snapshot' | 'live'>(() => {
    if (openedFromMapMany) return 'snapshot'
    try {
      return localStorage.getItem('monitoramento.viewMode') === 'live' ? 'live' : 'snapshot'
    } catch {
      return 'snapshot'
    }
  })
  const viewModeTouched = useRef(!openedFromMapMany)
  useEffect(() => {
    if (!viewModeTouched.current) {
      viewModeTouched.current = true
      return
    }
    try { localStorage.setItem('monitoramento.viewMode', viewMode) } catch {}
  }, [viewMode])

  // Câmeras com vídeo + análise contínua no backend (LIVE_CAPTURE_CODES):
  // nos dois modos mostram o frame ANALISADO (MJPEG, caixas e imagem do
  // mesmo instante) em vez do print do ciclo de 10s — sobrepor detecção
  // velha no player ao vivo deixava "fantasmas" de quem já tinha passado.
  // code → FPS da análise (só as que estão ao vivo)
  const [liveCams, setLiveCams] = useState<Record<string, number>>({})
  // code → instante (ms) do último frame analisado no servidor, para o selo de data/hora
  const [liveFrameTs, setLiveFrameTs] = useState<Record<string, number>>({})
  const [liveTick, setLiveTick] = useState(0)
  useEffect(() => {
    const load = async () => {
      try {
        const data = await api.getLiveStatus()
        const analysisFps: Record<string, number> = {}
        const frameTs: Record<string, number> = {}
        const fetchedAt = Date.now()
        for (const a of data.analysis ?? []) {
          analysisFps[a.camera_code] = a.analysis_fps
          if (a.last_analysis_age_s != null) frameTs[a.camera_code] = fetchedAt - a.last_analysis_age_s * 1000
        }
        setLiveFrameTs(frameTs)
        const next: Record<string, number> = {}
        for (const c of data.cameras ?? []) {
          if (c.state === 'ao vivo') next[c.camera_code] = analysisFps[c.camera_code] ?? 0
        }
        setLiveCams(next)
      } catch {
        setLiveCams({})
        setLiveFrameTs({})
      }
    }
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [])
  const visibleLiveCount = selectedCodes.filter((c) => c in liveCams).length
  const useMjpeg = visibleLiveCount <= MAX_MJPEG_STREAMS
  // MJPEG caiu (backend reiniciou, câmera sem análise): reconecta em 3s
  const [mjpegRetry, setMjpegRetry] = useState<Record<string, number>>({})
  const retryMjpeg = useCallback((code: string) => {
    setTimeout(() => setMjpegRetry((r) => ({ ...r, [code]: (r[code] || 0) + 1 })), 3000)
  }, [])
  // Fallback por polling (muitas câmeras ao vivo na tela)
  const pollLive = !useMjpeg && visibleLiveCount > 0
  useEffect(() => {
    if (!pollLive) return
    const id = setInterval(() => setLiveTick((t) => t + 1), 500)
    return () => clearInterval(id)
  }, [pollLive])
  // Resultado do "print" puro por câmera (sem YOLO/HF) — só pra checar
  // rapidamente se a câmera está entregando vídeo, isolado da IA.
  const [snapshots, setSnapshots] = useState<Record<string, {
    loading: boolean
    image?: string
    error?: string
  }>>({})
  const intervalRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const runningRef = useRef(false)
  // Som + mensagem de alerta ficam no AlertNotifier (global, App.tsx),
  // que só notifica o que o backend marca como relevante.

  useEffect(() => {
    api.getPatrimonios().then((data) => {
      setPatrimonios(data)
      const allCodes = Array.from(new Set(data.flatMap((p) => p.camera_codes)))
      if (allCodes.length > 0) {
        api.getCamerasByCodes(allCodes)
          .then((cams) => {
            const names: Record<string, string> = {}
            for (const cam of cams) {
              if (cam.code) names[cam.code] = cam.name || `Câmera ${cam.code}`
            }
            setCodeNames(names)
          })
          .catch(() => {})
      }
    }).catch(() => {})
  }, [])

  // Busca URLs reais dos streams (com JWT válido) sempre que selecionar câmeras
  const fetchStreamUrls = useCallback(async (codes: string[]) => {
    if (codes.length === 0) return
    try {
      const data = await api.getStreams(codes)
      if (data.success) {
        const urls: Record<string, string> = {}
        const names: Record<string, string> = {}
        const hls: Record<string, string> = {}
        const types: Record<string, string> = {}
        for (const cam of data.cameras) {
          if (cam.stream_url) urls[cam.camera_code] = cam.stream_url
          if (cam.hls_url) hls[cam.camera_code] = cam.hls_url
          names[cam.camera_code] = cam.camera_name
          types[cam.camera_code] = cam.stream_type || 'html'
        }
        setStreamUrls(urls)
        setHlsUrls(hls)
        setCameraNames(names)
        setStreamTypes(types)
      }
    } catch (err) {
      console.error('Erro ao buscar URLs dos streams:', err)
    }
  }, [])

  const toggleCamera = (codes: string[]) => {
    if (codes.length === 0) return
    const newSelected = [...selectedCodes]
    const alreadySelected = codes.every((c) => selectedCodes.includes(c))

    if (alreadySelected) {
      setSelectedCodes(selectedCodes.filter((c) => !codes.includes(c)))
    } else {
      codes.forEach((c) => {
        if (!newSelected.includes(c)) newSelected.push(c)
      })
      setSelectedCodes(newSelected)
    }
    setStreamErrors({})
  }

  const toggleSingleCode = (code: string) => {
    setSelectedCodes((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]
    )
    setStreamErrors({})
  }

  const selectAllCameras = () => {
    const allCodes = patrimonios.flatMap((p) => p.camera_codes)
    setSelectedCodes(allCodes)
    setStreamErrors({})
  }

  const clearSelection = () => {
    setSelectedCodes([])
    setStreamErrors({})
  }

  // Quando selectedCodes muda, busca URLs reais dos streams
  useEffect(() => {
    if (selectedCodes.length > 0) {
      fetchStreamUrls(selectedCodes)
    } else {
      setStreamUrls({})
      setCameraNames({})
    }
  }, [selectedCodes, fetchStreamUrls])

  // No modo "Ao vivo", a KEY embutida em cada stream_url vence em ~1h —
  // busca URLs novas a cada 45min (o backend já renova o token antes disso).
  useEffect(() => {
    if (viewMode !== 'live' || selectedCodes.length === 0) return
    const id = setInterval(() => fetchStreamUrls(selectedCodes), 45 * 60 * 1000)
    return () => clearInterval(id)
  }, [viewMode, selectedCodes, fetchStreamUrls])

  // Sem recarregamento automático do iframe — só manual, pelo botão
  // "🔄 Recarregar" em cada câmera (ou "🔗 Pop-up" numa janela separada).

  // Print puro de uma câmera — só captura e salva, sem YOLO/HF. Usado pra
  // checar rapidamente se a câmera está entregando vídeo de verdade.
  const handleSnapshot = useCallback(async (code: string) => {
    setSnapshots((s) => ({ ...s, [code]: { loading: true } }))
    try {
      const data = await api.captureSnapshot(code)
      if (data.success) {
        setSnapshots((s) => ({ ...s, [code]: { loading: false, image: data.image_base64 } }))
      } else {
        setSnapshots((s) => ({ ...s, [code]: { loading: false, error: data.error || 'Falha ao capturar' } }))
      }
    } catch (err) {
      setSnapshots((s) => ({ ...s, [code]: { loading: false, error: 'Erro na requisição' } }))
    }
  }, [])

  // Abre a URL real da câmera (direto da Tixxi, sem nenhuma camada nossa)
  // numa janela pop-up separada — útil pra testar/assistir isolado do resto
  // do app, e pra dar F5 manualmente quando o player travar.
  const openPopup = useCallback((code: string) => {
    const url = streamUrls[code]
    if (!url) return
    window.open(url, `camera-${code}`, 'width=420,height=340,noopener,noreferrer')
  }, [streamUrls])

  // Mesma URL do pop-up, mas em uma aba cheia do navegador em vez de uma
  // janela pequena — útil pra assistir a câmera em tela grande.
  const openNewTab = useCallback((code: string) => {
    const url = streamUrls[code]
    if (!url) return
    window.open(url, '_blank', 'noopener,noreferrer')
  }, [streamUrls])

  // Botão "🔄 Recarregar": busca uma stream_url NOVA do backend (a KEY de
  // cada câmera vence em ~1h — recarregar a mesma URL não resolve se ela já
  // expirou) e só então força o iframe a recarregar. Usa merge no estado em
  // vez de fetchStreamUrls (que substitui a lista inteira) pra não apagar as
  // outras câmeras do grid.
  const reloadCamera = useCallback(async (code: string) => {
    try {
      const data = await api.getStreams([code])
      if (data.success && data.cameras[0]) {
        const cam = data.cameras[0]
        if (cam.stream_url) setStreamUrls((s) => ({ ...s, [code]: cam.stream_url }))
        if (cam.hls_url) setHlsUrls((h) => ({ ...h, [code]: cam.hls_url }))
        setStreamTypes((t) => ({ ...t, [code]: cam.stream_type || 'html' }))
      }
    } catch (err) {
      console.error('Erro ao renovar stream_url:', err)
    }
    setStreamErrors((s) => ({ ...s, [code]: false }))
    setManualReload((m) => ({ ...m, [code]: (m[code] || 0) + 1 }))
  }, [])

  // Escaneia detecções (stream já roda separado no iframe)
  const scan = useCallback(async () => {
    if (selectedCodes.length === 0) return
    setLoading(true)
    try {
      const data = await api.monitorMulti(selectedCodes, 0.35)
      if (data.success) {
        setFrames((prev) => {
          const newFrames: Record<string, DetectionFrame> = {}
          for (const cam of data.cameras) {
            if (!cam.success) continue
            const old = prev[cam.camera_code]
            // Captura falhou neste ciclo: mantém o último print na tela
            // (marcado como desatualizado) em vez de deixar o quadro vazio.
            newFrames[cam.camera_code] = !cam.frame_captured && old?.image_base64
              ? { ...cam, image_base64: old.image_base64, timestamp: old.timestamp, stale: true }
              : cam
          }
          return newFrames
        })
      }
    } catch (err) {
      console.error('Erro no monitoramento:', err)
    } finally {
      setLoading(false)
    }
  }, [selectedCodes])

  const scanAndCompare = useCallback(async () => {
    await scan()
    // Se tiver referência, faz comparação automática
    for (const code of selectedCodes) {
      try {
        const data = await api.detectChanges(code, true)
        if (data.success && data.highlight_image_base64) {
          setFrames(prev => {
            const existing = prev[code]
            if (!existing) return prev
            return {
              ...prev,
              [code]: {
                ...existing,
                highlight_image_base64: data.highlight_image_base64,
                reference_roi_base64: data.reference_roi_base64,
                current_roi_base64: data.current_roi_base64,
                compared_area: data.compared_area,
                reference_time: data.reference_time,
                similarity_score: data.similarity_score,
                change_percentage: data.change_percentage,
                visible_percentage: data.visible_percentage,
                change_regions: data.change_regions,
                ssim_alert_level: data.ssim_alert_level,
                confirmation: data.confirmation,
                alert: data.alert || undefined,
                alert_level: data.alert_level || existing.alert_level || 'NORMAL',
                ignored_objects_count: data.ignored_objects_count,
                ignored_objects_alert: data.ignored_objects_alert || undefined,
                timestamp: Date.now() / 1000,
              }
            }
          })
        }
      } catch {}
    }
  }, [scan, selectedCodes])

  // Loop que se reagenda sozinho (em vez de setInterval fixo) — com muitas
  // câmeras selecionadas, um ciclo (scan + comparação) pode levar bem mais
  // que 10s (Playwright captura uma câmera de cada vez). Reagendar só depois
  // que o ciclo anterior terminar evita empilhar requisições simultâneas.
  const runAutoScanLoop = useCallback(async () => {
    if (!runningRef.current) return
    await scanAndCompare()
    if (runningRef.current) {
      intervalRef.current = setTimeout(runAutoScanLoop, 10000)
    }
  }, [scanAndCompare])

  // Salvar a imagem COM as marcações da IA (registro/evidência): grava no
  // servidor (assets/images/<câmera>/..._print_ia_<data>.jpg) e baixa uma cópia
  // Teste de detecção: dano simulado na estátua (servidor, sem gerar alerta)
  const [simScenario, setSimScenario] = useState('pichacao')
  const [simLoading, setSimLoading] = useState(false)
  const [simResult, setSimResult] = useState<SimResult | null>(null)
  const runSimulation = async () => {
    const code = selectedCodes[0]
    if (!code) return
    setSimLoading(true)
    try {
      setSimResult(await api.simulateDamage(code, simScenario))
    } catch (err: any) {
      window.alert(err?.response?.data?.detail || 'Não foi possível simular o dano agora')
    } finally {
      setSimLoading(false)
    }
  }

  // Imagem ampliada (clique nas miniaturas da comparação)
  const [zoomImage, setZoomImage] = useState<{ src: string; label: string } | null>(null)
  useEffect(() => {
    if (!zoomImage) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setZoomImage(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoomImage])
  const [aiSaving, setAiSaving] = useState<Record<string, boolean>>({})
  const [aiSaved, setAiSaved] = useState<Record<string, string>>({})
  const saveAiPrint = async (code: string) => {
    setAiSaving((s) => ({ ...s, [code]: true }))
    try {
      const b64 = code in liveCams ? undefined : frames[code]?.image_base64
      const r = await api.saveAiPrint(code, b64)
      setAiSaved((s) => ({ ...s, [code]: r.path }))
      // Cópia local: via blob (o atributo download é ignorado entre origens)
      const blob = await (await fetch(api.assetUrl(r.url))).blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = r.path.split(/[\\/]/).pop() || `IA_${code}.jpg`
      document.body.appendChild(a)
      a.click()
      a.remove()
      setTimeout(() => URL.revokeObjectURL(url), 5000)
    } catch (err: any) {
      window.alert(err?.response?.data?.detail || 'Não foi possível salvar a imagem da IA')
    } finally {
      setAiSaving((s) => ({ ...s, [code]: false }))
    }
  }

  const toggleAutoScan = () => {
    if (running) {
      runningRef.current = false
      if (intervalRef.current) clearTimeout(intervalRef.current)
      setRunning(false)
    } else {
      runningRef.current = true
      setRunning(true)
      runAutoScanLoop()
    }
  }

  useEffect(() => {
    return () => {
      runningRef.current = false
      if (intervalRef.current) clearTimeout(intervalRef.current)
    }
  }, [])

  // Vindo do Mapa ("Monitorar perímetro"): inicia o ciclo uma única vez
  const autoStartedRef = useRef(false)
  useEffect(() => {
    if (!autoStart || autoStartedRef.current || selectedCodes.length === 0) return
    autoStartedRef.current = true
    runningRef.current = true
    setRunning(true)
    runAutoScanLoop()
  }, [autoStart, selectedCodes, runAutoScanLoop])

  const getAlertColor = (level?: string) => {
    switch (level) {
      case 'CRÍTICO': return 'bg-red-600 text-white'
      case 'ALTO': return 'bg-orange-500 text-white'
      case 'MODERADO': return 'bg-yellow-500 text-white'
      default: return 'bg-green-500 text-white'
    }
  }

  return (
    <div className="flex gap-4 h-full">
      {/* Sidebar - Lista de patrimônios */}
      <div className="w-56 flex-shrink-0 space-y-2 overflow-y-auto">
        <div className="flex items-center justify-between px-1">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Câmeras disponíveis
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={selectAllCameras}
              className="text-[10px] text-cor-blue hover:underline"
            >
              Todas
            </button>
            <button
              onClick={clearSelection}
              className="text-[10px] text-gray-400 hover:underline"
            >
              Limpar
            </button>
          </div>
        </div>
        <div className="space-y-1.5">
          {patrimonios
            .filter((p) => p.camera_codes.length > 0)
            .map((p) => {
              if (p.camera_codes.length > 1) {
                return (
                  <CameraCodeGroup
                    key={p.id}
                    patrimony={p}
                    codeNames={codeNames}
                    selectedCodes={selectedCodes}
                    onToggleCode={toggleSingleCode}
                  />
                )
              }
              const anySelected = p.camera_codes.some((c) => selectedCodes.includes(c))
              return (
                <CameraQuadrant
                  key={p.id}
                  patrimony={p}
                  isSelected={anySelected}
                  onSelect={() => toggleCamera(p.camera_codes)}
                />
              )
            })}
        </div>

        <div className="pt-3 space-y-2">
          <button
            onClick={toggleAutoScan}
            disabled={selectedCodes.length === 0}
            className={`w-full py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-all ${
              running
                ? 'bg-red-500 text-white hover:bg-red-600'
                : 'bg-cor-blue text-white hover:bg-cor-blue-light disabled:opacity-40'
            }`}
          >
            {running ? <Pause size={16} /> : <Play size={16} />}
            {running ? 'Parar' : 'Iniciar Monitoramento'}
          </button>

          {running && (
            <div className="flex items-center gap-2 text-xs text-green-600 justify-center">
              <span className="status-dot online animate-pulse" />
              Monitorando {selectedCodes.length} câmera(s) a cada 10s
            </div>
          )}

          {selectedCodes.length > 0 && !running && (
            <button
              onClick={scan}
              disabled={loading}
              className="w-full py-2 rounded-lg text-sm font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
              Escanear agora
            </button>
          )}

          {/* Modo de exibição dos quadros */}
          <div className="flex rounded-lg bg-gray-100 p-0.5 text-xs font-medium">
            <button
              onClick={() => setViewMode('snapshot')}
              className={`flex-1 py-1.5 rounded-md transition-all ${
                viewMode === 'snapshot' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500 hover:text-gray-700'
              }`}
              title="Mostra o último print analisado pela IA — sem conexão contínua com a câmera"
            >
              📷 Prints (IA)
            </button>
            <button
              onClick={() => setViewMode('live')}
              className={`flex-1 py-1.5 rounded-md transition-all ${
                viewMode === 'live' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500 hover:text-gray-700'
              }`}
              title="Player de vídeo ao vivo embutido"
            >
              🎥 Ao vivo
            </button>
          </div>
        </div>

        {/* ─── Referência vs Comparação ──────────────────── */}
        <div className="pt-4 border-t border-gray-200">
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide px-1 mb-2">
            <Camera size={12} className="inline mr-1" />
            Referência do Monumento
          </h4>
          <p className="text-[10px] text-gray-400 px-1 mb-2 leading-relaxed">
            Salva o frame atual como base. Depois compara para detectar mudanças (pichação, danos, furto)
          </p>
          <div className="space-y-1.5">
            <button
              onClick={async () => {
                const code = selectedCodes[0] || '001175'
                setLoading(true)
                try {
                  const data = await api.setReference(code, true)
                  if (data.success) alert('✅ Referência salva! O sistema vai comparar os próximos frames.')
                } catch (err) { console.error(err) }
                finally { setLoading(false) }
              }}
              disabled={loading}
              className="w-full py-1.5 px-3 rounded-lg text-xs font-medium bg-green-50 text-green-700 hover:bg-green-100 border border-green-200 flex items-center gap-2 disabled:opacity-50"
            >
              <Camera size={12} />
              📸 Salvar como Referência
            </button>
            <button
              onClick={async () => {
                const code = selectedCodes[0] || '001175'
                setLoading(true)
                try {
                  const data = await api.detectChanges(code, true)
                  if (data.success) {
                    setFrames(prev => ({
                      ...prev,
                      [code]: {
                        ...prev[code],
                        camera_code: code,
                        highlight_image_base64: data.highlight_image_base64,
                        reference_roi_base64: data.reference_roi_base64,
                        current_roi_base64: data.current_roi_base64,
                        compared_area: data.compared_area,
                        reference_time: data.reference_time,
                        similarity_score: data.similarity_score,
                        change_percentage: data.change_percentage,
                visible_percentage: data.visible_percentage,
                        change_regions: data.change_regions,
                        significant_changes: data.significant_changes,
                        ssim_alert_level: data.ssim_alert_level,
                        confirmation: data.confirmation,
                        alert: data.alert || undefined,
                        alert_level: data.alert_level || 'NORMAL',
                        ignored_objects_count: data.ignored_objects_count,
                        ignored_objects_alert: data.ignored_objects_alert || undefined,
                        timestamp: Date.now() / 1000,
                      } as DetectionFrame
                    }))
                  }
                } catch (err) { console.error(err) }
                finally { setLoading(false) }
              }}
              disabled={loading}
              className="w-full py-1.5 px-3 rounded-lg text-xs font-medium bg-red-50 text-red-700 hover:bg-red-100 border border-red-200 flex items-center gap-2 disabled:opacity-50"
            >
              <RefreshCw size={12} />
              🔍 Comparar com Referência
            </button>
            <button
              onClick={() => setCalibratingCode(selectedCodes[0])}
              disabled={selectedCodes.length === 0}
              className="w-full py-1.5 px-3 rounded-lg text-xs font-medium bg-yellow-50 text-yellow-800 hover:bg-yellow-100 border border-yellow-200 flex items-center gap-2 disabled:opacity-50"
            >
              <Target size={12} />
              🎯 Calibrar Zona
            </button>
          </div>

          {/* Teste da detecção: dano simulado na imagem atual (não gera alerta) */}
          <div className="mt-3 pt-3 border-t border-gray-100 space-y-1.5">
            <p className="text-[10px] font-semibold text-gray-500 uppercase flex items-center gap-1">
              <FlaskConical size={11} /> Testar detecção de dano
            </p>
            <p className="text-[9px] text-gray-400">
              Aplica um dano simulado na estátua sobre a imagem atual da câmera {selectedCodes[0] ?? ''} e mostra se
              seria detectado. Não gera alerta nem altera o monitoramento.
            </p>
            <select
              value={simScenario}
              onChange={(e) => setSimScenario(e.target.value)}
              className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 bg-white"
            >
              {SIM_SCENARIOS.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
            </select>
            <button
              onClick={runSimulation}
              disabled={selectedCodes.length === 0 || simLoading}
              className="w-full py-1.5 px-3 rounded-lg text-xs font-medium bg-purple-50 text-purple-800 hover:bg-purple-100 border border-purple-200 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {simLoading ? <RefreshCw size={12} className="animate-spin" /> : <FlaskConical size={12} />}
              {simLoading ? 'Simulando…' : 'Simular dano'}
            </button>
          </div>
        </div>

      </div>

      {/* Grid de câmeras */}
      <div className="flex-1 flex flex-col space-y-3 overflow-y-auto">
        {selectedCodes.length > 0 ? (
          <>
            {/* Grid de quadrantes — rola verticalmente quando muitas câmeras estão ativas */}
            <div className={`grid gap-3 ${
              selectedCodes.length <= 1 ? 'grid-cols-1' :
              selectedCodes.length <= 4 ? 'grid-cols-2' :
              'grid-cols-2 xl:grid-cols-4'
            }`}>
              {selectedCodes.map((code) => {
                const frame = frames[code]
                // Presença contínua (loitering) é só informativa — não pinta o cabeçalho.
                // Interação MODERADA (mão na estátua pra foto, confirmando) também não.
                const interactionHigh = frame?.interaction_alert != null
                  && frame.interaction_alert.level !== 'MODERADO'
                const hasAlert = frame?.alert != null || frame?.risk_alert != null
                  || interactionHigh || frame?.surface_alert != null
                // Data/hora + situação da imagem exibida neste quadrante
                const clockInput = code in liveCams
                  ? { mode: 'live-analysis' as const, imageTs: liveFrameTs[code] ?? null }
                  : viewMode === 'snapshot'
                    ? {
                        mode: 'snapshot' as const,
                        imageTs: frame?.image_base64 ? frame.timestamp * 1000 : null,
                        running,
                        captureFailed: !!frame?.stale || frame?.frame_captured === false,
                      }
                    : { mode: 'player' as const, error: !!streamErrors[code] }
                const surfaceEvent = frame?.surface_event
                  && Date.now() - frame.surface_event.timestamp * 1000 < SURFACE_EVENT_VISIBLE_MS
                  ? frame.surface_event : null

                return (
                  <div key={code} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden flex flex-col">
                    {/* Header da câmera */}
                    <div className={`flex items-center justify-between px-3 py-2 text-white text-xs font-medium ${
                      hasAlert ? 'bg-red-600 animate-pulse' : 'bg-gray-800'
                    }`}>
                      <div className="flex items-center gap-2">
                        <Camera size={14} />
                        <span>{cameraNames[code] || frame?.camera_name || `Câmera ${code}`}</span>
                        <span className="opacity-60">cód. {code}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        {viewMode === 'live' && (
                        <button
                          onClick={() => reloadCamera(code)}
                          disabled={!streamUrls[code]}
                          className="text-[10px] px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 disabled:opacity-50"
                          title="Busca uma URL nova (KEY renovada) e força uma tentativa de conexão"
                        >
                          🔄 Recarregar
                        </button>
                        )}
                        <button
                          onClick={() => openPopup(code)}
                          disabled={!streamUrls[code]}
                          className="text-[10px] px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 disabled:opacity-50"
                          title="Abre a URL real da Tixxi numa janela pop-up pequena"
                        >
                          🔗 Pop-up
                        </button>
                        <button
                          onClick={() => openNewTab(code)}
                          disabled={!streamUrls[code]}
                          className="text-[10px] px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 disabled:opacity-50"
                          title="Abre a URL real da Tixxi em uma nova aba do navegador"
                        >
                          🔗 Nova aba
                        </button>
                        <button
                          onClick={() => saveAiPrint(code)}
                          disabled={aiSaving[code] || !(code in liveCams || frame?.image_base64)}
                          className="text-[10px] px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 disabled:opacity-50"
                          title="Salva a imagem com as marcações da IA (pessoas, objetos e % de confiança) no servidor e baixa uma cópia"
                        >
                          {aiSaving[code] ? '⏳' : '💾'} Salvar IA
                        </button>
                        <button
                          onClick={() => handleSnapshot(code)}
                          disabled={snapshots[code]?.loading}
                          className="text-[10px] px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 disabled:opacity-50"
                          title="Print puro da câmera, sem IA"
                        >
                          {snapshots[code]?.loading ? '⏳' : '📸'} Print
                        </button>
                        <CameraStatusDot {...clockInput} />
                        {frame?.processing_time_ms != null && (
                          <span className="opacity-60 text-[10px]">
                            {frame.processing_time_ms.toFixed(0)}ms
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Stream ao vivo + overlay — ocupa a maior parte da tela */}
                    <div className="relative bg-black flex-1" style={{ minHeight: selectedCodes.length > 4 ? '240px' : '420px' }}>
                      {code in liveCams ? (
                        <img
                          src={useMjpeg
                            ? api.liveAnalysisStreamUrl(code, mjpegRetry[code] || 0)
                            : api.liveAnalysisFrameUrl(code, liveTick)}
                          onError={() => useMjpeg && retryMjpeg(code)}
                          alt={`Análise ao vivo ${code}`}
                          className="absolute inset-0 w-full h-full object-contain"
                        />
                      ) : viewMode === 'snapshot' ? (
                        frame?.image_base64 ? (
                          <img
                            src={`data:image/jpeg;base64,${frame.image_base64}`}
                            alt={`Print analisado ${code}`}
                            className={`absolute inset-0 w-full h-full object-contain ${frame.stale ? 'opacity-50' : ''}`}
                          />
                        ) : (
                          <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">
                            <div className="text-center">
                              {loading ? (
                                <>
                                  <RefreshCw size={28} className="mx-auto mb-2 animate-spin" />
                                  <p>Capturando print...</p>
                                </>
                              ) : frame && !frame.frame_captured ? (
                                <>
                                  <WifiOff size={28} className="mx-auto mb-2" />
                                  <p>Não foi possível capturar a câmera</p>
                                </>
                              ) : (
                                <>
                                  <Camera size={28} className="mx-auto mb-2" />
                                  <p>Clique em "Iniciar Monitoramento" para capturar</p>
                                </>
                              )}
                            </div>
                          </div>
                        )
                      ) : (
                        // Player puro, sem sobrepor o print anotado: ele é do
                        // ciclo de 10s e aparecia como "fantasma" sobre o vídeo
                        <CameraStreamView
                          code={code}
                          streamUrl={streamUrls[code]}
                          hlsUrl={hlsUrls[code]}
                          streamType={streamTypes[code]}
                          reloadToken={manualReload[code] || 0}
                          className="absolute inset-0 w-full h-full object-contain border-none"
                          onLoad={() => setStreamErrors((s) => ({ ...s, [code]: false }))}
                          onError={() => setStreamErrors((s) => ({ ...s, [code]: true }))}
                        />
                      )}
                      {/* Fallback se iframe falhar */}
                      {viewMode === 'live' && !(code in liveCams) && streamErrors[code] && (
                        <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm bg-gray-900/90">
                          <div className="text-center">
                            <WifiOff size={28} className="mx-auto mb-2" />
                            <p>Stream indisponível</p>
                          </div>
                        </div>
                      )}

                      {/* Badge: pichação > interação com a estátua > objeto de risco > mudança física */}
                      {frame?.surface_alert ? (
                        <div className="absolute top-2 left-2 z-10">
                          <span className="text-[10px] px-2 py-0.5 rounded font-bold shadow-lg bg-red-600 text-white animate-pulse">
                            🎨 POSSÍVEL PICHAÇÃO
                          </span>
                        </div>
                      ) : interactionHigh ? (
                        <div className="absolute top-2 left-2 z-10">
                          <span className="text-[10px] px-2 py-0.5 rounded font-bold shadow-lg bg-red-600 text-white animate-pulse">
                            ✋ INTERAÇÃO COM A ESTÁTUA
                          </span>
                        </div>
                      ) : frame?.risk_alert ? (
                        <div className="absolute top-2 left-2 z-10">
                          <span className="text-[10px] px-2 py-0.5 rounded font-bold shadow-lg bg-red-600 text-white animate-pulse">
                            🔪 PREDITIVO: {frame.risk_alert.objects?.join('/').toUpperCase()}
                          </span>
                        </div>
                      ) : frame?.alert && (
                        <div className="absolute top-2 left-2 z-10">
                          <span className={`text-[10px] px-2 py-0.5 rounded font-bold shadow-lg ${
                            frame.alert.level === 'CRÍTICO' || frame.alert.level === 'ALTO'
                              ? 'bg-red-600 text-white animate-pulse'
                              : 'bg-yellow-500 text-black'
                          }`}>
                            🚨 {frame.alert.level}
                          </span>
                        </div>
                      )}

                      {/* Badge AO VIVO / horário do print */}
                      <div className="absolute top-2 right-2 z-10">
                        {code in liveCams ? (
                          <span
                            className="bg-red-600 text-white text-[10px] px-2 py-0.5 rounded font-bold shadow-lg"
                            title="Vídeo contínuo com YOLO + rastreamento rodando no servidor"
                          >
                            🔴 AO VIVO · IA {liveCams[code].toFixed(1)}/s
                          </span>
                        ) : viewMode === 'live' ? (
                          <span className="bg-red-600 text-white text-[10px] px-2 py-0.5 rounded font-bold animate-pulse shadow-lg">
                            🔴 AO VIVO
                          </span>
                        ) : frame?.image_base64 && (
                          <span className="bg-gray-900/80 text-white text-[10px] px-2 py-0.5 rounded font-bold shadow-lg">
                            📷 PRINT + IA
                          </span>
                        )}
                      </div>

                      {/* Data/hora da imagem + se está em dia, atrasada ou sem sinal */}
                      <CameraClockOverlay {...clockInput} />
                    </div>

                    {/* Painel de análise - SSIM + HF */}
                    <div className="p-3 space-y-2 text-xs">
                      {/* Resultado do "Print" puro (sem IA) — checagem rápida de vídeo */}
                      {snapshots[code] && !snapshots[code].loading && (
                        <div className={`rounded-lg p-2 ${snapshots[code].error ? 'bg-red-50' : 'bg-green-50'}`}>
                          {snapshots[code].error ? (
                            <p className="text-[10px] text-red-700">📸 Print falhou: {snapshots[code].error}</p>
                          ) : (
                            <>
                              <p className="text-[10px] text-green-700 font-medium mb-1">📸 Print capturado (sem IA):</p>
                              <img
                                src={`data:image/jpeg;base64,${snapshots[code].image}`}
                                alt={`Print ${code}`}
                                className="w-full rounded border border-green-200"
                              />
                            </>
                          )}
                        </div>
                      )}
                      {/* Superfície protegida alterada com pessoa suspeita (evidências) */}
                      {surfaceEvent && <SurfaceEventCard event={surfaceEvent} />}

                      {/* Alerta preditivo (YOLO — objeto de risco no quadrante) */}
                      {frame?.risk_alert && (
                        <div className="rounded-lg p-2 flex items-center gap-2 bg-red-100 text-red-800 animate-pulse">
                          <AlertTriangle size={16} className="shrink-0" />
                          <div>
                            <p className="font-bold text-[10px]">{frame.risk_alert.message}</p>
                          </div>
                        </div>
                      )}

                      {/* Interação com a estátua (pose) */}
                      {frame?.interaction_alert && (
                        <div className={`rounded-lg p-2 flex items-center gap-2 ${
                          frame.interaction_alert.level === 'MODERADO'
                            ? 'bg-slate-100 text-slate-700'
                            : 'bg-red-100 text-red-800 animate-pulse'
                        }`}>
                          <AlertTriangle size={16} className="shrink-0" />
                          <p className="font-bold text-[10px]">{frame.interaction_alert.message}</p>
                        </div>
                      )}

                      {/* Informativo: presença contínua de pessoas junto ao monumento */}
                      {frame?.loitering_alert && (
                        <div className="rounded-lg p-2 flex items-center gap-2 bg-gray-100 text-gray-700">
                          <Users size={16} className="shrink-0" />
                          <div>
                            <p className="font-bold text-[10px]">{frame.loitering_alert.message}</p>
                          </div>
                        </div>
                      )}

                      {/* Alerta (se houver) */}
                      {frame?.alert && (
                        <div className={`rounded-lg p-2 flex items-center gap-2 ${
                          frame.alert.level === 'CRÍTICO' ? 'bg-red-100 text-red-800 animate-pulse' :
                          frame.alert.level === 'ALTO' ? 'bg-orange-100 text-orange-800' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>
                          <AlertTriangle size={16} className="shrink-0" />
                          <div>
                            <p className="font-bold text-[10px]">{frame.alert.message}</p>
                          </div>
                        </div>
                      )}

                      {/* Info: pessoas/veículos passageiros ignorados na comparação */}
                      {frame?.ignored_objects_alert && (
                        <div className="rounded-lg p-2 flex items-center gap-2 bg-blue-50 text-blue-700">
                          <Users size={16} className="shrink-0" />
                          <div>
                            <p className="font-medium text-[10px]">{frame.ignored_objects_alert.message}</p>
                          </div>
                        </div>
                      )}

                      {/* 📸 Snapshot salvo */}
                      {frame?.snapshot_path && (
                        <div className="bg-gray-50 rounded-lg p-2 flex items-center gap-2">
                          <Camera size={14} className="text-gray-400 shrink-0" />
                          <div className="min-w-0 flex-1">
                            <p className="text-[9px] font-medium text-gray-500">📸 Snapshot salvo</p>
                            <a
                              // Caminho vem do Windows (assets\images\...): normaliza as barras.
                              // ?t= evita abrir a versão em cache — o arquivo é sobrescrito a cada ciclo
                              href={api.assetUrl(`/${frame.snapshot_path.replace(/\\/g, '/')}?t=${Math.round(frame.timestamp ?? Date.now() / 1000)}`)}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-[9px] text-blue-600 hover:text-blue-800 underline truncate block"
                              title={frame.snapshot_path}
                            >
                              {frame.snapshot_path.split(/[\\/]/).pop()}
                            </a>
                          </div>
                        </div>
                      )}

                      {/* 💾 Print com IA salvo pelo operador */}
                      {aiSaved[code] && (
                        <div className="bg-emerald-50 rounded-lg p-2 flex items-center gap-2">
                          <Camera size={14} className="text-emerald-500 shrink-0" />
                          <div className="min-w-0 flex-1">
                            <p className="text-[9px] font-medium text-emerald-700">💾 Print com IA salvo no servidor</p>
                            <a
                              href={api.assetUrl('/' + aiSaved[code].replace(/\\/g, '/'))}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-[9px] text-emerald-700 hover:text-emerald-900 underline truncate block"
                              title={aiSaved[code]}
                            >
                              {aiSaved[code].split(/[\\/]/).pop()}
                            </a>
                          </div>
                        </div>
                      )}

                      {/* SSIM: Similaridade / Mudança */}
                      {frame?.similarity_score != null && (
                        <div className="bg-blue-50 rounded-lg p-2">
                          <div className="grid grid-cols-3 gap-2 text-center">
                            <div>
                              <p className="text-lg font-bold text-blue-700">
                                {(frame.similarity_score * 100).toFixed(1)}%
                              </p>
                              <p className="text-[8px] text-gray-500" title="Quanto a estátua (só o contorno calibrado, sem as pessoas na frente) está igual à imagem de referência">Igual à referência</p>
                            </div>
                            <div>
                              <p className={`text-lg font-bold ${
                                // Cor pelo AUMENTO sobre o normal da câmera (mesmo critério do alerta)
                                (frame.confirmation?.increase_pct ?? frame.change_percentage ?? 0) >= 10 ? 'text-red-600' :
                                (frame.confirmation?.increase_pct ?? frame.change_percentage ?? 0) >= 5 ? 'text-yellow-600' :
                                'text-green-600'
                              }`}>
                                {frame.change_percentage?.toFixed(1)}%
                              </p>
                              <p className="text-[8px] text-gray-500" title="Parte visível da estátua (contorno calibrado) que mudou em relação à referência">Estátua alterada</p>
                            </div>
                            <div>
                              <p className="text-lg font-bold text-gray-700">
                                {frame.visible_percentage != null ? `${frame.visible_percentage.toFixed(0)}%` : (frame.change_regions || 0)}
                              </p>
                              <p className="text-[8px] text-gray-500" title="Quanto da estátua ficou visível nesta comparação (o resto estava encoberto por pessoas/veículos e não entra na conta)">
                                {frame.visible_percentage != null ? 'Estátua visível' : 'Regiões'}
                              </p>
                            </div>
                          </div>
                          <p className="mt-1 text-[9px] text-gray-500 text-center">
                            {frame.compared_area === 'silhueta da estátua'
                              ? 'Comparação só na silhueta da estátua (sem banco, chão e fundo), com a imagem alinhada à referência e sem quem está na frente'
                              : 'Comparação no contorno calibrado (estátua não reconhecida na referência), sem quem está na frente'}
                          </p>
                          {frame.reference_roi_base64 && frame.current_roi_base64 && (
                            <div className="mt-2 grid grid-cols-3 gap-1.5">
                              {[
                                { src: frame.reference_roi_base64, label: `Referência${frame.reference_time ? ` · ${new Date(frame.reference_time * 1000).toLocaleString('pt-BR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })}` : ''}` },
                                { src: frame.current_roi_base64, label: 'Agora' },
                                { src: frame.highlight_image_base64, label: 'Alterado (vermelho) · encoberto (azul)' },
                              ].map((im) => im.src && (
                                <figure
                                  key={im.label}
                                  onClick={() => setZoomImage({ src: `data:image/jpeg;base64,${im.src}`, label: im.label })}
                                  className="bg-white rounded ring-1 ring-blue-100 overflow-hidden cursor-zoom-in hover:ring-cor-blue"
                                  title="Clique para ampliar"
                                >
                                  <img src={`data:image/jpeg;base64,${im.src}`} alt={im.label} className="w-full h-28 object-contain bg-black" />
                                  <figcaption className="text-[8px] text-gray-500 text-center py-0.5">{im.label}</figcaption>
                                </figure>
                              ))}
                            </div>
                          )}
                          {frame.confirmation && (
                            <p className="mt-1 text-[9px] text-gray-600 text-center">
                              {frame.confirmation.learning
                                ? `Aprendendo o fundo que se mexe em volta da estátua (${frame.confirmation.learn_progress}) — sem alertas nesta fase`
                                : frame.confirmation.increase_pct >= 0.1
                                ? `+${frame.confirmation.increase_pct.toFixed(1)} p.p. sobre o normal (${frame.confirmation.baseline_pct.toFixed(1)}%)`
                                : `Dentro do normal da câmera (${frame.confirmation.baseline_pct.toFixed(1)}%)`}
                              {frame.confirmation.checks > 0 && !frame.confirmation.confirmed && (
                                <span className="ml-1 font-semibold text-amber-600">
                                  · confirmando {frame.confirmation.checks}/{frame.confirmation.checks_needed}
                                </span>
                              )}
                            </p>
                          )}
                          {frame.significant_changes && frame.significant_changes.length > 0 && (
                            <div className="mt-1 border-t border-blue-100 pt-1 space-y-0.5">
                              {frame.significant_changes.slice(0, 3).map((ch, i) => (
                                <p key={i} className="text-[9px] text-gray-500">
                                  • Região {i+1}: {ch.area_percent}% do monumento
                                </p>
                              ))}
                            </div>
                          )}
                        </div>
                      )}



                      {/* Objetos detectados pelo YOLO */}
                      {(frame?.yolo_detection?.total_objects ?? 0) > 0 && (
                        <div>
                          <p className="text-[9px] font-semibold text-gray-500 uppercase mb-1">
                            <ShieldAlert size={10} className="inline mr-0.5" />
                            Objetos detectados
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {Object.entries(frame.yolo_detection?.counts ?? {}).map(([name, count]) => (
                              <span key={name} className="bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded text-[9px]">
                                {name.replace(/_/g, ' ')}: {count}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Mensagem vazia */}
                      {!frame?.alert && !frame?.risk_alert && frame?.similarity_score == null && (
                        <div className="text-center text-gray-400 text-[10px] py-1">
                          {frame ? 'Aguardando verificação...' : 'Aguardando primeiro scan...'}
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Alerta geral */}
            {Object.values(frames).some((f) => f.alert != null || f.risk_alert != null) && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-3 flex items-center gap-3">
                <AlertTriangle size={20} className="text-red-500 shrink-0" />
                <div>
                  <p className="text-sm font-bold text-red-700">🚨 ALERTA DE VANDALISMO DETECTADO</p>
                  <p className="text-xs text-red-600">
                    {Object.values(frames)
                      .filter((f) => f.alert != null || f.risk_alert != null)
                      .map((f) => `${f.camera_name} (cód. ${f.camera_code})`)
                      .join(', ')}
                  </p>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-12 flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <Camera size={48} className="mx-auto mb-3 opacity-50" />
              <p className="text-lg font-medium">Selecione as câmeras</p>
              <p className="text-sm">Escolha os patrimônios ao lado para monitorar em tempo real</p>
            </div>
          </div>
        )}
      </div>

      {calibratingCode && (
        <ZoneCalibrator cameraCode={calibratingCode} onClose={() => setCalibratingCode(null)} />
      )}

      {/* Resultado do teste de dano simulado */}
      {simResult && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4" onClick={() => setSimResult(null)}>
          <div className="bg-white rounded-xl shadow-xl max-w-4xl w-full max-h-[92vh] overflow-y-auto p-4 space-y-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
                <FlaskConical size={16} className="text-purple-600" />
                Teste de detecção: {simResult.cenario_label} · câmera {simResult.camera_code}
              </h3>
              <button onClick={() => setSimResult(null)} className="text-gray-400 hover:text-gray-600"><X size={18} /></button>
            </div>

            {simResult.source === 'referencia' && (
              <div className="rounded-lg bg-sky-50 text-sky-800 text-xs p-2.5">
                ℹ️ Agora não dá pra testar na imagem ao vivo ({simResult.source_reason}). O teste foi feito na
                <strong> imagem de referência</strong>, com a estátua livre. Em operação é o mesmo: com gente na frente, o
                sistema pula a comparação e confere de novo quando a estátua fica livre.
              </div>
            )}

            {/* 1. O que foi simulado */}
            <section>
              <p className="text-xs font-semibold text-gray-700 mb-1.5">
                1. O dano simulado — {simResult.source === 'referencia' ? 'imagem de referência' : 'imagem atual da câmera'}
                <span className="font-normal text-gray-400"> (contorno amarelo = área da estátua que o sistema compara)</span>
              </p>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { src: simResult.before_base64, label: 'ANTES — como a câmera vê' },
                  { src: simResult.after_base64, label: `DEPOIS — com "${simResult.cenario_label}"` },
                ].map((im) => im.src && (
                  <figure key={im.label} className="rounded-lg ring-1 ring-gray-200 overflow-hidden cursor-zoom-in"
                    onClick={() => setZoomImage({ src: `data:image/jpeg;base64,${im.src}`, label: im.label })}>
                    <img src={`data:image/jpeg;base64,${im.src}`} alt={im.label} className="w-full h-72 object-contain bg-black" />
                    <figcaption className="text-[11px] font-medium text-gray-600 text-center py-1">{im.label}</figcaption>
                  </figure>
                ))}
              </div>
            </section>

            {/* 2. O que o sistema viu */}
            <section>
              <p className="text-xs font-semibold text-gray-700 mb-1.5">2. O que o sistema detectou na estátua</p>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { src: simResult.reference_roi_base64, label: 'Referência (estátua normal)' },
                  { src: simResult.current_roi_base64, label: 'Imagem analisada (com o dano)' },
                  { src: simResult.highlight_image_base64, label: 'Alterado em vermelho · encoberto em azul' },
                ].map((im) => im.src && (
                  <figure key={im.label} className="rounded ring-1 ring-gray-200 overflow-hidden cursor-zoom-in"
                    onClick={() => setZoomImage({ src: `data:image/jpeg;base64,${im.src}`, label: im.label })}>
                    <img src={`data:image/jpeg;base64,${im.src}`} alt={im.label} className="w-full h-48 object-contain bg-black" />
                    <figcaption className="text-[10px] text-gray-500 text-center py-1">{im.label}</figcaption>
                  </figure>
                ))}
              </div>
              <div className="mt-2 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="bg-gray-50 rounded-lg p-2">
                  <p className="text-lg font-bold text-gray-700">{simResult.sem_dano_pct != null ? `${simResult.sem_dano_pct.toFixed(1)}%` : '—'}</p>
                  <p className="text-[10px] text-gray-500">Estátua alterada ANTES</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-2">
                  <p className="text-lg font-bold text-red-600">{simResult.com_dano_pct.toFixed(1)}%</p>
                  <p className="text-[10px] text-gray-500">Estátua alterada DEPOIS</p>
                </div>
                <div className="bg-gray-50 rounded-lg p-2">
                  <p className="text-lg font-bold text-gray-700">{simResult.level}</p>
                  <p className="text-[10px] text-gray-500">Nível do alerta</p>
                </div>
              </div>
            </section>

            {/* 3. Veredito */}
            <div className={`rounded-lg p-3 text-sm font-medium ${simResult.would_alert ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-50 text-amber-800'}`}>
              3. {simResult.would_alert ? '✅ ' : '⚠️ '}{simResult.verdict}
            </div>

            {simResult.still_learning && (
              <p className="text-[11px] text-amber-700">
                A câmera ainda está na fase de aprendizado do fundo: o resultado pode incluir fundo (mar, areia) que depois sai da conta.
              </p>
            )}
            <p className="text-[10px] text-gray-400">Simulação feita numa cópia — nenhum alerta foi registrado e o monitoramento não foi alterado.</p>
          </div>
        </div>
      )}

      {/* Imagem ampliada — fecha no clique fora, no X ou com Esc */}
      {zoomImage && (
        <div
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 cursor-zoom-out"
          onClick={() => setZoomImage(null)}
        >
          <figure className="relative max-w-5xl w-full" onClick={(e) => e.stopPropagation()}>
            <img src={zoomImage.src} alt={zoomImage.label} className="w-full max-h-[85vh] object-contain rounded-lg bg-black" />
            <figcaption className="mt-2 text-center text-sm text-white/90">{zoomImage.label}</figcaption>
            <button
              onClick={() => setZoomImage(null)}
              className="absolute -top-3 -right-3 p-1.5 rounded-full bg-white text-gray-700 shadow hover:bg-gray-100"
              title="Fechar (Esc)"
            >
              <X size={16} />
            </button>
          </figure>
        </div>
      )}
    </div>
  )
}
