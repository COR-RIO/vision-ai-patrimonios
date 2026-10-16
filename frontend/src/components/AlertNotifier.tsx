import { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Bell, ImageOff, Video, Volume2, VolumeX, X } from 'lucide-react'
import { api } from '../api/client'

interface NotifyAlert {
  id: number
  timestamp: number
  camera_code: string
  camera_name: string
  level: string
  message: string
  source: string
  // Frame da câmera no momento do alerta (servido em /assets)
  image?: string | null
}

const POLL_MS = 5000
const MAX_ITEMS = 30
// Só ALTO/CRÍTICO tocam som; MODERADO só aparece no contador/lista.
const SOUND_LEVELS = new Set(['ALTO', 'CRÍTICO'])

const LEVEL_DOT: Record<string, string> = {
  'CRÍTICO': 'bg-red-600',
  'ALTO': 'bg-orange-500',
  'MODERADO': 'bg-yellow-400',
}

function loadMuted() {
  try { return localStorage.getItem('alertas.mudo') === '1' } catch { return false }
}

// Sino de alertas no cabeçalho — em vez de pop-ups sobre a tela: contador
// discreto + som para ALTO/CRÍTICO; a lista abre ao clicar. Consulta
// /api/alerts?notify=true — o backend já decide o que merece notificar
// (interação com a estátua, objeto de risco encostado, mudança física)
// e evita repetir o mesmo alerta em sequência.
export default function AlertNotifier({ onOpenMonitoramento }: { onOpenMonitoramento: () => void }) {
  const [items, setItems] = useState<NotifyAlert[]>([])
  const [unread, setUnread] = useState<Set<number>>(new Set())
  const [open, setOpen] = useState(false)
  const [viewing, setViewing] = useState<NotifyAlert | null>(null)
  const [muted, setMuted] = useState(loadMuted)
  const [soundBlocked, setSoundBlocked] = useState(false)
  const cursorRef = useRef<number | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const mutedRef = useRef(muted)

  useEffect(() => {
    mutedRef.current = muted
    try { localStorage.setItem('alertas.mudo', muted ? '1' : '0') } catch {}
  }, [muted])

  const playSound = useCallback(() => {
    const audio = audioRef.current
    if (!audio || mutedRef.current) return
    audio.currentTime = 0
    // Navegador bloqueia áudio antes de qualquer clique na página
    audio.play().then(() => setSoundBlocked(false)).catch(() => setSoundBlocked(true))
  }, [])

  useEffect(() => {
    let stopped = false

    const poll = async () => {
      try {
        if (cursorRef.current === null) {
          // Primeira consulta: só posiciona o cursor — não notifica
          // alertas que já existiam antes da página abrir.
          const data = await api.getAlerts({ limit: 1 })
          cursorRef.current = data.last_id ?? 0
          return
        }
        const data = await api.getAlerts({ afterId: cursorRef.current, notify: true })
        // Backend reiniciou (ids recomeçaram do zero)
        if ((data.last_id ?? 0) < cursorRef.current) {
          cursorRef.current = data.last_id ?? 0
          return
        }
        cursorRef.current = data.last_id ?? cursorRef.current
        const fresh: NotifyAlert[] = data.alerts ?? []
        if (fresh.length > 0 && !stopped) {
          setItems((prev) => [...fresh, ...prev].slice(0, MAX_ITEMS))
          setUnread((prev) => new Set([...prev, ...fresh.map((a) => a.id)]))
          if (fresh.some((a) => SOUND_LEVELS.has(a.level))) playSound()
        }
      } catch {
        // backend fora do ar — tenta de novo no próximo ciclo
      }
    }

    poll()
    const id = setInterval(poll, POLL_MS)
    return () => {
      stopped = true
      clearInterval(id)
    }
  }, [playSound])

  // Pop-up da imagem fecha com Esc
  useEffect(() => {
    if (!viewing) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setViewing(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [viewing])

  // Fecha a lista ao clicar fora
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('mousedown', onDown)
    return () => window.removeEventListener('mousedown', onDown)
  }, [open])

  const unreadItems = items.filter((a) => unread.has(a.id))
  const urgent = unreadItems.some((a) => SOUND_LEVELS.has(a.level))

  // Contador na aba do navegador — visível com o sistema em segundo plano
  useEffect(() => {
    const base = document.title.replace(/^\(🚨 \d+\) /, '')
    document.title = unreadItems.length > 0 ? `(🚨 ${unreadItems.length}) ${base}` : base
  }, [unreadItems.length])

  const toggleOpen = () => {
    setOpen((o) => !o)
    // Abrir a lista = operador viu os alertas
    setUnread(new Set())
    if (soundBlocked) playSound()
  }

  return (
    <div ref={boxRef} className="relative">
      <audio ref={audioRef} src="/alerta-cor.wav" preload="auto" />
      <button
        onClick={toggleOpen}
        className="relative p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
        title={soundBlocked ? 'Alertas — som bloqueado pelo navegador, clique para ativar' : 'Alertas'}
      >
        <Bell size={18} className={urgent ? 'text-red-600' : 'text-gray-600'} />
        {unreadItems.length > 0 && (
          <span className={`absolute -top-1 -right-1 min-w-[18px] h-[18px] px-1 rounded-full text-[10px] font-bold text-white flex items-center justify-center ${
            urgent ? 'bg-red-600 animate-pulse' : 'bg-yellow-500'
          }`}>
            {unreadItems.length}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-96 max-w-[calc(100vw-2rem)] bg-white rounded-lg shadow-lg border border-gray-200 z-50">
          <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100">
            <h3 className="text-xs font-semibold text-gray-500 uppercase">Alertas</h3>
            <button
              onClick={() => setMuted((m) => !m)}
              className="text-[11px] text-gray-500 hover:text-gray-700 flex items-center gap-1"
              title="Som para alertas ALTO/CRÍTICO"
            >
              {muted ? <VolumeX size={13} /> : <Volume2 size={13} />}
              {muted ? 'Som desligado' : 'Som ligado'}
            </button>
          </div>
          <div className="max-h-96 overflow-y-auto divide-y divide-gray-50">
            {items.length === 0 ? (
              <p className="text-xs text-gray-400 text-center py-6">Nenhum alerta desde que a página foi aberta</p>
            ) : items.map((a) => (
              <button
                key={a.id}
                onClick={() => { setViewing(a); setOpen(false) }}
                title={a.image ? 'Ver imagem da câmera no momento do alerta' : 'Ver detalhes do alerta'}
                className="w-full text-left px-3 py-2 hover:bg-gray-50 flex items-start gap-2"
              >
                <span className={`mt-1.5 w-2 h-2 rounded-full shrink-0 ${LEVEL_DOT[a.level] ?? 'bg-gray-400'}`} />
                <span className="flex-1 min-w-0">
                  <span className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-gray-800 truncate">{a.camera_name}</span>
                    <span className="text-[10px] text-gray-400 shrink-0">
                      {new Date(a.timestamp * 1000).toLocaleTimeString('pt-BR')}
                    </span>
                  </span>
                  <span className="block text-xs text-gray-600 leading-snug mt-0.5">{a.message}</span>
                </span>
                {a.image && (
                  <img
                    src={api.assetUrl(a.image)}
                    alt=""
                    loading="lazy"
                    className="w-14 h-10 object-cover rounded border border-gray-200 shrink-0"
                  />
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Pop-up com a imagem da câmera no momento do alerta — fecha no clique fora, no X ou com Esc.
          Portal no body: o backdrop-blur do header prenderia o "fixed" dentro dele. */}
      {viewing && createPortal(
        <div
          className="fixed inset-0 z-[60] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setViewing(null)}
        >
          <figure className="relative max-w-5xl w-full" onClick={(e) => e.stopPropagation()}>
            {viewing.image ? (
              <img
                src={api.assetUrl(viewing.image)}
                alt={viewing.camera_name}
                className="w-full max-h-[75vh] object-contain rounded-lg bg-black"
              />
            ) : (
              <div className="w-full h-64 rounded-lg bg-gray-900 flex flex-col items-center justify-center gap-2 text-white/60">
                <ImageOff size={28} />
                <span className="text-sm">Nenhuma imagem registrada para este alerta</span>
              </div>
            )}
            <figcaption className="mt-3 flex flex-wrap items-start justify-between gap-3 text-white">
              <span className="min-w-0">
                <span className="flex items-center gap-2 text-sm font-semibold">
                  <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${LEVEL_DOT[viewing.level] ?? 'bg-gray-400'}`} />
                  {viewing.camera_name}
                  <span className="text-xs font-normal text-white/60">
                    [{viewing.level}] {new Date(viewing.timestamp * 1000).toLocaleString('pt-BR')}
                  </span>
                </span>
                <span className="block text-sm text-white/85 mt-1">{viewing.message}</span>
              </span>
              <button
                onClick={() => { setViewing(null); onOpenMonitoramento() }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white text-gray-800 text-xs font-medium hover:bg-gray-100 shrink-0"
              >
                <Video size={14} />
                Abrir monitoramento
              </button>
            </figcaption>
            <button
              onClick={() => setViewing(null)}
              className="absolute -top-3 -right-3 p-1.5 rounded-full bg-white text-gray-700 shadow hover:bg-gray-100"
              title="Fechar (Esc)"
            >
              <X size={16} />
            </button>
          </figure>
        </div>,
        document.body,
      )}
    </div>
  )
}
