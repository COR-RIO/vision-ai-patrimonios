import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import net from 'net'

const START_PORT = 5173

// Checagem por CONEXÃO (cliente), não por bind (servidor): no Windows, dois
// processos conseguem fazer bind na mesma porta sem erro (falta de
// SO_EXCLUSIVEADDRUSE), então net.createServer().listen() mente dizendo que
// a porta está livre mesmo com outro processo já escutando nela. Tentar
// conectar como cliente é o jeito confiável de saber se alguém já responde
// naquela porta.
function isPortInUseOn(port: number, host: string): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = new net.Socket()
    const done = (result: boolean) => {
      socket.destroy()
      resolve(result)
    }
    socket.setTimeout(500)
    socket.once('connect', () => done(true))
    socket.once('timeout', () => done(false))
    socket.once('error', () => done(false))
    socket.connect(port, host)
  })
}

// Testa IPv4 e IPv6: no Windows o Vite com host padrão ('localhost') escuta
// só em ::1, então checar apenas 127.0.0.1 não enxerga essa instância.
async function isPortInUse(port: number): Promise<boolean> {
  const results = await Promise.all(['127.0.0.1', '::1'].map((h) => isPortInUseOn(port, h)))
  return results.some(Boolean)
}

async function findAvailablePort(startPort: number, maxAttempts = 10): Promise<number> {
  for (let port = startPort; port < startPort + maxAttempts; port++) {
    if (!(await isPortInUse(port))) {
      if (port !== startPort) {
        console.log(`⚠️  Porta ${startPort} já está em uso — subindo em ${port} no lugar`)
      }
      return port
    }
  }
  throw new Error(`Nenhuma porta livre entre ${startPort} e ${startPort + maxAttempts - 1}`)
}

export default defineConfig(async () => {
  const port = await findAvailablePort(START_PORT)

  return {
    plugins: [react()],
    server: {
      host: true,
      port,
      strictPort: true,
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        '/health': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        // Proxy da página de vídeo WebRTC (Tixxi) e das chamadas relativas
        // que o próprio player faz (/auth/refresh, /app/session/*,
        // /app/whep/*) — ver backend/app/api/video_proxy.py pro porquê.
        '/video': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        '/auth': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        '/app': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        // Imagens salvas pelo backend (snapshots, prints com IA, evidências).
        // Só essas subpastas: /assets/*.js do build do Vite não pode ir pro backend.
        '/assets/images': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        '/assets/evidence': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        '/assets/alerts': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
        '/logo.jpg': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
  }
})
