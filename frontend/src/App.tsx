import { useEffect, useState } from 'react'
import { useTheme } from 'next-themes'
import { Moon, Sun } from 'lucide-react'
import { AppSidebar, type PageId, PAGES } from '@/components/app-sidebar'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { api, type Health } from '@/lib/api'
import AdvisorPage from '@/pages/advisor'
import PerformancePage from '@/pages/performance'
import DatasetPage from '@/pages/dataset'
import FeedbackPage from '@/pages/feedback'
import AboutPage from '@/pages/about'

function StatusDot({ ok, label, title }: { ok: boolean | null; label: string; title: string }) {
  const color = ok == null ? 'bg-muted-foreground' : ok ? 'bg-[var(--series-3)]' : 'bg-[var(--series-2)]'
  return (
    <Badge variant="outline" className="gap-1.5 font-normal" title={title}>
      <span className={`size-2 rounded-full ${color}`} />
      {label}
    </Badge>
  )
}

export default function App() {
  // Deep link: ?page=performance opens a page directly (handy for presentations).
  const initial = new URLSearchParams(window.location.search).get('page') as PageId | null
  const [page, setPage] = useState<PageId>(PAGES.some((p) => p.id === initial) ? initial! : 'advisor')
  const [health, setHealth] = useState<Health | null>(null)
  const { resolvedTheme, setTheme } = useTheme()

  useEffect(() => {
    const check = () => api.health().then(setHealth).catch(() => setHealth(null))
    check()
    const t = setInterval(check, 15000)
    return () => clearInterval(t)
  }, [])

  const current = PAGES.find((p) => p.id === page)!
  return (
    <SidebarProvider>
      <AppSidebar page={page} onNavigate={setPage} />
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-14 shrink-0 items-center gap-3 border-b bg-background/90 px-4 backdrop-blur">
          <SidebarTrigger />
          <Separator orientation="vertical" className="h-5" />
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-semibold">{current.title}</h1>
            <p className="hidden truncate text-xs text-muted-foreground sm:block">{current.subtitle}</p>
          </div>
          <div className="hidden items-center gap-2 md:flex">
            <StatusDot ok={health ? health.models_loaded : null} label={health?.models_loaded ? 'Models ready' : 'Models'} title="NLP models loaded on the server" />
            <StatusDot ok={health ? health.llm_online : null} label={health?.llm_online ? `LLM online` : 'LLM offline'} title={health?.llm_model ?? 'Local LLM (Ollama)'} />
          </div>
          <Button variant="ghost" size="icon" aria-label="Toggle theme"
                  onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}>
            {resolvedTheme === 'dark' ? <Sun /> : <Moon />}
          </Button>
        </header>
        <main className="flex-1 p-4 md:p-6">
          {page === 'advisor' && <AdvisorPage />}
          {page === 'performance' && <PerformancePage />}
          {page === 'dataset' && <DatasetPage />}
          {page === 'feedback' && <FeedbackPage />}
          {page === 'about' && <AboutPage />}
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
