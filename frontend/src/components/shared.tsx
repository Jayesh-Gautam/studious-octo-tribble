import { useEffect, useState, type ReactNode } from 'react'
import { ExternalLink, FileText, Loader2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from '@/components/ui/dialog'
import { Skeleton } from '@/components/ui/skeleton'
import { api, fullDocUrl, pageUrl, pct, type CaseDetail, type Passage } from '@/lib/api'

export function StatCard({ label, value, hint, icon, children }: {
  label: string; value: ReactNode; hint?: ReactNode; icon?: ReactNode; children?: ReactNode
}) {
  return (
    <Card className="gap-2 py-4">
      <CardHeader className="px-4">
        <CardDescription className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide">
          {icon}
          {label}
        </CardDescription>
        <CardTitle className="text-xl leading-tight font-semibold tabular-nums">{value}</CardTitle>
      </CardHeader>
      {(hint || children) && (
        <CardContent className="space-y-2 px-4 text-xs text-muted-foreground">
          {children}
          {hint}
        </CardContent>
      )}
    </Card>
  )
}

/** Ranked horizontal bars: label on the left, bar + value on the right. */
export function BarList({ items, color = 'var(--series-1)', format = (v: number) => pct(v), max }: {
  items: { label: ReactNode; value: number; extra?: ReactNode; key?: string }[]
  color?: string
  format?: (v: number) => string
  max?: number
}) {
  const top = max ?? Math.max(...items.map((i) => i.value), 1e-9)
  return (
    <div className="space-y-2.5">
      {items.map((it, i) => (
        <div key={it.key ?? i} className="space-y-1">
          <div className="flex items-baseline justify-between gap-3 text-sm">
            <span className="min-w-0 truncate">{it.label}</span>
            <span className="shrink-0 tabular-nums text-muted-foreground">{format(it.value)}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-muted">
            <div className="h-full rounded-full" style={{ width: `${Math.max(2, (100 * it.value) / top)}%`, background: color }} />
          </div>
          {it.extra}
        </div>
      ))}
    </div>
  )
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-10 text-center">
      <div className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">{icon}</div>
      <p className="font-medium">{title}</p>
      {children && <div className="max-w-md text-sm text-muted-foreground">{children}</div>}
    </div>
  )
}

/** In-app viewer for the exact cited page of a public FAA document, plus "open full document". */
export function PdfDialog({ passage, onOpenChange }: { passage: Passage | null; onOpenChange: (o: boolean) => void }) {
  // The excerpt holds pages p-1..p+1, so the cited page is the 2nd page (or the 1st when p = 1).
  const inExcerpt = passage && passage.page > 1 ? 2 : 1
  return (
    <Dialog open={!!passage} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[90vh] max-w-5xl flex-col gap-3 sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="size-4" /> {passage?.doc} · page {passage?.page}
          </DialogTitle>
          <DialogDescription>
            {passage?.heading} · showing the cited page with one page of context either side. Public FAA guidance,
            not aircraft-specific: always follow the approved maintenance data.
          </DialogDescription>
        </DialogHeader>
        {passage && (
          <iframe title={`${passage.doc} page ${passage.page}`} className="w-full flex-1 rounded-md border bg-muted"
                  src={`${pageUrl(passage.doc, passage.page)}#page=${inExcerpt}&view=FitH`} />
        )}
        <DialogFooter>
          {passage && (
            <Button asChild variant="outline">
              <a href={fullDocUrl(passage.doc, passage.page)} target="_blank" rel="noreferrer">
                <ExternalLink /> Open full document at page {passage.page}
              </a>
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="text-sm font-medium">{value || '-'}</dd>
    </div>
  )
}

/** The full original FAA Service Difficulty Report behind a similar case. */
export function CaseDialog({ rowId, similarity, onOpenChange }: {
  rowId: number | null; similarity?: number; onOpenChange: (o: boolean) => void
}) {
  const [detail, setDetail] = useState<CaseDetail | null>(null)
  useEffect(() => {
    setDetail(null)
    if (rowId != null) api.caseDetail(rowId).then(setDetail).catch(() => setDetail(null))
  }, [rowId])
  return (
    <Dialog open={rowId != null} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>Original service difficulty report</DialogTitle>
          <DialogDescription>
            FAA SDR record #{rowId}{similarity != null && ` · similarity to your fault ${similarity.toFixed(2)}`}
          </DialogDescription>
        </DialogHeader>
        {!detail ? (
          <div className="space-y-2"><Skeleton className="h-16" /><Skeleton className="h-32" /></div>
        ) : (
          <div className="space-y-4">
            <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Field label="Aircraft" value={`${detail.family} · ${detail.variant}`} />
              <Field label="Date" value={detail.date ?? detail.year} />
              <Field label="System" value={`ATA ${detail.ata2} · ${detail.ata_title}`} />
              <Field label="JASC code" value={detail.jasc} />
              <Field label="Part" value={detail.part_name} />
              <Field label="Location" value={detail.part_location} />
              <Field label="Condition" value={detail.condition} />
              <Field label="Stage of operation" value={detail.stage} />
            </dl>
            <div className="max-h-72 overflow-y-auto rounded-md border bg-muted/40 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap">
              {detail.report}
            </div>
            {detail.refs.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-xs text-muted-foreground">Manual references cited:</span>
                {detail.refs.map((r) => <Badge key={r} variant="secondary">{r}</Badge>)}
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

export function Spinner() {
  return <Loader2 className="size-4 animate-spin" />
}
