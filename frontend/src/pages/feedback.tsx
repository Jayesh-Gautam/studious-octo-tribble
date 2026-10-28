import { useEffect, useState } from 'react'
import { CheckCircle2, FlaskConical, MessageSquareText, RefreshCw, XCircle } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { EmptyState, StatCard } from '@/components/shared'
import { api, type Row } from '@/lib/api'

const title = (s: string) => s.toLowerCase().replace(/(^|[\s/-])\w/g, (m) => m.toUpperCase())

export default function FeedbackPage() {
  const [data, setData] = useState<{ rows: Row[]; counts: Record<string, number> } | null>(null)
  const load = () => api.feedbackList().then(setData).catch(() => setData({ rows: [], counts: {} }))
  useEffect(() => { load() }, [])
  const rows = data?.rows ?? []
  const confirmed = rows.filter((r) => r.system_correct === 1).length
  const corrected = rows.filter((r) => r.system_correct === 0).length

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Feedback from the app" icon={<MessageSquareText className="size-3.5" />} value={data?.counts.app ?? 0} />
        <StatCard label="System confirmed" icon={<CheckCircle2 className="size-3.5" />} value={confirmed} />
        <StatCard label="System corrected" icon={<XCircle className="size-3.5" />} value={corrected} />
        <StatCard label="Simulated feedback (2024)" icon={<FlaskConical className="size-3.5" />} value={(data?.counts.simulated ?? 0).toLocaleString('en-US')}
                  hint="used for the learning-curve experiment" />
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Technician feedback</CardTitle>
          <CardDescription>Newest first. Confirmed reports join the search index; corrections retrain the models in the next update.</CardDescription>
          <CardAction><Button variant="outline" size="sm" onClick={load}><RefreshCw /> Refresh</Button></CardAction>
        </CardHeader>
        <CardContent>
          {rows.length === 0 ? (
            <EmptyState icon={<MessageSquareText />} title="No feedback yet">
              Analyse a fault on the Fault advisor page and send feedback from the bottom of the Overview tab.
            </EmptyState>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>When (UTC)</TableHead><TableHead>Aircraft</TableHead><TableHead>Fault</TableHead>
                  <TableHead>Predicted</TableHead><TableHead>Verdict</TableHead><TableHead>Action that fixed it</TableHead><TableHead>Comment</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={String(r.id)}>
                    <TableCell className="whitespace-nowrap text-muted-foreground">{String(r.ts).replace('T', ' ').slice(0, 16)}</TableCell>
                    <TableCell className="whitespace-nowrap">{r.family} {r.variant && `· ${r.variant}`}</TableCell>
                    <TableCell className="max-w-xs truncate font-mono text-xs" title={String(r.text)}>{r.text}</TableCell>
                    <TableCell>ATA {r.predicted_ata2}</TableCell>
                    <TableCell>
                      {r.system_correct === 1 ? <Badge variant="outline" className="text-[var(--series-3)]">Correct</Badge>
                        : r.system_correct === 0 ? <Badge variant="outline" className="text-[var(--series-2)]">Wrong → ATA {r.correct_ata2 ?? '?'}</Badge>
                        : <span className="text-muted-foreground">-</span>}
                    </TableCell>
                    <TableCell>{r.action_fixed ? title(String(r.action_fixed)) : '-'}</TableCell>
                    <TableCell className="max-w-[12rem] truncate text-muted-foreground">{r.comment ?? '-'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
