import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { toast } from 'sonner'
import {
  BookOpen, Bot, CheckCircle2, Clock, Cpu, ExternalLink, FileSearch, FileText, Gauge, History, ListChecks,
  ScanSearch, Send, ShieldAlert, Sparkles, TriangleAlert, Wrench,
} from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { BarList, CaseDialog, EmptyState, PdfDialog, Spinner, StatCard } from '@/components/shared'
import { api, fullDocUrl, pct, streamSummary, type Advice, type Meta, type Passage, type SummaryResult } from '@/lib/api'

const title = (s: string) => s.toLowerCase().replace(/(^|[\s/-])\w/g, (m) => m.toUpperCase())
const WORD = /[A-Z0-9][A-Z0-9/&-]*/g
const STOP = new Set(['THE', 'AND', 'WITH', 'FOR', 'FROM', 'WAS', 'ARE', 'NOT', 'REF', 'ON', 'AT', 'IN', 'OF', 'TO'])

/** Highlight words the past case shares with the query: shows WHY it was retrieved. */
function Highlight({ text, words }: { text: string; words: Set<string> }) {
  const parts: ReactNode[] = []
  let last = 0
  for (const m of text.matchAll(WORD)) {
    if (!words.has(m[0])) continue
    parts.push(text.slice(last, m.index), <mark key={m.index}>{m[0]}</mark>)
    last = (m.index ?? 0) + m[0].length
  }
  parts.push(text.slice(last))
  return <>{parts}</>
}

// ------------------------------------------------------------------ input panel
function InputPanel({ meta, busy, onSubmit }: {
  meta: Meta | null; busy: boolean
  onSubmit: (v: { family: string; variant: string; stage: string | null; text: string }) => void
}) {
  // Start from the deep-link values (?q=...&family=...&variant=...) so the form matches the shown result.
  const url = new URLSearchParams(window.location.search)
  const [family, setFamily] = useState(url.get('family') === 'B737' ? 'B737' : 'A320')
  const [variant, setVariant] = useState(url.get('variant') ?? (url.get('family') === 'B737' ? '737-800' : 'A320'))
  const [stage, setStage] = useState('none')
  const [text, setText] = useState(url.get('q') ?? 'L1 door damper leaking, emer cyl press low')
  const variants = meta?.variants[family] ?? []
  useEffect(() => { if (variants.length && !variants.includes(variant)) setVariant(variants[0]) }, [family, variants, variant])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Wrench className="size-4" /> New fault report</CardTitle>
        <CardDescription>Write the defect as it appears in the tech log; shorthand and typos are fine.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Aircraft family</Label>
          <ToggleGroup type="single" variant="outline" value={family} onValueChange={(v) => v && setFamily(v)} className="w-full">
            <ToggleGroupItem value="A320" className="flex-1">Airbus A320 family</ToggleGroupItem>
            <ToggleGroupItem value="B737" className="flex-1">Boeing 737</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-2">
            <Label>Variant</Label>
            <Select value={variant} onValueChange={setVariant}>
              <SelectTrigger className="w-full"><SelectValue placeholder="Variant" /></SelectTrigger>
              <SelectContent>{variants.map((v) => <SelectItem key={v} value={v}>{v}</SelectItem>)}</SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label>Stage of operation</Label>
            <Select value={stage} onValueChange={setStage}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="none">Not specified</SelectItem>
                {Object.entries(meta?.stages ?? {}).map(([k, v]) => <SelectItem key={k} value={k}>{v}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
        </div>
        <div className="space-y-2">
          <Label htmlFor="fault">Fault description</Label>
          <Textarea id="fault" rows={4} value={text} maxLength={2000} onChange={(e) => setText(e.target.value)}
                    placeholder="e.g. FWD LAV SMOKE DETECTOR INOP" className="font-mono text-sm" />
          <div className="flex flex-wrap gap-1.5">
            {(meta?.examples ?? []).map((ex) => (
              <button key={ex} type="button" onClick={() => setText(ex)}
                      className="rounded-full border px-2.5 py-0.5 text-xs text-muted-foreground transition hover:bg-muted hover:text-foreground">
                {ex}
              </button>
            ))}
          </div>
        </div>
        <Button className="w-full" size="lg" disabled={busy || text.trim().length < 3}
                onClick={() => onSubmit({ family, variant, stage: stage === 'none' ? null : stage, text })}>
          {busy ? <Spinner /> : <ScanSearch />} {busy ? 'Analysing...' : 'Analyse fault'}
        </Button>
      </CardContent>
    </Card>
  )
}

function HowItWorks() {
  const steps = [
    ['Clean & normalise', 'expand shorthand, fix typos, hide manual references'],
    ['Classify', 'fine-tuned DistilBERT predicts system, subsystem, condition'],
    ['Retrieve', 'search 507k past reports (keyword + semantic + reranker)'],
    ['Recommend', 'vote actions and references across the 20 closest cases'],
    ['Explain', 'local LLM writes a summary that cites its evidence'],
  ]
  return (
    <Card className="gap-3">
      <CardHeader><CardTitle className="text-sm">How a request is processed</CardTitle></CardHeader>
      <CardContent>
        <ol className="space-y-2.5">
          {steps.map(([t, d], i) => (
            <li key={t} className="flex gap-3 text-sm">
              <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium">{i + 1}</span>
              <span><span className="font-medium">{t}</span> <span className="text-muted-foreground">· {d}</span></span>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  )
}

// ------------------------------------------------------------------ summary with clickable citations
function CitedText({ text, onCase, onDoc, passages }: {
  text: string; onCase: (n: number) => void; onDoc: (p: Passage) => void; passages: Passage[]
}) {
  return (
    <p className="leading-7">
      {text.split(/(\[[^\]]+\])/g).map((part, i) => {
        const m = part.match(/^\[(.+)\]$/)
        if (!m) return <Fragment key={i}>{part}</Fragment>
        const label = m[1].trim()
        const caseNo = label.match(/^Case (\d+)$/)
        const passage = passages.find((p) => p.label === label)
        if (caseNo) {
          return <button key={i} onClick={() => onCase(Number(caseNo[1]))}
                         className="mx-0.5 inline-flex items-center rounded-md bg-[color-mix(in_oklab,var(--series-1)_14%,transparent)] px-1.5 text-xs font-medium text-[var(--series-1)] hover:underline">
            Case {caseNo[1]}</button>
        }
        if (passage) {
          return <button key={i} onClick={() => onDoc(passage)}
                         className="mx-0.5 inline-flex items-center gap-1 rounded-md bg-[color-mix(in_oklab,var(--series-3)_16%,transparent)] px-1.5 text-xs font-medium text-[var(--series-3)] hover:underline">
            <FileText className="size-3" />{label}</button>
        }
        return <Badge key={i} variant="destructive" className="mx-0.5">{label}?</Badge>
      })}
    </p>
  )
}

function SummaryCard({ advice, onCase, onDoc, onFinished }: {
  advice: Advice; onCase: (n: number) => void; onDoc: (p: Passage) => void; onFinished: (ms: number) => void
}) {
  const [text, setText] = useState('')
  const [result, setResult] = useState<SummaryResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    setText(''); setResult(null); setError(null)
    const t0 = performance.now()
    return streamSummary(advice.request_id, (tok) => setText((s) => s + tok),
      (r) => { setResult(r); onFinished(performance.now() - t0) }, setError)
  }, [advice.request_id, onFinished])

  return (
    <Card className="border-[color-mix(in_oklab,var(--series-1)_35%,var(--border))]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Sparkles className="size-4 text-[var(--series-1)]" /> AI summary</CardTitle>
        <CardDescription>Written by a local LLM from the evidence below only. Click a citation to open its source.</CardDescription>
        <CardAction>
          {result ? (
            result.ok
              ? <Badge variant="outline" className="gap-1 text-[var(--series-3)]"><CheckCircle2 className="size-3" />{result.n_citations} citations verified</Badge>
              : <Badge variant="outline" className="gap-1 text-[var(--series-2)]"><TriangleAlert className="size-3" />Check citations</Badge>
          ) : !error && <Badge variant="secondary" className="gap-1"><Spinner />Writing</Badge>}
        </CardAction>
      </CardHeader>
      <CardContent>
        {error ? (
          <Alert variant="destructive"><Bot /><AlertTitle>Summary unavailable</AlertTitle>
            <AlertDescription>The local LLM could not be reached ({error}). All other results are still valid.</AlertDescription></Alert>
        ) : result ? (
          <>
            <CitedText text={result.text} onCase={onCase} onDoc={onDoc} passages={advice.passages} />
            {!result.ok && (
              <p className="mt-2 text-xs text-[var(--series-2)]">
                {result.invalid_citations.length > 0 && `Unknown citation(s): ${result.invalid_citations.join(', ')}. `}
                {result.hallucinated_refs.length > 0 && `Reference(s) not in the evidence: ${result.hallucinated_refs.join(', ')}. `}
                {!result.has_citation && 'No citations given.'}
              </p>
            )}
          </>
        ) : text ? (
          <p className="leading-7 text-muted-foreground">{text}<span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-foreground align-middle" /></p>
        ) : (
          <div className="space-y-2"><Skeleton className="h-4 w-full" /><Skeleton className="h-4 w-11/12" /><Skeleton className="h-4 w-3/4" /></div>
        )}
      </CardContent>
    </Card>
  )
}

// ------------------------------------------------------------------ feedback
function FeedbackCard({ advice, meta }: { advice: Advice; meta: Meta | null }) {
  const [verdict, setVerdict] = useState('')
  const [correct, setCorrect] = useState('none')
  const [fixed, setFixed] = useState('none')
  const [comment, setComment] = useState('')
  const [sent, setSent] = useState(false)
  const pred = advice.system.ata2[0][0]
  const send = async () => {
    try {
      await api.sendFeedback({
        text: advice.input.text, family: advice.input.family, variant: advice.input.variant, stage: advice.input.stage,
        predicted_ata2: pred, predicted_condition: advice.condition[0][0],
        system_correct: verdict ? verdict === 'yes' : null,
        correct_ata2: verdict === 'yes' ? pred : correct !== 'none' ? correct : null,
        action_fixed: fixed !== 'none' ? fixed : null, comment: comment || null,
      })
      setSent(true)
      toast.success('Feedback saved', { description: 'It will be used in the next model update.' })
    } catch (e) {
      toast.error('Could not save feedback', { description: String(e) })
    }
  }
  if (sent) {
    return <Alert><CheckCircle2 /><AlertTitle>Thanks, feedback recorded</AlertTitle>
      <AlertDescription>See it in the Feedback log. Confirmed reports are added to the search index and used to refresh the models.</AlertDescription></Alert>
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Send className="size-4" /> Technician feedback</CardTitle>
        <CardDescription>Close the loop: tell AeroFix what was actually wrong and what fixed it.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <Label>Was ATA {pred} the right system?</Label>
          <ToggleGroup type="single" variant="outline" value={verdict} onValueChange={setVerdict}>
            <ToggleGroupItem value="yes">Correct</ToggleGroupItem>
            <ToggleGroupItem value="no">Wrong</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div className="space-y-2">
          <Label>If wrong, the correct chapter</Label>
          <Select value={correct} onValueChange={setCorrect} disabled={verdict !== 'no'}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="none">-</SelectItem>
              {Object.entries(meta?.ata_titles ?? {}).map(([k, v]) => <SelectItem key={k} value={k}>ATA {k} · {title(v)}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label>Which action fixed it?</Label>
          <Select value={fixed} onValueChange={setFixed}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="none">-</SelectItem>
              {advice.actions.map(([a]) => <SelectItem key={a} value={a}>{title(a)}</SelectItem>)}
              <SelectItem value="Something else">Something else</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label htmlFor="cmt">Comment</Label>
          <Input id="cmt" value={comment} onChange={(e) => setComment(e.target.value)} placeholder="Optional" />
        </div>
        <div className="md:col-span-2"><Button onClick={send} disabled={!verdict && fixed === 'none'}><Send /> Send feedback</Button></div>
      </CardContent>
    </Card>
  )
}

// ------------------------------------------------------------------ page
/** production = the clean operational view: no explanations, disclaimer banner or "under the hood" details. */
export default function AdvisorPage({ production = false }: { production?: boolean }) {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [advice, setAdvice] = useState<Advice | null>(null)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState('overview')
  const [openCase, setOpenCase] = useState<number | null>(null)
  const [openDoc, setOpenDoc] = useState<Passage | null>(null)
  const [llmMs, setLlmMs] = useState<number | null>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  useEffect(() => { api.meta().then(setMeta).catch(() => toast.error('Server not reachable')) }, [])
  // Deep link: ?q=<fault>&family=B737&variant=737-800&tab=documents runs an analysis on load.
  useEffect(() => {
    const p = new URLSearchParams(window.location.search)
    const q = p.get('q')
    if (!q) return
    const family = p.get('family') === 'B737' ? 'B737' : 'A320'
    run({ family, variant: p.get('variant') ?? (family === 'B737' ? '737-800' : 'A320'), stage: null, text: q })
      .then(() => p.get('tab') && !(production && p.get('tab') === 'hood') && setTab(p.get('tab')!))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const run = async (v: { family: string; variant: string; stage: string | null; text: string }) => {
    setBusy(true); setLlmMs(null)
    try {
      const a = await api.advise(v)
      setAdvice(a); setTab((t) => (t === 'overview' ? t : 'overview'))
      requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    } catch (e) {
      toast.error('Analysis failed', { description: String(e) })
    } finally {
      setBusy(false)
    }
  }
  const onFinished = useMemo(() => (ms: number) => setLlmMs(ms), [])
  const queryWords = useMemo(() => new Set(
    [...(advice?.under_the_hood.normalised ?? '').matchAll(WORD)].map((m) => m[0]).filter((w) => w.length > 2 && !STOP.has(w)),
  ), [advice])
  const caseByNumber = (n: number) => advice?.cases[n - 1]
  const openedCase = advice?.cases.find((c) => c.row_id === openCase)

  return (
    <div className={`grid gap-6 xl:grid-cols-[380px_minmax(0,1fr)] ${production ? '[&_[data-slot=card-description]]:hidden' : ''}`}>
      <div className="space-y-6">
        <InputPanel meta={meta} busy={busy} onSubmit={run} />
        {!production && <HowItWorks />}
      </div>

      <div ref={resultsRef} className="min-w-0 scroll-mt-20 space-y-6">
        {!production && (
          <Alert className="border-[color-mix(in_oklab,var(--series-4)_45%,var(--border))]">
            <ShieldAlert />
            <AlertTitle>Decision support only</AlertTitle>
            <AlertDescription>
              AeroFix summarises public FAA reports and guidance; it contains no OEM procedures. Always follow the approved AMM/SRM/FIM.
            </AlertDescription>
          </Alert>
        )}

        {!advice && !busy && (
          <EmptyState icon={<FileSearch />} title="No fault analysed yet">
            Pick the aircraft, describe the defect and press <b>Analyse fault</b>. Results appear in about a quarter of a
            second; the AI summary follows within a few seconds.
          </EmptyState>
        )}
        {busy && !advice && (
          <div className="grid gap-4 md:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-32" />)}</div>
        )}

        {advice && (
          <div className={busy ? 'pointer-events-none opacity-60 transition' : 'transition'}>
            {/* ---- headline predictions ---- */}
            <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-4">
              <StatCard label="Predicted system" icon={<Cpu className="size-3.5" />}
                        value={<>ATA {advice.system.ata2[0][0]} <span className="font-normal text-muted-foreground">· {title(advice.system.ata2[0][2])}</span></>}>
                <Progress value={advice.system.ata2[0][1] * 100} />
                <p>{pct(advice.system.ata2[0][1])} confidence · next: {advice.system.ata2.slice(1).map(([c, p]) => `ATA ${c} (${pct(p)})`).join(', ')}</p>
              </StatCard>
              <StatCard label="Part condition" icon={<Gauge className="size-3.5" />} value={title(advice.condition[0][0])}>
                <Progress value={advice.condition[0][1] * 100} />
                <p>{pct(advice.condition[0][1])} confidence · next: {advice.condition.slice(1).map(([c, p]) => `${title(c)} (${pct(p)})`).join(', ')}</p>
              </StatCard>
              <StatCard label="Subsystem (JASC)" icon={<ListChecks className="size-3.5" />} value={advice.system.ata4[0][0]}
                        hint={`${pct(advice.system.ata4[0][1])} confidence · alternatives ${advice.system.ata4.slice(1).map(([c]) => c).join(', ')}`} />
              <StatCard label="Response time" icon={<Clock className="size-3.5" />}
                        value={`${Math.round(advice.timings.total_ms)} ms`}
                        hint={llmMs ? `AI summary completed in ${(llmMs / 1000).toFixed(1)} s` : 'AI summary streaming...'} />
            </div>

            <Tabs value={tab} onValueChange={setTab} className="mt-6">
              <TabsList className="flex-wrap">
                <TabsTrigger value="overview"><Sparkles /> Overview</TabsTrigger>
                <TabsTrigger value="evidence"><History /> Similar cases <Badge variant="secondary" className="ml-1">{advice.cases.length}</Badge></TabsTrigger>
                <TabsTrigger value="documents"><BookOpen /> FAA documents <Badge variant="secondary" className="ml-1">{advice.passages.length}</Badge></TabsTrigger>
                {!production && <TabsTrigger value="hood"><Cpu /> Under the hood</TabsTrigger>}
              </TabsList>

              {/* ---- overview ---- */}
              <TabsContent value="overview" className="space-y-6 pt-2">
                <SummaryCard advice={advice} onFinished={onFinished}
                             onCase={(n) => { const c = caseByNumber(n); if (c) setOpenCase(c.row_id) }}
                             onDoc={setOpenDoc} />
                <div className="grid gap-6 lg:grid-cols-2">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2"><Wrench className="size-4" /> Recommended corrective actions</CardTitle>
                      <CardDescription>Share of the similarity-weighted vote of the 20 most similar past reports.</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {advice.actions.length ? (
                        <BarList items={advice.actions.map(([a, s], i) => ({ label: <span className={i === 0 ? 'font-medium' : ''}>{title(a)}</span>, value: s }))} />
                      ) : <p className="text-sm text-muted-foreground">No corrective actions recorded in the similar cases.</p>}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2"><BookOpen className="size-4" /> Manual references technicians used</CardTitle>
                      <CardDescription>Ranked by specificity: how much more often each is cited here than in all reports.</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader><TableRow>
                          <TableHead>Reference</TableHead><TableHead>Similar cases</TableHead><TableHead className="text-right">All reports</TableHead>
                        </TableRow></TableHeader>
                        <TableBody>
                          {advice.references.map((r) => (
                            <TableRow key={r.reference}>
                              <TableCell className="font-medium">
                                <span className="font-mono">{r.reference}</span>{' '}
                                {r.global_share > 0.02
                                  ? <Tooltip><TooltipTrigger asChild><Badge variant="outline" className="ml-1">generic</Badge></TooltipTrigger><TooltipContent>Cited in {pct(r.global_share, 1)} of all reports</TooltipContent></Tooltip>
                                  : <Badge variant="outline" className="ml-1 text-[var(--series-3)]">specific</Badge>}
                              </TableCell>
                              <TableCell>
                                <div className="flex items-center gap-2">
                                  <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted"><div className="h-full bg-[var(--series-1)]" style={{ width: `${r.cited_in_share * 100}%` }} /></div>
                                  <span className="tabular-nums">{pct(r.cited_in_share)}</span>
                                </div>
                              </TableCell>
                              <TableCell className="text-right tabular-nums text-muted-foreground">{pct(r.global_share, 1)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                      {!production && <p className="mt-3 text-xs text-muted-foreground">Pointers only: the procedure must come from the approved OEM maintenance data.</p>}
                    </CardContent>
                  </Card>
                </div>
                <FeedbackCard key={advice.request_id} advice={advice} meta={meta} />
              </TabsContent>

              {/* ---- evidence ---- */}
              <TabsContent value="evidence" className="space-y-4 pt-2">
                {!production && <p className="text-sm text-muted-foreground">The closest past reports from the same aircraft family (2024 and earlier). Highlighted words are shared with your fault.</p>}
                {advice.cases.map((c, i) => (
                  <Card key={c.row_id} className="gap-3">
                    <CardHeader>
                      <CardTitle className="flex flex-wrap items-center gap-2 text-sm">
                        <Badge>Case {i + 1}</Badge> {c.variant} · {c.year}
                      </CardTitle>
                      <CardAction className="flex items-center gap-2 text-xs text-muted-foreground">
                        similarity
                        <div className="h-1.5 w-20 overflow-hidden rounded-full bg-muted"><div className="h-full bg-[var(--series-1)]" style={{ width: `${Math.max(0, c.similarity) * 100}%` }} /></div>
                        <span className="tabular-nums">{c.similarity.toFixed(2)}</span>
                      </CardAction>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <p><span className="text-muted-foreground">Problem · </span><Highlight text={c.problem} words={queryWords} /></p>
                      <p><span className="text-muted-foreground">Action · </span>{c.action || <i className="text-muted-foreground">not recorded</i>}</p>
                      <div className="flex flex-wrap items-center gap-1.5 pt-1">
                        {c.actions.map((a) => <Badge key={a} variant="secondary">{title(a)}</Badge>)}
                        {c.refs.map((r) => <Badge key={r} variant="outline" className="font-mono">{r}</Badge>)}
                        <Button size="sm" variant="ghost" className="ml-auto" onClick={() => setOpenCase(c.row_id)}><FileText /> Original report</Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>

              {/* ---- documents ---- */}
              <TabsContent value="documents" className="space-y-4 pt-2">
                {!production && <p className="text-sm text-muted-foreground">Passages from public FAA guidance (AMT handbooks, AC 43.13-1B) related to the predicted system. Open the exact page they come from.</p>}
                {advice.passages.map((p) => (
                  <Card key={p.label} className="gap-3">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-sm"><FileText className="size-4" /> {p.doc} · page {p.page}</CardTitle>
                      <CardDescription>{p.heading}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <p className="line-clamp-5 text-sm leading-relaxed">{p.text}</p>
                      <div className="flex flex-wrap gap-2">
                        <Button size="sm" onClick={() => setOpenDoc(p)}><BookOpen /> View page {p.page}</Button>
                        <Button size="sm" variant="outline" asChild>
                          <a href={fullDocUrl(p.doc, p.page)} target="_blank" rel="noreferrer"><ExternalLink /> Open full document</a>
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </TabsContent>

              {/* ---- under the hood (presentation console only) ---- */}
              {!production && <TabsContent value="hood" className="space-y-6 pt-2">
                <Card>
                  <CardHeader><CardTitle>Pipeline timing</CardTitle><CardDescription>Server time per step for this request.</CardDescription></CardHeader>
                  <CardContent>
                    <BarList color="var(--series-5)" format={(v) => `${v.toFixed(v < 10 ? 1 : 0)} ms`}
                             items={[['Preprocess & normalise', 'preprocess_ms'], ['Classify (DistilBERT)', 'classify_ms'], ['Retrieve & rerank', 'retrieve_ms'],
                                     ['Recommend', 'recommend_ms'], ['FAA passages', 'passages_ms']]
                               .map(([l, k]) => ({ label: l, value: advice.timings[k] ?? 0 }))} />
                    <p className="mt-3 text-xs text-muted-foreground">
                      Total {Math.round(advice.timings.total_ms)} ms{llmMs ? ` · AI summary streamed in ${(llmMs / 1000).toFixed(1)} s` : ''}
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>Text processing</CardTitle><CardDescription>Exactly what each model sees.</CardDescription></CardHeader>
                  <CardContent>
                    <Table>
                      <TableBody>
                        {[['Input', advice.input.text], ['Cleaned', advice.under_the_hood.cleaned],
                          ['References masked', advice.under_the_hood.masked], ['Abbreviations + spelling', advice.under_the_hood.normalised],
                          ['Lemmatised (keyword search)', advice.under_the_hood.lemmatised]].map(([k, v]) => (
                          <TableRow key={k}><TableCell className="w-56 text-muted-foreground">{k}</TableCell><TableCell className="font-mono text-xs whitespace-normal">{v}</TableCell></TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
                <div className="grid gap-6 lg:grid-cols-3">
                  {[['ATA chapter', advice.system.ata2.map(([c, p, t]) => [`${c} ${title(t)}`, p] as [string, number])],
                    ['JASC subsystem', advice.system.ata4], ['Part condition', advice.condition.map(([c, p]) => [title(c), p] as [string, number])]]
                    .map(([name, rows]) => (
                      <Card key={name as string}>
                        <CardHeader><CardTitle className="text-sm">{name as string}: top 3</CardTitle></CardHeader>
                        <CardContent><BarList max={1} items={(rows as [string, number][]).map(([l, v]) => ({ label: l, value: v }))} /></CardContent>
                      </Card>
                    ))}
                </div>
              </TabsContent>}
            </Tabs>
          </div>
        )}
      </div>

      <CaseDialog rowId={openCase} similarity={openedCase?.similarity} onOpenChange={(o) => !o && setOpenCase(null)} />
      <PdfDialog passage={openDoc} onOpenChange={(o) => !o && setOpenDoc(null)} />
    </div>
  )
}
