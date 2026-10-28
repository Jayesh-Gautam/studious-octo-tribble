import { useEffect, useState } from 'react'
import { BookOpen, Brain, CheckCircle2, Search, Sparkles, Wrench } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { HBarChart, LineChartCard, VBarChart } from '@/components/charts'
import { StatCard } from '@/components/shared'
import { api, pct, type Metrics, type Row } from '@/lib/api'

const num = (r: Row | undefined, k: string) => Number(r?.[k] ?? 0)

function ChartCard({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <Card>
      <CardHeader><CardTitle>{title}</CardTitle><CardDescription>{description}</CardDescription></CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}

function DataTable({ rows, columns }: { rows: Row[]; columns: { key: string; label: string; fmt?: (v: Row[string]) => string }[] }) {
  return (
    <Table>
      <TableHeader><TableRow>{columns.map((c) => <TableHead key={c.key}>{c.label}</TableHead>)}</TableRow></TableHeader>
      <TableBody>
        {rows.map((r, i) => (
          <TableRow key={i}>
            {columns.map((c) => <TableCell key={c.key} className="tabular-nums">{c.fmt ? c.fmt(r[c.key]) : String(r[c.key] ?? '-')}</TableCell>)}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export default function PerformancePage() {
  const [m, setM] = useState<Metrics | null>(null)
  useEffect(() => { api.metrics().then(setM).catch(() => setM(null)) }, [])
  if (!m) return <div className="grid gap-4 md:grid-cols-3">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-36" />)}</div>

  const test = m.classification.filter((r) => r.split === 'test')
  const byModel = (target: string) => Object.fromEntries(test.filter((r) => r.target === target).map((r) => [String(r.model), r]))
  const ata2 = byModel('ata2'), cond = byModel('condition'), ata4 = byModel('ata4')
  const models = Object.keys(ata2).sort((a, b) => num(ata2[b], 'macro_F1') - num(ata2[a], 'macro_F1'))
  const best = ata2['DistilBERT (multi-task)'], tfidf = ata2['TF-IDF + LR']
  const ret = [...m.retrieval].sort((a, b) => num(b, 'NDCG@5') - num(a, 'NDCG@5'))
  const qwen = Object.fromEntries(m.rag.qwen.map((r) => [String(r.metric), String(r.value)]))
  const llama = Object.fromEntries(m.rag.llama.map((r) => [String(r.metric), String(r.value)]))
  const actBest = m.actions.find((r) => String(r.method).startsWith('Retrieval vote, top 20'))
  const refBest = m.references.find((r) => r.method === 'Retrieval + count')
  const short = (s: string) => s.replace('Hybrid RRF (BM25 + lemma + MiniLM fine-tuned)', 'Hybrid (BM25 + MiniLM)').replace(' (similarity-weighted)', '')

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <Badge variant="outline">Test set: 2025 reports (never used for tuning)</Badge>
        <Badge variant="outline">Model choices made on 2024</Badge>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <StatCard label="ATA system macro-F1" icon={<Brain className="size-3.5" />} value={num(best, 'macro_F1').toFixed(3)}
                  hint={`+${(num(best, 'macro_F1') - num(tfidf, 'macro_F1')).toFixed(3)} over TF-IDF baseline`} />
        <StatCard label="Top-3 accuracy" icon={<CheckCircle2 className="size-3.5" />} value={pct(num(best, 'top3_acc'), 1)}
                  hint="right system among the 3 shown" />
        <StatCard label="Part condition macro-F1" icon={<Wrench className="size-3.5" />} value={num(cond['DistilBERT (multi-task)'], 'macro_F1').toFixed(3)}
                  hint="21 classes, codes drift over time" />
        <StatCard label="Retrieval NDCG@5" icon={<Search className="size-3.5" />} value={num(ret[0], 'NDCG@5').toFixed(3)}
                  hint={`best: ${short(String(ret[0].method))}`} />
        <StatCard label="Action / reference hit@3" icon={<BookOpen className="size-3.5" />}
                  value={`${pct(num(actBest, 'hit@3'))} / ${pct(num(refBest, 'hit@3'))}`} hint="from the 20 most similar cases" />
        <StatCard label="Summary citations valid" icon={<Sparkles className="size-3.5" />} value={qwen['citation validity'] ?? '-'}
                  hint={`${qwen['summaries with a hallucinated reference'] ?? '-'} invented references`} />
      </div>

      <Tabs defaultValue="classification">
        <TabsList className="flex-wrap">
          <TabsTrigger value="classification">Classification</TabsTrigger>
          <TabsTrigger value="retrieval">Retrieval & recommendations</TabsTrigger>
          <TabsTrigger value="llm">AI summaries & learning</TabsTrigger>
          <TabsTrigger value="nlp">NLP foundations</TabsTrigger>
        </TabsList>

        <TabsContent value="classification" className="grid gap-6 pt-2 xl:grid-cols-2">
          <ChartCard title="ATA chapter: every model vs baselines" description="Macro-F1 on 2025 reports (each of 16 classes counts equally).">
            <HBarChart data={models.map((k) => ({ model: k, ata2: num(ata2[k], 'macro_F1'), ata4: num(ata4[k], 'macro_F1'), condition: num(cond[k], 'macro_F1') }))}
                       category="model" max={1} series={[{ key: 'ata2', label: 'ATA chapter' }, { key: 'condition', label: 'Part condition' }, { key: 'ata4', label: 'JASC subsystem' }]} />
          </ChartCard>
          <ChartCard title="Label leakage: why manual references are masked" description="DistilBERT macro-F1. References like AMM 52-10-00 contain the answer; the deployed model only sees the masked problem text.">
            <HBarChart data={m.leakage.map((r) => ({ input: String(r['DistilBERT input']).replace(/ \(.*\)/, ''), macro: num(r, 'macro_F1') }))}
                       category="input" max={1} series={[{ key: 'macro', label: 'macro-F1' }]} />
          </ChartCard>
          <ChartCard title="Cross-family transfer" description="Train on one fleet, test on each (equal training size). Training on both families matters.">
            <VBarChart category="test" domain={[0.6, 0.9]}
                       data={['A320', 'B737'].map((te) => ({ test: `Test on ${te}`, ...Object.fromEntries(m.cross_family.filter((r) => r['test on'] === te).map((r) => [`train_${r['train on']}`, num(r, 'macro_F1')])) }))}
                       series={[{ key: 'train_A320', label: 'Trained on A320' }, { key: 'train_B737', label: 'Trained on B737' }]} />
          </ChartCard>
          <ChartCard title="Per-model detail" description="ATA chapter on 2025 reports.">
            <DataTable rows={models.map((k) => ata2[k])} columns={[{ key: 'model', label: 'Model' }, { key: 'macro_F1', label: 'Macro-F1', fmt: (v) => Number(v).toFixed(3) },
              { key: 'accuracy', label: 'Accuracy', fmt: (v) => pct(Number(v), 1) }, { key: 'top3_acc', label: 'Top-3 acc.', fmt: (v) => pct(Number(v), 1) }]} />
          </ChartCard>
        </TabsContent>

        <TabsContent value="retrieval" className="grid gap-6 pt-2 xl:grid-cols-2">
          <ChartCard title="Finding similar past reports" description="NDCG@5 and precision@5 on 2,000 2025 queries (relevant = same subsystem and part condition).">
            <HBarChart data={ret.map((r) => ({ method: short(String(r.method)), ndcg: num(r, 'NDCG@5'), p5: num(r, 'P@5') }))} category="method" max={1}
                       series={[{ key: 'ndcg', label: 'NDCG@5' }, { key: 'p5', label: 'Precision@5' }]} />
          </ChartCard>
          <div className="grid gap-6">
            <ChartCard title="Corrective-action recommendation" description="Is the action that was actually done in the top 1 / top 3?">
              <HBarChart data={m.actions.map((r) => ({ method: short(String(r.method)), h1: num(r, 'hit@1'), h3: num(r, 'hit@3') }))} category="method" max={1}
                         format={(v) => pct(v)} series={[{ key: 'h1', label: 'hit@1' }, { key: 'h3', label: 'hit@3' }]} />
            </ChartCard>
            <ChartCard title="Manual-reference recommendation" description="Is a reference the technician actually cited in the top 1 / top 3?">
              <HBarChart data={m.references.map((r) => ({ method: short(String(r.method)).replace(' (TF-IDF + one-vs-rest LR, 300 refs)', ''), h1: num(r, 'hit@1'), h3: num(r, 'hit@3') }))}
                         category="method" max={1} format={(v) => pct(v)} series={[{ key: 'h1', label: 'hit@1' }, { key: 'h3', label: 'hit@3' }]} />
            </ChartCard>
          </div>
        </TabsContent>

        <TabsContent value="llm" className="grid gap-6 pt-2 xl:grid-cols-2">
          <ChartCard title="Cited AI summaries: automatic checks" description="50 random 2025 faults through the full pipeline.">
            <Table>
              <TableHeader><TableRow><TableHead>Check</TableHead><TableHead>Qwen 2.5 7B (used)</TableHead><TableHead>Llama 3.2 3B</TableHead></TableRow></TableHeader>
              <TableBody>
                {Object.keys(qwen).filter((k) => k !== 'queries').map((k) => (
                  <TableRow key={k}><TableCell>{k}</TableCell><TableCell className="tabular-nums font-medium">{qwen[k]}</TableCell><TableCell className="tabular-nums">{llama[k]}</TableCell></TableRow>
                ))}
              </TableBody>
            </Table>
          </ChartCard>
          <ChartCard title="Feedback loop: part condition" description="Quarterly 2024 feedback rounds vs a control trained the same amount on old data only.">
            <LineChartCard data={m.learning_curve.map((r) => ({ round: `R${r.round}`, feedback: num(r, 'cond_feedback_macroF1'), control: num(r, 'cond_control_macroF1') }))}
                           x="round" series={[{ key: 'feedback', label: 'With feedback' }, { key: 'control', label: 'Control: old data only' }]} />
          </ChartCard>
          <ChartCard title="Feedback loop: similar-case retrieval" description="Adding confirmed reports to the index is what helps most; fine-tuning adds a little.">
            <LineChartCard data={m.learning_curve.map((r) => ({ round: `R${r.round}`, tuned: num(r, 'ret_growth_finetune_Hit@5'), growth: num(r, 'ret_growth_Hit@5') }))}
                           x="round" series={[{ key: 'tuned', label: 'Index growth + fine-tune' }, { key: 'growth', label: 'Index growth only' }]} />
          </ChartCard>
          <ChartCard title="Feedback loop: ATA chapter" description="No measurable effect: chapters are stable and already learned (changes < 1 point).">
            <LineChartCard domain={[0.82, 0.9]} data={m.learning_curve.map((r) => ({ round: `R${r.round}`, feedback: num(r, 'clf_feedback_macroF1'), control: num(r, 'clf_control_macroF1') }))}
                           x="round" series={[{ key: 'feedback', label: 'With feedback' }, { key: 'control', label: 'Control' }]} />
          </ChartCard>
        </TabsContent>

        <TabsContent value="nlp" className="grid gap-6 pt-2 xl:grid-cols-2">
          <ChartCard title="N-gram language model (written from scratch)" description="Perplexity on 2024 text, lower is better. Laplace collapses at trigrams; Kneser-Ney keeps improving.">
            <VBarChart category="model" format={(v) => v.toFixed(0)} height={260}
                       data={['1-gram', '2-gram', '3-gram'].map((o) => ({ model: o, ...Object.fromEntries(m.language_model.filter((r) => r.model === o).map((r) => [r.smoothing === 'Laplace' ? 'laplace' : 'kn', num(r, 'val_perplexity')])) }))}
                       series={[{ key: 'laplace', label: 'Laplace' }, { key: 'kn', label: 'Kneser-Ney' }]} />
          </ChartCard>
          <ChartCard title="Problem / action segmentation" description="Span F1 on 60 test reports (labels made by the local LLM: silver).">
            <HBarChart data={m.segmentation.filter((r) => r.split === 'test').map((r) => ({ method: String(r.method).replace(/ \(.*\)/, ''), span: num(r, 'span_micro_F1'), token: num(r, 'token_macro_F1') }))}
                       category="method" max={1} series={[{ key: 'span', label: 'Span F1 (strict)' }, { key: 'token', label: 'Token macro-F1' }]} />
          </ChartCard>
        </TabsContent>
      </Tabs>
    </div>
  )
}
