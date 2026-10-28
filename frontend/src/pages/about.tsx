import { ArrowRight, BookOpen, Brain, Database, FileText, Layers, Search, ShieldAlert, Sparkles, Wrench } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const STEPS = [
  { icon: FileText, title: 'Fault text', text: 'Technician enters aircraft and defect, free text with shorthand.' },
  { icon: Layers, title: 'NLP preprocessing', text: 'Abbreviations, spelling (noisy channel + n-gram LM), reference masking, lemmas.' },
  { icon: Brain, title: 'Classification', text: 'Fine-tuned multi-task DistilBERT: ATA chapter, JASC subsystem, part condition.' },
  { icon: Search, title: 'Retrieval', text: 'BM25 + fine-tuned MiniLM, fused and reranked by a cross-encoder.' },
  { icon: Wrench, title: 'Recommendation', text: 'Similarity-weighted actions; references ranked by specificity.' },
  { icon: Sparkles, title: 'Cited summary', text: 'Local LLM (Qwen 2.5 7B) writes from evidence only; citations are verified.' },
]

const STACK = ['Python 3.12', 'PyTorch + CUDA', 'Hugging Face Transformers', 'sentence-transformers', 'scikit-learn',
  'sklearn-crfsuite', 'gensim', 'NLTK', 'FAISS', 'PyMuPDF', 'Ollama', 'FastAPI', 'React', 'shadcn/ui', 'Tailwind CSS', 'Recharts']

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">AeroFix: maintenance fault advisor</CardTitle>
          <CardDescription className="text-base">
            A final-year NLP project (B.Tech ICT). It reads 31 years of public FAA Service Difficulty Reports for the Airbus
            A320 family and Boeing 737, and turns a technician's free-text defect into the likely system, the fixes and manual
            references used for similar faults, the evidence behind them, and a short cited summary.
          </CardDescription>
        </CardHeader>
      </Card>

      <Card>
        <CardHeader><CardTitle>How it works</CardTitle><CardDescription>One request, end to end (about 0.2 s plus a streamed summary).</CardDescription></CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-6">
            {STEPS.map((s, i) => (
              <div key={s.title} className="relative rounded-lg border p-3">
                <div className="mb-2 flex items-center gap-2">
                  <div className="flex size-7 items-center justify-center rounded-md bg-muted"><s.icon className="size-4" /></div>
                  <span className="text-xs text-muted-foreground">Step {i + 1}</span>
                </div>
                <p className="text-sm font-medium">{s.title}</p>
                <p className="mt-1 text-xs text-muted-foreground">{s.text}</p>
                {i < STEPS.length - 1 && <ArrowRight className="absolute top-1/2 -right-3 hidden size-4 -translate-y-1/2 text-muted-foreground xl:block" />}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><Database className="size-4" /> Data sources</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p><b>FAA Service Difficulty Reports</b>, 1995-2025: 540,008 A320-family and 737 reports after cleaning and deduplication.
              Split by time: train ≤ 2023, validation 2024, test 2025.</p>
            <p><b>Public FAA guidance</b>: AMT Handbooks (General, Airframe, Powerplant) and AC 43.13-1B, split into 7,237 cited passages.
              The Documents tab opens the exact page of each passage.</p>
            <p className="text-muted-foreground">No proprietary Airbus or Boeing manuals (AMM, SRM, FIM) are used. AeroFix only points to the references
              technicians cited in public reports.</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><BookOpen className="size-4" /> Limitations</CardTitle></CardHeader>
          <CardContent>
            <ul className="list-disc space-y-1.5 pl-4 text-sm text-muted-foreground">
              <li>US operators only; reporting habits may differ elsewhere.</li>
              <li>Retrieval and recommendation scores use proxy relevance labels.</li>
              <li>Segmentation training labels were produced by a local LLM (silver labels).</li>
              <li>Reporters code the same structure to different chapters, which caps accuracy.</li>
              <li>Part-condition codes drift over time; periodic updates are needed.</li>
              <li>FAA handbooks are generic and rarely cover aircraft-specific systems.</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle>Technology</CardTitle><CardDescription>Runs fully locally on a laptop GPU (RTX 4060, 8 GB).</CardDescription></CardHeader>
        <CardContent className="flex flex-wrap gap-2">{STACK.map((s) => <Badge key={s} variant="secondary">{s}</Badge>)}</CardContent>
      </Card>

      <Alert>
        <ShieldAlert />
        <AlertTitle>Decision support only</AlertTitle>
        <AlertDescription>AeroFix never replaces approved maintenance data or engineering judgement. Always follow the AMM/SRM/FIM and your organisation's procedures.</AlertDescription>
      </Alert>
    </div>
  )
}
