// Typed client for the AeroFix FastAPI service (all routes under /api).

export type Prob = [string, number]

export interface Passage {
  label: string
  heading: string
  text: string
  doc: string
  page: number
}

export interface CaseSummary {
  row_id: number
  problem: string
  action: string
  refs: string[]
  actions: string[]
  year: number
  variant: string
  similarity: number
  text: string
}

export interface Advice {
  request_id: string
  input: { family: string; variant: string; stage: string | null; text: string }
  under_the_hood: { cleaned: string; context_prefix: string; masked: string; normalised: string; lemmatised: string }
  system: { ata2: [string, number, string][]; ata4: Prob[] }
  condition: Prob[]
  actions: Prob[]
  references: { reference: string; score: number; cited_in_share: number; global_share: number }[]
  cases: CaseSummary[]
  passages: Passage[]
  disclaimer: string
  timings: Record<string, number>
}

export interface SummaryResult {
  text: string
  ok: boolean
  n_citations: number
  invalid_citations: string[]
  hallucinated_refs: string[]
  has_citation: boolean
}

export interface CaseDetail {
  row_id: number
  family: string
  variant: string
  year: number
  date: string | null
  ata2: string
  ata_title: string
  jasc: string
  condition: string
  part_name: string
  part_location: string
  stage: string
  refs: string[]
  report: string
  split: string
}

export interface Meta {
  families: string[]
  variants: Record<string, string[]>
  stages: Record<string, string>
  examples: string[]
  ata_titles: Record<string, string>
  documents: string[]
}

export interface Health {
  status: string
  models_loaded: boolean
  llm_online: boolean
  llm_model: string
}

export type Row = Record<string, string | number | null>

export interface Metrics {
  classification: Row[]
  leakage: Row[]
  cross_family: Row[]
  retrieval: Row[]
  retrieval_llm_check: Row[]
  actions: Row[]
  references: Row[]
  segmentation: Row[]
  language_model: Row[]
  rag: { qwen: Row[]; llama: Row[] }
  learning_curve: Row[]
}

export interface Dataset {
  counts: Record<string, number | Record<string, number>>
  per_year: Row[]
  ata_share: Row[]
  conditions: Row[]
  references: Row[]
  variants: Row[]
  split_sizes: Record<string, number>
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`/api${path}`)
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

export const api = {
  health: () => get<Health>('/health'),
  meta: () => get<Meta>('/meta'),
  advise: (body: { family: string; variant: string; text: string; stage: string | null }) =>
    post<Advice>('/advise', body),
  caseDetail: (id: number) => get<CaseDetail>(`/cases/${id}`),
  metrics: () => get<Metrics>('/metrics'),
  dataset: () => get<Dataset>('/dataset'),
  feedbackList: () => get<{ rows: Row[]; counts: Record<string, number> }>('/feedback'),
  sendFeedback: (body: Record<string, unknown>) => post<{ id: number }>('/feedback', body),
}

// Public FAA PDFs: the cited page with one page of context, or the whole document opened at that page.
export const pageUrl = (doc: string, page: number) =>
  `/api/documents/${encodeURIComponent(doc)}/pages/${page}`
export const fullDocUrl = (doc: string, page: number) =>
  `/api/documents/${encodeURIComponent(doc)}#page=${page}`

/** Stream the cited LLM summary. Returns a function that cancels the stream. */
export function streamSummary(
  requestId: string,
  onToken: (t: string) => void,
  onDone: (s: SummaryResult) => void,
  onError: (msg: string) => void,
): () => void {
  const es = new EventSource(`/api/summary/${requestId}`)
  es.onmessage = (e) => onToken(JSON.parse(e.data).token)
  es.addEventListener('done', (e) => {
    onDone(JSON.parse((e as MessageEvent).data))
    es.close()
  })
  es.addEventListener('error', (e) => {
    const data = (e as MessageEvent).data
    onError(data ? JSON.parse(data).error : 'connection lost')
    es.close()
  })
  return () => es.close()
}

export const pct = (x: number | null | undefined, digits = 0) =>
  x == null ? '-' : `${(x * 100).toFixed(digits)}%`
