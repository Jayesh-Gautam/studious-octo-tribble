# Viva notes: design decisions and why

## Phase 0: Setup

**src layout + `pip install -e .`**
The code lives in `src/aerofix/` and is installed in editable mode. Tests and scripts then import the *installed* package, so they behave the same from any folder. This avoids `sys.path` hacks and catches packaging mistakes early.

**One config file (`configs/config.yaml`)**
Seeds, paths, split years, regexes and hyperparameters live in one place. Any result can be traced to a config, and nothing depends on where the project sits on disk. `load_config()` turns relative paths into absolute ones under the project root.

**Fixed seeds (`set_seed`)**
Python `random`, NumPy and PyTorch (CPU + CUDA) are all seeded from `seed: 42`. GPU kernels can still differ slightly between runs, so reported numbers are from single seeded runs. Small differences are noise, not signal.

**Time-based split (train ≤ 2023, val 2024, test 2025)**
In deployment the model always predicts *future* reports from *past* ones. A random split would leak information, because the same operator writes near-identical reports in the same month. Splitting by year matches real use and is the stricter test.

**Python 3.12 instead of 3.11**
3.12 was already installed, and every library in the stack ships 3.12 wheels (PyTorch CUDA, faiss-cpu, sklearn-crfsuite). Changing version would add setup work for no benefit.

**PyTorch installed separately from `requirements.txt`**
The default PyPI `torch` wheel on Windows is CPU-only. The CUDA build comes from PyTorch's own index (`cu124`). Keeping it as a separate, documented step makes sure the GPU build is used. `requirements-lock.txt` records the exact versions installed.

**faiss-cpu over chromadb**
FAISS is a single library doing exact or approximate vector search. Our corpus (~tens of thousands of reports plus a few thousand doc chunks) fits in memory, so we don't need a database server. FAISS is also easier to explain: it is just an index over a matrix of embeddings.

**FAA documents: current "B" handbook editions only**
FAA-H-8083-30B/31B/32B replace the older split volumes. Indexing both editions would put near-duplicate passages in the RAG index and crowd out other results.

## Phase 1: Data pipeline

**Read only the needed columns, as strings, and filter at once.** Each CSV has 76 columns and ~60k rows. Reading 22 columns with `dtype=str` avoids pandas guessing types (part numbers like `0012` would lose zeros) and keeps memory low. Per-year Parquet caches make re-runs take seconds.

**UTF-8, checked rather than assumed.** All 32 files decode as strict UTF-8, so they are read that way. Latin-1 would silently garble the ~15k non-ASCII characters in later years.

**Deduplicate before splitting, keep the earliest copy.** The same operator often files the same text again in a later year. If the 2023 copy is in train and the 2025 copy in test, the model is tested on a report it has already seen. 11,924 such cross-year copies were removed. The dedup key ignores punctuation and spacing ("R/H" = "R H").

**Split on file year, not a random split.** Every `DifficultyDate` falls inside its file's year (0 mismatches), so file year is a clean time stamp.

**Why mask references (label leakage).** "PER AMM 52-10-00" literally contains the answer (ATA 52). 64% of 2025 reports with a reference cite the label's chapter. A model trained on raw text would learn to read reference numbers instead of understanding the fault. We keep the references in a `refs` column (used by the Phase 6 recommender) and train on `text_masked`. Phase 4 measures F1 with and without masking to show the effect.

**Why the regex grew.** The brief's regex missed real formats (`M/M 33-51-06`, chained `SRM 51-10-02; 51-40-02`, `SRM B737-700 51-20-01`). Each new format has a unit test (`tests/test_refs.py`), so a regex change can't silently break an older case. A final safety net masks any leftover `NN-NN-NN` number without adding it to `refs`, because a few could be dates.

**Boilerplate kept as a feature.** "AIRCRAFT WAS GROUNDED:" says something about severity, not about which system failed. It is stripped from the text (so it doesn't dominate similarity) and kept in `context`.

**MAX variant mapping is evidence-based.** Bare codes `7378`/`7379` appear only from 2017/2018 (MAX entry into service), and where the engine field is filled it is LEAP-1B (MAX engine) rather than CFM56-7B (NG engine).

**Why top-K chapters come from train only.** Choosing classes by looking at the test year would be a (mild) form of tuning on test.

## Phase 2: Text normalisation

**Tokeniser built for the domain.** Generic tokenisers split `R&R`, `P/N`, `A/C` and `737-700` into meaningless pieces. Ours keeps alphanumerics joined by `& / ' . -` together, keeps `[REF]` whole, and treats `W/` ("with") as one token.

**Abbreviations: mined, then reviewed by a human.** Two mining rules on train:
1. **Shape:** no vowels, or contains `&` or `/`.
2. **Contraction:** the token's letters appear in order inside a much more frequent word with the same first letter (RPLCD ⊂ REPLACED).

The automatic suggestions are often wrong (`MIA` is an airport, not MAINTENANCE), so only the hand-reviewed `abbrev.csv` is applied. Rows marked ambiguous (`LT` = left or light) are never expanded, because a wrong expansion is worse than none.

**N-gram LM written from scratch, with Kneser-Ney.**
- *Laplace* adds 1 to every count. With ~80k vocabulary words, almost all trigram contexts are rare, so the +1s swamp the real counts and the trigram ends up *worse* than the unigram (perplexity 2,834 vs 862).
- *Kneser-Ney* subtracts a small discount D = 0.75 from seen counts and gives that mass to a lower-order model. The lower-order model is built on *continuation counts*: how many different contexts a word follows, not how often it occurs. Trigram KN reaches perplexity ~41.
- Unit tests check that every model's probabilities sum to 1 for seen and unseen contexts. That is the correctness property of a smoothing method.

**Spelling: noisy channel = edit model × language model.**
- Candidates come from a SymSpell-style index of all strings 1–2 deletions away from a dictionary word. This is fast, and the delete trick finds every word within 2 edits.
- Scoring: `log P_LM(word | context) - penalty × edit distance`.
- The penalty was tuned on half of a set of synthetic typos and reported on the other half (no tuning on the reported numbers).
- The best penalty is high (15), which means "fewest edits first, then let context break ties". Context breaks ties better than word frequency does.
- A WordNet guard was added after the first run "corrected" valid rare words (LOADABLE → READABLE). Precision matters more than recall here, because a wrong correction changes the meaning.

**Stemming vs lemmatisation.**
- **Porter stemming** applies suffix rules: fast (8 s for the corpus) but gives non-words (VALVES → VALV).
- **WordNet lemmatisation** needs a POS tag (REPLACED as a verb → REPLACE). It gives readable output but is about 70× slower (10 min).

**Honest negative result.** On TF-IDF + logistic regression, none of the normalisation steps changes val macro-F1 by more than noise. A linear model with 100k training reports sees each common variant often enough to weight it on its own. We keep normalisation because it is expected to help retrieval of short queries (Phase 5) and the embedding-based models (Phase 4), and those phases measure it.

## Phase 3: Problem / action segmentation

**Why segment at all?** At query time the technician only knows the *problem*. If the classifier is trained on the full report, it learns from the *action* ("REPLACED EMERGENCY LIGHT BATTERY") and looks better in evaluation than it will be in use. Training on `problem_masked` matches what the deployed system actually sees.

**Annotation design.**
- Chunk-level clicking (split at punctuation, step markers and action verbs) with an optional split *inside* a chunk. This keeps 400 reports at a few hours of work while still giving token-level labels.
- Train/dev reports are pre-filled by the rule baseline to save time.
- Test reports are **not** pre-filled. Otherwise the annotator might accept rule mistakes on test and the rule baseline would look better than it is (anchoring bias).

**BIO tags.** Labels are encoded as B-X (first token of a span) / I-X (inside). Without B tags, two adjacent spans of the same type couldn't be told apart, and the CRF transition weights couldn't learn "a new span starts here".

**CRF vs BERT.**
- The **CRF** combines hand-made features with *transition* weights between tags, and Viterbi decoding returns the best whole sequence. Features encode human cues: past tense, an action-verb lexicon, punctuation before the word, position in the report, and whether an action verb has appeared yet.
- **DistilBERT** learns its own features from context, but 280 reports is very little data for a 66M-parameter model. The learning curve shows whether more annotation would help.

**Own training loop instead of HF Trainer.**
- AdamW with weight decay, 10% linear warm-up then linear decay, fp16 autocast with GradScaler, gradient clipping, and early stopping on dev with the best weights restored.
- Every step is visible and explainable, and there's no extra dependency.

**Sub-word labels.** WordPiece splits a word into pieces. Only the first piece is trained and predicted; the others get label -100, which the loss ignores. So each word is counted exactly once.

**Two metrics.**
- **Token F1** gives partial credit for nearly-right boundaries.
- **Span F1** is strict: start, end and label must all match. A segmenter that's off by one word gets decent token F1 but zero span credit for that span. Both are reported because they answer different questions.

**Change of plan: LLM-assisted (silver) annotation.** Hand-labelling 400 reports (3–4 hours) wasn't possible in the time available, so the local LLM (Qwen 2.5 7B via Ollama) labelled them. Be ready to defend these points:
- **Chunk-level, not word-level.** A 7B model is unreliable at counting words but good at judging phrases. It labels the same chunks the human tool shows.
- **Constrained output.** One *required* JSON key per chunk, enforced by Ollama's schema-constrained decoding, so the number of labels can never be wrong (0 fallbacks out of 399).
- **Deterministic constraints where rules are reliable.** References come from the Phase 1 regex (forced to REF), and punctuation follows the previous word.
- **Worked examples (few-shot)** come from train reports and the guide, never from test reports.
- **Human labels always win.** They live in a separate file and override the LLM.
- **Honest reporting.** The report states that test scores measure *agreement with the LLM labels*. Human verification of the 60 test reports is the planned next step and would make the evaluation fully gold-standard.

**Result: CRF beats BERT on spans, BERT edges it on tokens.**
- DistilBERT's softmax predicts each token on its own, so it fragments spans (P P A P A). Token F1 forgives that; strict span F1 doesn't.
- The CRF's transition weights and Viterbi decoding produce whole, clean spans.
- The standard fix is a CRF layer on top of BERT (BERT-CRF), listed as future work.
- With only 280 training reports, a 66M-parameter model can't beat good features plus structured decoding. The learning curve shows BERT is still improving at 280.

## Phase 4: Classification

**Why `problem_masked` as input.** At prediction time the technician has only the fault description. Training on the full report (which includes the fix) or unmasked references would inflate scores without helping real use. The leakage tables quantify both effects.

**Macro-F1 as the headline.** ATA 53 is 46% of train. Accuracy rewards predicting 53; macro-F1 averages per-class F1, so ATA 34 (139 test reports) counts as much as ATA 53 (11,507).

**Label spaces from train only.** Top-K chapters, JASC codes ≥ 100 and top-20 conditions are counted on train. UNKNOWN/NONE are placeholders, not conditions, so they go to OTHER.

**Class weights.**
- LR uses fully "balanced" weights (1/frequency).
- The neural models use (1/frequency)^0.5. Full inverse weights with 226 ATA4 classes make rare-class gradients huge and training unstable; the square root is a common compromise.

**Multi-task learning.** One encoder, three heads, summed losses. The chapter, the JASC code and the condition share evidence ("CRACKED FRAME" informs both 53 and CRACKED), and one model is cheaper to serve than three.

**Character n-grams.** These make TF-IDF robust to typos (CORRISION shares most 3–5-grams with CORROSION). A control experiment showed they must be taken from tokenised text: punctuation glued to words ("LEAKING.") split the evidence.

**Word2Vec learned the typos.** The nearest neighbours of CORROSION are CORRSION, COROSION, CORROSSION… Skip-gram puts words used in the same contexts close together, which is why the BiLSTM doesn't need spelling correction to cope with typos.

**Length-grouped batches.** Median problem text is 21 sub-words, the maximum 128. Sorting by length inside random mega-batches cuts padding, so an epoch takes 8 min instead of ~25, and batch order stays random.

**Why not RoBERTa?** It matched DistilBERT (val 0.844 vs 0.848) at twice the cost. The texts are short and full of jargon and abbreviations unlike RoBERTa's pre-training data, so extra capacity has little to work with. DistilBERT is also faster at inference, which matters for the < 5 s app target.

**Error analysis.**
- Most errors are genuine ambiguity: fume events (APU vs air conditioning), door-slide parts (52 vs 25), structure near doors (53 vs 52).
- A few come from segmentation: action text left inside the problem ("FOUND AND CORRECTED").

## Phase 5: Retrieval

**Proxy relevance and metric choice.** There are no human relevance labels for 500k reports. "Same JASC code + same part condition" is a cheap stand-in for "the same kind of fault". Because each query has many such matches, classical recall (share of *all* relevant found) is meaningless, so we use Hit@k, P@5, MRR@10 and graded NDCG@5 (same JASC only = partial credit).

**BM25 on sparse matrices.** The per-(term, document) BM25 weights are pre-computed once. A query is then the sum of its terms' columns, touching only their posting lists: ~10 ms over 400k reports, against ~0.5 s for the rank-bm25 loop.
- k1 controls term-frequency saturation; b controls length normalisation.
- A unit test checks the ranking against rank-bm25.

**Bi-encoder vs cross-encoder.**
- A bi-encoder embeds the query and each report *separately*, so all report vectors are computed once and search is a fast dot product (FAISS).
- A cross-encoder reads the query *and* a candidate together, so attention compares them word by word. It's more accurate but must run once per candidate, which is why it only reranks the top 50.

**Multiple Negatives Ranking loss.** For a batch of (anchor, positive, hard negative) triples, every other positive in the batch acts as a free negative.
- The loss is cross-entropy over the scaled cosine-similarity matrix.
- Hard negatives (same chapter, different JASC code) teach the model fine distinctions that random negatives never would.
- Batches never contain two anchors from the same group, so a true match is never treated as a negative.

**Reciprocal rank fusion.** BM25 scores are unbounded and cosine is between -1 and 1, so adding scores would be meaningless. RRF uses ranks only: score = Σ 1/(60 + rank).

**Honest caveat: optimising the proxy.** The fine-tuned model was trained with the same definition the proxy metric uses, so proxy metrics favour it. The LLM judge (a different, reader-like notion of relevance) preferred BM25 on 50 queries, and the two agree on only 63% of pairs. Choosing the hybrid for the app, which mixes lexical and semantic evidence, hedges against this. A small human-judged set is the proper fix.

## Phase 6: Recommenders

**Why recommend from retrieved cases rather than a classifier?** A classifier over the predicted JASC code gives the *typical* actions for that subsystem. Similar past cases carry the specific part and fix. The numbers confirm it: action hit@1 is 0.31 for the retrieval vote vs 0.12 for the JASC-based baseline.

**Canonical actions.**
- Free-text actions are reduced to verb + head noun so that "R&R LH AFT EMERG LIGHT PWR PACK M1673" and "REPLACED POWER PACK" count as the same recommendation.
- Rules handle the grammar (verb lemmas, "removed and replaced" = replace, objects end at prepositions and punctuation).
- Clustering handles vocabulary variants (plurals, word order, synonyms) that rules would miss.
- Follow-up checks ("OPS CHECK GOOD") are excluded because they follow almost every repair and would top every list.

**Similarity-weighted vote.** Each retrieved case votes for its actions with its cosine similarity to the query, so near-identical cases count more. The percentage shown is that action's share of the total vote.

**Specificity (log-lift).** lift = (share of retrieved cases citing r) / (share of all reports citing r). Ranking by support × log(lift) favours references that are unusually common *for this fault*. Honest result: it reduces how often the generic reference comes first (13.8% → 7.7%) but lowers the hit rate, because generic references really are cited. That's a trade-off between informativeness and matching what technicians wrote, and the app shows both numbers.

## Phase 7: RAG over public documents + cited LLM summary

**Only public data.**
- The corpus is the FAA AMT Handbooks (General, Airframe, Powerplant) and AC 43.13-1B: 7,237 passages.
- The TCDS PDFs are scanned images with no text, so they are excluded rather than OCR'd.
- No OEM manual text is used anywhere. The LLM may only *point* to references technicians cited ("refer to AMM 52-10-00 in the approved maintenance data").

**Chunking by headings.**
- A heading is detected as a short, all-bold line at or above the document's body font size.
- Sections are cut into 300-word chunks with a 50-word overlap, so a sentence split at a boundary still appears whole in one chunk.
- Every chunk keeps its document, page and heading, which become the citation label `[AC 43.13-1B p.212]`.

**Passage query.** The query is the predicted ATA chapter's title plus the fault text ("DOORS: DAMPER LEAKING"). Without the title, "damper" matched shimmy dampers on landing gear.

**Generation design.**
- The LLM sees only assembled evidence, every item labelled.
- Temperature is 0.2.
- No procedures, values or part numbers beyond the evidence.
- Short output: 3 sentences, at most 2 labels per sentence.

**Citation validator** (mechanical, not LLM-judged):
1. Every `[label]` must exist in the context.
2. Every manual reference written in the summary must appear in the context.
3. At least one citation must be present.

**Key fix:** the first version cited in only 24% of summaries. The model stated facts from the *aggregated* lines ("replace EPSU, 25%"), which had no labels to cite. Tagging each aggregated action and reference with its supporting cases ("see [Case 1] [Case 3]") and adding one style example raised it to 100% with ~96% validity. Lesson: the model can only cite what the context makes citable.

**Faithfulness:** the automatic checks catch invented citations and references, not subtle misreadings. A 1–5 rubric sheet (`reports/rag_faithfulness_sheet.csv`) is provided for human scoring.

**Latency trade-off.**
- Everything except the LLM takes ~0.1–0.15 s. Generation dominates.
- Qwen 2.5 7B takes ~5.5–6 s end to end, with 0 hallucinated references.
- Llama 3.2 3B takes ~4.4 s, with 4% hallucinated references.
- In the app the structured results appear immediately and the summary streams, so the technician isn't waiting on the LLM.
