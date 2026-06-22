# LegalAssist / TRACE-Law — Full Review

*A walk-through of the whole project (paper + report + code + action guide), the
bug you hit, every inconsistency I found across the four documents, and concrete
fixes. Written to be readable without deep coding knowledge.*

---

## 0. The bug you hit first (step 1.7 → "Not enough data available")

**What was happening, in plain English.** Step 1.7 builds a tiny 2-document
"corpus" and searches it with BM25 (a keyword search method). BM25 has a quirk:
it scores a word by how *rare* it is across the corpus. Its rarity formula gives
a word a score of **exactly 0** when that word appears in half (or more) of the
documents. With only 2 documents, "bail", "cheating", "anticipatory" each appear
in 1 of 2 docs → each scores 0 → the whole document scores 0.

The retriever then had a line that said *"only keep documents with a score above
0."* Because every score was exactly 0, it threw **everything** away, returned an
empty list, and the generator printed its empty-handed message: *"Not enough
data available in the retrieved documents."* So nothing was wrong with what you
typed — the demo simply could never return a result as written.

**The fix.** I changed the keep-rule in `src/trace_law/bm25_index.py` from
"keep docs scoring above 0" to **"keep docs that share at least one word with the
query."** That keeps the original intent (drop totally unrelated documents) but
stops it from wrongly deleting real matches on tiny corpora. After the fix, 1.7
returns a full FIRAC brief with the statutory-transition note, exactly as the
guide promises. All 36 tests still pass.

**If you ever want to unblock yourself without re-downloading the zip**, paste
this into a Colab cell *before* running 1.7 — it patches the same rule live:

```python
import sys; sys.path.insert(0, "src")
import trace_law.bm25_index as b
def _search(self, query, top_k=10):
    q = set(b.tokenize(query))
    scores = self._bm25.get_scores(b.tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(self.doc_ids[i], float(scores[i])) for i in order
            if q & set(self._tokenized[i])]
b.BM25Index.search = _search
```

But the zip I'm handing back already has the proper fix baked in, so you don't
need the patch.

---

## 1. The single most important problem: the IL-PCR data won't download the way the guide says

This is the one that would have wasted hours, so it's first.

**Action Guide step 2.1** tells you to run
`git clone https://github.com/Exploration-Lab/IL-PCR.git` and says it
*"downloads 7,070 judgments + the citation links."* **It does not.** That GitHub
repo is the *U-CREAT* model code; its data folders contain only empty
`.gitkeep` placeholder files. I cloned it and confirmed: there is not a single
judgment or citation label in it. So step 2.2 ("turn the files into three
objects") has no files to work with.

**Where the data actually is.** The real IL-PCR corpus and its gold citation
links live on **Hugging Face**, inside the IL-TUR benchmark under the `pcr`
config — which your own `config.py` already points at
(`Exploration-Lab/IL-TUR`, config `pcr`). The structure is:

- `test_candidates` (~1.7k cases) → the searchable corpus
- `test_queries` (237 cases), each with a `relevant_candidates` list → the gold answers
- (also `train_*` and `dev_*` splits)

**The fix.** I wrote you a ready-made loader, `scripts/load_ilpcr.py`, that does
all of step 2.2 for you. Replace guide steps 2.1 + 2.2 with:

```python
!pip install datasets -q
from scripts.load_ilpcr import load_ilpcr
# quick smoke run first (30 queries); drop max_queries for the full set
id_to_text, queries, qrels = load_ilpcr(split="test", max_queries=30)
```

Then steps 2.3 (build the dense index) and 2.4 (evaluate) work unchanged. Once
the 30-query smoke run looks right, re-run with `load_ilpcr(split="test")` for
the numbers that go in the paper.

> Note: IL-TUR is gated/under a non-commercial research licence. You'll need to
> be logged in with your Hugging Face token (`from huggingface_hub import login;
> login()`), and you may need to accept the dataset terms on its HF page once.

---

## 2. Inconsistencies *between* your four documents

These are places where the paper, the report, the guide, and the code disagree
with each other. Reviewers notice these instantly.

| # | Where | The inconsistency | What to do |
|---|-------|-------------------|------------|
| 2.1 | Paper vs Report/Code | The **paper** still describes the *old* system: max-cosine fusion, τ=0.35 decision, GPT-4o-mini, and "top-1 cosine similarity / +4.26%" as the headline result. The **code** is the *new* system: RRF fusion, cross-encoder rerank, authority, temporal, verification, abstention. | The paper must be rewritten to the new pipeline (this is exactly what Action-Guide Part 3 says). Until then, paper ≠ code. |
| 2.2 | Report vs Code (test count) | The **TRACE-Law report** says *"27 / 27 tests pass."* The **shipped code** has **34 tests** (and the guide's step 1.5 says "34 passed"). | Harmless, but update the report/paper to say 34 so the numbers line up. The report describes an earlier code drop than the zip you have. |
| 2.3 | Guide vs Reality (IL-PCR) | Covered in §1 — the guide's git-clone won't get the data. | Use `load_ilpcr.py`. |
| 2.4 | Report's demo table | Appendix A shows **every system scoring identically** (0.800 / 0.926 / …). The report honestly flags these as throwaway demo numbers, **but they must never appear in the paper.** | Only put *real* IL-PCR numbers (from §1) in the paper. Don't let the all-identical demo row leak in. |
| 2.5 | FIRAC "A" | The paper expands FIRAC's "A" two ways (Application vs Analysis); the offline brief prints "**A - Analysis**". | Pick one — the report recommends **Application** — and make the code, paper, and figure all say the same word. (Small code edit in `extractive.py`.) |
| 2.6 | §2.5 benchmark names | The paper mis-expands acronyms: IL-TUR and ILDC are given wrong long-forms (e.g. "Indian Legal Translation Using Machine Learning", "International Law Documentation Centre"). | Correct them: ILDC = *Indian Legal Documents Corpus*; IL-TUR = *Indian Legal Text Understanding and Reasoning*; AILA = *Artificial Intelligence for Legal Assistance* (not "Asian Information Litigation Association"). |
| 2.7 | Reference lists | The report notes the literature review cites sources (Reuter, the agentic-RAG survey, LawPal, SCaLe-QA, LexGLUE, LEXTREME) that are missing from the paper's reference list, and vice-versa. | Align the two lists before submission. |

---

## 3. Code-level findings (beyond the 1.7 bug)

The code is genuinely well-structured: lazy imports, offline fallbacks, clean
separation. The fixes already claimed in the report (authority double-count,
dead temporal penalty, inflated margin, brittle authority regex, negation-blind
entailment, unused date dimension) are all present and test-covered. Remaining
items:

**3.1 — The hybrid BM25+dense RRF is *not actually wired into the ladder*
(medium).** The report's checklist item #6 calls weighted BM25+dense RRF a
"likely cheap, strong win" and says it's "already implemented." The *function*
`weighted_rrf` exists in `fusion.py`, but `eval_ladder.run_ladder` never calls
it — the "Dense-multi-RRF" rung fuses only the 4 dense query-variants, never
BM25 with dense. So the paper can't yet report a hybrid number. **Either** add a
hybrid rung to the ladder **or** soften the report's "already implemented"
wording. (I can wire the rung in for you if you want it — say the word.)

**3.2 — `support_rate` defaults to 1.0 when nothing is cited (low/medium).** In
`verification.py`, if the generated answer cites no `[Judgment k]` at all,
support is recorded as a perfect 1.0, which then *raises* the reliability score.
A no-citations answer arguably shouldn't be rewarded. Consider treating "no
citations" as neutral (e.g. 0.5) or as its own flag. Minor, but a sharp reviewer
could poke at it.

**3.3 — A few temporal-mapping rows are legally contestable (medium, and a
*legal* reviewer will check).** The report already flags the two worst:
IPC 124A→BNS 152 (sedition was repealed; BNS 152 is a differently-scoped
offence) and IPC 420→BNS 318(4). Your code does contain exactly those rows. Add
a footnote citing the official Government-of-India correspondence tables and mark
the genuinely contested rows, rather than presenting all mappings as settled.

**3.4 — Every threshold is a "magic number" (medium).** RRF k=60, authority
α=0.3 (and its 0.45/0.30/0.25 sub-weights), reliability weights
0.35/0.10/0.40/0.15, the 0.15 temporal penalty, the 0.45 abstain threshold, the
0.5 support threshold, the 45-char window. None are justified or tuned in the
paper. You don't have to *learn* them, but you should show a small sensitivity
check (e.g. vary α and the abstain threshold, show the metric barely moves) and
state plainly that the test set was never used to pick them.

**3.5 — Reproducibility pinning (medium).** `requirements.txt` should pin exact
versions, and the paper needs to name the exact cross-encoder, NLI model, and
generator revisions. Otherwise "reproducible" — your headline claim — isn't
literally true. Also pin a seed for the corpus sample (already `RANDOM_SEED=42`,
good) and record the exact IL-PCR split.

---

## 4. The evaluation gap that decides accept vs reject

This isn't a bug — it's the thing the report calls the "reject-on-sign risk,"
and it's worth restating because it governs your whole Part 2:

**No number produced by an offline *fallback* may appear in a results table.**
The test suite and demo run on deterministic stand-ins (lexical overlap instead
of a real cross-encoder, token-overlap instead of a real NLI model, extractive
text instead of a real generator). Those exist so the code *runs* anywhere — but
the moment a reviewer opens the repo and sees the headline scores were produced
by token-overlap, the contribution collapses. So when you run Part 2, you must
plug in the **real** models:

```python
from trace_law.rerank import load_cross_encoder
from trace_law.verification import load_nli
report = pipeline.evaluate(
    queries, qrels, id_to_text, dense=dense, bm25=bm25,
    rerank_score_fn=load_cross_encoder(),     # real cross-encoder
)
# and run generation/verification with generation_backend="local" + load_nli()
```

The cross-encoder (`ms-marco-MiniLM-L-6-v2`) and NLI model
(`DeBERTa-v3-base-mnli`) are small and download automatically the first time.
For generation, your config already targets `Qwen/Qwen2.5-3B-Instruct`, which
fits your 4 GB RTX 3050. Report Qwen-vs-GPT-4o-mini faithfulness side by side to
back the "local & reproducible" claim.

---

## 5. Suggested order of work (lowest effort → highest payoff)

1. **Use the fixed zip** (1.7 now works; 36 tests pass).
2. **Load real IL-PCR** with `scripts/load_ilpcr.py` (§1) — 30-query smoke run first.
3. **Build the dense index once** (guide 2.3) and save `faiss.index` to Drive.
4. **Run the ladder with real models** (§4) — these become your results table.
5. **Run leave-one-out ablations** (drop RRF / rerank / authority / temporal one
   at a time) so you can claim each part helps.
6. **Label ~50 judgments** for the temporal detector and report its precision/
   recall — this turns your one genuinely novel idea into evidence.
7. **Rewrite the paper** around the temporal-validity layer (Part 3), fix the
   small inconsistencies in §2 above, and align the references.
8. *(Optional)* decide on the hybrid-RRF rung (§3.1) and a sensitivity table
   (§3.4) — both are cheap reviewer-pleasers.

---

## 6. What I changed in the code I'm handing back

- **Fixed** `src/trace_law/bm25_index.py` — the small-corpus zero-score bug that
  caused "Not enough data available" in step 1.7.
- **Added** `scripts/load_ilpcr.py` — a working IL-PCR/IL-TUR loader that
  replaces the broken git-clone in guide steps 2.1–2.2.
- Nothing else was touched, so none of your research logic changed. All 36 tests
  still pass.

Everything else above is a *recommendation*, not a change I made — the items that
affect your results (hybrid rung, FIRAC wording, temporal footnotes, thresholds)
are yours to decide, and I can implement any of them on request.
