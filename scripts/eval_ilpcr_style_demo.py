"""Run the REAL TRACE-Law ladder + leave-one-out ablation on a non-trivial
IL-PCR-style benchmark with known gold labels. Every number is produced by the
actual trace_law code (fusion, metrics, rerank, bm25), not hand-written."""
import sys, random, re
sys.path.insert(0, "src")
random.seed(42)

from trace_law import fusion, metrics, rerank, eval_ladder
from trace_law.bm25_index import BM25Index
from trace_law.trace_pipeline import default_expansions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---- 1. Build a synthetic-but-realistic prior-case corpus -------------------
# Each "topic" has a query case + several relevant prior cases that share legal
# concepts but deliberately use DIFFERENT surface wording (so lexical != gold),
# plus many hard-negative distractors. This mirrors IL-PCR's challenge.
TOPICS = {
 "dowry_death": dict(
   query="The deceased wife died of burns within seven years of marriage amid persistent demands for money and household articles from her in-laws; the prosecution invokes the statutory presumption against the husband.",
   gold=[
     "Where a woman dies of unnatural causes within seven years of wedlock and cruelty linked to demands for property is shown, the court must presume the husband caused the death.",
     "The presumption as to dowry-related fatalities shifts the onus onto the accused spouse once soon-after-marriage death and harassment for valuables are established.",
     "A bride's demise by injury shortly after the nuptials, coupled with coercion for gifts, attracts a rebuttable statutory inference of culpability against the matrimonial family.",
   ]),
 "anticipatory_bail": dict(
   query="The applicant apprehends arrest in a cheating and criminal breach of trust matter and seeks a direction that he be released on bail in the event of being taken into custody.",
   gold=[
     "A person fearing apprehension in a non-bailable accusation may move the higher court for a pre-arrest direction safeguarding personal liberty pending investigation.",
     "Pre-arrest protection is discretionary and turns on the gravity of the accusation, the antecedents of the seeker, and the risk of the investigation being hampered.",
     "Where custodial interrogation appears unnecessary, courts may grant protective relief enabling release upon being detained in a fraud-related complaint.",
   ]),
 "cheque_dishonour": dict(
   query="A post-dated instrument issued to discharge a debt was returned unpaid for want of sufficient funds, and the drawer ignored the statutory demand notice within the prescribed period.",
   gold=[
     "Failure to honour a negotiable instrument presented within validity, after a written demand, renders the issuer liable on the presumption of a legally enforceable liability.",
     "Once a draft towards repayment bounces for inadequate balance and the maker stays silent past the notice window, the offence of dishonour is complete.",
     "The holder of a returned payment order need only prove issuance and unmet demand; the burden then lies on the signatory to rebut the debt presumption.",
   ]),
 "land_acquisition": dict(
   query="Landowners challenge the quantum of compensation awarded for compulsory acquisition of their agricultural plots, contending the market value was undervalued by the collector.",
   gold=[
     "Indemnity for state taking of private holdings must reflect the genuine market worth on the notification date, with solatium and statutory interest added.",
     "Where the assessing officer's valuation of expropriated farmland is inadequate, the reference court may enhance the award on comparable sale exemplars.",
     "Fair recompense for requisitioned property is a constitutional expectation; under-assessment of agrarian land warrants upward revision with accretions.",
   ]),
 "fundamental_rights": dict(
   query="A petitioner invokes the writ jurisdiction of the constitutional court alleging that an executive order infringes the guarantee of equality and the freedom to carry on trade.",
   gold=[
     "An arbitrary state action that offends the equal-protection clause and the liberty of occupation is amenable to correction by prerogative writ.",
     "The apex constitutional forum may strike down administrative fiat that violates the charter of basic liberties, including non-discrimination and free commerce.",
     "Executive directions that unreasonably curtail a citizen's vocational freedom and parity before law are void and quashable in writ proceedings.",
   ]),
}
# distractors: legal text from OTHER domains (hard negatives)
DISTRACTORS = [
 "Specific performance of a contract for sale of immovable property is discretionary and depends on readiness and willingness of the plaintiff.",
 "A registered trademark confers exclusive rights and deceptive similarity is judged from the standpoint of an average consumer with imperfect recollection.",
 "Income escaping assessment may be reopened within the limitation only on tangible material indicating under-disclosure by the assessee.",
 "Industrial disputes concerning wrongful termination are referred to the labour tribunal which may order reinstatement with back wages.",
 "A testamentary instrument must be attested by two witnesses and suspicious circumstances surrounding its execution must be dispelled by the propounder.",
 "Environmental clearance for a project requires assessment of cumulative ecological impact and public consultation under the applicable regulations.",
 "Arbitration awards may be set aside only on narrow grounds such as patent illegality going to the root of the matter.",
 "Custody of a minor is decided on the paramount consideration of the welfare of the child rather than the rights of the parents.",
 "A consumer complaint for deficiency in service must be filed within two years of the cause of action before the appropriate forum.",
 "Defamation requires publication of a false imputation lowering the reputation of the complainant in the estimation of right-thinking members of society.",
 "The dissolution of a partnership entitles each partner to an account of the assets and a share of the surplus after settlement of liabilities.",
 "A government servant facing departmental inquiry is entitled to a reasonable opportunity of being heard before imposition of a major penalty.",
]

id_to_text, queries, qrels = {}, {}, {}
cid = 0
for topic, d in TOPICS.items():
    qid = f"q_{topic}"
    queries[qid] = d["query"]
    rel = set()
    for g in d["gold"]:
        gid = f"d{cid}"; id_to_text[gid] = g; rel.add(gid); cid += 1
    qrels[qid] = rel
for dist in DISTRACTORS:
    id_to_text[f"d{cid}"] = dist; cid += 1
# add near-duplicate distractors of each gold (same domain, wrong specifics) to make it harder
for topic, d in TOPICS.items():
    id_to_text[f"d{cid}"] = d["query"][:120] + " However the facts here concern a wholly unrelated procedural objection."; cid += 1

print(f"Benchmark: {len(id_to_text)} candidates, {len(queries)} queries, "
      f"{sum(len(v) for v in qrels.values())} gold links.\n")

# ---- 2. Real retrievers ------------------------------------------------------
ids = list(id_to_text); texts = [id_to_text[i] for i in ids]
bm25 = BM25Index(texts, doc_ids=ids)
def bm25_retriever(q, k): return bm25.search(q, k)

# TF-IDF "dense" stand-in (offline analogue of MiniLM dense vectors)
vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=1)
M = vec.fit_transform(texts)
def dense_retriever(q, k):
    qv = vec.transform([q])
    sims = cosine_similarity(qv, M)[0]
    order = np.argsort(-sims)[:k]
    return [(ids[j], float(sims[j])) for j in order]

# a deterministic cross-encoder stand-in (lexical re-scorer) for the rerank rung
from trace_law.rerank import lexical_overlap_score

# ---- 3. Full ladder ----------------------------------------------------------
report = eval_ladder.run_ladder(
    queries, qrels, id_to_text,
    dense_retriever=dense_retriever, bm25_retriever=bm25_retriever,
    top_k=10, rerank_score_fn=lexical_overlap_score, ks=(1,5,10),
)
print("=== FULL LADDER (real harness, IL-PCR-style benchmark) ===")
print(eval_ladder.format_ladder(report))

# ---- 4. Leave-one-out ablation on the strongest rung -------------------------
print("\n=== LEAVE-ONE-OUT ABLATION (effect on nDCG@10 / MRR@10 / Recall@10) ===")
def evaluate_run(runs): return metrics.evaluate(runs, qrels, ks=(1,5,10))

def build(use_bm25=True, use_dense=True, use_multi=True, use_rrf=True, use_rerank=True):
    out = {}
    for qid, qtext in queries.items():
        variants = default_expansions(qtext) if use_multi else [qtext]
        lists = []
        if use_dense: lists += [dense_retriever(v,10) for v in variants]
        if use_bm25:  lists += [bm25_retriever(v,10) for v in variants]
        if use_rrf and len(lists)>1:
            fused = fusion.reciprocal_rank_fusion(lists)
        elif len(lists)>1:
            fused = fusion.max_merge(lists)
        else:
            fused = lists[0]
        if use_rerank:
            cand = fused[:20]
            ctext = [(id_to_text.get(d,""), s) for d,s in cand]
            rr = rerank.rerank(qtext, ctext, score_fn=lexical_overlap_score)
            t2id = {id_to_text.get(d,""):d for d,_ in cand}
            ranked = [t2id.get(t,t) for t,_ in rr]
        else:
            ranked = [d for d,_ in fused]
        out[qid] = ranked
    return evaluate_run(out)

configs = {
 "FULL (BM25+dense+multi+RRF+rerank)": dict(),
 "  - rerank":            dict(use_rerank=False),
 "  - RRF (use max-merge)":dict(use_rrf=False),
 "  - multi-query":       dict(use_multi=False),
 "  - BM25 (dense only)":  dict(use_bm25=False),
 "  - dense (BM25 only)":  dict(use_dense=False),
}
hdr = "Configuration".ljust(34)+"nDCG@10".rjust(10)+"MRR@10".rjust(10)+"Recall@10".rjust(11)+"MAP@10".rjust(10)
print(hdr); print("-"*len(hdr))
full=None
for name,kw in configs.items():
    m=build(**kw)
    if full is None: full=m
    d_nd = m["nDCG@10"]-full["nDCG@10"]
    tag = "" if name.startswith("FULL") else f"  (Δ nDCG {d_nd:+.3f})"
    print(name.ljust(34)+f'{m["nDCG@10"]:10.3f}{m["MRR@10"]:10.3f}{m["Recall@10"]:11.3f}{m["MAP@10"]:10.3f}'+tag)
