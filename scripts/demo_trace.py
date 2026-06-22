"""Offline demo of the TRACE-Law additions.

Runs the baseline ladder (BM25 / dense-single / dense-multi-max / dense-multi-RRF
/ +rerank) against gold labels, then runs one full pipeline query showing
temporal-statutory flagging, citation verification and reliability/abstention.

Uses BM25 + a TF-IDF cosine retriever so it runs with no GPU, no API key and no
model downloads. Swap in FAISS + a real cross-encoder/NLI/Qwen for production.

    python scripts/demo_trace.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from trace_law.bm25_index import BM25Index
from trace_law.eval_ladder import run_ladder, format_ladder
from trace_law.trace_pipeline import run_trace_pipeline, TraceConfig

CORPUS = {
    "d0": "Supreme Court of India, Constitution Bench. Dishonour of cheque under Section 138 of the Negotiable Instruments Act, 1881; the complaint was held maintainable and the conviction affirmed.",
    "d1": "High Court, single judge. Anticipatory bail under Section 438 CrPC for an offence punishable under Section 420 IPC for cheating and dishonesty. Bail granted with conditions.",
    "d2": "Supreme Court division bench. Grounds of divorce under the Hindu Marriage Act, 1955; irretrievable breakdown of marriage discussed.",
    "d3": "Supreme Court. Dowry death under Section 304B IPC with the presumption under Section 113B of the Indian Evidence Act, 1872; conviction affirmed.",
    "d4": "Motor Accident Claims Tribunal. Compensation under the Motor Vehicles Act for rash and negligent driving; just compensation awarded.",
    "d5": "High Court. Quashing of FIR under Section 482 CrPC; inherent powers to prevent abuse of process discussed.",
    "d6": "Supreme Court Constitution Bench. Maintenance under Section 125 CrPC for a divorced wife; scope of the provision explained.",
    "d7": "A blog post about cooking recipes and kitchen utensils, entirely unrelated to law.",
    "d8": "High Court. Section 138 of the Customs Act, 1962 on confiscation and penalty; not about cheque dishonour.",
    "d9": "Supreme Court. Cheating and criminal breach of trust under Section 406 IPC; bail and custody discussed.",
    "d10": "Tribunal. General principles of negligence and compensation; maintenance of dependants noted in passing.",
}
QUERIES = {
    "q_cheque": "section 138 cheque dishonour notice",
    "q_bail": "anticipatory bail cheating IPC 420",
    "q_dowry": "dowry death IPC 304B presumption",
    "q_fir": "quashing FIR section 482 CrPC",
    "q_maint": "maintenance under section 125 CrPC",
}
QRELS = {
    "q_cheque": {"d0"}, "q_bail": {"d1"}, "q_dowry": {"d3"},
    "q_fir": {"d5"}, "q_maint": {"d6"},
}


def make_retrievers():
    ids = list(CORPUS)
    texts = [CORPUS[i] for i in ids]
    bm25 = BM25Index(texts, doc_ids=ids)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vec.fit_transform(texts)

    def dense(q, k):
        sims = cosine_similarity(vec.transform([q]), matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [(ids[i], float(sims[i])) for i in order if sims[i] > 0]

    return dense, (lambda q, k: bm25.search(q, k))


def main():
    dense, bm25 = make_retrievers()

    print("=" * 78)
    print("BASELINE LADDER  (gold-label IR metrics; TF-IDF stands in for FAISS dense)")
    print("=" * 78)
    report = run_ladder(QUERIES, QRELS, CORPUS, dense_retriever=dense,
                        bm25_retriever=bm25, top_k=5, ks=(1, 5))
    print(format_ladder(report, ks=(1, 5)))

    print("\n" + "=" * 78)
    print("FULL TRACE-LAW PIPELINE  (one query)")
    print("=" * 78)
    cfg = TraceConfig(generation_backend="extractive", use_rrf=True, final_evidence=3)
    res = run_trace_pipeline("dowry death IPC 304B presumption", [dense, bm25], CORPUS, cfg)
    d = res.as_dict()
    print(f"Query        : {d['query']}")
    print(f"Evidence ids : {d['evidence_ids']}")
    print(f"Temporal     : stale={d['temporal']['stale']}")
    for rep in d["temporal"]["reports"]:
        for f in rep["flags"]:
            print(f"   - {f['message']}")
    print(f"Verification : support_rate={d['verification']['support_rate']:.2f} "
          f"({d['verification']['num_supported']}/{d['verification']['num_cited_claims']} cited claims)")
    print(f"Reliability  : {d['reliability']['reliability']:.2f} -> {d['reliability']['decision']}")
    print("\n--- ANSWER (first 600 chars) ---")
    print(d["answer"][:600])


if __name__ == "__main__":
    main()
