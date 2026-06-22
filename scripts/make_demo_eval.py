"""Write tiny demo CSVs and run the gold-label ladder CLI end-to-end (offline)."""
import subprocess, sys, csv, os
os.makedirs("data/demo", exist_ok=True)
corpus = [("d0","Supreme Court. Anticipatory bail under Section 438 CrPC for cheating under Section 420 IPC; bail granted."),
          ("d1","Supreme Court. Dowry death under Section 304B IPC; conviction affirmed."),
          ("d2","High Court. Quashing of FIR under Section 482 CrPC."),
          ("d3","A cooking blog, unrelated to law.")]
with open("data/demo/corpus.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["doc_id","text"]); w.writerows(corpus)
with open("data/demo/queries.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["query_id","text"]); w.writerow(["q0","anticipatory bail cheating 420"]); w.writerow(["q1","dowry death 304B"])
with open("data/demo/qrels.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["query_id","relevant_id"]); w.writerow(["q0","d0"]); w.writerow(["q1","d1"])
sys.exit(subprocess.call([sys.executable,"src/evaluation.py",
    "--corpus-csv","data/demo/corpus.csv","--queries-csv","data/demo/queries.csv","--qrels-csv","data/demo/qrels.csv"]))
