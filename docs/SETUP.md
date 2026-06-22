# Setup & Running — step by step (for beginners)

Two ways to run this. **Use Google Colab** unless you specifically want it on your laptop.
Colab is a free website that runs Python in your browser and gives you a free GPU — nothing
to install.

---
## What you need to download / sign up for (once)
1. A **Google account** (for Colab) — you already have one if you use Gmail.
2. The **code**: this project folder (the zip you were given, e.g. `LegalRAG_Combined.zip`).
3. (For real data) a **Hugging Face account** + a free *Read* access token
   (huggingface.co → Settings → Access Tokens).
4. (Optional) an **OpenAI API key** only if you want GPT-4o-mini to write answers (costs a
   few dollars). You can skip this and use the free offline writer or a local model.

---
## TRACK A — Google Colab (recommended)

### A1. Open Colab and a new notebook
Go to `colab.research.google.com` → *File ▸ New notebook*.
Turn on the GPU: *Runtime ▸ Change runtime type ▸ T4 GPU ▸ Save*.

### A2. Upload the code
Run this cell (▶), then click "Choose Files" and pick your `LegalRAG_Combined.zip`:
```python
from google.colab import files; files.upload()
!unzip -o LegalRAG_Combined.zip -d .
%cd LegalRAG_Combined        # or the folder name that was unzipped
```

### A3. Install libraries
```python
!pip install -r requirements.txt rank-bm25 sentence-transformers faiss-cpu -q
```

### A4. Prove it works (no data/keys needed)
```python
!python -m pytest tests/ -q          # expect: 36 passed
!python scripts/demo_trace.py        # prints the ladder + one full answer
```
If you see **36 passed**, the combined code is healthy.

### A5. Run a real answer on a tiny corpus
```python
import sys; sys.path.insert(0, "src")
import pipeline
texts = ["Supreme Court. Anticipatory bail under Section 438 CrPC for cheating under Section 420 IPC.",
         "Supreme Court. Dowry death under Section 304B IPC; conviction affirmed."]
id_to_text, dense, bm25 = pipeline.build_engine(texts)      # dense=None here (BM25 only)
res = pipeline.answer("anticipatory bail cheating", [bm25], id_to_text)
print(res.answer)
print(res.as_dict()["temporal"])     # see the IPC->BNS flags
```

### A6. Build the real search over a big corpus (with GPU)
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
texts = [...]                                  # your judgments (list of strings)
m = SentenceTransformer("all-MiniLM-L6-v2")
emb = m.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
faiss.write_index(index, "faiss.index")        # SAVE so you never rebuild
# wire it in:
import config; config.RETRIEVER_BACKEND = "dense"
id_to_text, dense, bm25 = pipeline.build_engine(texts, dense_index=index)
res = pipeline.answer("dowry death presumption", [dense, bm25], id_to_text)
```
**Tip:** mount Google Drive (*Files ▸ mount*) and save `faiss.index` there so it survives.

### A7. Evaluate on IL-PCR (the paper numbers)
Get the data and build `queries / qrels / id_to_text` (see `docs/EVALUATION.md`), then:
```python
from trace_law.rerank import load_cross_encoder
report = pipeline.evaluate(queries, qrels, id_to_text, dense=dense, bm25=bm25,
                           rerank_score_fn=load_cross_encoder())
print(pipeline.format_ladder(report))
```

---
## TRACK B — Local on your laptop (VS Code)

### B1. Install the basics (once)
- **Python 3.10+**: python.org/downloads (tick "Add Python to PATH" on Windows).
- **VS Code**: code.visualstudio.com, then install the **Python** extension (Extensions panel).
- (Optional, for a local LLM on RTX 3050) **Ollama**: ollama.com → then `ollama pull qwen2.5:3b`.

### B2. Open the project
VS Code → *File ▸ Open Folder* → choose the unzipped `LegalRAG_Combined` folder.

### B3. Make a virtual environment and install
Open the terminal in VS Code (*Terminal ▸ New Terminal*) and run:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt rank-bm25 sentence-transformers faiss-cpu
```

### B4. Test it
```bash
python -m pytest tests/ -q          # expect 36 passed
python scripts/demo_trace.py
```

### B5. Run the app (optional UI)
```bash
streamlit run src/app.py
```
First create a `.env` file in the project root if you want GPT-4o-mini:
```
OPENAI_API_KEY=sk-...
```
Without a key it uses the free offline writer automatically.

### B6. Local LLM on the RTX 3050 (4 GB)
```bash
ollama pull qwen2.5:3b
```
Then set `GENERATION_BACKEND=local` and `LOCAL_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct` (or wire
`generation.py`'s local backend to your Ollama endpoint). The embedder, reranker and NLI
models are small and run fine on the 3050; only the LLM is heavy.

---
## What "passing" looks like
- `pytest` → **36 passed**.
- `demo_trace.py` → a metrics table + a FIRAC answer with a "Statutory-transition note".
- `pipeline.answer(...).as_dict()` → keys `evidence_ids, temporal, verification, reliability, answer`.

## Common issues
- *`ModuleNotFoundError: trace_law`* → run from the project root, or `sys.path.insert(0,"src")`.
- *faiss / torch errors offline* → those only load when you use dense/embeddings; the tests
  and BM25 path don't need them.
- *Colab session reset* → re-run A2–A3; keep your FAISS index on Google Drive.
