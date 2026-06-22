# Contributing

- Keep modules **import-safe**: heavy deps (faiss/torch/transformers/openai) load lazily.
- Every new logic module needs an offline fallback so tests run without models.
- Run `python -m pytest tests/ -q` before committing (must stay green, offline).
- Pin model revisions when reporting results.
