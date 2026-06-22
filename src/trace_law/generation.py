"""Generation backends (TRACE-Law).

Abstracts answer generation behind one interface with three backends:
  * "openai"  : GPT-4o-mini (or any chat model) via the OpenAI API
  * "local"   : a local HF causal LM (e.g. Qwen/Qwen2.5-3B-Instruct) -- the
                proposed reproducible, API-free option
  * "extractive" : deterministic offline FIRAC brief (no model needed)

The factory picks a backend by name and gracefully degrades to "extractive"
when the requested backend's dependencies/keys are unavailable, so the pipeline
always returns something and is testable offline.
"""
from __future__ import annotations

import os
import re
from typing import List, Protocol, Sequence, Tuple

FIRAC_SYSTEM = (
    "You are an Indian legal research assistant. Use ONLY the retrieved judgment "
    "excerpts as authority. Produce a FIRAC analysis (Facts, Issues, Rule, "
    "Application, Conclusion). Cite sources as [Judgment k]. Never invent facts, "
    "citations, or holdings not present in the excerpts. This is research "
    "assistance, not legal advice."
)


class Generator(Protocol):
    name: str
    def generate(self, query: str, evidence: Sequence[Tuple[str, float]]) -> str: ...


def _context_block(evidence: Sequence[Tuple[str, float]]) -> str:
    return "\n\n".join(
        f"[Judgment {i+1}] (score={score:.3f})\n{doc}"
        for i, (doc, score) in enumerate(evidence)
    )


class ExtractiveGenerator:
    """Offline, deterministic FIRAC brief. No external model required."""

    name = "extractive"

    def generate(self, query: str, evidence: Sequence[Tuple[str, float]]) -> str:
        from trace_law.extractive import firac_brief
        return firac_brief(query, evidence)


class OpenAIGenerator:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2, api_key: str | None = None):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._fallback = ExtractiveGenerator()

    def generate(self, query: str, evidence: Sequence[Tuple[str, float]]) -> str:
        if not evidence:
            return "Not enough data available in the retrieved documents."
        if not self.api_key:
            return self._fallback.generate(query, evidence)
        try:  # pragma: no cover - needs network/key
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            user = (
                f"Top Retrieved Cases:\n{_context_block(list(evidence)[:3])}\n\n"
                f"Query:\n{query}\n\nProduce a FIRAC analysis using only the text above."
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": FIRAC_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=self.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return self._fallback.generate(query, evidence)


class LocalQwenGenerator:
    """Reproducible, API-free generation with a local HF causal LM (e.g. Qwen)."""

    name = "local"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature
        self._fallback = ExtractiveGenerator()
        self._pipe = None

    def _ensure(self):  # pragma: no cover - needs model weights
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-generation", model=self.model_name)

    def generate(self, query: str, evidence: Sequence[Tuple[str, float]]) -> str:
        if not evidence:
            return "Not enough data available in the retrieved documents."
        try:  # pragma: no cover - needs model weights
            self._ensure()
            prompt = (
                f"<|system|>\n{FIRAC_SYSTEM}\n<|user|>\n"
                f"Top Retrieved Cases:\n{_context_block(list(evidence)[:3])}\n\nQuery: {query}\n<|assistant|>\n"
            )
            out = self._pipe(prompt, max_new_tokens=700, temperature=self.temperature, do_sample=self.temperature > 0)
            return out[0]["generated_text"][len(prompt):].strip()
        except Exception:
            return self._fallback.generate(query, evidence)


def get_generator(backend: str = "extractive", **kwargs) -> Generator:
    backend = (backend or "extractive").lower()
    if backend == "openai":
        return OpenAIGenerator(**kwargs)
    if backend in ("local", "qwen"):
        return LocalQwenGenerator(**kwargs)
    return ExtractiveGenerator()
