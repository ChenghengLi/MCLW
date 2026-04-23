"""
Quality metrics: perplexity (reference-model), semantic similarity, diversity.

PPL reference: Llama-2-7B (community default). Provide a GPT-2-XL fallback
for low-VRAM runs.

Semantic similarity via sentence-transformers/all-mpnet-base-v2 (cheap).
Self-BLEU / distinct-n for diversity (pure Python / nltk).
"""

from __future__ import annotations

import gc
from typing import List, Optional, Sequence

import numpy as np
import torch


class PerplexityScorer:
    def __init__(
        self,
        model_name: str = "gpt2-xl",  # cheaper default; pass 'meta-llama/Llama-2-7b-hf' for standard
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PPL] Loading reference model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def ppl(self, text: str, max_length: int = 1024) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2:
            return float("nan")
        out = self.model(ids, labels=ids)
        loss = float(out.loss.item())
        return float(np.exp(loss))

    def unload(self):
        del self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SemanticSimilarity:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def cosine(self, a: str, b: str) -> float:
        import torch.nn.functional as F
        emb = self.model.encode([a, b], convert_to_tensor=True)
        return float(F.cosine_similarity(emb[0:1], emb[1:2]).item())

    def batch_cosine(self, pairs: Sequence[tuple]) -> List[float]:
        import torch.nn.functional as F
        texts = [t for pair in pairs for t in pair]
        emb = self.model.encode(texts, convert_to_tensor=True, batch_size=64)
        out = []
        for i in range(0, len(texts), 2):
            out.append(float(F.cosine_similarity(emb[i : i + 1], emb[i + 1 : i + 2]).item()))
        return out


def distinct_n(text: str, n: int = 2) -> float:
    toks = text.split()
    if len(toks) < n:
        return 0.0
    ngrams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return len(set(ngrams)) / max(1, len(ngrams))


def self_bleu(texts: List[str], n: int = 4) -> float:
    """
    Self-BLEU score per Zhu et al. 2018. Higher = less diverse.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        raise RuntimeError("nltk required for self_bleu")
    if len(texts) < 2:
        return 0.0
    smoother = SmoothingFunction().method1
    scores = []
    tokenized = [t.split() for t in texts]
    weights = tuple([1.0 / n] * n)
    for i in range(len(tokenized)):
        refs = tokenized[:i] + tokenized[i + 1 :]
        scores.append(sentence_bleu(refs, tokenized[i], weights=weights, smoothing_function=smoother))
    return float(np.mean(scores))
