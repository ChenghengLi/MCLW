"""
Unified attack module for watermark robustness evaluation.

Implements (in priority order for TAIGR submission):
  - random_substitution (token-level, uniform over vocab) -- FREE
  - random_masked_word (original v1 word-level masked) -- FREE
  - synonym_substitution (WordNet) -- FREE
  - dipper_paraphrase (kalpeshk2011/dipper-paraphraser-xxl, ~22GB bf16)
  - back_translation (NLLB-200 or MarianMT)
  - sira (Self-Information Rewrite Attack, local Mistral as rewriter)

DIPPER and SIRA are lazy-loaded (heavy models). Call `unload()` between attacks
to free GPU memory.
"""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


# ======================================================================
# Cheap attacks (no external model)
# ======================================================================
def random_substitution_tokens(
    token_ids: Sequence[int],
    vocab_size: int,
    rate: float,
    seed: int = 0,
) -> List[int]:
    """Replace `rate` fraction of tokens with UNIFORMLY random token IDs."""
    rng = random.Random(seed)
    ids = list(token_ids)
    n = len(ids)
    if n == 0 or rate <= 0:
        return ids
    k = max(0, int(round(n * rate)))
    indices = rng.sample(range(n), min(k, n))
    for i in indices:
        ids[i] = rng.randrange(vocab_size)
    return ids


def word_level_masked(text: str, rate: float, seed: int = 0) -> str:
    """Replace `rate` fraction of whitespace-split words with 'masked'."""
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return text
    k = max(0, int(round(len(words) * rate)))
    indices = rng.sample(range(len(words)), min(k, len(words)))
    for i in indices:
        words[i] = "masked"
    return " ".join(words)


_WORDNET = None


def _load_wordnet():
    global _WORDNET
    if _WORDNET is None:
        import nltk
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            from nltk.corpus import wordnet as wn  # noqa: F401
        _WORDNET = wn
    return _WORDNET


def synonym_substitution(text: str, rate: float, seed: int = 0) -> str:
    """
    WordNet-based synonym replacement. Replaces `rate` fraction of whitespace
    tokens with a same-POS WordNet lemma when one exists.
    """
    wn = _load_wordnet()
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return text
    k = max(0, int(round(len(words) * rate)))
    candidates = rng.sample(range(len(words)), min(k, len(words)))
    for i in candidates:
        w = words[i]
        syns = wn.synsets(w)
        lemmas = {l.name().replace("_", " ") for s in syns for l in s.lemmas() if l.name().lower() != w.lower()}
        if lemmas:
            words[i] = rng.choice(sorted(lemmas))
    return " ".join(words)


# ======================================================================
# DIPPER paraphraser (HF: kalpeshk2011/dipper-paraphraser-xxl)
# ======================================================================
class DipperAttacker:
    """
    Wraps DIPPER. Uses bf16 + device_map='auto' to fit in ~22GB VRAM.
    Control codes are multiples of 20 in [0..100].
    """

    def __init__(
        self,
        model_name: str = "kalpeshk2011/dipper-paraphraser-xxl",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DIPPER] Loading {model_name} ({dtype})...", flush=True)
        # Use AutoTokenizer to pick the fast (tokenizers-library) variant when
        # available, avoiding the legacy T5Tokenizer's sentencepiece dependency.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        self.model.eval()
        print(f"[DIPPER] loaded ({sum(p.numel() for p in self.model.parameters())/1e9:.1f}B params)", flush=True)

    @torch.no_grad()
    def paraphrase(
        self,
        text: str,
        lex_diversity: int = 60,
        order_diversity: int = 60,
        prefix: str = "",
        sent_interval: int = 3,
        max_new_tokens: int = 256,
    ) -> str:
        assert lex_diversity in range(0, 101, 20), "lex_diversity in {0,20,40,60,80,100}"
        assert order_diversity in range(0, 101, 20), "order_diversity in {0,20,40,60,80,100}"
        lex_code = 100 - lex_diversity
        order_code = 100 - order_diversity

        # Sentence-wise chunking. We use a regex-based splitter rather than
        # nltk.sent_tokenize because nltk-3.9+ requires the `punkt_tab`
        # resource which is not always available offline; the regex split
        # is good enough for paraphrase chunking and removes the dependency.
        import re
        text = " ".join(text.split())
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sents:
            # Fall back to whole text as one chunk if no sentence terminators.
            sents = [text] if text else []
        output = prefix
        for i in range(0, len(sents), sent_interval):
            chunk = " ".join(sents[i : i + sent_interval])
            prompt = (
                f"lexical = {lex_code}, order = {order_code}"
                + (f" <sent> {output} </sent>" if output else "")
                + f" <sent> {chunk} </sent>"
            )
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            out = self.model.generate(
                **enc,
                do_sample=True,
                top_p=0.75,
                top_k=None,
                max_new_tokens=max_new_tokens,
            )
            dec = self.tokenizer.decode(out[0], skip_special_tokens=True)
            output = (output + " " + dec).strip()
        return output

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ======================================================================
# NLLB back-translation (cheap; ~600M model)
# ======================================================================
class BackTranslationAttacker:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        pivot_lang: str = "fra_Latn",
        device: Optional[str] = None,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BackTrans] Loading {model_name} (pivot={pivot_lang})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.pivot = pivot_lang
        self.src = "eng_Latn"

    @torch.no_grad()
    def _translate(self, text: str, src: str, tgt: str, max_new_tokens: int = 512) -> str:
        self.tokenizer.src_lang = src
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        tgt_id = self.tokenizer.convert_tokens_to_ids(tgt)
        out = self.model.generate(**enc, forced_bos_token_id=tgt_id, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def paraphrase(self, text: str, max_new_tokens: int = 512) -> str:
        mid = self._translate(text, self.src, self.pivot, max_new_tokens)
        return self._translate(mid, self.pivot, self.src, max_new_tokens)

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ======================================================================
# SIRA (Self-Information Rewrite Attack)
# ======================================================================
class SIRAAttacker:
    """
    Minimal re-implementation of SIRA (Cheng et al. ICML 2025):
      1. Score each token by self-information -log p(t_i | t_<i) under a
         small reference LM (default: Llama-3.2-1B).
      2. Mask the top-k fraction of high-self-information tokens.
      3. Re-fill via a rewriter LM (default: Mistral-7B-Instruct in 4-bit).

    This is a lightweight, reproducible stand-in for the original SIRA.
    If you want exact SIRA numbers, use the author code at
    https://github.com/Allencheng97/Self-information-Rewrite-Attack.
    """

    def __init__(
        self,
        scorer_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        rewriter_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SIRA] Loading scorer {scorer_model_name}...")
        self.s_tok = AutoTokenizer.from_pretrained(scorer_model_name)
        self.s_model = AutoModelForCausalLM.from_pretrained(
            scorer_model_name, torch_dtype=torch.float16
        ).to(self.device).eval()

        print(f"[SIRA] Loading rewriter {rewriter_model_name} (4-bit={load_in_4bit})...")
        quant = None
        if load_in_4bit:
            try:
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                quant = None
        self.r_tok = AutoTokenizer.from_pretrained(rewriter_model_name)
        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        if quant is not None:
            kwargs["quantization_config"] = quant
        self.r_model = AutoModelForCausalLM.from_pretrained(rewriter_model_name, **kwargs).eval()

    @torch.no_grad()
    def paraphrase(self, text: str, top_k_frac: float = 0.25) -> str:
        """
        Rewrite the top-k_frac fraction of highest-self-information tokens.
        """
        # Score self-information
        enc = self.s_tok(text, return_tensors="pt").to(self.device)
        ids = enc["input_ids"][0]
        if ids.numel() < 3:
            return text
        out = self.s_model(**enc)
        logits = out.logits[0, :-1, :]  # predict next from prefix
        probs = torch.softmax(logits, dim=-1)
        targets = ids[1:]
        tok_probs = probs.gather(1, targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        self_info = -tok_probs.log().cpu().numpy()  # (n-1,)

        # Which token positions are "high self-info"?
        n = len(self_info)
        k = max(1, int(round(n * top_k_frac)))
        mask_idx = set(np.argsort(-self_info)[:k].tolist())

        # Build mask-marked text and ask rewriter to fill in.
        masked_tokens = []
        for i, tid in enumerate(targets.tolist()):
            if i in mask_idx:
                masked_tokens.append("[MASK]")
            else:
                masked_tokens.append(self.s_tok.decode([tid]))
        masked_text = "".join(masked_tokens).strip()

        prompt = (
            "Rewrite the following text to replace every [MASK] with a "
            "natural, meaning-preserving word. Output ONLY the rewritten text.\n\n"
            f"Text: {masked_text}\n\nRewrite:"
        )
        r_enc = self.r_tok(prompt, return_tensors="pt").to(self.r_model.device)
        out = self.r_model.generate(
            **r_enc,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            max_new_tokens=min(1024, int(1.5 * n)),
            pad_token_id=self.r_tok.eos_token_id,
        )
        full = self.r_tok.decode(out[0], skip_special_tokens=True)
        rewritten = full.split("Rewrite:", 1)[-1].strip()
        return rewritten or text

    def unload(self):
        del self.s_model, self.r_model, self.s_tok, self.r_tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
