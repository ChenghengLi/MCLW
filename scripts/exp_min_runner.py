#!/usr/bin/env python3
"""Minimal experiment runner for the MCL paper workshop subset.

Dispatches one of three experiments on a single model and writes a per-cell
JSONL + summary.json under data/v7_min/<exp>_<model_short>/.

  exp 2 — Head-to-head MCL vs KGW vs SWEET
          on 4 domains × 3 attacks × 100 prompts × n=200, ρ=0.5.
  exp 3 — MCL-only (1−δ_eff)² translation curve
          on 4 domains × 4 translation pivots × 25 subsample prompts × n=200.
  exp 4 — MCL k=2 (soft_cycle) single-point check
          wiki × 100 prompts × n=200 × {clean, ZH-roundtrip}.

Each invocation runs ONE (exp, model) cell so jobs are independently
launchable in parallel via runai.

Usage:
    python3 scripts/exp_min_runner.py --exp 2 --model meta-llama/Llama-3.1-8B-Instruct
    python3 scripts/exp_min_runner.py --exp 4 --model Qwen/Qwen2.5-7B-Instruct
    python3 scripts/exp_min_runner.py --exp 2 --model meta-llama/Llama-3.1-70B-Instruct --int8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse existing KGW / SWEET implementation -- proven on Gemma already.
from run_kgw_sweet_baselines import (
    kgw_generate,
    kgw_detect,
    init_green,
    random_substitute,
)

from ltw_watermark.attacks import DipperAttacker, BackTranslationAttacker
from ltw_watermark.prompts_4domain import get_prompts, get_subsample, DOMAINS

try:
    import Levenshtein  # python-levenshtein
    _HAVE_LEV = True
except ImportError:
    _HAVE_LEV = False


SECRET_KEY = "mclw_v7_min_2026"
MCL_S = 5
MCL_RHO = 0.5          # ρ=0.5 with G_H_high (high-entropy gate) — matched to
                       # SWEET's gating strategy at the same budget for an
                       # apples-to-apples head-to-head comparison.
KGW_GAMMA = 0.5
KGW_DELTA = 2.0
TEMPERATURE = 0.7
ALPHA = 0.01
Z_ALPHA = 2.326  # one-sided 99% threshold


# ---------------------------------------------------------------------------
# Self-contained MCL generator + detector (replaces EnhancedMCLGenerator API
# which doesn't accept pre-loaded model/tokenizer and returns a different
# shape than this runner needs).
# ---------------------------------------------------------------------------

_TOK_STATES: dict[tuple[str, int, int], np.ndarray] = {}


def init_state_partition(secret_key: str, vocab_size: int, S: int) -> np.ndarray:
    """Precompute state(t) = SHA256(key||'state'||t) mod S for every token t."""
    key = (secret_key, vocab_size, S)
    cached = _TOK_STATES.get(key)
    if cached is not None:
        return cached
    print(f"[mcl] precomputing state partition (V={vocab_size}, S={S}) ...", flush=True)
    states = np.zeros(vocab_size, dtype=np.int16)
    for t in range(vocab_size):
        digest = hashlib.sha256(f"{secret_key}|state|{t}".encode()).digest()
        states[t] = int.from_bytes(digest[:4], "little") % S
    _TOK_STATES[key] = states
    return states


def allowed_next_states(s: int, S: int, chain_key: str) -> tuple[int, ...]:
    if chain_key == "clockwork":
        return ((s + 1) % S,)
    if chain_key == "soft_cycle":
        return ((s + 1) % S, (s + 2) % S)
    raise ValueError(f"unknown chain_key={chain_key!r}")


def random_baseline(S: int, chain_key: str) -> float:
    if chain_key == "clockwork":
        return 1.0 / S
    if chain_key == "soft_cycle":
        return 2.0 / S
    raise ValueError(chain_key)


def mcl_generate(
    model,
    tokenizer,
    prompt: str,
    secret_key: str,
    num_states: int,
    chain_key: str,
    max_tokens: int,
    temperature: float,
    device: str,
    rho: float = 1.0,
    seed: int = 42,
    gate_entropy_thr: float | None = None,
) -> dict:
    """Generate one MCL-watermarked sample and self-detect.

    Gate logic at each step:
      * if ρ ≥ 1.0  → always gate (G_all);
      * elif `gate_entropy_thr` is not None → gate iff position entropy
        H_i > τ  (high-entropy gate, matched to SWEET); the threshold is
        calibrated externally from a pilot so that ⟨g_i⟩ ≈ ρ;
      * else → fall back to a Bernoulli(ρ) random gate.
    When gated, mask logits to allowed next-state tokens and pick argmax.
    When not gated, sample at temperature T.
    Returns: dict with token_ids, text, ppl, phi, z, n_tokens, gate_rate.
    """
    vocab_size = model.config.vocab_size
    states = init_state_partition(secret_key, vocab_size, num_states)
    states_dev = torch.from_numpy(states.astype(np.int64)).to(device)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]
    cur_state = int(states[int(input_ids[0, -1].item())])

    log_probs_chosen: list[float] = []
    n_gated = 0
    rng = np.random.default_rng(seed)
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_tokens):
            out = model(input_ids)
            logits = out.logits[0, -1].float()

            if rho >= 1.0:
                apply_gate = True
            elif gate_entropy_thr is not None:
                probs_t1 = F.softmax(logits, dim=-1)
                h = float(-(probs_t1 * torch.log(probs_t1.clamp_min(1e-12))).sum().item())
                apply_gate = h > gate_entropy_thr
            else:
                apply_gate = rng.random() < rho
            if apply_gate:
                n_gated += 1
            if apply_gate:
                allowed = allowed_next_states(cur_state, num_states, chain_key)
                allowed_mask = torch.zeros_like(states_dev, dtype=torch.bool)
                for s in allowed:
                    allowed_mask |= (states_dev == s)
                masked = logits.clone()
                masked[~allowed_mask] = float("-inf")
                if torch.isinf(masked).all():
                    masked = logits  # fallback: target partition empty
                next_token = int(torch.argmax(masked).item())
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            nat_lp = F.log_softmax(logits, dim=-1)
            log_probs_chosen.append(float(nat_lp[next_token].item()))

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=device)], dim=1,
            )
            cur_state = int(states[next_token])
            if eos_id is not None and next_token == eos_id:
                break

    gen_ids = input_ids[0, prompt_len:].cpu().tolist()
    n = len(gen_ids)
    if n == 0:
        return {"token_ids": [], "text": "", "n_tokens": 0,
                "ppl": float("nan"), "phi": float("nan"), "z": float("nan")}
    ppl = float(np.exp(-float(np.mean(log_probs_chosen))))
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    det = mcl_detect(gen_ids, secret_key, num_states, chain_key, vocab_size)
    return {
        "token_ids": gen_ids, "text": text, "n_tokens": n, "ppl": ppl,
        "phi": det["phi"], "z": det["z"],
    }


def mcl_detect(
    token_ids: list[int],
    secret_key: str,
    num_states: int,
    chain_key: str,
    vocab_size: int,
) -> dict:
    n = len(token_ids)
    if n < 2:
        return {"phi": float("nan"), "z": float("nan"),
                "n_valid": 0, "n_eval": 0, "p0": float("nan")}
    states = init_state_partition(secret_key, vocab_size, num_states)
    p0 = random_baseline(num_states, chain_key)
    n_valid = 0
    n_eval = n - 1
    for i in range(1, n):
        prev_s = int(states[token_ids[i - 1]])
        cur_s = int(states[token_ids[i]])
        if cur_s in allowed_next_states(prev_s, num_states, chain_key):
            n_valid += 1
    phi = n_valid / n_eval
    z = (phi - p0) / np.sqrt(p0 * (1.0 - p0) / n_eval)
    return {"phi": float(phi), "z": float(z),
            "n_valid": int(n_valid), "n_eval": int(n_eval), "p0": float(p0)}


def mcl_detect_from_text(
    tokenizer, text: str, secret_key: str,
    num_states: int, chain_key: str, vocab_size: int,
) -> dict:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return mcl_detect(ids, secret_key, num_states, chain_key, vocab_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def short_name(model_id: str) -> str:
    base = model_id.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", base).strip("-")


def load_model(model_id: str, int8: bool = False):
    """Load tokenizer + LM. Uses int8 weights for >40B params."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    if int8:
        # Llama-70B in int8 fits ~70 GB on a single H100-80GB.
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=qcfg, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
    model.eval()
    return tok, model, device


def build_pilot_entropies(model, tok, prompts: list[str], device: str,
                          max_tokens: int = 50) -> list[float]:
    """Pilot run for SWEET entropy threshold."""
    ents: list[float] = []
    for p in prompts:
        out = kgw_generate(
            model, tok, p, SECRET_KEY,
            gamma=0.5, delta=0.0,
            max_tokens=max_tokens, temperature=TEMPERATURE,
            device=device, method="kgw",
        )
        ents.extend(out["entropies"])
    return ents


# NOTE: MCL generation/detection now uses the self-contained `mcl_generate` /
# `mcl_detect` / `mcl_detect_from_text` helpers defined above. The old
# EnhancedMCLGenerator/Detector adapters have been removed because their
# constructor and return-type signatures don't match what this runner needs
# (see review-agent findings: no `model=`/`tokenizer=` kwargs, no z-score in
# detector return, etc.).


# ---------------------------------------------------------------------------
# Attack adapters
# ---------------------------------------------------------------------------


class AttackBundle:
    """Lazy holder for DIPPER + per-pivot back-translation attackers.

    Models load on first use to keep startup fast. Call .unload_all() when
    the experiment finishes."""

    def __init__(self, device: str):
        self.device = device
        self._dipper: DipperAttacker | None = None
        self._bt: dict[str, BackTranslationAttacker] = {}

    def dipper(self, text: str) -> str:
        if self._dipper is None:
            print("[attacks] loading DIPPER (~22 GB) ...", flush=True)
            self._dipper = DipperAttacker(device=self.device)
        return self._dipper.paraphrase(text)

    def back_translate(self, text: str, pivot_lang: str) -> str:
        if pivot_lang not in self._bt:
            print(f"[attacks] loading NLLB-200 (pivot={pivot_lang}) ...", flush=True)
            self._bt[pivot_lang] = BackTranslationAttacker(
                pivot_lang=pivot_lang, device=self.device
            )
        return self._bt[pivot_lang].paraphrase(text)

    def unload_all(self) -> None:
        if self._dipper is not None:
            self._dipper.unload()
            self._dipper = None
        for k, atk in list(self._bt.items()):
            atk.unload()
            del self._bt[k]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Token-level edit-distance for δ_eff measurement.
def token_edit_fraction(tok, before_ids: list[int], after_text: str) -> float:
    """Token-level normalised Levenshtein distance between the original
    generation token-id sequence and the re-tokenised attacked text.

    Uses python-Levenshtein when available (handles insertions/deletions
    correctly); otherwise falls back to a positional-mismatch upper bound."""
    after_ids = tok.encode(after_text, add_special_tokens=False)
    n_before, n_after = len(before_ids), len(after_ids)
    if n_before == 0:
        return 1.0
    if _HAVE_LEV:
        # Levenshtein on integer sequences via opaque-string trick:
        # python-Levenshtein operates on str; encode each id as a 2-byte
        # codepoint to avoid collisions in the ASCII range.
        s_before = "".join(chr(i % 0xFFFF + 1) for i in before_ids)
        s_after  = "".join(chr(i % 0xFFFF + 1) for i in after_ids)
        d = Levenshtein.distance(s_before, s_after)
        return float(max(0.0, min(1.0, d / max(n_before, n_after))))
    # Fallback: positional-mismatch upper bound (the original implementation;
    # length-divergent cases are flagged by saturation near 1.0).
    matched = 0
    for i in range(min(n_before, n_after)):
        if before_ids[i] == after_ids[i]:
            matched += 1
    delta = 1.0 - matched / n_before
    return float(max(0.0, min(1.0, delta)))


# ---------------------------------------------------------------------------
# Experiment 2 — head-to-head
# ---------------------------------------------------------------------------


PIVOT_NLLB = {
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
}


def run_exp2(args, tok, model, device: str, out_dir: Path) -> None:
    """3 methods × 4 domains × 3 attacks × 100 prompts."""
    vocab_size = model.config.vocab_size
    init_green(SECRET_KEY, vocab_size, device)

    # ---- prompts ----
    prompts_per_domain: dict[str, list[str]] = {}
    for d in DOMAINS:
        try:
            prompts_per_domain[d] = get_prompts(d, n=args.n_prompts)
        except Exception as e:
            print(f"[exp2] WARNING: failed to load domain {d}: {e}", flush=True)
            prompts_per_domain[d] = []

    # ---- pilot entropy for SWEET ----
    pilot = (prompts_per_domain.get("wiki", [])[:10]
             + prompts_per_domain.get("factual", [])[:10])
    pilot_entropies = build_pilot_entropies(model, tok, pilot, device) if pilot else []
    if pilot_entropies:
        sweet_thr = float(np.quantile(pilot_entropies, 1.0 - 0.5))
    else:
        sweet_thr = -1.0  # always-on fallback
    print(f"[exp2] SWEET threshold = {sweet_thr:.3f} from pilot n={len(pilot_entropies)}", flush=True)

    # MCL state partition warmed up once (one entry per (S, chain_key) used).
    init_state_partition(SECRET_KEY, vocab_size, MCL_S)

    attacks = AttackBundle(device=device)
    rng = np.random.default_rng(args.seed)

    # ---- methods ----
    # MCL uses the same high-entropy threshold as SWEET (top-ρ fraction by H)
    # → matched-budget, matched-strategy comparison.
    mcl_gate_thr = sweet_thr if sweet_thr > 0 else None
    print(f"[exp2] MCL gate (G_H_high) threshold = {mcl_gate_thr}", flush=True)

    def run_method_on_prompt(method: str, prompt: str) -> dict | None:
        if method == "mcl":
            res = mcl_generate(
                model, tok, prompt, SECRET_KEY,
                num_states=MCL_S, chain_key="clockwork",
                max_tokens=args.max_tokens, temperature=TEMPERATURE,
                device=device, rho=MCL_RHO, seed=args.seed,
                gate_entropy_thr=mcl_gate_thr,
            )
            if res["n_tokens"] < 2:
                return None
            return res
        kind = "sweet" if method == "sweet" else "kgw"
        gen = kgw_generate(
            model, tok, prompt, SECRET_KEY,
            gamma=KGW_GAMMA, delta=KGW_DELTA,
            max_tokens=args.max_tokens, temperature=TEMPERATURE,
            device=device, method=kind,
            sweet_rho=0.5, sweet_entropy_pilot=pilot_entropies,
        )
        if gen["n_tokens"] < 2:
            return None
        det = kgw_detect(gen["token_ids"], SECRET_KEY,
                         gamma=KGW_GAMMA, vocab_size=vocab_size, device=device)
        text = tok.decode(gen["token_ids"], skip_special_tokens=True)
        return {
            "text": text, "token_ids": gen["token_ids"],
            "phi": det["phi"], "z": det["z"], "n_tokens": gen["n_tokens"],
            "ppl": gen["ppl"],
        }

    def detect_attacked(method: str, ids_or_text, original_token_ids) -> dict:
        if method == "mcl":
            text = ids_or_text if isinstance(ids_or_text, str) else tok.decode(ids_or_text, skip_special_tokens=True)
            return mcl_detect_from_text(tok, text, SECRET_KEY, MCL_S, "clockwork", vocab_size)
        token_ids = ids_or_text if isinstance(ids_or_text, list) else tok.encode(ids_or_text, add_special_tokens=False)
        return kgw_detect(token_ids, SECRET_KEY, gamma=KGW_GAMMA,
                          vocab_size=vocab_size, device=device)

    # ---- main loop ----
    records_path = out_dir / "records.jsonl"
    summary: dict[str, Any] = {}
    n_records = 0

    methods_to_run = ("mcl", "kgw", "sweet")
    if getattr(args, "only_method", None):
        methods_to_run = (args.only_method,)
        print(f"[exp2] restricting to method={args.only_method} only", flush=True)
    with records_path.open("w") as fh_records:
        for method in methods_to_run:
            for domain, prompts in prompts_per_domain.items():
                if not prompts:
                    continue
                print(f"\n[exp2] === method={method} domain={domain} N={len(prompts)} ===", flush=True)
                for i, p in enumerate(prompts):
                    base = run_method_on_prompt(method, p)
                    if base is None:
                        continue
                    # Clean detection — also persist the watermarked text + ids
                    # so post-hoc attack recovery doesn't require re-generation.
                    rec = {
                        "method": method, "domain": domain, "idx": i,
                        "n_tokens": base["n_tokens"],
                        "z_clean": base["z"], "phi_clean": base["phi"],
                        "ppl": base.get("ppl"),
                        "text": base.get("text", ""),
                        "token_ids": base.get("token_ids", []),
                    }
                    # Random-substitution attack (cheap, gives parity with old data)
                    attacked_ids = random_substitute(
                        base["token_ids"], 0.20, vocab_size, rng,
                    )
                    if method == "mcl":
                        attacked_text = tok.decode(attacked_ids, skip_special_tokens=True)
                        det_r = detect_attacked(method, attacked_text, base["token_ids"])
                    else:
                        det_r = detect_attacked(method, attacked_ids, base["token_ids"])
                    rec["z_random"] = det_r["z"]
                    # ZH back-translation
                    try:
                        zh_text = attacks.back_translate(base["text"], PIVOT_NLLB["zh"])
                        det_zh = detect_attacked(
                            method,
                            zh_text if method == "mcl" else tok.encode(zh_text, add_special_tokens=False),
                            base["token_ids"],
                        )
                        rec["z_zh"] = det_zh["z"]
                        rec["delta_eff_zh"] = token_edit_fraction(tok, base["token_ids"], zh_text)
                    except Exception as e:
                        print(f"  [exp2] zh-attack failed on {domain}/{i}: {e}", flush=True)
                        rec["z_zh"] = float("nan")
                    # DIPPER
                    try:
                        dip_text = attacks.dipper(base["text"])
                        det_dip = detect_attacked(
                            method,
                            dip_text if method == "mcl" else tok.encode(dip_text, add_special_tokens=False),
                            base["token_ids"],
                        )
                        rec["z_dipper"] = det_dip["z"]
                        rec["delta_eff_dipper"] = token_edit_fraction(tok, base["token_ids"], dip_text)
                    except Exception as e:
                        print(f"  [exp2] dipper-attack failed on {domain}/{i}: {e}", flush=True)
                        rec["z_dipper"] = float("nan")
                    fh_records.write(json.dumps(rec) + "\n")
                    fh_records.flush()
                    n_records += 1
                    if (i + 1) % 25 == 0:
                        print(f"  [exp2 {method}/{domain}] {i+1}/{len(prompts)}  "
                              f"z_clean={rec['z_clean']:.2f}  z_zh={rec.get('z_zh', float('nan')):.2f}",
                              flush=True)

    attacks.unload_all()

    # ---- aggregate ----
    summary = aggregate_records(records_path, ("z_clean", "z_random", "z_zh", "z_dipper"))
    write_summary(out_dir, "exp2", args, summary, n_records)


# ---------------------------------------------------------------------------
# Experiment 3 — translation curve (MCL only, 25-prompt subsample)
# ---------------------------------------------------------------------------


def run_exp3(args, tok, model, device: str, out_dir: Path) -> None:
    """MCL only × 4 domains × 4 translation pivots × 25 prompts."""
    vocab_size = model.config.vocab_size
    init_green(SECRET_KEY, vocab_size, device)

    sub: dict[str, list[str]] = {}
    for d in DOMAINS:
        try:
            sub[d] = get_subsample(d)
        except Exception as e:
            print(f"[exp3] WARN domain {d}: {e}", flush=True)
            sub[d] = []

    init_state_partition(SECRET_KEY, vocab_size, MCL_S)
    # ---- entropy pilot for matched-budget G_H_high gate ----
    pilot_pool: list[str] = []
    for d in ("wiki", "factual"):
        if sub.get(d):
            pilot_pool.extend(sub[d][:5])
    pilot_entropies = build_pilot_entropies(model, tok, pilot_pool, device) if pilot_pool else []
    mcl_gate_thr = (float(np.quantile(pilot_entropies, 1.0 - MCL_RHO))
                    if pilot_entropies else None)
    print(f"[exp3] MCL G_H_high threshold = {mcl_gate_thr} from pilot n={len(pilot_entropies)}", flush=True)

    attacks = AttackBundle(device=device)

    records_path = out_dir / "records.jsonl"
    n_records = 0

    with records_path.open("w") as fh:
        for domain, prompts in sub.items():
            if not prompts:
                continue
            print(f"\n[exp3] domain={domain} N={len(prompts)}", flush=True)
            for i, p in enumerate(prompts):
                base = mcl_generate(
                    model, tok, p, SECRET_KEY,
                    num_states=MCL_S, chain_key="clockwork",
                    max_tokens=args.max_tokens, temperature=TEMPERATURE,
                    device=device, rho=MCL_RHO, seed=args.seed,
                    gate_entropy_thr=mcl_gate_thr,
                )
                if base["n_tokens"] < 2:
                    continue
                rec: dict[str, Any] = {
                    "method": "mcl", "domain": domain, "idx": i,
                    "n_tokens": base["n_tokens"],
                    "z_clean": base["z"], "phi_clean": base["phi"],
                    "ppl": base.get("ppl"),
                }
                for short, nllb in PIVOT_NLLB.items():
                    try:
                        attacked_text = attacks.back_translate(base["text"], nllb)
                        det = mcl_detect_from_text(
                            tok, attacked_text, SECRET_KEY, MCL_S, "clockwork", vocab_size,
                        )
                        rec[f"z_{short}"]     = det["z"]
                        rec[f"phi_{short}"]   = det["phi"]
                        rec[f"delta_eff_{short}"] = token_edit_fraction(
                            tok, base["token_ids"], attacked_text,
                        )
                    except Exception as e:
                        print(f"  [exp3] pivot={short} failed: {e}", flush=True)
                        rec[f"z_{short}"] = float("nan")
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_records += 1
                if (i + 1) % 5 == 0:
                    print(f"  [exp3 {domain}] {i+1}/{len(prompts)}", flush=True)

    attacks.unload_all()

    summary = aggregate_records(
        records_path,
        ("z_clean",) + tuple(f"z_{p}" for p in PIVOT_NLLB),
    )
    write_summary(out_dir, "exp3", args, summary, n_records)


# ---------------------------------------------------------------------------
# Experiment 4 — k=2 soft_cycle single point
# ---------------------------------------------------------------------------


def run_exp4(args, tok, model, device: str, out_dir: Path) -> None:
    """MCL k=2 (soft_cycle) on wiki × 100 prompts × {clean, zh}.

    Verifies Proposition 1 at k=2: random baseline p₀ ≈ 2/S = 0.40,
    and δ⋆ = 1−1/√2 still holds."""
    vocab_size = model.config.vocab_size
    prompts = get_prompts("wiki", n=args.n_prompts)
    init_state_partition(SECRET_KEY, vocab_size, MCL_S)

    # ---- pilot for G_H_high gate at ρ=0.5 ----
    pilot_entropies = build_pilot_entropies(model, tok, prompts[:10], device) if prompts else []
    mcl_gate_thr = (float(np.quantile(pilot_entropies, 1.0 - MCL_RHO))
                    if pilot_entropies else None)
    print(f"[exp4] MCL G_H_high threshold = {mcl_gate_thr} from pilot n={len(pilot_entropies)}", flush=True)

    attacks = AttackBundle(device=device)
    records_path = out_dir / "records.jsonl"
    n_records = 0

    with records_path.open("w") as fh:
        print(f"\n[exp4] k=2 soft_cycle, S={MCL_S}, ρ={MCL_RHO}, n_prompts={len(prompts)}", flush=True)
        for i, p in enumerate(prompts):
            base = mcl_generate(
                model, tok, p, SECRET_KEY,
                num_states=MCL_S, chain_key="soft_cycle",
                max_tokens=args.max_tokens, temperature=TEMPERATURE,
                device=device, rho=MCL_RHO, seed=args.seed,
                gate_entropy_thr=mcl_gate_thr,
            )
            if base["n_tokens"] < 2:
                continue
            rec: dict[str, Any] = {
                "method": "mcl_k2", "domain": "wiki", "idx": i,
                "n_tokens": base["n_tokens"],
                "z_clean": base["z"], "phi_clean": base["phi"],
                "ppl": base.get("ppl"),
            }
            try:
                zh_text = attacks.back_translate(base["text"], PIVOT_NLLB["zh"])
                det = mcl_detect_from_text(
                    tok, zh_text, SECRET_KEY, MCL_S, "soft_cycle", vocab_size,
                )
                rec["z_zh"] = det["z"]
                rec["phi_zh"] = det["phi"]
                rec["delta_eff_zh"] = token_edit_fraction(tok, base["token_ids"], zh_text)
            except Exception as e:
                print(f"  [exp4] zh failed: {e}", flush=True)
                rec["z_zh"] = float("nan")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_records += 1
            if (i + 1) % 25 == 0:
                print(f"  [exp4] {i+1}/{len(prompts)}", flush=True)

    attacks.unload_all()
    summary = aggregate_records(records_path, ("z_clean", "z_zh"))
    write_summary(out_dir, "exp4", args, summary, n_records)


# ---------------------------------------------------------------------------
# Experiment 5 — calibration sweep (S × model): watermarked vs non-watermarked
# at varied S so empirical FPR / TPR can be plotted against the closed-form
# Theorem 1 prediction S*(n, ρ, α).
# ---------------------------------------------------------------------------


def run_exp5(args, tok, model, device, out_dir: Path) -> None:
    DOMAINS = ("code", "factual", "wiki", "writing")
    vocab_size = model.config.vocab_size
    n_per_dom = args.n_prompts  # default 25 — wrapper sets this
    prompts_per_domain: dict[str, list[str]] = {}
    for d in DOMAINS:
        try:
            prompts_per_domain[d] = get_prompts(d, n=n_per_dom)
        except Exception as e:
            print(f"[exp5] WARNING: failed to load domain {d}: {e}", flush=True)
            prompts_per_domain[d] = []

    # entropy pilot for matched-budget high-entropy gate
    pilot_pool = (prompts_per_domain.get("wiki", [])[:5]
                  + prompts_per_domain.get("factual", [])[:5])
    pilot_entropies = build_pilot_entropies(model, tok, pilot_pool, device) if pilot_pool else []
    mcl_gate_thr = (float(np.quantile(pilot_entropies, 1.0 - MCL_RHO))
                    if pilot_entropies else None)
    print(f"[exp5] S={MCL_S} ρ={MCL_RHO} gate_thr={mcl_gate_thr} pilot_n={len(pilot_entropies)}",
          flush=True)

    init_state_partition(SECRET_KEY, vocab_size, MCL_S)

    records_path = out_dir / "records.jsonl"
    n_records = 0
    with records_path.open("w") as fh:
        for d in DOMAINS:
            prompts = prompts_per_domain.get(d, [])
            if not prompts:
                continue
            print(f"\n[exp5 S={MCL_S}] domain={d} N={len(prompts)}", flush=True)
            for i, p in enumerate(prompts):
                # watermarked: MCL at this S
                try:
                    wm = mcl_generate(
                        model, tok, p, SECRET_KEY,
                        num_states=MCL_S, chain_key="clockwork",
                        max_tokens=args.max_tokens, temperature=TEMPERATURE,
                        device=device, rho=MCL_RHO, seed=args.seed,
                        gate_entropy_thr=mcl_gate_thr,
                    )
                except Exception as e:
                    print(f"  [exp5] wm gen failed at {d}/{i}: {e}", flush=True)
                    continue
                if wm["n_tokens"] < 2:
                    continue

                # non-watermarked baseline: KGW δ=0 (no green-list bias) → natural LM output
                try:
                    nwm = kgw_generate(
                        model, tok, p, SECRET_KEY,
                        gamma=KGW_GAMMA, delta=0.0,
                        max_tokens=args.max_tokens, temperature=TEMPERATURE,
                        device=device, method="kgw",
                    )
                except Exception as e:
                    print(f"  [exp5] nwm gen failed at {d}/{i}: {e}", flush=True)
                    nwm = None

                rec = {
                    "exp": 5, "S": MCL_S, "rho": MCL_RHO, "domain": d,
                    "model": args.model, "method": "mcl",
                    "n_tokens_wm": wm["n_tokens"],
                    "phi_wm": wm.get("phi"),
                    "z_wm": wm.get("z"),
                    "ppl_wm": wm.get("ppl"),
                }
                if nwm and nwm.get("n_tokens", 0) >= 2:
                    try:
                        nwm_text = tok.decode(nwm["token_ids"], skip_special_tokens=True)
                        det = mcl_detect_from_text(
                            tok, nwm_text, SECRET_KEY,
                            num_states=MCL_S, chain_key="clockwork",
                            vocab_size=vocab_size,
                        )
                        rec["n_tokens_nwm"] = det.get("n_tokens", nwm["n_tokens"])
                        rec["phi_nwm"] = det.get("phi")
                        rec["z_nwm"] = det.get("z")
                        rec["ppl_nwm"] = nwm.get("ppl")
                    except Exception as e:
                        print(f"  [exp5] nwm detect failed: {e}", flush=True)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_records += 1
                if (i + 1) % 5 == 0:
                    print(f"  [exp5 S={MCL_S}] {d} {i+1}/{len(prompts)}", flush=True)

    print(f"[exp5] DONE S={MCL_S} n_records={n_records}", flush=True)
    summary = {
        "exp": 5, "S": MCL_S, "rho": MCL_RHO, "model": args.model,
        "n_prompts_per_domain": n_per_dom, "n_records": n_records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Aggregation + summary writer
# ---------------------------------------------------------------------------


def aggregate_records(records_path: Path, score_fields: tuple[str, ...]) -> dict:
    """Per (method, domain) and per attack: mean z, mean phi, TPR @ Z_ALPHA."""
    by_cell: dict[tuple, list[dict]] = {}
    with records_path.open() as fh:
        for line in fh:
            r = json.loads(line)
            key = (r.get("method", "mcl"), r.get("domain", "wiki"))
            by_cell.setdefault(key, []).append(r)
    out: dict[str, dict] = {}
    for (method, domain), records in by_cell.items():
        cell_key = f"{method}__{domain}"
        cell: dict[str, Any] = {"method": method, "domain": domain, "n": len(records)}
        for fld in score_fields:
            zs = np.array([r.get(fld, float("nan")) for r in records], dtype=float)
            zs_clean = zs[~np.isnan(zs)]
            if len(zs_clean) == 0:
                cell[fld] = {"mean_z": float("nan"), "tpr": float("nan"), "n_valid": 0}
            else:
                cell[fld] = {
                    "mean_z": float(zs_clean.mean()),
                    "std_z":  float(zs_clean.std(ddof=1)) if len(zs_clean) > 1 else 0.0,
                    "tpr":    float((zs_clean > Z_ALPHA).mean()),
                    "n_valid": int(len(zs_clean)),
                }
        out[cell_key] = cell
    return out


def write_summary(out_dir: Path, exp: str, args, summary: dict, n_records: int) -> None:
    sm = {
        "exp":         exp,
        "model":       args.model,
        "model_short": short_name(args.model),
        "max_tokens":  args.max_tokens,
        "n_prompts":   args.n_prompts,
        "secret_key":  SECRET_KEY,
        "S":           MCL_S,
        "rho":         MCL_RHO,
        "alpha":       ALPHA,
        "z_alpha":     Z_ALPHA,
        "n_records":   n_records,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "cells":       summary,
    }
    p = out_dir / "summary.json"
    with p.open("w") as fh:
        json.dump(sm, fh, indent=2)
    print(f"\n[done] wrote {p}\n", flush=True)
    # Headline print
    print("=== HEADLINE ===")
    for k, v in summary.items():
        msg = f"  {k:30s}  n={v['n']:3d}"
        for fld, sub in v.items():
            if isinstance(sub, dict) and "tpr" in sub:
                msg += f"  {fld}: z={sub['mean_z']:+6.2f} tpr={sub['tpr']:.2f}"
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["2", "3", "4", "5"])
    ap.add_argument("--S", type=int, default=None,
                    help="Override MCL_S (used by exp 5 calibration sweep)")
    ap.add_argument("--model", required=True,
                    help="HuggingFace model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--int8", action="store_true",
                    help="Load model with bitsandbytes 8-bit quantisation (for >40B)")
    ap.add_argument("--only-method", default=None, choices=[None, "mcl", "kgw", "sweet"],
                    help="If set (exp 2 only), restrict to this watermarking method "
                         "so jobs can be split across (method, model) cells in parallel.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--n-prompts", type=int, default=100,
                    help="Per-domain prompt count for exps 2 and 4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default=str(REPO / "data" / "v7_min"))
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.S is not None:
        global MCL_S
        MCL_S = int(args.S)
        print(f"[runner] MCL_S overridden to {MCL_S}", flush=True)

    method_tag = f"_{args.only_method}" if args.only_method else ""
    s_tag = f"_S{MCL_S}" if args.exp == "5" else ""
    out_dir = Path(args.out_root) / f"exp{args.exp}_{short_name(args.model)}{method_tag}{s_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[runner] exp={args.exp} model={args.model} int8={args.int8} out={out_dir}", flush=True)
    t0 = time.time()
    tok, model, device = load_model(args.model, int8=args.int8)
    print(f"[runner] model loaded in {time.time()-t0:.1f}s on device={device}", flush=True)

    if args.exp == "2":
        run_exp2(args, tok, model, device, out_dir)
    elif args.exp == "3":
        run_exp3(args, tok, model, device, out_dir)
    elif args.exp == "4":
        run_exp4(args, tok, model, device, out_dir)
    elif args.exp == "5":
        run_exp5(args, tok, model, device, out_dir)
    else:
        raise ValueError(args.exp)

    print(f"[runner] DONE exp={args.exp} model={args.model} elapsed={(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
