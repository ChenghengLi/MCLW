#!/usr/bin/env python3
"""Head-to-head baselines vs MCL on google/gemma-3-270m-it.

Implements:
- KGW (Kirchenbauer et al. 2023, "A Watermark for Large Language Models"):
  per-step green/red split via SHA-256(secret_key || prev_token), bias delta
  added to green logits, sample at temperature T.
- SWEET (Lee et al. 2023): KGW with bias applied only at high-entropy
  positions (top-rho fraction by next-token entropy).

Generates max_tokens=100 on the same 173 wiki + 20 factual prompts as
v5_baseline_N200, computes per-text z-scores and perplexities, applies
the same random-substitution attack at delta=0.20, and aggregates.

Usage:
    python3 scripts/run_kgw_sweet_baselines.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ltw_watermark import load_prompts
from ltw_watermark.factual_prompts import FACTUAL_CLOSED_PROMPTS


# ---------------------------------------------------------------------------
# KGW / SWEET helpers
# ---------------------------------------------------------------------------

# --- Vectorised hash-membership green-list -------------------------------
# Precompute per-token 32-bit SHA-256 hashes once at startup. Per-step green
# mask is then a vectorised arithmetic op on these hashes, avoiding any
# per-step 256k-element permutation (which was the bottleneck in the
# permutation-based KGW implementation).

_TOK_HASH: torch.Tensor | None = None  # (V,) uint32, fixed for the run


def _build_token_hashes_cpu(secret_key: str, vocab_size: int) -> np.ndarray:
    h = np.zeros(vocab_size, dtype=np.uint32)
    for t in range(vocab_size):
        digest = hashlib.sha256(f"{secret_key}|tok|{t}".encode()).digest()
        h[t] = int.from_bytes(digest[:4], "little")
    return h.astype(np.int64)  # widen so addition can't overflow


def init_green(secret_key: str, vocab_size: int, device: str) -> None:
    global _TOK_HASH
    if _TOK_HASH is None or _TOK_HASH.shape[0] != vocab_size:
        print(f"[baselines] precomputing token hashes (V={vocab_size}) ...", flush=True)
        _TOK_HASH = _build_token_hashes_cpu(secret_key, vocab_size)


def green_mask(secret_key: str, prev_token: int, gamma: float, vocab_size: int,
               device: str) -> torch.Tensor:
    """Boolean mask of size (V,). A token t is in the green list given prev p
    iff ((H(t) + H(p)) mod 2^32) < gamma * 2^32. Marginal Pr[green | prev] =
    gamma under the random-oracle model. Computed on CPU (numpy) then
    transferred to device once per call - on MPS, integer arithmetic over
    250k elements is much slower than on CPU + a single 32 KB host-to-device
    copy."""
    if _TOK_HASH is None:
        init_green(secret_key, vocab_size, device)
    h_p = int(_TOK_HASH[int(prev_token)])
    threshold = int(round(gamma * (1 << 32)))
    rotated = (_TOK_HASH + h_p) & 0xFFFFFFFF                   # (V,) numpy int64
    mask = rotated < threshold                                 # (V,) numpy bool
    return torch.from_numpy(mask).to(device)


def kgw_generate(
    model,
    tokenizer,
    prompt: str,
    secret_key: str,
    gamma: float,
    delta: float,
    max_tokens: int,
    temperature: float,
    device: str,
    method: str = "kgw",
    sweet_rho: float = 0.5,
    sweet_entropy_pilot: list[float] | None = None,
) -> dict:
    """Generate tokens under KGW or SWEET. Returns dict with token_ids,
    perplexity, and per-step metadata."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    log_probs_chosen: list[float] = []
    entropies: list[float] = []
    is_green_per_step: list[bool] = []
    is_gated_per_step: list[bool] = []  # SWEET: did we apply bias?

    # SWEET threshold from pilot entropies (top-rho fraction by entropy).
    if method == "sweet" and sweet_entropy_pilot is not None and len(sweet_entropy_pilot) > 0:
        sweet_thr = float(np.quantile(sweet_entropy_pilot, 1.0 - sweet_rho))
    else:
        sweet_thr = -1.0  # always-on fallback

    eos_id = tokenizer.eos_token_id
    vocab_size = model.config.vocab_size

    with torch.no_grad():
        for step in range(max_tokens):
            out = model(input_ids)
            logits = out.logits[0, -1].float()  # (V,)

            # Per-step entropy of unmodified distribution (used for SWEET gate).
            base_probs = F.softmax(logits / temperature, dim=-1)
            ent = -(base_probs * torch.log(base_probs.clamp_min(1e-12))).sum().item()
            entropies.append(ent)

            prev_token = int(input_ids[0, -1].item())
            mask = green_mask(secret_key, prev_token, gamma, vocab_size, device)

            # Decide whether to apply bias (SWEET gate).
            apply_bias = (method == "kgw") or (method == "sweet" and ent > sweet_thr)
            is_gated_per_step.append(apply_bias)

            if apply_bias:
                logits = logits + delta * mask.float()

            probs = F.softmax(logits / temperature, dim=-1)
            # Sample (multinomial).
            idx = torch.multinomial(probs, num_samples=1)
            next_token = int(idx.item())

            # Track perplexity: log p of the chosen token under the MODEL's
            # natural distribution at T=1.0 (not biased, not temperature-scaled).
            # This matches the convention used by the existing v5_baseline_N200
            # script and lets us compare PPL across methods.
            nat_log_probs = F.log_softmax(out.logits[0, -1].float(), dim=-1)
            log_probs_chosen.append(float(nat_log_probs[next_token].item()))

            is_green_per_step.append(bool(mask[next_token].item()))

            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=device)], dim=1
            )
            if eos_id is not None and next_token == eos_id:
                break

    gen_ids = input_ids[0, prompt_len:].cpu().tolist()
    n = len(gen_ids)
    if n == 0:
        return {
            "token_ids": [], "n_tokens": 0, "ppl": float("nan"),
            "n_green": 0, "n_eval": 0, "is_green": [], "is_gated": [],
        }

    nll = -float(np.mean(log_probs_chosen))
    ppl = float(np.exp(nll))
    return {
        "token_ids": gen_ids,
        "n_tokens": n,
        "ppl": ppl,
        "n_green": int(sum(is_green_per_step)),
        "n_eval": n,
        "is_green": is_green_per_step,
        "is_gated": is_gated_per_step,
        "entropies": entropies,
    }


def kgw_detect(
    token_ids: list[int],
    secret_key: str,
    gamma: float,
    vocab_size: int,
    device: str = "cpu",
    only_indices: list[int] | None = None,
) -> dict:
    """Score a sequence under KGW: count green tokens, compute z-score.

    If `only_indices` is given (SWEET-detector mode), restrict the count to
    those positions; otherwise score every i >= 1 transition."""
    n = len(token_ids)
    if n < 2:
        return {"phi": float("nan"), "z": float("nan"), "n_green": 0, "n_eval": 0}
    indices = list(range(1, n)) if only_indices is None else only_indices
    n_green = 0
    for i in indices:
        prev = token_ids[i - 1]
        cur = token_ids[i]
        mask = green_mask(secret_key, prev, gamma, vocab_size, device)
        if bool(mask[cur].item()):
            n_green += 1
    n_eval = max(1, len(indices))
    phi = n_green / n_eval
    z = (n_green - gamma * n_eval) / np.sqrt(max(1e-12, n_eval * gamma * (1 - gamma)))
    return {"phi": phi, "z": float(z), "n_green": n_green, "n_eval": n_eval}


# ---------------------------------------------------------------------------
# Random-substitution attack
# ---------------------------------------------------------------------------


def random_substitute(token_ids: list[int], delta: float, vocab_size: int,
                      rng: np.random.Generator) -> list[int]:
    out = list(token_ids)
    for i in range(len(out)):
        if rng.random() < delta:
            out[i] = int(rng.integers(0, vocab_size))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate(records: list[dict], alpha: float, z_alpha: float) -> dict:
    arr = np.array
    ppl  = arr([r["ppl"]  for r in records if not np.isnan(r["ppl"])])
    z    = arr([r["z"]    for r in records])
    z_post = arr([r["z_post"] for r in records])
    detected = z > z_alpha
    detected_post = z_post > z_alpha
    return {
        "n":             int(len(records)),
        "mean_ppl":      float(ppl.mean()) if len(ppl) else float("nan"),
        "std_ppl":       float(ppl.std(ddof=1)) if len(ppl) > 1 else 0.0,
        "mean_z":        float(z.mean()),
        "mean_z_post":   float(z_post.mean()),
        "ratio_post_pre": float((z_post / np.maximum(z, 1e-9)).mean()),
        "tpr":           float(detected.mean()),
        "tpr_post":      float(detected_post.mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--secret-key", default="mclw_v4_2026")
    ap.add_argument("--n-wiki", type=int, default=173)
    ap.add_argument("--n-factual", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack-delta", type=float, default=0.20)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else REPO / "data" / "v6_kgw_sweet_N200"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device.
    if torch.backends.mps.is_available():
        device = "mps"; dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"; dtype = torch.float16
    else:
        device = "cpu"; dtype = torch.float32

    print(f"[baselines] device={device} dtype={dtype}")
    print(f"[baselines] loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    vocab_size = model.config.vocab_size
    z_alpha = float(np.percentile(np.random.standard_normal(10**6), 99.0))  # ~2.326

    wiki = load_prompts(args.n_wiki)
    fact = FACTUAL_CLOSED_PROMPTS[: args.n_factual]
    domains = {"wiki": wiki, "factual": fact}

    # Methods to run.
    methods = [
        # name, gamma, delta_bias, sweet_rho (or None)
        ("kgw_g25_d2",   0.25, 2.0, None),
        ("kgw_g50_d2",   0.50, 2.0, None),
        ("sweet_g50_d2_r50", 0.50, 2.0, 0.50),
    ]

    # Pilot pass: run a small unwatermarked sample to estimate the entropy
    # threshold used by SWEET (top 50% entropy quantile).
    print("[baselines] running entropy pilot (unwatermarked, 20 prompts)...")
    pilot_entropies: list[float] = []
    for p in (wiki[:10] + fact[:10]):
        out = kgw_generate(
            model, tokenizer, p, args.secret_key,
            gamma=0.5, delta=0.0,           # no bias = unwatermarked
            max_tokens=50, temperature=args.temperature,
            device=device, method="kgw",
        )
        pilot_entropies.extend(out["entropies"])
    print(f"  pilot entropies: n={len(pilot_entropies)} "
          f"median={np.median(pilot_entropies):.3f} "
          f"q50={np.quantile(pilot_entropies, 0.5):.3f}")

    rng_attack = np.random.default_rng(args.seed)

    summary: dict[str, dict] = {}
    t0 = time.time()
    for method_name, gamma, delta_bias, sweet_rho in methods:
        method_kind = "sweet" if sweet_rho is not None else "kgw"
        print(f"\n[baselines] === {method_name} (kind={method_kind} "
              f"gamma={gamma} delta={delta_bias} sweet_rho={sweet_rho}) ===")
        domain_records: dict[str, list[dict]] = {"wiki": [], "factual": []}

        for dom, prompts in domains.items():
            print(f"\n[baselines] domain={dom}  N={len(prompts)}")
            for i, p in enumerate(prompts):
                gen = kgw_generate(
                    model, tokenizer, p, args.secret_key,
                    gamma=gamma, delta=delta_bias,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    device=device, method=method_kind,
                    sweet_rho=(sweet_rho or 0.0),
                    sweet_entropy_pilot=pilot_entropies,
                )
                if gen["n_tokens"] < 2:
                    continue
                det = kgw_detect(
                    gen["token_ids"], args.secret_key, gamma=gamma,
                    vocab_size=vocab_size, device=device,
                )
                attacked = random_substitute(
                    gen["token_ids"], args.attack_delta, vocab_size, rng_attack,
                )
                det_post = kgw_detect(
                    attacked, args.secret_key, gamma=gamma,
                    vocab_size=vocab_size, device=device,
                )
                domain_records[dom].append({
                    "ppl":        gen["ppl"],
                    "n_tokens":   gen["n_tokens"],
                    "phi":        det["phi"],
                    "z":          det["z"],
                    "z_post":     det_post["z"],
                    "is_gated_rate": float(np.mean(gen["is_gated"])),
                })
                if (i + 1) % 25 == 0 or i == len(prompts) - 1:
                    print(f"  [{dom} {i+1}/{len(prompts)}]  "
                          f"ppl={gen['ppl']:.2f}  z={det['z']:+.2f}  "
                          f"z_post={det_post['z']:+.2f}", flush=True)

        summary[method_name] = {
            "method":   method_name,
            "kind":     method_kind,
            "gamma":    gamma,
            "delta":    delta_bias,
            "sweet_rho": sweet_rho,
            "wiki":     aggregate(domain_records["wiki"], args.alpha, z_alpha),
            "factual":  aggregate(domain_records["factual"], args.alpha, z_alpha),
            "combined": aggregate(domain_records["wiki"] + domain_records["factual"],
                                  args.alpha, z_alpha),
        }
        # incrementally save
        elapsed = time.time() - t0
        out_path = out_dir / "summary.json"
        with out_path.open("w") as f:
            json.dump({
                "run":          "v6_kgw_sweet_N200",
                "model":        args.model,
                "n_wiki":       args.n_wiki,
                "n_factual":    args.n_factual,
                "max_tokens":   args.max_tokens,
                "temperature":  args.temperature,
                "alpha":        args.alpha,
                "z_alpha":      z_alpha,
                "attack_delta": args.attack_delta,
                "seed":         args.seed,
                "elapsed_s":    round(elapsed, 2),
                "timestamp":    datetime.now().isoformat(timespec="seconds"),
                "methods":      summary,
            }, f, indent=2)
        print(f"  [{method_name}] elapsed so far: {elapsed/60:.1f} min")

    print(f"\n[baselines] DONE. wrote {out_dir/'summary.json'}")
    print("\n=== HEADLINE ===")
    for name, s in summary.items():
        c = s["combined"]
        print(f"  {name:25s}  ppl={c['mean_ppl']:6.2f}  "
              f"z={c['mean_z']:+6.2f}  z_post={c['mean_z_post']:+6.2f}  "
              f"tpr={c['tpr']:.3f}  tpr_post={c['tpr_post']:.3f}")


if __name__ == "__main__":
    main()
