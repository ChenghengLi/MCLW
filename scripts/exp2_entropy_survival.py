#!/usr/bin/env python
"""
Experiment 2 -- HEADLINE: Entropy vs paraphrase survival.

This is the primary novel empirical contribution; Proposition 8 is tested here.

Pipeline:
  1. Generate N entropy-gated MCL texts, recording per-position entropy H_i,
     state sigma(t_i), watermark-gate W_i, and valid-transition flag.
  2. Paraphrase each text with each configured attack.
  3. Re-tokenize paraphrase under generator's tokenizer and align to original.
  4. For each original position, record {survived, substituted, deleted}.
  5. Bin by H_i and report survival probability.
  6. Report post-attack detection AUC.

Outputs: data/exp2_<timestamp>/{attack}/{sample}.json + summary.json

Usage:
    uv run python scripts/exp2_entropy_survival.py \
        --n-prompts 100 \
        --attacks random_sub30 synonym30 dipper_60 \
        --states 7 --tau-quantile Q50
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark.entropy_gated_mcl import (
    EntropyGatedMCLGenerator,
    pilot_measure_entropy_quantiles,
)
from ltw_watermark.detection_stats import detect
from ltw_watermark.alignment import align_token_sequences
from ltw_watermark.attacks import (
    random_substitution_tokens,
    synonym_substitution,
    DipperAttacker,
    BackTranslationAttacker,
    SIRAAttacker,
)


def load_prompts(n: int) -> List[str]:
    from scripts.generate_curated_dataset import WIKIPEDIA_CONCEPTS  # type: ignore
    tmpl = "Explain {c} in a comprehensive way."
    return [tmpl.format(c=c.replace("_", " ")) for c in WIKIPEDIA_CONCEPTS[:n]]


# ---------- Attack plumbing ----------
ATTACK_SPECS = {
    "random_sub10": {"kind": "random_sub", "rate": 0.10},
    "random_sub20": {"kind": "random_sub", "rate": 0.20},
    "random_sub30": {"kind": "random_sub", "rate": 0.30},
    "synonym20":    {"kind": "synonym",    "rate": 0.20},
    "synonym30":    {"kind": "synonym",    "rate": 0.30},
    "dipper_40":    {"kind": "dipper",     "lex": 40, "order": 40},
    "dipper_60":    {"kind": "dipper",     "lex": 60, "order": 60},
    "backtrans_fr": {"kind": "backtrans",  "pivot": "fra_Latn"},
    "sira25":       {"kind": "sira",       "top_k": 0.25},
}


def build_attack(spec: Dict, tokenizer, vocab_size: int) -> Callable[[str, List[int], int], Dict]:
    """Returns a function (text, token_ids, seed) -> {text, token_ids}."""
    kind = spec["kind"]

    if kind == "random_sub":
        rate = spec["rate"]

        def _attack(text, ids, seed):
            new_ids = random_substitution_tokens(ids, vocab_size, rate, seed=seed)
            return {"text": tokenizer.decode(new_ids, skip_special_tokens=True),
                    "token_ids": new_ids}
        return _attack

    if kind == "synonym":
        rate = spec["rate"]

        def _attack(text, ids, seed):
            new_text = synonym_substitution(text, rate, seed=seed)
            new_ids = tokenizer.encode(new_text, add_special_tokens=False)
            return {"text": new_text, "token_ids": new_ids}
        return _attack

    if kind == "dipper":
        lex, order = spec["lex"], spec["order"]
        atk = DipperAttacker()

        def _attack(text, ids, seed):
            torch.manual_seed(seed)
            new_text = atk.paraphrase(text, lex_diversity=lex, order_diversity=order)
            new_ids = tokenizer.encode(new_text, add_special_tokens=False)
            return {"text": new_text, "token_ids": new_ids}
        _attack.cleanup = atk.unload  # type: ignore
        return _attack

    if kind == "backtrans":
        atk = BackTranslationAttacker(pivot_lang=spec["pivot"])

        def _attack(text, ids, seed):
            new_text = atk.paraphrase(text)
            new_ids = tokenizer.encode(new_text, add_special_tokens=False)
            return {"text": new_text, "token_ids": new_ids}
        _attack.cleanup = atk.unload  # type: ignore
        return _attack

    if kind == "sira":
        atk = SIRAAttacker()
        top_k = spec["top_k"]

        def _attack(text, ids, seed):
            torch.manual_seed(seed)
            new_text = atk.paraphrase(text, top_k_frac=top_k)
            new_ids = tokenizer.encode(new_text, add_special_tokens=False)
            return {"text": new_text, "token_ids": new_ids}
        _attack.cleanup = atk.unload  # type: ignore
        return _attack

    raise ValueError(f"unknown attack kind: {kind}")


def bin_entropies(entropies: List[float]) -> str:
    h = entropies
    avg = float(np.mean(h)) if h else 0.0  # per-position binning used downstream; this is summary
    return f"avg_H={avg:.2f}"


def main():
    parser = argparse.ArgumentParser(description="E2: entropy-vs-survival under paraphrase")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--secret-key", default="mclw_entropy_gated_2026")
    parser.add_argument("--states", type=int, default=7)
    parser.add_argument("--chain-key", default="clockwork")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--tau-quantile", default="Q50",
                        help="Entropy percentile used for tau_H; pilot-measured per run.")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--n-pilot", type=int, default=20)
    parser.add_argument("--attacks", nargs="+",
                        default=["random_sub30", "synonym30", "dipper_60"],
                        help="Any of: " + ", ".join(ATTACK_SPECS.keys()))
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/exp2_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[E2] Writing to {out_dir}")

    prompts = load_prompts(args.n_prompts)
    pilot = prompts[: args.n_pilot]

    # ============================== 1. Generate watermarked texts ==============================
    gen = EntropyGatedMCLGenerator(
        model_name=args.model,
        secret_key=args.secret_key,
        num_states=args.states,
        chain_key=args.chain_key,
        overlap_ratio=args.overlap,
        tau_H=float("inf"),
    )
    print("[E2] Pilot for tau_H percentiles...")
    qs = pilot_measure_entropy_quantiles(gen, pilot, max_new_tokens=args.max_tokens)
    tau = qs[args.tau_quantile]
    gen.tau_H = tau
    print(f"[E2] Using tau_H = {tau:.2f} (= {args.tau_quantile})")

    gen_records = []
    for i, p in enumerate(prompts):
        res = gen.generate(p, max_new_tokens=args.max_tokens, greedy=True)
        gen_records.append(res)
        if (i + 1) % 10 == 0:
            print(f"[E2] Generated {i+1}/{len(prompts)}")
    generator_tokenizer = gen.tokenizer
    vocab_size = gen.vocab_size

    # Free generator VRAM before loading attack models (DIPPER etc).
    del gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================== 2 & 3. Attacks + alignment + analysis ==============================
    all_summary = {
        "experiment": "E2_entropy_survival",
        "timestamp": stamp,
        "model": args.model,
        "num_states": args.states,
        "tau_H": tau,
        "tau_quantile": args.tau_quantile,
        "n_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "attacks": {},
    }

    # Pre-save generation records (useful for resuming).
    gen_dir = out_dir / "generations"
    gen_dir.mkdir(exist_ok=True)
    with (gen_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in gen_records:
            d = r.to_dict()
            # positions list can be long; keep it -- it's the whole point
            f.write(json.dumps(d) + "\n")

    for attack_name in args.attacks:
        if attack_name not in ATTACK_SPECS:
            print(f"[E2] Unknown attack '{attack_name}', skipping.")
            continue
        spec = ATTACK_SPECS[attack_name]
        print(f"\n[E2] === Attack: {attack_name} ({spec}) ===")
        attack_dir = out_dir / attack_name
        attack_dir.mkdir(exist_ok=True)

        attack_fn = build_attack(spec, generator_tokenizer, vocab_size)

        # per-position pooled arrays, for survival binning.
        pooled_H: List[float] = []
        pooled_W: List[int] = []
        pooled_V: List[int] = []
        pooled_survived: List[int] = []
        post_phi, post_z, post_det = [], [], 0

        for idx, res in enumerate(gen_records):
            out = attack_fn(res.text, res.token_ids, seed=args.seed + idx)
            attacked_text = out["text"]
            attacked_ids = out["token_ids"]

            try:
                align = align_token_sequences(res.token_ids, attacked_ids)
            except RuntimeError:
                print("[E2] python-Levenshtein not installed; aborting E2.")
                sys.exit(2)

            # per-original-position pooled contributions
            for pos, lab in zip(res.positions, align.labels):
                pooled_H.append(pos.entropy)
                pooled_W.append(int(pos.watermarked))
                pooled_V.append(int(pos.valid_transition))
                pooled_survived.append(int(lab == "survived"))

            # post-attack detection
            det = detect(
                attacked_ids,
                args.states, args.chain_key, args.secret_key,
                overlap_ratio=args.overlap,
                alpha=args.alpha,
                exact=len(attacked_ids) < 200,
            )
            post_phi.append(det.phi)
            post_z.append(det.z)
            if det.is_watermarked:
                post_det += 1

            # persist per-sample record
            with (attack_dir / f"sample_{idx:04d}.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "prompt": res.prompt,
                    "original_text": res.text,
                    "attacked_text": attacked_text,
                    "n_orig": align.n_orig,
                    "n_para": align.n_para,
                    "edit_distance": align.edit_distance,
                    "labels": align.labels,
                    "post_attack_phi": det.phi,
                    "post_attack_z": det.z,
                    "post_attack_p": det.p_value,
                }, f)

            if (idx + 1) % 10 == 0:
                print(f"[E2][{attack_name}] {idx+1}/{len(gen_records)}  post-z mean={np.mean(post_z):.2f}")

        # cleanup heavy attack models
        if hasattr(attack_fn, "cleanup"):
            try:
                attack_fn.cleanup()  # type: ignore[attr-defined]
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # entropy-binned survival analysis
        H = np.array(pooled_H, dtype=np.float64)
        W = np.array(pooled_W, dtype=np.int32)
        V = np.array(pooled_V, dtype=np.int32)
        S = np.array(pooled_survived, dtype=np.int32)

        bin_edges = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, np.inf]
        bins = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (H >= lo) & (H < hi)
            if mask.sum() == 0:
                continue
            bins.append({
                "H_lo": lo,
                "H_hi": hi if np.isfinite(hi) else None,
                "n_positions": int(mask.sum()),
                "survival_rate": float(S[mask].mean()),
                "watermarked_fraction": float(W[mask].mean()),
                "valid_transition_fraction": float(V[mask].mean()),
            })

        # Survival rate for WATERMARKED vs non-watermarked positions (key chart).
        wm_mask = W == 1
        free_mask = W == 0
        overall = {
            "n_positions": int(len(H)),
            "survival_all": float(S.mean()) if len(S) else 0.0,
            "survival_watermarked": float(S[wm_mask].mean()) if wm_mask.any() else None,
            "survival_free": float(S[free_mask].mean()) if free_mask.any() else None,
            "mean_H": float(H.mean()) if len(H) else 0.0,
        }

        attack_summary = {
            "attack": attack_name,
            "spec": spec,
            "n_samples": len(gen_records),
            "post_attack_tpr": post_det / max(1, len(gen_records)),
            "post_attack_mean_z": float(np.mean(post_z)) if post_z else 0.0,
            "post_attack_median_phi": float(np.median(post_phi)) if post_phi else 0.0,
            "overall": overall,
            "bins": bins,
        }
        all_summary["attacks"][attack_name] = attack_summary

        print(f"[E2][{attack_name}] TPR={attack_summary['post_attack_tpr']:.3f} "
              f"survival wm={overall['survival_watermarked']} free={overall['survival_free']}")
        print("[E2] Bin summary:")
        for b in bins:
            print(f"  H in [{b['H_lo']}, {b['H_hi']}) -> survive={b['survival_rate']:.3f} (n={b['n_positions']})")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\n[E2] Done. Summary: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
