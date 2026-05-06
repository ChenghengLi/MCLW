#!/usr/bin/env python
"""
Paraphrase Robustness Attack

Tests whether MCL detection survives semantic paraphrasing attacks. This
complements the per-token attacks in robustness_attack.py: paraphrasing
preserves meaning but rewrites the surface form, which is the realistic
adversary in practice.

Two methods are supported:

  --method dipper
      DIPPER paraphraser (Krishna et al. 2023). Loads
      kalpeshk2011/dipper-paraphraser-xxl (~11B params; needs ~22 GB VRAM).
      Use --lex and --order to control lexical and order diversity (each
      0/20/40/60/80/100; 60 is the standard reported in their paper).

  --method backtranslation
      Round-trip translation through a pivot language using
      Helsinki-NLP/opus-mt-en-{pivot} and Helsinki-NLP/opus-mt-{pivot}-en.
      Default pivots: de, fr. Cheaper than DIPPER (~300M params each way)
      but still needs a GPU for sensible runtime.

Detection is then re-run on the paraphrased text using the same
EnhancedMCLDetector configuration that produced the original watermark.

Usage:
    # DIPPER paraphrase, standard 60/60 setting
    uv run python scripts/paraphrase_attack.py \\
        --data-dir data/curated_wiki_dataset_XXXXXXXX_XXXXXX \\
        --config states7_overlap0pct \\
        --method dipper --lex 60 --order 60

    # German pivot back-translation
    uv run python scripts/paraphrase_attack.py \\
        --data-dir data/curated_wiki_dataset_XXXXXXXX_XXXXXX \\
        --config states7_overlap0pct \\
        --method backtranslation --pivots de fr

The script writes paraphrase_<config>_<method>.json next to the dataset
and prints a summary table.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcl_watermark.enhanced_mcl import EnhancedMCLDetector


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch is required for paraphrase attacks. `uv sync` first.")


# -----------------------------------------------------------------------------
# DIPPER paraphraser (Krishna et al. 2023)
# -----------------------------------------------------------------------------

class DipperParaphraser:
    """Wrapper around kalpeshk2011/dipper-paraphraser-xxl.

    DIPPER is a fine-tuned T5-XXL that takes a control prefix encoding
    desired lexical diversity (lex) and word-order diversity (order) on
    a 0/20/40/60/80/100 scale and rewrites the input. We follow the
    prompt format from the original DIPPER repo.
    """

    MODEL_NAME = "kalpeshk2011/dipper-paraphraser-xxl"

    def __init__(self, device: str = "cuda", torch_dtype: str = "float16"):
        _require_torch()
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        dtype = getattr(torch, torch_dtype)
        # AutoTokenizer correctly resolves the FAST T5 tokenizer; the legacy
        # T5Tokenizer + newer sentencepiece raises "not a string" on Load.
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, use_fast=True)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.MODEL_NAME, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.device = device

    def paraphrase(
        self,
        text: str,
        lex: int = 60,
        order: int = 60,
        max_new_tokens: int = 256,
        prefix: str = "",
    ) -> str:
        import torch

        assert lex in {0, 20, 40, 60, 80, 100}, "lex must be one of 0/20/40/60/80/100"
        assert order in {0, 20, 40, 60, 80, 100}, "order must be one of 0/20/40/60/80/100"

        prompt = (
            f"lexical = {lex}, order = {order} "
            f"{prefix.strip()} <sent> {text.strip()} </sent>"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=0.75,
                top_k=None,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Round-trip back-translation
# -----------------------------------------------------------------------------

class BackTranslator:
    """English -> pivot -> English using the Helsinki-NLP Opus models.

    These are small (~300M params) MarianMT models. Two passes are needed
    per pivot, but both directions reuse the same architecture so the
    overhead is modest.
    """

    def __init__(self, pivots: List[str], device: str = "cuda"):
        _require_torch()
        import torch  # noqa: F401
        from transformers import MarianMTModel, MarianTokenizer

        self.pivots = pivots
        self.device = device
        self.fwd: Dict[str, Any] = {}
        self.bwd: Dict[str, Any] = {}
        for p in pivots:
            fwd_name = f"Helsinki-NLP/opus-mt-en-{p}"
            bwd_name = f"Helsinki-NLP/opus-mt-{p}-en"
            self.fwd[p] = (
                MarianTokenizer.from_pretrained(fwd_name),
                MarianMTModel.from_pretrained(fwd_name).to(device).eval(),
            )
            self.bwd[p] = (
                MarianTokenizer.from_pretrained(bwd_name),
                MarianMTModel.from_pretrained(bwd_name).to(device).eval(),
            )

    def _translate(self, text: str, tok, model, max_length: int = 512) -> str:
        import torch

        inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(
            self.device
        )
        with torch.no_grad():
            out = model.generate(**inputs, num_beams=4, max_length=max_length)
        return tok.decode(out[0], skip_special_tokens=True)

    def paraphrase(self, text: str, pivot: str) -> str:
        f_tok, f_mod = self.fwd[pivot]
        b_tok, b_mod = self.bwd[pivot]
        intermediate = self._translate(text, f_tok, f_mod)
        return self._translate(intermediate, b_tok, b_mod)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def _parse_config_name(config: str) -> Dict[str, Any]:
    parts = config.split("_")
    num_states = int(parts[0].replace("states", ""))
    overlap_pct = int(parts[1].replace("overlap", "").replace("pct", ""))
    return {"num_states": num_states, "overlap": overlap_pct / 100.0}


def main():
    parser = argparse.ArgumentParser(description="Paraphrase robustness attack")
    parser.add_argument("--data-dir", required=True, help="Path to curated dataset directory")
    parser.add_argument("--config", default="states7_overlap0pct", help="Watermark config to attack")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Tokenizer model")
    parser.add_argument(
        "--method", required=True, choices=["dipper", "backtranslation"], help="Paraphrase method"
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--lex", type=int, default=60, choices=[0, 20, 40, 60, 80, 100])
    parser.add_argument("--order", type=int, default=60, choices=[0, 20, 40, 60, 80, 100])
    parser.add_argument("--pivots", nargs="+", default=["de", "fr"], help="Backtranslation pivots")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (debug)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_file = data_dir / f"{args.config}.jsonl"
    if not config_file.exists():
        sys.exit(f"Config file not found: {config_file}")

    samples = []
    with open(config_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.limit:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} samples from {args.config}")

    cfg = _parse_config_name(args.config)
    chain_key = "soft_cycle"
    secret_key = "curated_wiki_dataset_2024"
    summary_file = data_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            s = json.load(f)
        if s.get("configs"):
            chain_key = s["configs"][0].get("chain_key", chain_key)

    detector = EnhancedMCLDetector(
        tokenizer_name=args.model,
        secret_key=secret_key,
        num_states=cfg["num_states"],
        chain_key=chain_key,
        overlap_ratio=cfg["overlap"],
        detection_threshold=0.5,
    )

    # Score originals first so we can report degradation
    orig_scores = []
    orig_detected = 0
    for s in samples:
        r = detector.detect(s["text"])
        orig_scores.append(r.chain_score)
        orig_detected += int(r.is_watermarked)
    orig_avg = float(np.mean(orig_scores))
    orig_rate = orig_detected / len(samples)
    print(f"Original  avg score = {orig_avg:.4f}, detection = {orig_rate*100:.1f}%")

    if args.method == "dipper":
        para = DipperParaphraser(device=args.device)
        runs = [{"setting": f"lex{args.lex}_order{args.order}",
                 "fn": lambda t: para.paraphrase(t, lex=args.lex, order=args.order)}]
    else:
        para = BackTranslator(pivots=args.pivots, device=args.device)
        runs = [
            {"setting": f"backtrans_{p}", "fn": (lambda p_: lambda t: para.paraphrase(t, pivot=p_))(p)}
            for p in args.pivots
        ]

    all_results = []
    for run in runs:
        print(f"\n=== Paraphrasing with {run['setting']} ===")
        scores: List[float] = []
        detected = 0
        examples: List[Dict[str, str]] = []
        for i, s in enumerate(samples):
            paraphrased = run["fn"](s["text"])
            r = detector.detect(paraphrased)
            scores.append(r.chain_score)
            detected += int(r.is_watermarked)
            if i < 3:
                examples.append({"original": s["text"][:200], "paraphrased": paraphrased[:200]})
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(samples)}] running avg = {float(np.mean(scores)):.4f}")
        avg = float(np.mean(scores))
        rate = detected / len(samples)
        print(f"  -> avg score = {avg:.4f} (was {orig_avg:.4f}), "
              f"detection = {rate*100:.1f}% (was {orig_rate*100:.1f}%)")
        all_results.append(
            {
                "setting": run["setting"],
                "avg_score": avg,
                "detection_rate": float(rate),
                "n_samples": len(samples),
                "n_detected": detected,
                "score_drop_vs_original": orig_avg - avg,
                "examples": examples,
            }
        )

    output = {
        "config": args.config,
        "num_states": cfg["num_states"],
        "overlap": cfg["overlap"],
        "chain_key": chain_key,
        "model": args.model,
        "method": args.method,
        "n_samples": len(samples),
        "original_avg_score": orig_avg,
        "original_detection_rate": orig_rate,
        "results": all_results,
    }
    out_file = data_dir / f"paraphrase_{args.config}_{args.method}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_file}")


if __name__ == "__main__":
    main()
