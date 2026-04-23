"""
Entropy-Gated MCL Generator.

Extends EnhancedMCLGenerator with entropy gating: at each position, compute
the model's next-token entropy H_i, and only apply the MCL state mask when
H_i < tau_H. High-entropy positions generate freely (no watermark).

Records per-position metadata needed for Experiment 2 (entropy-vs-survival):
- token_id
- state (SHA-based partition label)
- entropy H_i (nats)
- W_i (1 if watermarked at this position, 0 otherwise)
- valid_transition (did (state_{i-1}, state_i) follow T?)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ltw_watermark.enhanced_mcl import (
    generate_transition_matrix,
    get_token_state_soft,
    precompute_soft_masks,
)


@dataclass
class PositionRecord:
    """Per-token record kept by the entropy-gated generator."""
    token_id: int
    state: int
    entropy: float
    watermarked: bool
    valid_transition: bool


@dataclass
class GenerationResult:
    text: str
    prompt: str
    token_ids: List[int]
    positions: List[PositionRecord]
    rho_empirical: float
    tau_H: float
    perplexity: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "prompt": self.prompt,
            "token_ids": self.token_ids,
            "positions": [p.__dict__ for p in self.positions],
            "rho_empirical": self.rho_empirical,
            "tau_H": self.tau_H,
            "perplexity": self.perplexity,
            **self.meta,
        }


def entropy_from_logits(logits: torch.Tensor) -> float:
    """Shannon entropy (nats) of softmax(logits). logits: (vocab,)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return float(-(probs * log_probs).sum().item())


class EntropyGatedMCLGenerator:
    """
    Low-entropy-gated Markov-chain-lock watermark.

    At position i:
      - Compute H_i from model logits.
      - If H_i < tau_H: MASK to allowed-next-state tokens (watermark ON).
        Otherwise: generate freely from full distribution (watermark OFF).
      - Sample / argmax among allowed tokens.
      - Record metadata.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        secret_key: str = "mclw_entropy_gated_2026",
        num_states: int = 5,
        chain_key: str = "clockwork",
        overlap_ratio: float = 0.0,
        tau_H: float = 1.5,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.secret_key = secret_key
        self.num_states = num_states
        self.chain_key = chain_key
        self.overlap_ratio = overlap_ratio
        self.tau_H = tau_H

        print(f"[EntropyGatedMCL] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = self.model.config.vocab_size
        self.transition_matrix = generate_transition_matrix(num_states, chain_key, secret_key)
        self.state_masks = precompute_soft_masks(
            self.vocab_size, num_states, secret_key, overlap_ratio, self.device
        )
        print(f"[EntropyGatedMCL] S={num_states} tau_H={tau_H} chain={chain_key}")

    def _allowed_states(self, current_state: int) -> List[int]:
        return [i for i, p in enumerate(self.transition_matrix[current_state]) if p > 0]

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> GenerationResult:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        prompt_len = input_ids.shape[1]

        last_token_id = int(input_ids[0, -1].item())
        current_state = get_token_state_soft(
            last_token_id, self.num_states, self.secret_key, self.overlap_ratio
        )[0]

        positions: List[PositionRecord] = []
        log_probs: List[float] = []
        token_ids: List[int] = []

        past = None
        cur = input_ids
        for _ in range(max_new_tokens):
            out = self.model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()  # (vocab,)

            H_i = entropy_from_logits(logits)
            watermark_on = H_i < self.tau_H

            allowed = self._allowed_states(current_state)
            if watermark_on:
                combined = torch.full((self.vocab_size,), float("-inf"), device=self.device)
                for s in allowed:
                    combined = torch.maximum(combined, self.state_masks[s])
                eff_logits = logits + combined
            else:
                eff_logits = logits

            if temperature != 1.0:
                eff_logits = eff_logits / temperature

            if greedy:
                next_tok = int(torch.argmax(eff_logits).item())
            else:
                probs = F.softmax(eff_logits, dim=-1)
                next_tok = int(torch.multinomial(probs, 1).item())

            # log-prob under the ORIGINAL (unmasked) distribution for PPL
            orig_log_probs = F.log_softmax(logits, dim=-1)
            log_probs.append(float(orig_log_probs[next_tok].item()))

            # state of next token
            next_states = get_token_state_soft(
                next_tok, self.num_states, self.secret_key, self.overlap_ratio
            )
            # pick primary-consistent state
            chosen = next_states[0]
            for s in allowed:
                if s in next_states:
                    chosen = s
                    break
            valid = self.transition_matrix[current_state][chosen] > 0

            positions.append(
                PositionRecord(
                    token_id=next_tok,
                    state=chosen,
                    entropy=H_i,
                    watermarked=bool(watermark_on),
                    valid_transition=bool(valid),
                )
            )
            token_ids.append(next_tok)
            current_state = chosen

            cur = torch.tensor([[next_tok]], device=self.device)

            if next_tok == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        ppl = float(np.exp(-np.mean(log_probs))) if log_probs else 0.0
        rho = float(np.mean([p.watermarked for p in positions])) if positions else 0.0

        return GenerationResult(
            text=text,
            prompt=prompt,
            token_ids=token_ids,
            positions=positions,
            rho_empirical=rho,
            tau_H=self.tau_H,
            perplexity=ppl,
            meta={
                "model_name": getattr(self.model.config, "_name_or_path", "unknown"),
                "num_states": self.num_states,
                "chain_key": self.chain_key,
                "overlap_ratio": self.overlap_ratio,
                "secret_key": self.secret_key,
                "n_tokens": len(token_ids),
            },
        )


def pilot_measure_entropy_quantiles(
    generator: EntropyGatedMCLGenerator,
    prompts: List[str],
    max_new_tokens: int = 100,
    quantiles: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> Dict[str, float]:
    """
    Generate a small pilot with watermark OFF (tau_H = inf) to measure the
    empirical distribution of entropies on this model/prompts.
    Returns percentile values suitable as candidate tau_H thresholds.
    """
    saved_tau = generator.tau_H
    generator.tau_H = float("inf")  # never watermark, just measure H
    all_H: List[float] = []
    for p in prompts:
        res = generator.generate(p, max_new_tokens=max_new_tokens, greedy=True)
        all_H.extend([pos.entropy for pos in res.positions])
    generator.tau_H = saved_tau
    arr = np.array(all_H)
    return {f"Q{int(q*100)}": float(np.quantile(arr, q)) for q in quantiles}
