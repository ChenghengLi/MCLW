"""
Gated MCL generator (v3) -- replaces entropy-only gating with a pluggable Gate.

Any `Gate` from ltw_watermark.gates controls per-position watermark insertion.
Records per-position metadata:
  - token_id, state, entropy H, surprisal-gap Δ, p1, p2
  - watermarked (bool, from the gate)
  - valid_transition (bool, wrt MCL transition matrix)

This is the v3 drop-in. v2's EntropyGatedMCLGenerator is now a thin wrapper
around this that pins the gate to GateEntropyLow for backward compat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ltw_watermark.enhanced_mcl import (
    generate_transition_matrix,
    get_token_state_soft,
    precompute_soft_masks,
)
from ltw_watermark.gates import Gate, GateEntropyLow, logits_stats


@dataclass
class PositionRecord:
    token_id: int
    state: int
    entropy: float
    delta: float
    p1: float
    p2: float
    watermarked: bool
    valid_transition: bool


@dataclass
class GenerationResult:
    text: str
    prompt: str
    token_ids: List[int]
    positions: List[PositionRecord]
    rho_empirical: float
    gate_name: str
    perplexity: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "prompt": self.prompt,
            "token_ids": self.token_ids,
            "positions": [p.__dict__ for p in self.positions],
            "rho_empirical": self.rho_empirical,
            "gate_name": self.gate_name,
            "perplexity": self.perplexity,
            **self.meta,
        }


class GatedMCLGenerator:
    """
    Markov-Chain-Lock watermark generator gated by an arbitrary Gate.

    Usage:
        gen = GatedMCLGenerator(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            num_states=5, chain_key="clockwork",
            gate=GateDelta(tau=0.1),
        )
        res = gen.generate("Explain X in a comprehensive way.")
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        secret_key: str = "mclw_gated_v3_2026",
        num_states: int = 5,
        chain_key: str = "clockwork",
        overlap_ratio: float = 0.0,
        gate: Optional[Gate] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.secret_key = secret_key
        self.num_states = num_states
        self.chain_key = chain_key
        self.overlap_ratio = overlap_ratio
        self.gate = gate or GateEntropyLow(tau=1.5)

        print(f"[GatedMCL] Loading {model_name}...")
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
        print(f"[GatedMCL] S={num_states} chain={chain_key} gate={self.gate.name}")

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

        last = int(input_ids[0, -1].item())
        current_state = get_token_state_soft(last, self.num_states, self.secret_key, self.overlap_ratio)[0]

        positions: List[PositionRecord] = []
        log_probs: List[float] = []
        token_ids: List[int] = []

        past = None
        cur = input_ids
        for step in range(max_new_tokens):
            out = self.model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()

            stats = logits_stats(logits)
            # Gate may be GatePSurviveOracle (needs step); handle both.
            try:
                watermark_on = bool(self.gate(logits, step=step))
            except TypeError:
                watermark_on = bool(self.gate(logits))

            allowed = self._allowed_states(current_state)
            if watermark_on:
                combined = torch.full((self.vocab_size,), float("-inf"), device=self.device)
                for s in allowed:
                    combined = torch.maximum(combined, self.state_masks[s])
                eff = logits + combined
            else:
                eff = logits

            if temperature != 1.0:
                eff = eff / temperature

            if greedy:
                next_tok = int(torch.argmax(eff).item())
            else:
                probs = F.softmax(eff, dim=-1)
                next_tok = int(torch.multinomial(probs, 1).item())

            # PPL log-probs from ORIGINAL (unmasked) distribution
            orig_lp = F.log_softmax(logits, dim=-1)
            log_probs.append(float(orig_lp[next_tok].item()))

            # State of the chosen token
            next_states = get_token_state_soft(next_tok, self.num_states, self.secret_key, self.overlap_ratio)
            chosen = next_states[0]
            for s in allowed:
                if s in next_states:
                    chosen = s
                    break
            valid = self.transition_matrix[current_state][chosen] > 0

            positions.append(PositionRecord(
                token_id=next_tok,
                state=chosen,
                entropy=stats["entropy"],
                delta=stats["delta"],
                p1=stats["p1"],
                p2=stats["p2"],
                watermarked=bool(watermark_on),
                valid_transition=bool(valid),
            ))
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
            gate_name=self.gate.name,
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
