#!/usr/bin/env python
"""
Markov Chain-Lock (MCL) Watermarking - "The Clockwork Method"

A deterministic, robust watermarking method that forces tokens to follow
a hidden color cycle pattern: 0 -> 1 -> 2 -> 3 -> 0 -> 1 -> ...

Key advantages over LTW:
- 100% deterministic (greedy decoding)
- Trivial detection (just check color sequence)
- Robust to local edits
- Statistically impossible to reproduce by chance

Based on the Markov Chain-Lock concept for watermarking.
"""

import hashlib
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from tqdm import tqdm
import json


def get_token_color(token_id: int, secret_key: str = "mcl_secret", num_colors: int = 4) -> int:
    """
    Assign a deterministic "color" (0 to num_colors-1) to a token.
    
    Uses SHA-256 hash of (secret_key, token_id) for deterministic assignment.
    """
    data = f"{secret_key}-{token_id}".encode()
    hash_val = int(hashlib.sha256(data).hexdigest(), 16)
    return hash_val % num_colors


def precompute_color_masks(
    vocab_size: int, 
    secret_key: str = "mcl_secret", 
    num_colors: int = 4,
    device: str = "cpu"
) -> Dict[int, torch.Tensor]:
    """
    Precompute masks for each color to speed up generation.
    
    Returns a dict mapping color -> mask tensor where valid tokens have 0
    and invalid tokens have -inf.
    """
    print(f"Precomputing color masks for vocab size {vocab_size}...")
    
    # Assign colors to all tokens
    token_colors = np.array([get_token_color(t, secret_key, num_colors) for t in range(vocab_size)])
    
    masks = {}
    for color in range(num_colors):
        mask = torch.full((vocab_size,), float('-inf'), device=device)
        valid_indices = np.where(token_colors == color)[0]
        mask[valid_indices] = 0
        masks[color] = mask
        print(f"  Color {color}: {len(valid_indices)} tokens ({100*len(valid_indices)/vocab_size:.1f}%)")
    
    return masks


@dataclass
class MCLResult:
    """Result of MCL watermark detection."""
    is_watermarked: bool
    clockwork_score: float
    expected_score: float  # Random baseline (1/num_colors)
    sequence: List[int]
    matches: int
    total_pairs: int


class MCLGenerator:
    """
    Markov Chain-Lock Watermark Generator.
    
    Generates text with a hidden "clockwork" color pattern that cycles
    through colors deterministically: 0 -> 1 -> 2 -> 3 -> 0 -> ...
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        secret_key: str = "mcl_secret_key",
        num_colors: int = 4,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.secret_key = secret_key
        self.num_colors = num_colors
        
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get actual vocab size from model (tokenizer may report different size)
        self.vocab_size = self.model.config.vocab_size
        print(f"Model vocab size: {self.vocab_size}")
        
        # Precompute color masks for fast generation
        self.color_masks = precompute_color_masks(
            self.vocab_size,  # Use model's vocab size, not tokenizer's
            secret_key,
            num_colors,
            self.device
        )
        
        print(f"MCL Generator ready (colors={num_colors})")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,  # Use 1.0 for pure greedy within color
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate watermarked text using the Clockwork method.
        
        The text will follow a deterministic color cycle pattern.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]
        
        # Get color of last prompt token to start the clock
        last_token = input_ids[0, -1].item()
        last_color = get_token_color(last_token, self.secret_key, self.num_colors)
        
        color_sequence = [last_color]
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model predictions
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]
                
                # THE CLOCKWORK MECHANISM
                # 1. Determine REQUIRED color (next in cycle)
                required_color = (last_color + 1) % self.num_colors
                
                # 2. Apply mask to block all non-required-color tokens
                masked_logits = logits + self.color_masks[required_color]
                
                # 3. Apply temperature (optional)
                if temperature != 1.0:
                    masked_logits = masked_logits / temperature
                
                # 4. Pick best token (greedy within color class)
                next_token = torch.argmax(masked_logits, dim=-1)
                
                # 5. Update state
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                last_color = required_color
                color_sequence.append(required_color)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(
            input_ids[0, input_length:],
            skip_special_tokens=True
        )
        
        metadata = {
            "prompt": prompt,
            "watermarked": True,
            "method": "MCL",
            "num_colors": self.num_colors,
            "tokens_generated": len(color_sequence) - 1,
            "color_sequence": color_sequence[:20],  # First 20 for reference
        }
        
        return generated_text, metadata
    
    def generate_batch(
        self,
        prompts: List[str],
        n_per_prompt: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple watermarked samples."""
        results = []
        for prompt in tqdm(prompts, desc="Generating MCL watermarked text"):
            for _ in range(n_per_prompt):
                text, metadata = self.generate(prompt, **kwargs)
                results.append({
                    "id": f"mcl_{len(results):04d}",
                    "text": text,
                    **metadata
                })
        return results


class MCLDetector:
    """
    Markov Chain-Lock Watermark Detector.
    
    Detection is trivially simple: map tokens to colors and check
    if they follow the clockwork pattern 0 -> 1 -> 2 -> 3 -> 0 -> ...
    """
    
    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        secret_key: str = "mcl_secret_key",
        num_colors: int = 4,
        detection_threshold: float = 0.5  # Score above this = watermarked
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.secret_key = secret_key
        self.num_colors = num_colors
        self.threshold = detection_threshold
        self.expected_random = 1.0 / num_colors  # ~0.25 for 4 colors
        
        print(f"MCL Detector ready (threshold={detection_threshold}, random baseline={self.expected_random:.2f})")
    
    def detect(self, text: str) -> MCLResult:
        """
        Detect clockwork watermark in text.
        
        Returns the proportion of token pairs that follow the 
        expected color cycle pattern.
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 2:
            return MCLResult(
                is_watermarked=False,
                clockwork_score=0.0,
                expected_score=self.expected_random,
                sequence=[],
                matches=0,
                total_pairs=0
            )
        
        # Map tokens to colors
        colors = [get_token_color(t, self.secret_key, self.num_colors) for t in tokens]
        
        # Check clockwork pattern: each color should be (prev_color + 1) % num_colors
        matches = 0
        for i in range(1, len(colors)):
            expected = (colors[i - 1] + 1) % self.num_colors
            if colors[i] == expected:
                matches += 1
        
        total_pairs = len(colors) - 1
        score = matches / total_pairs if total_pairs > 0 else 0.0
        
        return MCLResult(
            is_watermarked=score > self.threshold,
            clockwork_score=score,
            expected_score=self.expected_random,
            sequence=colors[:20],  # First 20 for display
            matches=matches,
            total_pairs=total_pairs
        )
    
    def batch_detect(self, texts: List[str]) -> List[MCLResult]:
        """Detect watermarks in multiple texts."""
        return [self.detect(text) for text in texts]


if __name__ == "__main__":
    # Quick test
    print("\n" + "="*60)
    print("MCL Watermarking Test")
    print("="*60)
    
    # This is just for testing the color assignment
    print("\nTesting color assignment...")
    for i in range(10):
        color = get_token_color(i, "test_key")
        print(f"  Token {i} -> Color {color}")
