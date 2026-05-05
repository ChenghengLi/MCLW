#!/usr/bin/env python
"""
Enhanced MCL (Markov Chain-Lock) Watermarking with Soft Partitions

Improvements over basic MCL:
1. Configurable Markov chain transitions (custom state machine via key)
2. Soft partitions: overlapping token sets instead of hard 25% splits
3. Intersection percentage control for softer generation
4. Multiple chain configurations via transition matrix

The key innovation: Instead of forcing tokens into disjoint color sets,
we allow configurable overlap between sets, making generation smoother
while maintaining detectability.

Parameters:
- num_states: Number of states in the chain (e.g., 4)
- transition_key: Secret key that determines state transitions
- overlap_ratio: How much sets overlap (0 = hard partition, 1 = full overlap)
- temperature: Sampling temperature within valid tokens

Example transition matrices:
- "clockwork": 0->1->2->3->0 (original)
- "binary": 0->1->0->1 (two states)
- "random_walk": each state can go to 2 neighbors
"""

import hashlib
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from tqdm import tqdm
import json


def generate_transition_matrix(
    num_states: int = 4,
    chain_key: str = "clockwork",
    secret_key: str = "mcl_secret"
) -> np.ndarray:
    """
    Generate a Markov chain transition matrix from a key.
    
    The matrix defines which state transitions are allowed.
    matrix[i][j] = 1 means transition from state i to state j is allowed.
    
    Built-in patterns:
    - "clockwork": strict cycle 0->1->2->3->0
    - "binary": alternating 0->1->0->1
    - "soft_cycle": cycle with random neighbor allowed
    - Any other key: generates random deterministic transitions
    """
    matrix = np.zeros((num_states, num_states), dtype=np.float32)
    
    if chain_key == "clockwork":
        # Strict clockwork: each state goes to next
        for i in range(num_states):
            next_state = (i + 1) % num_states
            matrix[i][next_state] = 1.0
            
    elif chain_key == "binary":
        # Binary alternation
        matrix[0][1] = 1.0
        matrix[1][0] = 1.0
        for i in range(2, num_states):
            matrix[i][1 - (i % 2)] = 1.0
            
    elif chain_key == "soft_cycle":
        # Soft Cycle Matrix (Example 3.5 / def in paper):
        # T^soft_ij = 2/3 if j ≡ i+1 (mod S), 1/3 if j ≡ i+2 (mod S), 0 otherwise
        # Properties: stochastic, aperiodic, doubly stochastic, p_random = 2/S.
        # Detection only checks whether T_ij > 0, so the specific 2/3 / 1/3 split
        # affects sampling but not the chain_score; we use exact fractions to
        # match the paper's Definition exactly.
        for i in range(num_states):
            next1 = (i + 1) % num_states
            next2 = (i + 2) % num_states
            matrix[i][next1] = 2.0 / 3.0  # Immediate successor
            matrix[i][next2] = 1.0 / 3.0  # Skip one state
            
    else:
        # Generate deterministic transitions from key
        rng = np.random.default_rng(
            int(hashlib.sha256(f"{secret_key}-{chain_key}".encode()).hexdigest(), 16) % (2**32)
        )
        # Each state allows 1-2 random next states
        for i in range(num_states):
            n_allowed = rng.integers(1, 3)  # 1 or 2 allowed transitions
            allowed = rng.choice(num_states, size=n_allowed, replace=False)
            for j in allowed:
                matrix[i][j] = 1.0 / n_allowed
    
    # Normalize rows (make proper transition probabilities)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums
    
    return matrix


def get_token_state_soft(
    token_id: int,
    num_states: int,
    secret_key: str,
    overlap_ratio: float = 0.0
) -> List[int]:
    """
    Get the state(s) a token belongs to with soft partitioning.
    
    With overlap_ratio=0: each token belongs to exactly 1 state (hard partition)
    With overlap_ratio=0.5: each token belongs to 1-2 states (50% overlap)
    With overlap_ratio=1.0: each token belongs to all states
    
    Returns list of states this token is valid for.
    """
    # Primary state (always assigned)
    data = f"{secret_key}-{token_id}".encode()
    hash_val = int(hashlib.sha256(data).hexdigest(), 16)
    primary_state = hash_val % num_states
    
    if overlap_ratio <= 0:
        return [primary_state]
    
    # Secondary state assignment based on overlap ratio
    states = [primary_state]
    secondary_hash = int(hashlib.sha256(f"{secret_key}-secondary-{token_id}".encode()).hexdigest(), 16)
    
    # Probability of belonging to additional states
    for i in range(num_states):
        if i != primary_state:
            # Use hash to deterministically decide if token belongs to this state
            state_hash = int(hashlib.sha256(f"{secret_key}-{token_id}-{i}".encode()).hexdigest(), 16)
            if (state_hash % 100) < (overlap_ratio * 100):
                states.append(i)
    
    return states


def precompute_soft_masks(
    vocab_size: int,
    num_states: int,
    secret_key: str,
    overlap_ratio: float,
    device: str = "cpu"
) -> Dict[int, torch.Tensor]:
    """
    Precompute soft masks for each state.
    
    With soft partitions, tokens can belong to multiple states,
    making generation smoother while maintaining detectability.
    """
    print(f"Precomputing soft masks (overlap={overlap_ratio:.0%})...")
    
    # Create mask for each state
    masks = {}
    state_counts = [0] * num_states
    
    for state in range(num_states):
        mask = torch.full((vocab_size,), float('-inf'), device=device)
        
        for token_id in range(vocab_size):
            token_states = get_token_state_soft(token_id, num_states, secret_key, overlap_ratio)
            if state in token_states:
                mask[token_id] = 0
                state_counts[state] += 1
        
        masks[state] = mask
    
    for state in range(num_states):
        pct = 100 * state_counts[state] / vocab_size
        print(f"  State {state}: {state_counts[state]} tokens ({pct:.1f}%)")
    
    return masks


@dataclass
class EnhancedMCLResult:
    """Result of enhanced MCL watermark detection."""
    is_watermarked: bool
    chain_score: float
    expected_random: float
    state_sequence: List[int]
    valid_transitions: int
    total_transitions: int
    perplexity: float = 0.0  # Added for comparison


class EnhancedMCLGenerator:
    """
    Enhanced Markov Chain-Lock Generator with soft partitions.
    
    Features:
    - Configurable transition matrix (chain_key parameter)
    - Soft partitions with overlap_ratio
    - Perplexity measurement
    - Deterministic or sampled generation
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        secret_key: str = "enhanced_mcl_key",
        num_states: int = 4,
        chain_key: str = "clockwork",  # Pattern: clockwork, binary, soft_cycle, or custom
        overlap_ratio: float = 0.0,    # 0 = hard partition, 0.3 = 30% overlap
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.secret_key = secret_key
        self.num_states = num_states
        self.chain_key = chain_key
        self.overlap_ratio = overlap_ratio
        
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab_size = self.model.config.vocab_size
        
        # Generate transition matrix
        self.transition_matrix = generate_transition_matrix(num_states, chain_key, secret_key)
        print(f"Transition matrix ({chain_key}):")
        print(self.transition_matrix)
        
        # Precompute soft masks
        self.state_masks = precompute_soft_masks(
            self.vocab_size,
            num_states,
            secret_key,
            overlap_ratio,
            self.device
        )
        
        print(f"Enhanced MCL Generator ready (states={num_states}, overlap={overlap_ratio:.0%})")
    
    def get_allowed_next_states(self, current_state: int) -> List[int]:
        """Get list of valid next states from transition matrix."""
        transitions = self.transition_matrix[current_state]
        return [i for i, prob in enumerate(transitions) if prob > 0]
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        greedy: bool = True,  # True = argmax, False = sample
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate watermarked text with enhanced MCL.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]
        
        # Get initial state from last prompt token
        last_token = input_ids[0, -1].item()
        token_states = get_token_state_soft(last_token, self.num_states, self.secret_key, self.overlap_ratio)
        current_state = token_states[0]  # Use primary state
        
        state_sequence = [current_state]
        log_probs = []
        # Per-step KL of P_MCL || P_M, where Z_s is the unmasked prob mass on
        # the allowed-state vocabulary subset. Aggregated below.
        kl_per_step = []
        past_key_values = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                # KV cache: full pass on the prompt for step 0, then feed only
                # the newly generated token thereafter. This drops generation
                # complexity from O(n^2) to O(n) and is ~10x faster at n=512.
                if past_key_values is None:
                    outputs = self.model(input_ids, use_cache=True)
                else:
                    outputs = self.model(
                        input_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                # Get allowed next states
                allowed_states = self.get_allowed_next_states(current_state)

                # Combine masks for allowed states
                combined_mask = torch.full((self.vocab_size,), float('-inf'), device=self.device)
                for state in allowed_states:
                    # Union of allowed token sets
                    combined_mask = torch.maximum(combined_mask, self.state_masks[state])

                # Empirical Z_s = sum of unmasked prob mass on the allowed set;
                # KL(P_MCL || P_M) = log(1/Z_s). This is what Theorem 4 bounds
                # by log(S). Compute BEFORE temperature scaling so the KL is
                # against the model's natural distribution.
                base_probs = torch.softmax(logits.float(), dim=-1)
                allowed_indicator = (combined_mask > float('-inf')).float()
                Z_s = (base_probs * allowed_indicator).sum(dim=-1).clamp(min=1e-30)
                kl_per_step.append(float((-torch.log(Z_s)).mean().item()))

                # Apply mask
                masked_logits = logits + combined_mask

                # Apply temperature
                if temperature != 1.0:
                    masked_logits = masked_logits / temperature

                # Select next token
                if greedy:
                    next_token = torch.argmax(masked_logits, dim=-1)
                else:
                    probs = torch.softmax(masked_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Track log probability for perplexity
                log_prob = torch.log(base_probs[0, next_token.item()] + 1e-10)
                log_probs.append(log_prob.item())
                
                # Update state
                next_token_id = next_token.item()
                token_states = get_token_state_soft(next_token_id, self.num_states, self.secret_key, self.overlap_ratio)
                
                # Find which state this token actually belongs to (for tracking)
                for state in allowed_states:
                    if state in token_states:
                        current_state = state
                        break
                
                state_sequence.append(current_state)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Calculate perplexity
        perplexity = np.exp(-np.mean(log_probs)) if log_probs else 0.0

        # Empirical KL bound diagnostic for Theorem 4: log(S) is the
        # theoretical upper bound on D_KL(P_MCL || P_M); we report mean and
        # 95th-percentile per-step KL so a paper-side analysis can compare
        # against log(S).
        if kl_per_step:
            kl_arr = np.asarray(kl_per_step, dtype=np.float64)
            empirical_mean_kl = float(kl_arr.mean())
            empirical_p95_kl = float(np.quantile(kl_arr, 0.95))
            empirical_max_kl = float(kl_arr.max())
        else:
            empirical_mean_kl = empirical_p95_kl = empirical_max_kl = 0.0

        generated_text = self.tokenizer.decode(
            input_ids[0, input_length:],
            skip_special_tokens=True
        )

        metadata = {
            "prompt": prompt,
            "watermarked": True,
            "method": "EnhancedMCL",
            "num_states": self.num_states,
            "chain_key": self.chain_key,
            "overlap_ratio": self.overlap_ratio,
            "tokens_generated": len(state_sequence) - 1,
            "state_sequence": state_sequence[:20],
            "perplexity": perplexity,
            "empirical_mean_kl_nats": empirical_mean_kl,
            "empirical_p95_kl_nats": empirical_p95_kl,
            "empirical_max_kl_nats": empirical_max_kl,
            "log_S_bound_nats": float(np.log(self.num_states)),
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
        for prompt in tqdm(prompts, desc="Generating Enhanced MCL text"):
            for _ in range(n_per_prompt):
                text, metadata = self.generate(prompt, **kwargs)
                results.append({
                    "id": f"emcl_{len(results):04d}",
                    "text": text,
                    **metadata
                })
        return results


class EnhancedMCLDetector:
    """
    Enhanced MCL Detector with configurable transition validation.
    
    Checks if token state sequences follow the expected Markov chain transitions.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        secret_key: str = "enhanced_mcl_key",
        num_states: int = 4,
        chain_key: str = "clockwork",
        overlap_ratio: float = 0.0,
        detection_threshold: float = 0.5
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.secret_key = secret_key
        self.num_states = num_states
        self.chain_key = chain_key
        self.overlap_ratio = overlap_ratio
        self.threshold = detection_threshold
        
        # Generate same transition matrix
        self.transition_matrix = generate_transition_matrix(num_states, chain_key, secret_key)
        
        # Calculate expected random baseline
        # For random text, probability of valid transition = sum of non-zero entries / num_states
        self.expected_random = np.mean(self.transition_matrix > 0)
        
        print(f"Enhanced MCL Detector ready (threshold={detection_threshold}, baseline={self.expected_random:.2f})")
    
    def detect(self, text: str) -> EnhancedMCLResult:
        """
        Detect watermark by checking transition validity.
        """
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) < 2:
            return EnhancedMCLResult(
                is_watermarked=False,
                chain_score=0.0,
                expected_random=self.expected_random,
                state_sequence=[],
                valid_transitions=0,
                total_transitions=0
            )
        
        # Get state sequence (using primary state)
        states = []
        for token_id in tokens:
            token_states = get_token_state_soft(token_id, self.num_states, self.secret_key, self.overlap_ratio)
            states.append(token_states[0])  # Primary state
        
        # Check transitions against allowed transitions
        valid = 0
        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            if self.transition_matrix[current][next_state] > 0:
                valid += 1
        
        total = len(states) - 1
        score = valid / total if total > 0 else 0.0
        
        return EnhancedMCLResult(
            is_watermarked=score > self.threshold,
            chain_score=score,
            expected_random=self.expected_random,
            state_sequence=states[:20],
            valid_transitions=valid,
            total_transitions=total
        )
    
    def batch_detect(self, texts: List[str]) -> List[EnhancedMCLResult]:
        """Detect watermarks in multiple texts."""
        return [self.detect(text) for text in texts]


def calculate_perplexity(
    text: str,
    model,
    tokenizer,
    device: str = "cuda"
) -> float:
    """Calculate perplexity of text using the model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    return np.exp(loss.item())


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Enhanced MCL Watermarking Test")
    print("="*60)
    
    # Test transition matrix generation
    print("\nTesting transition matrices...")
    
    for key in ["clockwork", "binary", "soft_cycle", "custom_key_123"]:
        matrix = generate_transition_matrix(4, key, "test_secret")
        print(f"\n{key}:")
        print(matrix)
