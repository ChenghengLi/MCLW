"""
LTW Generator - Full Watermark Injection During Text Generation

This module implements the complete Latent Trajectory Watermarking (LTW) system
that injects watermarks during text generation by biasing token selection
towards tokens that follow the secret rotation pattern in embedding space.

Key Components:
1. Token-level embedding computation
2. Rotation alignment scoring
3. Logit manipulation to bias towards high-alignment tokens
"""

import hashlib
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from ltw_watermark.rotation import OrthogonalRotation


class LTWLogitsProcessor:
    """
    Custom logits processor that biases token selection towards
    tokens following the secret rotation pattern.
    
    This is the core of the LTW watermarking system.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        embedder: SentenceTransformer,
        rotation: OrthogonalRotation,
        watermark_strength: float = 2.0,
        top_k_candidates: int = 50,
        device: str = "cpu"
    ):
        """
        Initialize the LTW logits processor.
        
        Args:
            tokenizer: The tokenizer for decoding tokens
            embedder: Sentence transformer for computing embeddings
            rotation: The secret rotation for watermarking
            watermark_strength: Bias strength (higher = stronger watermark)
            top_k_candidates: Number of top tokens to consider for alignment
            device: Device for computation
        """
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.rotation = rotation
        self.watermark_strength = watermark_strength
        self.top_k_candidates = top_k_candidates
        self.device = device
        
        # Cache for previous token embedding
        self._prev_embedding: Optional[np.ndarray] = None
        self._context_text: str = ""
        
        # Precompute token embeddings for vocabulary (optional optimization)
        self._token_embeddings_cache: Dict[int, np.ndarray] = {}
    
    def reset(self):
        """Reset state for new generation."""
        self._prev_embedding = None
        self._context_text = ""
    
    def get_token_embedding(self, token_id: int) -> np.ndarray:
        """Get embedding for a single token."""
        if token_id in self._token_embeddings_cache:
            return self._token_embeddings_cache[token_id]
        
        token_text = self.tokenizer.decode([token_id])
        embedding = self.embedder.encode(token_text, convert_to_numpy=True)
        
        self._token_embeddings_cache[token_id] = embedding
        return embedding
    
    def get_context_embedding(self, text: str) -> np.ndarray:
        """Get embedding for current context."""
        return self.embedder.encode(text, convert_to_numpy=True)
    
    def compute_alignment_scores(
        self,
        candidate_token_ids: List[int],
        prev_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Compute alignment scores for candidate tokens.
        
        Higher score = better alignment with rotated previous embedding.
        """
        # Compute expected next embedding (rotated previous)
        expected_embedding = self.rotation.rotate(prev_embedding)
        expected_norm = expected_embedding / (np.linalg.norm(expected_embedding) + 1e-8)
        
        scores = []
        for token_id in candidate_token_ids:
            token_embedding = self.get_token_embedding(token_id)
            token_norm = token_embedding / (np.linalg.norm(token_embedding) + 1e-8)
            
            # Cosine similarity with expected (rotated) embedding
            alignment = np.dot(expected_norm, token_norm)
            scores.append(alignment)
        
        return np.array(scores)
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Modify logits to inject watermark.
        
        Args:
            input_ids: Current input token IDs [batch, seq_len]
            logits: Logits from the model [batch, vocab_size]
            
        Returns:
            Modified logits with watermark bias
        """
        batch_size = input_ids.shape[0]
        modified_logits = logits.clone()
        
        for b in range(batch_size):
            # Get current context
            current_ids = input_ids[b].tolist()
            current_text = self.tokenizer.decode(current_ids)
            
            # Get embedding of current context
            if len(current_text.strip()) > 0:
                current_embedding = self.get_context_embedding(current_text)
                
                if self._prev_embedding is not None:
                    # Get top-k candidate tokens
                    batch_logits = logits[b]
                    top_k_values, top_k_indices = torch.topk(
                        batch_logits, 
                        min(self.top_k_candidates, batch_logits.shape[0])
                    )
                    
                    # Compute alignment scores for candidates
                    candidate_ids = top_k_indices.cpu().tolist()
                    alignment_scores = self.compute_alignment_scores(
                        candidate_ids, self._prev_embedding
                    )
                    
                    # Convert to bias (scale and add to logits)
                    # Higher alignment = positive bias
                    alignment_bias = torch.tensor(
                        alignment_scores * self.watermark_strength,
                        device=self.device,
                        dtype=logits.dtype
                    )
                    
                    # Apply bias to top-k tokens
                    modified_logits[b, top_k_indices] += alignment_bias
                
                # Update previous embedding for next step
                self._prev_embedding = current_embedding
            
        return modified_logits


class LTWGenerator:
    """
    Text generator with full LTW watermark injection.
    
    This generator modifies the language model's output distribution
    at each step to bias token selection towards tokens that align
    with the secret rotation pattern.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        secret_key: str = "secret-key",
        rotation_strength: float = 0.3,
        watermark_strength: float = 2.0,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the LTW generator.
        
        Args:
            model_name: HuggingFace model name for text generation
            secret_key: Secret key for watermark (keep this private!)
            rotation_strength: Rotation intensity in embedding space
            watermark_strength: How strongly to bias towards watermark
            embedding_model: Sentence transformer for embeddings
            device: Device to use (None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.secret_key = secret_key
        
        # Load language model
        print(f"Loading language model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Setup rotation
        self.rotation = OrthogonalRotation(
            secret_key=secret_key,
            dim=embedding_dim,
            rotation_strength=rotation_strength
        )
        
        # Setup logits processor
        self.logits_processor = LTWLogitsProcessor(
            tokenizer=self.tokenizer,
            embedder=self.embedder,
            rotation=self.rotation,
            watermark_strength=watermark_strength,
            device=self.device
        )
        
        self.watermark_strength = watermark_strength
        print(f"LTW Generator initialized (watermark_strength={watermark_strength})")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate watermarked text.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to sample (vs greedy)
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        # Reset processor state
        self.logits_processor.reset()
        
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Custom generation loop with watermark injection
        generated_ids = inputs['input_ids'].clone()
        
        alignment_scores = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get model output
                outputs = self.model(generated_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply watermark bias
                modified_logits = self.logits_processor(
                    generated_ids, next_token_logits
                )
                
                # Apply temperature
                if temperature > 0:
                    modified_logits = modified_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = modified_logits < torch.topk(modified_logits, top_k)[0][..., -1, None]
                    modified_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(modified_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    modified_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(modified_logits, dim=-1)
                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Track alignment (for debugging/analysis)
                if self.logits_processor._prev_embedding is not None:
                    token_emb = self.logits_processor.get_token_embedding(next_token.item())
                    alignment = self.rotation.compute_alignment_score(
                        self.logits_processor._prev_embedding, token_emb
                    )
                    alignment_scores.append(alignment)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_length:], 
            skip_special_tokens=True
        )
        
        metadata = {
            "prompt": prompt,
            "watermarked": True,
            "watermark_strength": self.watermark_strength,
            "secret_key_hash": hash(self.secret_key) % 100000,
            "tokens_generated": generated_ids.shape[1] - input_length,
            "mean_alignment": float(np.mean(alignment_scores)) if alignment_scores else 0.0,
        }
        
        return generated_text, metadata
    
    def generate_batch(
        self,
        prompts: List[str],
        n_per_prompt: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple watermarked samples."""
        from tqdm import tqdm
        
        results = []
        for prompt in tqdm(prompts, desc="Generating watermarked text"):
            for i in range(n_per_prompt):
                text, metadata = self.generate(prompt, **kwargs)
                results.append({
                    "id": f"wm_{len(results):04d}",
                    "text": text,
                    **metadata
                })
        
        return results


class NonWatermarkedGenerator:
    """Standard generator without watermarking (control group)."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text without watermark."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        metadata = {
            "prompt": prompt,
            "watermarked": False,
        }
        
        return generated_text, metadata
    
    def generate_batch(
        self,
        prompts: List[str],
        n_per_prompt: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple non-watermarked samples."""
        from tqdm import tqdm
        
        results = []
        for prompt in tqdm(prompts, desc="Generating non-watermarked text"):
            for i in range(n_per_prompt):
                text, metadata = self.generate(prompt, **kwargs)
                results.append({
                    "id": f"nwm_{len(results):04d}",
                    "text": text,
                    **metadata
                })
        
        return results
