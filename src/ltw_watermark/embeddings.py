"""
Embedding extraction module for Latent Trajectory Watermarking.

Uses sentence-transformers to extract semantic embeddings from text.
Supports both word-level and sentence-level embeddings with caching.
"""

import hashlib
from typing import Optional, List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingExtractor:
    """
    Extract semantic embeddings from text using sentence-transformers.
    
    Attributes:
        model_name: Name of the sentence-transformer model
        embedding_dim: Dimension of the embedding vectors
        device: Device to run the model on (cuda/cpu)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            model_name: Sentence-transformer model to use
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_embeddings: Whether to cache computed embeddings
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.cache_embeddings = cache_embeddings
        self._cache: dict = {}
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if self.cache_embeddings:
            key = self._cache_key(text)
            if key in self._cache:
                return self._cache[key]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if self.cache_embeddings:
            self._cache[key] = embedding
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Embeddings array of shape (n_texts, embedding_dim)
        """
        # Check cache for all texts
        if self.cache_embeddings:
            uncached_indices = []
            uncached_texts = []
            for i, text in enumerate(texts):
                key = self._cache_key(text)
                if key not in self._cache:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
            
            # Compute uncached embeddings
            if uncached_texts:
                new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
                for i, (text, emb) in enumerate(zip(uncached_texts, new_embeddings)):
                    self._cache[self._cache_key(text)] = emb
            
            # Reconstruct full array
            embeddings = np.array([self._cache[self._cache_key(t)] for t in texts])
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        return embeddings
    
    def embed_words(self, text: str) -> tuple[List[str], np.ndarray]:
        """
        Get word-level embeddings by embedding each word separately.
        
        Note: This is a simplified approach. For better word embeddings,
        consider using contextualized embeddings from BERT.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (words list, embeddings array)
        """
        words = text.split()
        if not words:
            return [], np.array([])
        
        embeddings = self.embed_texts(words)
        return words, embeddings
    
    def embed_sentences(self, text: str) -> tuple[List[str], np.ndarray]:
        """
        Get sentence-level embeddings.
        
        Args:
            text: Input text with multiple sentences
            
        Returns:
            Tuple of (sentences list, embeddings array)
        """
        # Simple sentence splitting (could be improved with nltk/spacy)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [], np.array([])
        
        embeddings = self.embed_texts(sentences)
        return sentences, embeddings
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache = {}
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
