"""
Unit tests for the embedding extraction module.
"""

import pytest
import numpy as np
from ltw_watermark.embeddings import EmbeddingExtractor


class TestEmbeddingExtractor:
    """Tests for embedding extraction."""
    
    @pytest.fixture
    def extractor(self):
        return EmbeddingExtractor(
            model_name="all-MiniLM-L6-v2",
            cache_embeddings=True
        )
    
    def test_initialization(self, extractor):
        """Extractor should initialize with correct dimension."""
        assert extractor.embedding_dim == 384  # MiniLM dimension
        assert extractor.cache_embeddings is True
    
    def test_embed_text_shape(self, extractor):
        """Single text embedding should have correct shape."""
        text = "This is a test sentence."
        embedding = extractor.embed_text(text)
        
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64
    
    def test_embed_texts_shape(self, extractor):
        """Multiple text embeddings should have correct shape."""
        texts = ["First text.", "Second text.", "Third text."]
        embeddings = extractor.embed_texts(texts)
        
        assert embeddings.shape == (3, 384)
    
    def test_caching_works(self, extractor):
        """Caching should return same embedding for same text."""
        text = "Unique test sentence for caching."
        
        emb1 = extractor.embed_text(text)
        emb2 = extractor.embed_text(text)
        
        assert np.allclose(emb1, emb2)
    
    def test_different_texts_different_embeddings(self, extractor):
        """Different texts should produce different embeddings."""
        emb1 = extractor.embed_text("The cat sat on the mat.")
        emb2 = extractor.embed_text("Quantum physics is fascinating.")
        
        similarity = extractor.cosine_similarity(emb1, emb2)
        assert similarity < 0.9  # Should be somewhat different
    
    def test_similar_texts_similar_embeddings(self, extractor):
        """Similar texts should produce similar embeddings."""
        emb1 = extractor.embed_text("I love machine learning.")
        emb2 = extractor.embed_text("I enjoy machine learning a lot.")
        
        similarity = extractor.cosine_similarity(emb1, emb2)
        assert similarity > 0.7  # Should be quite similar
    
    def test_embed_words(self, extractor):
        """Word-level embedding should work."""
        text = "Hello world from Python"
        words, embeddings = extractor.embed_words(text)
        
        assert len(words) == 4
        assert embeddings.shape == (4, 384)
    
    def test_embed_sentences(self, extractor):
        """Sentence-level embedding should work."""
        text = "First sentence here. Second sentence there. Third one too!"
        sentences, embeddings = extractor.embed_sentences(text)
        
        assert len(sentences) == 3
        assert embeddings.shape == (3, 384)
    
    def test_clear_cache(self, extractor):
        """Cache should be clearable."""
        text = "Text for cache test."
        extractor.embed_text(text)
        assert len(extractor._cache) > 0
        
        extractor.clear_cache()
        assert len(extractor._cache) == 0
    
    def test_cosine_similarity_range(self, extractor):
        """Cosine similarity should be in [-1, 1]."""
        emb1 = np.random.randn(384)
        emb2 = np.random.randn(384)
        
        similarity = extractor.cosine_similarity(emb1, emb2)
        assert -1 <= similarity <= 1
    
    def test_cosine_similarity_identical(self, extractor):
        """Identical vectors should have similarity 1."""
        emb = np.random.randn(384)
        similarity = extractor.cosine_similarity(emb, emb)
        assert np.isclose(similarity, 1.0)
    
    def test_empty_text_handling(self, extractor):
        """Empty text should be handled."""
        words, embeddings = extractor.embed_words("")
        assert len(words) == 0


class TestEmbeddingNormalization:
    """Tests for embedding properties."""
    
    def test_embeddings_are_normalized(self):
        """Check if embeddings are unit vectors (model-dependent)."""
        extractor = EmbeddingExtractor()
        emb = extractor.embed_text("Test sentence")
        
        # Many sentence transformers return normalized embeddings
        # This is model-specific, so we just check it's not all zeros
        assert np.linalg.norm(emb) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
