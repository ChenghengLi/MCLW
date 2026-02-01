"""
Unit tests for the watermark detection module.
"""

import pytest
import numpy as np
from ltw_watermark.watermark import LTWWatermarker, WatermarkResult


class TestLTWWatermarker:
    """Tests for LTW watermark detection."""
    
    @pytest.fixture
    def detector(self):
        return LTWWatermarker(
            secret_key="test-secret-key",
            rotation_strength=0.3,
            detection_threshold=0.1,
            use_differential_scoring=True,
            unit="sentence"
        )
    
    def test_initialization(self, detector):
        """Detector should initialize correctly."""
        assert detector.secret_key == "test-secret-key"
        assert detector.threshold == 0.1
        assert detector.use_differential_scoring is True
    
    def test_detect_returns_result(self, detector):
        """Detection should return a WatermarkResult."""
        text = "This is a test sentence. Here is another one. And a third."
        result = detector.detect(text)
        
        assert isinstance(result, WatermarkResult)
        assert isinstance(result.is_watermarked, bool)
        assert isinstance(result.confidence, float)
        assert isinstance(result.differential_score, float)
    
    def test_short_text_handling(self, detector):
        """Very short text should be handled gracefully."""
        result = detector.detect("Too short")
        
        assert result.is_watermarked is False
        assert "error" in result.details or result.details.get("units", 0) < 2
    
    def test_different_keys_different_results(self):
        """Different secret keys should produce different scores."""
        text = "The quick brown fox jumps over the lazy dog. It was a beautiful day."
        
        detector1 = LTWWatermarker(secret_key="key-1")
        detector2 = LTWWatermarker(secret_key="key-2")
        
        result1 = detector1.detect(text)
        result2 = detector2.detect(text)
        
        # Scores should differ (not exactly equal)
        assert result1.differential_score != result2.differential_score
    
    def test_same_key_same_results(self):
        """Same key should produce identical scores for same text."""
        text = "Machine learning is fascinating. It transforms data into insights."
        
        detector1 = LTWWatermarker(secret_key="same-key")
        detector2 = LTWWatermarker(secret_key="same-key")
        
        result1 = detector1.detect(text)
        result2 = detector2.detect(text)
        
        assert result1.differential_score == result2.differential_score
    
    def test_analyze_trajectory(self, detector):
        """analyze_trajectory should return appropriate data."""
        text = "First sentence here. Second sentence follows. Third sentence ends."
        
        units, embeddings, align_scores, diff_scores = detector.analyze_trajectory(text)
        
        assert len(units) == 3
        assert embeddings.shape[0] == 3
        assert len(align_scores) == 2  # n-1 transitions
        assert len(diff_scores) == 2
    
    def test_batch_detect(self, detector):
        """batch_detect should process multiple texts."""
        texts = [
            "Text one with sentences. More here.",
            "Text two is different. Very different.",
            "Text three completes the set. All done.",
        ]
        
        results = detector.batch_detect(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, WatermarkResult) for r in results)
    
    def test_word_unit_mode(self):
        """Word-level detection should work."""
        detector = LTWWatermarker(
            secret_key="test",
            unit="word"
        )
        text = "The quick brown fox jumps over the lazy dog"
        result = detector.detect(text)
        
        assert result.details.get("n_units", 0) > 3


class TestDetectionStats:
    """Tests for detection statistics calculation."""
    
    def test_get_detection_stats(self):
        """Statistics calculation should be correct."""
        detector = LTWWatermarker(
            secret_key="test",
            detection_threshold=0.0  # Always detect
        )
        
        texts = [
            "Sentence one here. Sentence two there.",
            "Another text sample. With more sentences.",
        ]
        labels = [True, False]
        
        stats = detector.get_detection_stats(texts, labels)
        
        assert "tp" in stats
        assert "fp" in stats
        assert "tn" in stats
        assert "fn" in stats
        assert "precision" in stats
        assert "recall" in stats
        assert "f1" in stats
    
    def test_perfect_classification(self):
        """Perfect classification should yield perfect metrics."""
        detector = LTWWatermarker(secret_key="test", detection_threshold=999)
        
        # With impossible threshold, nothing should be detected
        texts = ["Text one. More.", "Text two. More."]
        labels = [False, False]  # All should be "not detected"
        
        stats = detector.get_detection_stats(texts, labels)
        
        # All true negatives
        assert stats["tn"] == 2
        assert stats["fp"] == 0


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""
    
    def test_confidence_range(self):
        """Confidence should be in [0, 1]."""
        detector = LTWWatermarker(secret_key="test")
        
        for _ in range(10):
            text = "Random test sentence. With some variation."
            result = detector.detect(text)
            assert 0 <= result.confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
