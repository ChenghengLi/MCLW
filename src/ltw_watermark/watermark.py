"""
Core watermarking logic for Latent Trajectory Watermarking.

Implements the LTW algorithm for:
- Watermark detection in text
- Differential scoring to reduce false positives
- Trajectory analysis
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from ltw_watermark.embeddings import EmbeddingExtractor
from ltw_watermark.rotation import OrthogonalRotation


@dataclass
class WatermarkResult:
    """Result of watermark detection."""
    is_watermarked: bool
    confidence: float
    alignment_score: float
    differential_score: float
    threshold: float
    details: Dict[str, Any]


class LTWWatermarker:
    """
    Latent Trajectory Watermarking detector.
    
    Detects watermarks by checking if consecutive text units
    (words or sentences) follow a secret rotation pattern in
    embedding space.
    
    Attributes:
        secret_key: The secret key for watermark verification
        embedding_extractor: Extractor for text embeddings
        rotation: Orthogonal rotation handler
        threshold: Detection threshold for differential score
    """
    
    def __init__(
        self,
        secret_key: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        rotation_strength: float = 0.3,
        detection_threshold: float = 0.1,
        use_differential_scoring: bool = True,
        unit: str = "sentence"  # "word" or "sentence"
    ):
        """
        Initialize the LTW watermarker.
        
        Args:
            secret_key: Secret key for watermark verification
            embedding_model: Name of sentence-transformer model
            rotation_strength: Strength of rotation (0-1)
            detection_threshold: Threshold for watermark detection
            use_differential_scoring: Use differential scoring to reduce FPR
            unit: Text unit for analysis ("word" or "sentence")
        """
        self.secret_key = secret_key
        self.threshold = detection_threshold
        self.use_differential_scoring = use_differential_scoring
        self.unit = unit
        
        # Initialize embedding extractor
        self.embedding_extractor = EmbeddingExtractor(
            model_name=embedding_model,
            cache_embeddings=True
        )
        
        # Initialize rotation with correct dimension
        self.rotation = OrthogonalRotation(
            secret_key=secret_key,
            dim=self.embedding_extractor.embedding_dim,
            rotation_strength=rotation_strength,
            use_anisotropy_correction=True
        )
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into units (words or sentences)."""
        if self.unit == "word":
            return text.split()
        else:  # sentence
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s.strip() for s in sentences if s.strip()]
    
    def detect(self, text: str) -> WatermarkResult:
        """
        Detect watermark in text.
        
        Args:
            text: Input text to check for watermark
            
        Returns:
            WatermarkResult with detection details
        """
        # Split text into units
        units = self._split_text(text)
        
        if len(units) < 2:
            return WatermarkResult(
                is_watermarked=False,
                confidence=0.0,
                alignment_score=0.0,
                differential_score=0.0,
                threshold=self.threshold,
                details={"error": "Text too short", "units": len(units)}
            )
        
        # Get embeddings for all units
        embeddings = self.embedding_extractor.embed_texts(units)
        
        # Calculate scores for consecutive pairs
        alignment_scores = []
        differential_scores = []
        
        for i in range(len(embeddings) - 1):
            source_emb = embeddings[i]
            target_emb = embeddings[i + 1]
            
            # Watermark alignment
            alignment = self.rotation.compute_alignment_score(
                source_emb, target_emb
            )
            alignment_scores.append(alignment)
            
            # Differential score (watermark vs natural coherence)
            if self.use_differential_scoring:
                differential = self.rotation.compute_differential_score(
                    source_emb, target_emb
                )
                differential_scores.append(differential)
        
        # Aggregate scores
        mean_alignment = np.mean(alignment_scores)
        std_alignment = np.std(alignment_scores)
        
        if self.use_differential_scoring:
            mean_differential = np.mean(differential_scores)
            std_differential = np.std(differential_scores)
            
            # Use differential score for detection
            detection_score = mean_differential
        else:
            mean_differential = 0.0
            std_differential = 0.0
            detection_score = mean_alignment
        
        # Determine if watermarked
        is_watermarked = detection_score > self.threshold
        
        # Calculate confidence (how far above/below threshold)
        confidence = self._calculate_confidence(detection_score)
        
        return WatermarkResult(
            is_watermarked=is_watermarked,
            confidence=confidence,
            alignment_score=mean_alignment,
            differential_score=mean_differential,
            threshold=self.threshold,
            details={
                "n_units": len(units),
                "n_pairs": len(alignment_scores),
                "alignment_std": std_alignment,
                "differential_std": std_differential,
                "all_alignment_scores": alignment_scores,
                "all_differential_scores": differential_scores if self.use_differential_scoring else [],
            }
        )
    
    def _calculate_confidence(self, score: float) -> float:
        """
        Calculate confidence based on detection score.
        
        Returns a value between 0 and 1 indicating confidence
        in the detection result.
        """
        # Sigmoid-like transformation centered at threshold
        distance = abs(score - self.threshold)
        confidence = 1 - np.exp(-5 * distance)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def analyze_trajectory(
        self,
        text: str
    ) -> Tuple[List[str], np.ndarray, List[float], List[float]]:
        """
        Analyze the embedding trajectory of text.
        
        Useful for visualization and debugging.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (units, embeddings, alignment_scores, differential_scores)
        """
        units = self._split_text(text)
        
        if len(units) < 2:
            return units, np.array([]), [], []
        
        embeddings = self.embedding_extractor.embed_texts(units)
        
        alignment_scores = []
        differential_scores = []
        
        for i in range(len(embeddings) - 1):
            alignment = self.rotation.compute_alignment_score(
                embeddings[i], embeddings[i + 1]
            )
            differential = self.rotation.compute_differential_score(
                embeddings[i], embeddings[i + 1]
            )
            alignment_scores.append(alignment)
            differential_scores.append(differential)
        
        return units, embeddings, alignment_scores, differential_scores
    
    def batch_detect(
        self,
        texts: List[str]
    ) -> List[WatermarkResult]:
        """
        Detect watermarks in multiple texts.
        
        Args:
            texts: List of texts to check
            
        Returns:
            List of WatermarkResult objects
        """
        return [self.detect(text) for text in texts]
    
    def get_detection_stats(
        self,
        texts: List[str],
        labels: List[bool]
    ) -> Dict[str, float]:
        """
        Calculate detection statistics on labeled data.
        
        Args:
            texts: List of texts
            labels: True if watermarked, False otherwise
            
        Returns:
            Dictionary with TP, FP, TN, FN, precision, recall, F1
        """
        results = self.batch_detect(texts)
        
        tp = fp = tn = fn = 0
        
        for result, label in zip(results, labels):
            if result.is_watermarked and label:
                tp += 1
            elif result.is_watermarked and not label:
                fp += 1
            elif not result.is_watermarked and not label:
                tn += 1
            else:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / len(texts) if texts else 0.0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        }
