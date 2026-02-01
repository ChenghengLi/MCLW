"""
Perplexity-based detection for AI-generated text.

This module implements the traditional "passive" detection method
used by tools like GPTZero, serving as a baseline comparison for LTW.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass


@dataclass
class PerplexityResult:
    """Result of perplexity-based detection."""
    is_ai_generated: bool
    perplexity: float
    burstiness: float
    threshold: float
    confidence: float
    details: Dict[str, Any]


def calculate_perplexity(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cpu"
) -> float:
    """
    Calculate perplexity of text using a language model.
    
    Lower perplexity = more predictable = more likely AI-generated.
    
    Args:
        text: Input text
        model: Pretrained language model
        tokenizer: Corresponding tokenizer
        device: Device to run on
        
    Returns:
        Perplexity score
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity


def calculate_burstiness(text: str) -> float:
    """
    Calculate burstiness of text.
    
    Burstiness measures the variation in sentence lengths.
    Human text tends to have higher burstiness (more variation).
    
    Args:
        text: Input text
        
    Returns:
        Burstiness score (higher = more human-like)
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.0
    
    lengths = [len(s.split()) for s in sentences]
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    # Coefficient of variation as burstiness measure
    if mean_length == 0:
        return 0.0
    
    burstiness = std_length / mean_length
    return float(burstiness)


class PerplexityDetector:
    """
    Perplexity-based AI text detector.
    
    Uses perplexity and burstiness metrics to detect AI-generated text.
    This serves as a baseline comparison for the LTW method.
    
    Attributes:
        model_name: Name of the language model used
        perplexity_threshold: Texts below this are flagged as AI
        burstiness_threshold: Texts below this are flagged as AI
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        perplexity_threshold: float = 30.0,
        burstiness_threshold: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the perplexity detector.
        
        Args:
            model_name: Language model for perplexity calculation
            perplexity_threshold: Threshold for AI detection
            burstiness_threshold: Threshold for burstiness check
            device: Device to run on (None for auto)
        """
        self.model_name = model_name
        self.perplexity_threshold = perplexity_threshold
        self.burstiness_threshold = burstiness_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Handle tokenizer padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def detect(self, text: str) -> PerplexityResult:
        """
        Detect if text is AI-generated using perplexity.
        
        Args:
            text: Input text to analyze
            
        Returns:
            PerplexityResult with detection details
        """
        if len(text.strip()) < 10:
            return PerplexityResult(
                is_ai_generated=False,
                perplexity=float('inf'),
                burstiness=0.0,
                threshold=self.perplexity_threshold,
                confidence=0.0,
                details={"error": "Text too short"}
            )
        
        # Calculate perplexity
        perplexity = calculate_perplexity(
            text, self.model, self.tokenizer, self.device
        )
        
        # Calculate burstiness
        burstiness = calculate_burstiness(text)
        
        # Combined detection logic
        # Low perplexity AND low burstiness = AI
        is_ai_perplexity = perplexity < self.perplexity_threshold
        is_ai_burstiness = burstiness < self.burstiness_threshold
        
        # Conservative: require both conditions
        is_ai_generated = is_ai_perplexity and is_ai_burstiness
        
        # Calculate confidence
        perplexity_distance = abs(perplexity - self.perplexity_threshold)
        confidence = 1 - np.exp(-0.1 * perplexity_distance)
        
        return PerplexityResult(
            is_ai_generated=is_ai_generated,
            perplexity=perplexity,
            burstiness=burstiness,
            threshold=self.perplexity_threshold,
            confidence=float(confidence),
            details={
                "is_ai_by_perplexity": is_ai_perplexity,
                "is_ai_by_burstiness": is_ai_burstiness,
                "burstiness_threshold": self.burstiness_threshold,
            }
        )
    
    def batch_detect(self, texts: List[str]) -> List[PerplexityResult]:
        """
        Detect AI generation in multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of PerplexityResult objects
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
            labels: True if AI-generated, False otherwise
            
        Returns:
            Dictionary with detection metrics
        """
        results = self.batch_detect(texts)
        
        tp = fp = tn = fn = 0
        
        for result, label in zip(results, labels):
            if result.is_ai_generated and label:
                tp += 1
            elif result.is_ai_generated and not label:
                fp += 1
            elif not result.is_ai_generated and not label:
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
    
    def calculate_perplexity_distribution(
        self,
        texts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate perplexity statistics across texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Statistics dictionary
        """
        perplexities = []
        burstiness_scores = []
        
        for text in texts:
            result = self.detect(text)
            perplexities.append(result.perplexity)
            burstiness_scores.append(result.burstiness)
        
        return {
            "perplexity_mean": np.mean(perplexities),
            "perplexity_std": np.std(perplexities),
            "perplexity_median": np.median(perplexities),
            "perplexity_min": np.min(perplexities),
            "perplexity_max": np.max(perplexities),
            "burstiness_mean": np.mean(burstiness_scores),
            "burstiness_std": np.std(burstiness_scores),
        }
