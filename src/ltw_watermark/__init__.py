"""
Latent Trajectory Watermarking (LTW) for AI-generated text detection.

This package provides tools for:
- Watermark injection during text generation (full LTW)
- Embedding-based watermark detection with differential scoring
- Perplexity-based detection (baseline comparison)
- Analysis and visualization utilities
"""

from ltw_watermark.embeddings import EmbeddingExtractor
from ltw_watermark.rotation import generate_rotation_matrix, OrthogonalRotation
from ltw_watermark.watermark import LTWWatermarker
from ltw_watermark.perplexity import PerplexityDetector
from ltw_watermark.generator import LTWGenerator, NonWatermarkedGenerator, LTWLogitsProcessor

__version__ = "0.1.0"
__all__ = [
    "EmbeddingExtractor",
    "generate_rotation_matrix",
    "OrthogonalRotation",
    "LTWWatermarker",
    "PerplexityDetector",
    "LTWGenerator",
    "NonWatermarkedGenerator",
    "LTWLogitsProcessor",
]

