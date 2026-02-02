"""
MCL (Markov Chain-Lock) Watermarking for AI-generated text detection.

This package provides:
- MCL watermark injection during text generation
- MCL watermark detection with configurable chains
- Enhanced MCL with soft partitions and flexible transitions
"""

from ltw_watermark.mcl_watermark import MCLGenerator, MCLDetector
from ltw_watermark.enhanced_mcl import EnhancedMCLGenerator, EnhancedMCLDetector

__version__ = "0.2.0"
__all__ = [
    "MCLGenerator",
    "MCLDetector",
    "EnhancedMCLGenerator",
    "EnhancedMCLDetector",
]
