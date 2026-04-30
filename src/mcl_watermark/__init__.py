"""
MCL (Markov Chain-Lock) Watermarking for AI-generated text detection.

This package provides:
- MCL watermark injection during text generation
- MCL watermark detection with configurable chains
- Enhanced MCL with soft partitions and flexible transitions
"""

from mcl_watermark.mcl_watermark import MCLGenerator, MCLDetector
from mcl_watermark.enhanced_mcl import EnhancedMCLGenerator, EnhancedMCLDetector

__version__ = "0.2.0"
__all__ = [
    "MCLGenerator",
    "MCLDetector",
    "EnhancedMCLGenerator",
    "EnhancedMCLDetector",
]
