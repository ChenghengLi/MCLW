"""
MCL (Markov Chain-Lock) watermarking for AI-generated text.

Package name `ltw_watermark` is historical; the actual mechanism is MCL
(SHA-256 vocabulary partition into S states + forced Markov transitions).

Public API:

    EnhancedMCLGenerator / EnhancedMCLDetector
        Original MCL generator/detector with configurable transition topology
        and soft partitions. Used for v1 baseline (always-on watermarking).

    EntropyGatedMCLGenerator
        v2 generator: gates MCL insertion by next-token entropy (low-H only).
        Records per-position metadata used by Experiment 2.

    detect, observed_z, expected_z_H1, p_value_one_sided, s_min_calibration,
    critical_delta_midpoint, critical_delta_alpha
        Corrected statistical detection utilities (see docs/research_plan_v2.md
        Theorems 1, 2, 4, 7 for the corrected math).

    align_token_sequences
        Token-level edit-distance alignment for paraphrase-survival analysis.

    attacks module
        random_substitution_tokens, synonym_substitution, DipperAttacker,
        BackTranslationAttacker, SIRAAttacker.

    metrics module
        PerplexityScorer, SemanticSimilarity, distinct_n, self_bleu.

    prompts
        load_prompts / WIKIPEDIA_CONCEPTS for experiment inputs.
"""

from ltw_watermark.enhanced_mcl import (
    EnhancedMCLGenerator,
    EnhancedMCLDetector,
    generate_transition_matrix,
    get_token_state_soft,
    precompute_soft_masks,
)
from ltw_watermark.entropy_gated_mcl import (
    EntropyGatedMCLGenerator,
    pilot_measure_entropy_quantiles,
)
# v3 generic gated generator + gates (headline)
from ltw_watermark.gated_mcl import (
    GatedMCLGenerator,
    GenerationResult,
    PositionRecord,
)
from ltw_watermark.gates import (
    Gate,
    GateAll,
    GateNone,
    GateEntropyHigh,
    GateEntropyLow,
    GateDelta,
    GatePSurviveOracle,
    logits_stats,
    make_gate,
)
from ltw_watermark.detection_stats import (
    detect,
    observed_z,
    expected_z_H1,
    p_value_one_sided,
    s_min_calibration,
    critical_delta_midpoint,
    critical_delta_alpha,
    random_baseline_for_topology,
    hdd_lambda,
    hdd_p_value,
    DetectionReport,
)
from ltw_watermark.alignment import (
    align_token_sequences,
    AlignmentResult,
    validate_alignment_identity,
    validate_alignment_shuffle,
)
from ltw_watermark.prompts import (
    WIKIPEDIA_CONCEPTS,
    PROMPT_TEMPLATE,
    make_prompt,
    load_prompts,
)

__version__ = "0.3.0"
__all__ = [
    # v3 generic generator
    "GatedMCLGenerator",
    "GenerationResult",
    "PositionRecord",
    # Gates
    "Gate",
    "GateAll",
    "GateNone",
    "GateEntropyHigh",
    "GateEntropyLow",
    "GateDelta",
    "GatePSurviveOracle",
    "logits_stats",
    "make_gate",
    # Legacy generators / detectors
    "EnhancedMCLGenerator",
    "EnhancedMCLDetector",
    "EntropyGatedMCLGenerator",
    "pilot_measure_entropy_quantiles",
    # Core primitives
    "generate_transition_matrix",
    "get_token_state_soft",
    "precompute_soft_masks",
    # Detection stats
    "detect",
    "observed_z",
    "expected_z_H1",
    "p_value_one_sided",
    "s_min_calibration",
    "critical_delta_midpoint",
    "critical_delta_alpha",
    "random_baseline_for_topology",
    "hdd_lambda",
    "hdd_p_value",
    "DetectionReport",
    # Alignment
    "align_token_sequences",
    "AlignmentResult",
    "validate_alignment_identity",
    "validate_alignment_shuffle",
    # Prompts
    "WIKIPEDIA_CONCEPTS",
    "PROMPT_TEMPLATE",
    "make_prompt",
    "load_prompts",
]
