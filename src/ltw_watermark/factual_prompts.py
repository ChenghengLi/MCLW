"""
Closed-class, near-deterministic prompts for stressing the (H ≈ 0, Δ ≈ 1)
corner of the gate-equivalence experiment.

These prompts FORCE argmax to be effectively unique, so entropy H → 0 and
surprisal-gap Δ → 1 simultaneously. Used in v4 Experiment B as a "corner
probe" — if entropy and Δ gates ever diverge, it's here or in the complementary
diffuse-tail open-end regime.
"""

FACTUAL_CLOSED_PROMPTS = [
    "The chemical symbol for gold is",
    "The capital of France is",
    "Two plus two equals",
    "The largest planet in our solar system is",
    "The author of Hamlet is",
    "The speed of light is approximately",
    "The President of the United States in 2024 is",
    "The Pacific Ocean is the world's",
    "Einstein's famous equation is",
    "The square root of 144 is",
    "The chemical formula for water is",
    "The currency of Japan is",
    "The tallest mountain in the world is",
    "The first person to walk on the Moon was",
    "The official language of Brazil is",
    "The chemical symbol for iron is",
    "The Great Wall is located in",
    "The year World War II ended was",
    "The capital of Australia is",
    "The human body has",
]
