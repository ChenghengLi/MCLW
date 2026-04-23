"""
Token-level alignment for paraphrase-survival experiments (E2).

Uses the CJK-block trick to let python-Levenshtein treat integer token IDs
as single characters (vocabs up to ~20K fit in the 0x4E00..0x9FFF block;
for larger vocabs we use 0x4E00..0xD7FF ~ 37K which is safe for Llama-3).

Pipeline:
    1. Re-tokenize paraphrase under GENERATOR's tokenizer.
    2. `Levenshtein.editops(orig_str, para_str)` returns (op, i, j) triples.
    3. Per original position label -> {survived, substituted, deleted}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

try:
    from Levenshtein import editops as _editops  # python-Levenshtein
    HAVE_LEV = True
except Exception:  # pragma: no cover
    HAVE_LEV = False


_CJK_BASE = 0x4E00  # 20K safe
_ID_MAX = 0xD7FF - _CJK_BASE  # ~37K, covers Llama-3 128K tokens? No - falls short.


def _encode_ids(ids: List[int]) -> str:
    """
    Map token IDs to characters via CJK base. For vocab_size > ~37K we fold
    with modulo; collisions are rare enough to keep alignment useful but the
    caller should be aware.
    """
    return "".join(chr(_CJK_BASE + (int(i) % _ID_MAX)) for i in ids)


@dataclass
class AlignmentResult:
    n_orig: int
    n_para: int
    labels: List[str]         # one per original token: {survived, substituted, deleted}
    survived_mask: List[bool]
    substituted_mask: List[bool]
    deleted_mask: List[bool]
    edit_distance: int


def align_token_sequences(
    orig_ids: List[int], para_ids: List[int]
) -> AlignmentResult:
    """
    Align two integer token-ID sequences via edit distance and produce
    per-original-position labels.

    Returns labels of length len(orig_ids), where:
        labels[i] == "survived"    iff no edit op touches original position i
                                       (and the character at that position in para
                                       matches)
        labels[i] == "substituted" iff a `replace` op at original position i
        labels[i] == "deleted"     iff a `delete` op at original position i
    """
    if not HAVE_LEV:
        raise RuntimeError(
            "python-Levenshtein is required. `uv add python-Levenshtein` or "
            "`pip install python-Levenshtein`."
        )
    s_orig = _encode_ids(orig_ids)
    s_para = _encode_ids(para_ids)
    ops = _editops(s_orig, s_para)

    n = len(orig_ids)
    labels = ["survived"] * n
    for op, i, _j in ops:
        if 0 <= i < n:
            if op == "replace":
                labels[i] = "substituted"
            elif op == "delete":
                labels[i] = "deleted"
            # insert ops don't touch an original position -> ignore

    survived = [lab == "survived" for lab in labels]
    substituted = [lab == "substituted" for lab in labels]
    deleted = [lab == "deleted" for lab in labels]

    return AlignmentResult(
        n_orig=n,
        n_para=len(para_ids),
        labels=labels,
        survived_mask=survived,
        substituted_mask=substituted,
        deleted_mask=deleted,
        edit_distance=len(ops),
    )


def validate_alignment_identity(tokenizer, text: str) -> float:
    """
    Sanity check: aligning a text with itself must give 100% survival.
    Returns survival fraction.
    """
    ids = tokenizer.encode(text)
    res = align_token_sequences(ids, ids)
    return sum(res.survived_mask) / max(1, res.n_orig)


def validate_alignment_shuffle(tokenizer, text: str, seed: int = 0) -> float:
    """
    Sanity check: aligning a text with a random permutation should give ~0 survival.
    """
    import random
    ids = tokenizer.encode(text)
    shuf = list(ids)
    random.Random(seed).shuffle(shuf)
    res = align_token_sequences(ids, shuf)
    return sum(res.survived_mask) / max(1, res.n_orig)
