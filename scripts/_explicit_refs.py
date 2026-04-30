#!/usr/bin/env python3
"""Replace \cref{...} and \Cref{...} with explicit Type~\ref{...} forms.

Rationale: in the user's compiled PDF, cleveref renders \cref{thm:foo} as
just "4.1" instead of "theorem 4.1". To avoid relying on cleveref's name
mappings, we substitute every cref call with an explicit "Theorem~\ref{...}"
(or Proposition / Definition / Table / Section / etc.) form.

Multi-arg refs are split with proper conjunctions:
  \Cref{a,b}     -> "TypeA~\ref{a} and TypeB~\ref{b}"
  \Cref{a,b,c}   -> "TypeA~\ref{a}, TypeB~\ref{b}, and TypeC~\ref{c}"

Equation refs are converted to \eqref so they render as "Equation (N)".
Comments (lines starting with %) are left alone.
"""
from __future__ import annotations
import glob, os, re, sys

PREFIX = {
    "thm":     "Theorem",
    "prop":    "Proposition",
    "lem":     "Lemma",
    "cor":     "Corollary",
    "def":     "Definition",
    "ex":      "Example",
    "rem":     "Remark",
    "ass":     "Assumption",
    "alg":     "Algorithm",
    "fig":     "Figure",
    "tab":     "Table",
    "sec":     "Section",
    "subsec":  "Section",
    "app":     "Appendix",
}

CREF_RE = re.compile(r"\\([cC])ref\{([^}]+)\}")


def render_one(label: str) -> str:
    label = label.strip()
    head = label.split(":", 1)[0]
    if head == "eq":
        return f"\\eqref{{{label}}}"
    type_name = PREFIX.get(head)
    if type_name is None:
        # unknown prefix; fall back to plain \ref so the number at least appears
        return f"\\ref{{{label}}}"
    return f"{type_name}~\\ref{{{label}}}"


def render_many(labels: list[str]) -> str:
    parts = [render_one(k) for k in labels]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def transform(line: str) -> str:
    # Skip comment lines entirely (treat the line as a comment if it's
    # whitespace-only-then-%).
    stripped = line.lstrip()
    if stripped.startswith("%"):
        return line

    def repl(match: re.Match) -> str:
        labels = [k.strip() for k in match.group(2).split(",")]
        return render_many(labels)

    return CREF_RE.sub(repl, line)


def main() -> None:
    base = "/Users/chenghengli/Desktop/MCLW/icml2026"
    files = sorted(glob.glob(os.path.join(base, "0[0-9]_*.tex"))
                   + glob.glob(os.path.join(base, "1[0-9]_*.tex")))
    changed = 0
    for path in files:
        with open(path) as f:
            src = f.read()
        new = "".join(transform(ln) for ln in src.splitlines(keepends=True))
        if new != src:
            with open(path, "w") as f:
                f.write(new)
            changed += 1
            print(f"updated {os.path.basename(path)}")
    print(f"{changed} files updated")


if __name__ == "__main__":
    main()
