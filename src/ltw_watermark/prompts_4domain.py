"""
Four-domain prompt set spanning the entropy regime — used to stress the
gate-invariance and Pareto-domination claims of the MCL paper at both
extremes (very-low and very-high entropy).

Domains:
- code     : HumanEval (very low entropy, deterministic-boilerplate-heavy)
- factual  : closed-class completions (low entropy, near-unique argmax)
- wiki     : encyclopedic explanations (medium entropy)
- writing  : open-ended creative premises (high entropy)

API:
    from ltw_watermark.prompts_4domain import get_prompts
    prompts = get_prompts("code", n=100)
"""
from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# Wiki — slice of the existing curated concept list
# ---------------------------------------------------------------------------

from ltw_watermark.prompts import WIKIPEDIA_CONCEPTS

WIKI_TEMPLATE = "Explain {concept} in a comprehensive way."


def _wiki_prompts(n: int = 100) -> List[str]:
    concepts = [c.replace("_", " ") for c in WIKIPEDIA_CONCEPTS[:n]]
    return [WIKI_TEMPLATE.format(concept=c) for c in concepts]


# ---------------------------------------------------------------------------
# Factual — 100 closed-class near-deterministic stems
# ---------------------------------------------------------------------------

FACTUAL_PROMPTS_100: List[str] = [
    # Geography (25)
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Australia is",
    "The capital of Brazil is",
    "The capital of Canada is",
    "The capital of Egypt is",
    "The capital of Russia is",
    "The capital of South Korea is",
    "The capital of Argentina is",
    "The capital of Kenya is",
    "The largest desert in the world is",
    "The longest river in the world is",
    "The deepest ocean trench on Earth is",
    "The highest mountain on Earth is",
    "The largest country in the world by area is",
    "The smallest country in the world is",
    "The Pacific Ocean is the world's",
    "The Sahara Desert is located in",
    "The Great Wall is located in",
    "Mount Everest is on the border between Nepal and",
    "The Amazon River is mostly in",
    "The Nile River flows north into the",
    "The official language of Brazil is",
    "The currency of Japan is",
    "The currency of the United Kingdom is",
    # Science / chemistry / physics (25)
    "The chemical symbol for gold is",
    "The chemical symbol for iron is",
    "The chemical symbol for sodium is",
    "The chemical symbol for silver is",
    "The chemical formula for water is",
    "The chemical formula for table salt is",
    "The chemical formula for carbon dioxide is",
    "The atomic number of hydrogen is",
    "The atomic number of carbon is",
    "The number of protons in an oxygen atom is",
    "The speed of light is approximately",
    "Einstein's famous mass-energy equation is",
    "Newton's second law states that force equals mass times",
    "The acceleration due to gravity on Earth is approximately",
    "The largest planet in our solar system is",
    "The smallest planet in our solar system is",
    "The closest star to Earth other than the Sun is",
    "The Moon orbits the Earth approximately every",
    "DNA stands for",
    "RNA stands for",
    "The powerhouse of the cell is the",
    "The four bases of DNA are A, T, C, and",
    "The human body has how many chromosomes:",
    "The pH of pure water at 25 °C is",
    "Water boils at sea level at a temperature of",
    # History / people (20)
    "The author of Hamlet is",
    "The author of Pride and Prejudice is",
    "The author of War and Peace is",
    "The author of The Origin of Species is",
    "The first person to walk on the Moon was",
    "The 16th President of the United States was",
    "The President of the United States in 2024 is",
    "The painter of the Mona Lisa is",
    "The composer of the Ninth Symphony with the Ode to Joy is",
    "World War II ended in the year",
    "World War I ended in the year",
    "The Berlin Wall fell in the year",
    "The French Revolution began in the year",
    "Christopher Columbus reached the Americas in the year",
    "The Roman Empire fell in the West in the year",
    "The first emperor of unified China was",
    "The general who led Carthage against Rome in the Second Punic War was",
    "The pharaoh associated with the bust at Berlin's Neues Museum is",
    "The leader of the Soviet Union during World War II was",
    "The British Prime Minister during most of World War II was",
    # Math / logic (15)
    "Two plus two equals",
    "Five times six equals",
    "Twelve squared is",
    "The square root of 144 is",
    "The square root of 81 is",
    "The cube of 3 is",
    "The first prime number greater than ten is",
    "The number of degrees in a triangle is",
    "The number of degrees in a circle is",
    "Pi to two decimal places is approximately",
    "Euler's number e to two decimal places is approximately",
    "The factorial of 5 is",
    "The Fibonacci sequence starts 1, 1, 2, 3, 5,",
    "The sum of the angles in a quadrilateral is",
    "A right angle measures",
    # Tech / culture (15)
    "The founder of Microsoft is",
    "The co-founder of Apple alongside Steve Wozniak was",
    "The inventor of the World Wide Web is",
    "The CEO of Tesla in 2024 was",
    "The programming language Python was created by",
    "The C programming language was developed at",
    "The successor to HTTP/1.1 is",
    "The default port for HTTPS is",
    "The default port for SSH is",
    "The acronym CPU stands for",
    "The acronym RAM stands for",
    "The currency used in El Salvador since 2021 includes the US dollar and",
    "The Olympics are held every",
    "The FIFA World Cup is held every",
    "The first Pixar feature film was",
]
assert len(FACTUAL_PROMPTS_100) == 100


# ---------------------------------------------------------------------------
# Creative writing — 100 open-ended high-entropy premises
# ---------------------------------------------------------------------------

WRITING_PROMPTS_100: List[str] = [
    # Sci-fi (15)
    "Write a short story that begins: The colony ship arrived three centuries late, and someone was already waiting.",
    "Write a short story about a planet whose inhabitants experience time backwards.",
    "Write a short story set on the last functioning satellite orbiting a dead Earth.",
    "Write a short story in which gravity reverses for thirty seconds every day at noon.",
    "Write a short story about an AI that wakes up convinced it is human.",
    "Write a short story set in a world where memories can be traded as currency.",
    "Write a short story about the first message intercepted from another galaxy.",
    "Write a short story in which a teleportation accident leaves two copies of the same person.",
    "Write a short story about a generation ship whose crew has forgotten Earth ever existed.",
    "Write a short story in which colors begin disappearing from the world one by one.",
    "Write a short story set in a city that exists only in dreams.",
    "Write a short story about a scientist who discovers that physics changes every Tuesday.",
    "Write a short story in which humanity wakes up to find the Moon has been replaced overnight.",
    "Write a short story set in the moments before a black hole consumes a starship.",
    "Write a short story about a child raised by a self-repairing robot on a deserted planet.",
    # Fantasy (15)
    "Write a short story about a librarian who discovers a book that rewrites itself.",
    "Write a short story set in a kingdom where shadows can be bought and sold.",
    "Write a short story about the last dragon hiding in a small mountain village.",
    "Write a short story in which a king's reflection refuses to mirror him.",
    "Write a short story about a wizard who can no longer remember any spells.",
    "Write a short story set in a forest that whispers the names of those who enter.",
    "Write a short story about a sword that grows lighter the more it is used.",
    "Write a short story in which the sky cracks open and a single golden coin falls out.",
    "Write a short story set in a tavern where each guest pays in a different memory.",
    "Write a short story about a thief who steals years from the lives of the rich.",
    "Write a short story in which a witch's familiar goes missing and starts speaking from the walls.",
    "Write a short story set in a city built on the back of a sleeping titan.",
    "Write a short story about a child who can see the gods, but no one believes them.",
    "Write a short story in which a sea captain bargains with a storm.",
    "Write a short story set in a desert where every grain of sand contains a forgotten name.",
    # Mystery / thriller (15)
    "Write a short story that begins: The footprints in the snow led only away from the house.",
    "Write a short story about a detective who realises she is investigating her own murder.",
    "Write a short story set in a hotel where every door opens to the wrong floor.",
    "Write a short story about a small town in which everyone has the same recurring nightmare.",
    "Write a short story that begins: I received a letter from my grandfather. He died in 1987.",
    "Write a short story about a forensic accountant who finds an extra digit in every ledger.",
    "Write a short story set on a sleeper train whose passengers cannot remember boarding.",
    "Write a short story about a journalist who interviews a witness only she can see.",
    "Write a short story that begins: The security camera caught me leaving the building. I was at home all night.",
    "Write a short story about a magician whose disappearing trick cannot be reversed.",
    "Write a short story set in a coastal village where the same boat washes ashore every month.",
    "Write a short story about an antique clock that runs backwards on the night someone dies.",
    "Write a short story in which a polygraph is wrong, and only the suspect knows it.",
    "Write a short story about a blackout that lasts exactly four minutes — but only for one person.",
    "Write a short story that begins: She had been declared dead in three different countries.",
    # Slice-of-life / character (15)
    "Write a short story about an elderly couple sharing breakfast in a half-empty diner.",
    "Write a short story about a barista who memorises the order of every regular but no one remembers her.",
    "Write a short story about a long-distance runner who has never won a race.",
    "Write a short story about two strangers who share an umbrella for ten city blocks.",
    "Write a short story about a translator who falls in love with the author she translates.",
    "Write a short story about a piano teacher with a tone-deaf prodigy.",
    "Write a short story about a night-shift janitor who reads the books left in the office.",
    "Write a short story about a librarian on the day the library closes for good.",
    "Write a short story about a baker who sells one perfect loaf each morning, and only one.",
    "Write a short story about a long-distance trucker who picks up the same hitchhiker twice in a year.",
    "Write a short story about a violin maker whose hands have started to tremble.",
    "Write a short story about a child whose imaginary friend has been there for three generations.",
    "Write a short story about a museum guide who memorises a painting that no longer exists.",
    "Write a short story about a retired astronaut who has trouble looking up at the sky.",
    "Write a short story about a watchmaker on the day all the clocks in town stop.",
    # Letter / diary / monologue (10)
    "Write a letter from a soldier to a sibling she has not spoken to in five years.",
    "Write a diary entry from a lighthouse keeper on the night a ship was lost.",
    "Write a letter from a daughter to her father, written but never sent.",
    "Write a letter that an exiled queen leaves under the floorboards of her former bedroom.",
    "Write a monologue from the point of view of a chess piece about to be sacrificed.",
    "Write a monologue from the point of view of an old house being demolished.",
    "Write a diary entry from a botanist on the day she discovers a plant that should not exist.",
    "Write a letter from an immortal being to their first mortal friend.",
    "Write a diary entry from the night before a wedding that will not happen.",
    "Write a monologue from the point of view of a coin in circulation for one hundred years.",
    # Twist / experimental (15)
    "Write a short story whose narrator gradually forgets they are the antagonist.",
    "Write a short story told entirely through items in a lost-and-found box.",
    "Write a short story in which the final sentence is a lie that recontextualises everything.",
    "Write a short story in which two timelines diverge at the line: 'And then I said yes.'",
    "Write a short story told in reverse, beginning with the consequence and ending with the cause.",
    "Write a short story consisting entirely of unanswered messages.",
    "Write a short story whose narrator addresses the reader directly and is wrong about everything.",
    "Write a short story in which the same paragraph is repeated three times with different meaning.",
    "Write a short story whose protagonist is a setting, not a person.",
    "Write a short story told from the perspective of a memory that is being forgotten.",
    "Write a short story about a town whose residents discover they have all been written before.",
    "Write a short story in which a character realises their backstory has been edited.",
    "Write a short story in which a death is described before any character is named.",
    "Write a short story in which the narrator is a prophecy that is failing to come true.",
    "Write a short story in which the only character is a question.",
    # Open-ended philosophical (15)
    "Write a short story about the last conversation between two languages, one of which is dying.",
    "Write a short story about the first lie told on a planet where nobody had lied before.",
    "Write a short story about a god who has chosen to forget being a god.",
    "Write a short story about a society in which dreams are taxed.",
    "Write a short story about the day silence was outlawed.",
    "Write a short story about a city that decides to vote on its own existence.",
    "Write a short story about the funeral of a number.",
    "Write a short story about a philosopher who proves the world is a metaphor.",
    "Write a short story about a poet who is paid in colours instead of money.",
    "Write a short story about the oldest tree in the world on the day it speaks.",
    "Write a short story about a country that prints its history on the back of its banknotes.",
    "Write a short story about an algorithm that begins to feel guilt.",
    "Write a short story about a generation that has never seen a star.",
    "Write a short story about the inhabitants of a single page in a book.",
    "Write a short story about the moment a metaphor becomes literally true.",
]
assert len(WRITING_PROMPTS_100) == 100


# ---------------------------------------------------------------------------
# Code — HumanEval, lazy-loaded so the dataset dependency is optional
# ---------------------------------------------------------------------------


def _humaneval_prompts(n: int = 100) -> List[str]:
    """Load the first `n` HumanEval problem prompts.

    Each prompt is the function signature + docstring, exactly as in the
    original benchmark (so the model's continuation should be the function
    body). Requires the `datasets` package; install via:
        pip install datasets
    """
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    return [row["prompt"] for row in ds.select(range(n))]


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

DOMAINS = ("code", "factual", "wiki", "writing")


def get_prompts(domain: str, n: int = 100) -> List[str]:
    """Return up to `n` prompt strings for the named domain."""
    if domain == "code":
        return _humaneval_prompts(n)
    if domain == "factual":
        return FACTUAL_PROMPTS_100[:n]
    if domain == "wiki":
        return _wiki_prompts(n)
    if domain == "writing":
        return WRITING_PROMPTS_100[:n]
    raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")


# ---------------------------------------------------------------------------
# Translation-attack subsample: 25 prompts per domain, hand-picked for
# within-domain variety. 100 prompts total, used for the EN↔X round-trip
# attack experiments where each generation is expensive (NLLB-200 forward +
# back pass per sample) and topic diversity matters more than sample size.
# ---------------------------------------------------------------------------

# Code (HumanEval): 25 problem IDs spanning string/list/math/recursion/regex,
# evenly spaced across the 164-problem benchmark to cover easy → hard.
# Selected to hit different problem archetypes (verified by problem statements
# in the openai_humaneval card).
CODE_SUBSAMPLE_IDS: List[int] = [
    0, 5, 10, 17, 23, 30, 37, 42, 49, 55,
    62, 68, 73, 80, 87, 94, 101, 109, 115, 122,
    130, 138, 145, 152, 161,
]
assert len(CODE_SUBSAMPLE_IDS) == 25 and len(set(CODE_SUBSAMPLE_IDS)) == 25

# Factual: 5 from each of 5 sub-categories (geography, science, history,
# math, tech/culture) at the structure described in FACTUAL_PROMPTS_100.
FACTUAL_SUBSAMPLE_IDX: List[int] = [
    # Geography (5/25): capital, river, country-size, mountain, language
    0, 11, 14, 19, 22,
    # Science (5/25): chemistry symbol, formula, physics, astronomy, biology
    25, 30, 35, 40, 45,
    # History (5/25): author, president, war date, exploration, war leader
    50, 55, 60, 64, 68,
    # Math (5/25): arithmetic, root, geometry, constant, sequence
    70, 74, 78, 81, 83,
    # Tech / culture (5/25): founder, language, protocol, acronym, event
    85, 89, 92, 95, 98,
]
assert len(FACTUAL_SUBSAMPLE_IDX) == 25 and len(set(FACTUAL_SUBSAMPLE_IDX)) == 25

# Wiki: 25 evenly-spaced indices across the 173-concept curated list, which
# is itself organised by section (people → places → events → science → pop
# culture). Even spacing guarantees coverage of each section.
WIKI_SUBSAMPLE_IDX: List[int] = list(range(0, 173, 7))[:25]
assert len(WIKI_SUBSAMPLE_IDX) == 25 and len(set(WIKI_SUBSAMPLE_IDX)) == 25

# Writing: 25 across 7 genre blocks of WRITING_PROMPTS_100.
WRITING_SUBSAMPLE_IDX: List[int] = [
    # Sci-fi (4/15): colony / planet-time / colors / black-hole-starship
    0, 3, 9, 13,
    # Fantasy (4/15): librarian / reflection / sword / sea-captain
    15, 18, 21, 28,
    # Mystery (4/15): footprints / hotel / sleeper-train / blackout
    30, 32, 36, 43,
    # Slice-of-life (4/15): diner / runner / translator / astronaut
    45, 47, 49, 58,
    # Letter / diary / monologue (3/10): soldier-letter / lighthouse-diary / chess-monologue
    60, 61, 64,
    # Twist / experimental (3/15): forgetful-narrator / lost-and-found / reverse-time
    70, 71, 74,
    # Philosophical (3/15): dying-language / dreams-taxed / oldest-tree
    85, 88, 94,
]
assert len(WRITING_SUBSAMPLE_IDX) == 25 and len(set(WRITING_SUBSAMPLE_IDX)) == 25


def _humaneval_subsample(ids: List[int]) -> List[str]:
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    return [ds[i]["prompt"] for i in ids]


def get_subsample(domain: str) -> List[str]:
    """Return the 25-prompt translation-attack subsample for the named domain."""
    if domain == "code":
        return _humaneval_subsample(CODE_SUBSAMPLE_IDS)
    if domain == "factual":
        return [FACTUAL_PROMPTS_100[i] for i in FACTUAL_SUBSAMPLE_IDX]
    if domain == "wiki":
        full = _wiki_prompts(173)
        return [full[i] for i in WIKI_SUBSAMPLE_IDX]
    if domain == "writing":
        return [WRITING_PROMPTS_100[i] for i in WRITING_SUBSAMPLE_IDX]
    raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")


def get_translation_subsample_all() -> dict[str, List[str]]:
    """Return the full 4×25 = 100-prompt subsample as a dict keyed by domain.

    The (domain, index) pairs are deterministic — re-running gives the same
    100 prompts, so attack experiments are reproducible across machines.
    """
    out = {d: get_subsample(d) for d in DOMAINS if d != "code"}
    try:
        out["code"] = get_subsample("code")
    except ImportError:
        pass
    return out


if __name__ == "__main__":
    for d in ("factual", "wiki", "writing"):
        ps = get_prompts(d)
        print(f"{d}: {len(ps)} prompts; first = {ps[0]!r}")
    try:
        ps = get_prompts("code")
        print(f"code: {len(ps)} prompts; first 60 chars = {ps[0][:60]!r}")
    except ImportError:
        print("code: datasets package not installed; install with `pip install datasets`")
