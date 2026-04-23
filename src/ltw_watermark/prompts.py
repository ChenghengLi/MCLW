"""
Curated Wikipedia-concept prompts used across the MCL watermarking experiments.

Previously duplicated in scripts/generate_curated_dataset.py; consolidated here
so that experiment scripts (E1-E6) import from the package rather than one
script importing from another script.
"""

from __future__ import annotations

from typing import List

WIKIPEDIA_CONCEPTS: List[str] = [
    # --- Famous People (Modern & Historical) ---
    "Donald_Trump", "Elon_Musk", "Taylor_Swift", "Joe_Biden", "Kamala_Harris",
    "Vladimir_Putin", "Elizabeth_II", "Charles_III", "Princess_Diana", "Marilyn_Monroe",
    "Albert_Einstein", "Adolf_Hitler", "Winston_Churchill", "J._Robert_Oppenheimer",
    "Leonardo_da_Vinci", "Vincent_van_Gogh", "Frida_Kahlo", "Pablo_Picasso",
    "William_Shakespeare", "Elvis_Presley", "Michael_Jackson", "Freddie_Mercury",
    "Tupac_Shakur", "Eminem", "Cristiano_Ronaldo", "Lionel_Messi", "LeBron_James",
    "Kobe_Bryant", "Michael_Jordan", "Tom_Cruise", "Dwayne_Johnson", "Johnny_Depp",
    "Keanu_Reeves", "Kim_Kardashian", "Steve_Jobs", "Bill_Gates", "Mark_Zuckerberg",
    "Nikola_Tesla", "Isaac_Newton", "Charles_Darwin", "Sigmund_Freud", "Jesus",
    "Muhammad", "Gautama_Buddha", "Cleopatra", "Julius_Caesar", "Alexander_the_Great",
    "Genghis_Khan", "Napoleon", "Jeffrey_Dahmer", "Ted_Bundy", "Pablo_Escobar",

    # --- Places & Geography ---
    "United_States", "India", "China", "Russia", "Ukraine", "Israel", "Japan",
    "United_Kingdom", "Germany", "France", "North_Korea", "Vatican_City", "Dubai",
    "Chernobyl_Exclusion_Zone", "Area_51", "Bermuda_Triangle", "Mount_Everest",
    "Mariana_Trench", "Antarctica", "Mars", "Moon", "Sun", "Earth", "Milky_Way",

    # --- Major Events & Historical Concepts ---
    "World_War_II", "World_War_I", "The_Holocaust", "September_11_attacks",
    "Vietnam_War", "American_Civil_War", "French_Revolution", "Cold_War",
    "Space_Race", "Titanic", "Hindenburg_disaster", "Black_Death", "Spanish_flu",
    "COVID-19",

    # --- Science, Nature & Technology ---
    "Artificial_intelligence", "ChatGPT", "Bitcoin", "Cryptocurrency", "Internet",
    "YouTube", "Google", "Nuclear_weapon", "Black_hole", "Big_Bang",
    "Theory_of_relativity", "Evolution", "DNA", "Periodic_table", "Human_brain",
    "Heart", "Cancer", "Virus", "Dinosaur", "Tyrannosaurus", "Blue_whale",
    "Lion", "Wolf", "Shark", "Dog", "Cat",

    # --- Pop Culture, Fiction & Entertainment ---
    "Star_Wars", "Darth_Vader", "Harry_Potter", "Game_of_Thrones", "Lord_of_the_Rings",
    "Marvel_Cinematic_Universe", "Spider-Man", "Batman", "Superman", "Joker_(character)",
    "Wonder_Woman", "James_Bond", "Sherlock_Holmes", "Mickey_Mouse", "SpongeBob_SquarePants",
    "Pikachu", "Mario", "Barbie", "Oppenheimer_(film)", "Avatar_(2009_film)",
    "Titanic_(1997_film)", "The_Godfather", "Friends_(TV_series)", "Breaking_Bad",
    "Stranger_Things", "Squid_Game", "One_Piece", "Naruto", "Anime",

    # --- Sports & Games ---
    "Football", "Basketball", "Cricket", "American_football", "Tennis", "Chess",
    "Formula_One", "UFC", "WWE", "Olympics", "FIFA_World_Cup", "Super_Bowl",

    # --- General Objects & Concepts ---
    "Bible", "Quran", "Democracy", "Communism", "Capitalism", "Socialism",
    "Philosophy", "Psychology", "Zodiac", "Tarot", "Illuminati", "UFO",
    "Extraterrestrial_life", "Ghost", "Vampire", "Zombie",
]


PROMPT_TEMPLATE = "Explain {concept} in a comprehensive way."


def make_prompt(concept: str) -> str:
    """Convert a raw concept name into the final LLM prompt."""
    readable = concept.replace("_", " ").replace("(", "- ").replace(")", "")
    return PROMPT_TEMPLATE.format(concept=readable)


def load_prompts(n: int = 176) -> List[str]:
    """Return the first `n` prompts."""
    return [make_prompt(c) for c in WIKIPEDIA_CONCEPTS[:n]]
