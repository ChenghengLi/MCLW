#!/usr/bin/env python
"""
Curated Wikipedia MCL Dataset Generator

Generates text for ~176 curated Wikipedia concepts with:
- Non-watermarked baseline
- 28 MCL configurations (states 2,4,5,7,9,11,15 × overlaps 0%, 5%, 10%, 15%)

Usage:
    uv run python scripts/generate_curated_dataset.py
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import torch
from tqdm import tqdm
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcl_watermark.enhanced_mcl import EnhancedMCLGenerator, EnhancedMCLDetector
from transformers import AutoModelForCausalLM, AutoTokenizer


# Curated 176 concepts from user
WIKIPEDIA_CONCEPTS = [
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
    "Extraterrestrial_life", "Ghost", "Vampire", "Zombie"
]

def build_mcl_configs(topology: str = "soft_cycle", states=None, overlaps=None):
    """Build MCL configurations for given state/overlap combinations."""
    if states is None:
        states = [2, 4, 5, 7, 9, 11, 15]
    if overlaps is None:
        overlaps = [0, 5, 10, 15]
    configs = []
    for num_states in states:
        for overlap_pct in overlaps:
            overlap = overlap_pct / 100.0
            configs.append({
                "name": f"states{num_states}_overlap{overlap_pct}pct",
                "num_states": num_states,
                "chain_key": topology,
                "overlap": overlap,
            })
    return configs


def generate_prompt(concept: str) -> str:
    """Create a prompt from a Wikipedia concept."""
    readable = concept.replace("_", " ").replace("(", "- ").replace(")", "")
    return f"Explain {readable} in a comprehensive way."


# -----------------------------------------------------------------------------
# Cross-domain prompt pools (used with --domain {wiki, news, social, abstract})
# -----------------------------------------------------------------------------

NEWS_PROMPTS = [
    "Write a 200-word news brief about recent advances in renewable energy storage.",
    "Write a wire-style report covering today's developments in central bank policy.",
    "Summarise this week's major geopolitical events affecting global trade.",
    "Write a short BBC-style article on a recent breakthrough in cancer research.",
    "Report on the outcome of the latest United Nations climate summit.",
    "Cover a major sporting event final in 200 words.",
    "Write a market-recap piece on today's tech stock performance.",
    "Describe a recent space mission and its scientific objectives.",
    "Write a news piece on a notable cyber-attack on critical infrastructure.",
    "Cover a recent natural disaster and the international response.",
    "Report on a recent supreme court ruling and its political implications.",
    "Write an obituary for a fictional Nobel laureate in physics.",
    "Cover a recent diplomatic incident between two major powers.",
    "Write a profile of a tech CEO announcing a major product launch.",
    "Report on a vaccine rollout in a developing country.",
    "Write a feature on a recent labour strike at a major automaker.",
    "Cover the launch of a new high-speed rail line in Asia.",
    "Write a piece on rising food prices and their causes.",
    "Report on the discovery of a previously unknown archaeological site.",
    "Write a news article on a major art-forgery investigation.",
    "Cover a recent reshuffle in a major government cabinet.",
    "Write a piece on regulatory action against a large social-media platform.",
    "Report on a high-profile corporate merger announcement.",
    "Cover the impact of a new tariff regime on global supply chains.",
    "Write a news article on the spread of an emerging infectious disease.",
    # --- expanded set (75 more) ---
    "Write a wire-style report on a contested election outcome in a major democracy.",
    "Cover a high-profile court verdict in a long-running corporate fraud case.",
    "Report on the bankruptcy of a once-dominant retail chain.",
    "Write a news brief about a new free-trade agreement between two continents.",
    "Cover an oil tanker accident in a sensitive marine ecosystem.",
    "Write a report on the discovery of a new exoplanet within the habitable zone.",
    "Cover a major hostage situation resolved by negotiation.",
    "Report on the release of an annual global hunger index.",
    "Write a news piece on a record-breaking drought in a major agricultural region.",
    "Cover a mass evacuation following a volcanic eruption.",
    "Report on a scientific consensus statement about microplastics in drinking water.",
    "Write a news brief about a coup attempt foiled by security services.",
    "Cover a celebrity's death and its impact on a creative industry.",
    "Report on a major sports league's decision to expand into a new continent.",
    "Write a news article about a state-sponsored cyber operation revealed by researchers.",
    "Cover a record-setting Initial Public Offering of a chip-design company.",
    "Report on a transit strike that paralyzed a major capital city.",
    "Write a news piece on ratification of a treaty banning autonomous weapons.",
    "Cover the unveiling of a new generation of fusion reactor prototype.",
    "Report on the criminal indictment of a former head of state.",
    "Write a news brief about a heat dome breaking temperature records across a continent.",
    "Cover an industrial accident at a chemical plant near a populated area.",
    "Report on a milestone in interstellar communication research.",
    "Write a news article about a major airline ordering a fleet of supersonic jets.",
    "Cover the resignation of a long-serving central bank governor.",
    "Report on widespread blackouts caused by a storm in a developed country.",
    "Write a news brief about a new vaccine receiving emergency-use authorization.",
    "Cover a class-action settlement against a social network for privacy violations.",
    "Report on a scientific expedition reaching the deepest part of the ocean.",
    "Write a news piece on a controversial monument being removed from a public square.",
    "Cover a religious leader's first foreign trip and its diplomatic implications.",
    "Report on the dismantling of a major drug-trafficking network.",
    "Write a news article about a record-breaking philanthropic donation.",
    "Cover the launch of a publicly-funded universal basic income pilot.",
    "Report on a Nobel Prize announcement in chemistry.",
    "Write a news brief about a major dam failure and downstream consequences.",
    "Cover the world's first commercial passenger flight on hydrogen fuel.",
    "Report on a high-profile arrest of a fugitive after a decade on the run.",
    "Write a news piece on a new international agreement on AI governance.",
    "Cover a scandal in a major academic publishing house.",
    "Report on a wildlife reintroduction program declared a success.",
    "Write a news brief about the release of a long-classified intelligence file.",
    "Cover a high-stakes diplomatic standoff over disputed islands.",
    "Report on the discovery of an ancient city buried beneath a desert.",
    "Write a news article about a record minimum sea-ice extent in the Arctic.",
    "Cover a major shareholder revolt at an automaker over EV strategy.",
    "Report on the first manned mission to land on Mars.",
    "Write a news piece on a controversial AI-generated film winning a major festival.",
    "Cover the launch of a continent-wide digital currency pilot.",
    "Report on the conclusion of a decade-long climate adaptation study.",
    "Write a news brief about a deadly fire at a high-rise residential tower.",
    "Cover the testimony of a tech CEO before a parliamentary inquiry.",
    "Report on a treaty banning seabed mining in international waters.",
    "Write a news article about a successful asteroid-deflection demonstration.",
    "Cover a stock-market flash crash blamed on algorithmic trading.",
    "Report on a major recall of a popular consumer electronics product.",
    "Write a news piece on a government-sponsored mass relocation due to coastal erosion.",
    "Cover the swearing-in of the youngest prime minister in a country's history.",
    "Report on a deadly stampede at a religious gathering.",
    "Write a news brief about a trans-oceanic submarine cable being severed.",
    "Cover a corporate espionage trial involving foreign nationals.",
    "Report on a country becoming the first to outlaw single-use plastics nationwide.",
    "Write a news article about a record number of refugees crossing a contested border.",
    "Cover the appointment of a new Secretary-General of the United Nations.",
    "Report on the launch of a deep-space probe targeting an outer-system moon.",
    "Write a news piece on the global response to a new strain of avian flu.",
    "Cover the legalization of euthanasia in a major Western country.",
    "Report on the unveiling of a privately-funded asteroid mining station.",
    "Write a news brief about a high-profile resignation at a central bank over inflation.",
    "Cover a contested referendum on a region's independence.",
    "Report on the world's largest battery storage facility coming online.",
    "Write a news article about a deadly chemical leak at a fertilizer plant.",
    "Cover the indictment of a major sports league official on bribery charges.",
    "Report on a record-setting volunteer effort to clean a polluted river.",
    "Write a news piece on the discovery of a mass grave from a recent conflict.",
    "Cover the international response to a previously unknown deep-sea organism.",
]

SOCIAL_PROMPTS = [
    "Write a Reddit r/AskScience top-comment answer about why the sky is blue.",
    "Compose an X (Twitter) thread of 6 tweets explaining inflation to a layperson.",
    "Write a Reddit r/Cooking comment sharing a personal sourdough recipe.",
    "Compose a Reddit r/Personalfinance answer about whether to pay off debt or invest.",
    "Write a sarcastic but informative X thread on why JavaScript dates are broken.",
    "Compose a Reddit r/legaladvice top comment carefully outlining tenant rights.",
    "Write a Reddit r/AmItheAsshole story from a workplace conflict.",
    "Compose a Reddit r/relationships comment offering empathetic advice.",
    "Write a Reddit r/programming comment discussing Rust vs Go for a backend service.",
    "Compose a viral X thread about your worst flight delay experience.",
    "Write a Reddit r/explainlikeimfive top answer about how vaccines work.",
    "Compose a Mastodon-style post about a hike you went on last weekend.",
    "Write a Reddit r/buildapc comment recommending a $1500 gaming build.",
    "Compose a heartfelt LinkedIn post about a recent job change.",
    "Write a Reddit r/MaliciousCompliance story from your time in retail.",
    "Compose an X thread debunking a common nutrition myth.",
    "Write a Reddit r/MechanicalKeyboards comment reviewing a new switch.",
    "Compose a parenting-subreddit post about a difficult bedtime routine.",
    "Write a Reddit r/travel comment recommending an itinerary for Lisbon.",
    "Compose an X thread on lessons learned from launching a side project.",
    "Write a Reddit r/MachineLearning discussion comment on a recent paper.",
    "Compose a Reddit r/Frugal comment with five concrete grocery-saving tips.",
    "Write a Reddit r/AskHistorians-style explanation of medieval guild structures.",
    "Compose an X thread on why your favourite local cafe just closed.",
    "Write a Reddit r/UpliftingNews comment summarising a feel-good story.",
    # --- expanded set (75 more) ---
    "Compose a Reddit r/wallstreetbets DD on a small-cap battery materials company.",
    "Write a Reddit r/keto comment summarising a 90-day weight-loss attempt.",
    "Compose a Reddit r/woodworking post asking for advice on a warped tabletop.",
    "Write a Reddit r/OutOfTheLoop top-comment explaining a sudden meme craze.",
    "Compose a viral X thread about being mistakenly added to a celebrity group chat.",
    "Write a Reddit r/changemyview comment arguing the unpopular side of a tax debate.",
    "Compose a heartfelt Reddit r/Parenting post about a teen's college rejection.",
    "Write a Reddit r/dataisbeautiful comment critiquing the colour scheme of a viral chart.",
    "Compose a Reddit r/AskCulinary thread on emulsifying without lecithin.",
    "Write a sarcastic X thread on a corporate rebranding gone wrong.",
    "Compose a Reddit r/EcoFriendly comment with a 30-day low-waste challenge plan.",
    "Write a Reddit r/CarTalk post diagnosing a recurring engine misfire.",
    "Compose an Instagram caption for a sunset hike in the Dolomites.",
    "Write a Reddit r/FinancialIndependence comment on FIRE math at age 35.",
    "Compose a Reddit r/AmITheAsshole story from a wedding seating dispute.",
    "Write an X thread defending a controversial casting choice for a remake.",
    "Compose a Reddit r/Bouldering comment with technique tips for slab climbs.",
    "Write a Reddit r/Buddhism post on integrating mindfulness into office work.",
    "Compose a Reddit r/PhotoCritique top comment on composition rules to break.",
    "Write a Reddit r/Frugal_Jerk satirical comment on extreme penny-pinching.",
    "Compose an X thread on common grant-writing pitfalls for academics.",
    "Write a Reddit r/legaladvice post asking about landlord retaliation in Europe.",
    "Compose a Reddit r/Showerthoughts top comment about cumulative sleep over a lifetime.",
    "Write a Reddit r/Coffee comment ranking pour-over brewers under $50.",
    "Compose an X thread roasting your former self's resume from five years ago.",
    "Write a Reddit r/Stoicism comment on applying Marcus Aurelius to chronic pain.",
    "Compose a Reddit r/recipes post asking for an heirloom tomato sauce variation.",
    "Write a Reddit r/Antiwork story about a successful union drive at a small cafe.",
    "Compose an X thread about an unexpectedly emotional graduation ceremony.",
    "Write a Reddit r/PetCare comment on transitioning a dog to a raw diet.",
    "Compose a Reddit r/HomeImprovement post asking how to retile a bathroom shower.",
    "Write a Reddit r/Math comment intuitively explaining the central limit theorem.",
    "Compose an X thread breaking down a recent biotech IPO's S-1.",
    "Write a Reddit r/AskScience comment on why time perception slows in emergencies.",
    "Compose a Reddit r/ZeroWaste post sharing a year's worth of compost yields.",
    "Write a Reddit r/WritingPrompts response opening with 'The lighthouse keeper...'.",
    "Compose a Reddit r/Chess analysis of a brilliancy from a recent grandmaster game.",
    "Write a sympathetic Reddit r/JustNoMIL comment offering boundary-setting language.",
    "Compose an X thread on a hidden bug in a popular web framework's router.",
    "Write a Reddit r/PostalPride comment defending the value of physical mail.",
    "Compose a Reddit r/Yoga post for beginners with anxiety-reducing flows.",
    "Write a Reddit r/SkincareAddiction comment on simplifying a 12-step routine.",
    "Compose an X thread on why most enterprise cybersecurity training is useless.",
    "Write a Reddit r/IRL comment about reconnecting with an estranged sibling.",
    "Compose a Reddit r/DataHoarder post on best practices for cold-storage backups.",
    "Write a Reddit r/AskBaking comment on rescuing under-proofed sourdough.",
    "Compose an X thread documenting a one-week digital detox experiment.",
    "Write a Reddit r/EngineeringStudents comment on surviving thermodynamics.",
    "Compose a Reddit r/rugby tactical analysis of a recent Six Nations match.",
    "Write a Reddit r/Meditation post about a difficult Vipassana retreat experience.",
    "Compose an X thread on the best (and worst) airport lounges in Asia.",
    "Write a Reddit r/SQL comment debugging a window-function ordering issue.",
    "Compose a Reddit r/Sneakers post evaluating the resale market for a recent drop.",
    "Write a Reddit r/Fitness comment giving a balanced 4-day push/pull split.",
    "Compose an X thread on a podcast episode that changed your career.",
    "Write a Reddit r/MealPrepSunday post with five $30 weekday-lunch recipes.",
    "Compose a Reddit r/cogsci comment summarising recent work on episodic memory.",
    "Write an Instagram-style caption for a candid wedding photo at golden hour.",
    "Compose a Reddit r/MachineLearning comment on reproducibility issues in NLP.",
    "Write a Reddit r/Indoorgardening post on lighting setups for tomato seedlings.",
    "Compose an X thread about negotiating a remote-work salary across time zones.",
    "Write a Reddit r/ExperiencedDevs comment on transitioning from IC to manager.",
    "Compose a Reddit r/Cycling post on commuting with electric assist in winter.",
    "Write a Reddit r/UrbanExploration tale of finding a forgotten subway tunnel.",
    "Compose an X thread on why climate communication needs better metaphors.",
    "Write a Reddit r/Aviation comment explaining go-arounds to a nervous flier.",
    "Compose a Reddit r/Anki post sharing a five-year learning streak retrospective.",
    "Write a Reddit r/CulinaryHistory comment on the origin of pho noodles.",
    "Compose an X thread on the biggest open problems in formal verification.",
    "Write a Reddit r/AskMen comment on grief after losing a long-time mentor.",
    "Compose a Reddit r/SeriousConversation post on the ethics of surveillance capitalism.",
    "Write a Reddit r/Spotify comment defending an under-appreciated album.",
    "Compose an X thread about a viral book you finally read and disliked.",
    "Write a Reddit r/Boardgames comment ranking the best two-player abstract games.",
    "Compose a Reddit r/Productivity post on quitting time-tracking apps for good.",
    "Write a Reddit r/Backpacking comment with packing-list optimisations under 8 lbs.",
]

ABSTRACT_PROMPTS = [
    "Write a 250-word academic abstract for a paper on graph neural networks for protein folding.",
    "Compose an abstract for a study on long-context transformers for legal document analysis.",
    "Write an abstract for an empirical study of LLM hallucinations under retrieval augmentation.",
    "Compose an abstract for a paper introducing a new benchmark for code generation.",
    "Write an abstract for a paper on differential privacy in federated learning.",
    "Compose an abstract for a clinical study of a novel SGLT2 inhibitor in heart failure.",
    "Write an abstract for an economics paper on minimum-wage effects in retail employment.",
    "Compose an abstract for a paper proposing a new attention mechanism for long sequences.",
    "Write an abstract for a paper on quantum error correction in surface codes.",
    "Compose an abstract for a paper on fairness in algorithmic credit scoring.",
    "Write an abstract for a paper on metal-organic frameworks for carbon capture.",
    "Compose an abstract for a sociological study on remote-work adoption post-2020.",
    "Write an abstract for a paper on parameter-efficient fine-tuning for large language models.",
    "Compose an abstract for a study on microbiome composition and depression.",
    "Write an abstract for a paper on hardware-aware neural architecture search.",
    "Compose an abstract for an HCI paper on voice-assistant accessibility.",
    "Write an abstract for a paper on robust optimisation under demand uncertainty.",
    "Compose an abstract for a paper on watermarking image-generation models.",
    "Write an abstract for a study on adversarial robustness of speech recognition.",
    "Compose an abstract for a paper on lattice-based post-quantum cryptography.",
    "Write an abstract for a paper on mechanism design for online ad auctions.",
    "Compose an abstract for a study on deep-learning-based weather forecasting.",
    "Write an abstract for a paper on neural radiance fields for autonomous driving.",
    "Compose an abstract for a study on cosmological constraints from JWST observations.",
    "Write an abstract for a paper on continuous-time reinforcement learning for robotics.",
    # --- expanded set (75 more) ---
    "Compose an abstract for a paper introducing a new dataset for multilingual code generation.",
    "Write an abstract for a study on long-tail evaluation of vision-language models.",
    "Compose an abstract for a paper on mixed-precision training for trillion-parameter models.",
    "Write an abstract for a study on causal inference under unobserved confounders in healthcare.",
    "Compose an abstract for a paper on retrieval-augmented dialogue with episodic memory.",
    "Write an abstract for a study on Bayesian model averaging in macroeconomic forecasting.",
    "Compose an abstract for a paper on latent-diffusion-based protein design.",
    "Write an abstract for a clinical trial of a CRISPR-based sickle-cell therapy.",
    "Compose an abstract for an information-theoretic analysis of in-context learning.",
    "Write an abstract for a paper on equivariant graph networks for crystal property prediction.",
    "Compose an abstract for a study on adaptive sampling in variational quantum eigensolvers.",
    "Write an abstract for a paper on continual learning under non-stationary user preferences.",
    "Compose an abstract for an econometrics paper on zero-lower-bound monetary policy.",
    "Write an abstract for a study on learned indexes for memory-bounded vector search.",
    "Compose an abstract for a sociolinguistics paper on dialect drift in online communities.",
    "Write an abstract for a paper on reinforcement learning from holographic reduced representations.",
    "Compose an abstract for a paper on differential equations governing herd-immunity dynamics.",
    "Write an abstract for a study on topological data analysis of fMRI connectivity.",
    "Compose an abstract for a paper on fair allocation of indivisible goods with subsidy.",
    "Write an abstract for a paper introducing a benchmark for theorem proving with Lean 4.",
    "Compose an abstract for a study on hardware fingerprinting via cache side channels.",
    "Write an abstract for a paper on neural compression of high-dynamic-range video.",
    "Compose an abstract for a paper on fine-grained authorship attribution under stylometric attack.",
    "Write an abstract for a study on lattice QCD constraints on quark-gluon plasma viscosity.",
    "Compose an abstract for a paper on spectrum-efficient over-the-air federated learning.",
    "Write an abstract for an HCI study of voice-cloning watermark perception by lay listeners.",
    "Compose an abstract for a paper on convergence guarantees for Polyak-Lojasiewicz objectives.",
    "Write an abstract for a paper on graph neural networks for combinatorial optimisation.",
    "Compose an abstract for a study on differential gene expression in long-COVID patients.",
    "Write an abstract for a paper on probabilistic forecasting of extreme precipitation events.",
    "Compose an abstract for a paper on adversarial robustness of large multi-modal foundation models.",
    "Write an abstract for a study on power-law scaling of compute-optimal pre-training.",
    "Compose an abstract for a paper on mixed-effects modeling of phonetic variation in bilinguals.",
    "Write an abstract for a paper on model-based reinforcement learning for nuclear plasma control.",
    "Compose an abstract for a paper on programmable matter using DNA origami.",
    "Write an abstract for a study on semantic drift in long-context summarisation.",
    "Compose an abstract for a paper on minimax-optimal off-policy evaluation in contextual bandits.",
    "Write an abstract for a study on bias-variance tradeoffs in self-distillation.",
    "Compose an abstract for a paper on quantum advantage for the maximum-clique problem.",
    "Write an abstract for a paper on CRISPR base editing in primary T cells.",
    "Compose an abstract for a sociology paper on the gig economy's effect on local labour markets.",
    "Write an abstract for a study on the cosmological lithium problem revisited with new abundance data.",
    "Compose an abstract for a paper on Lipschitz-continuous representations for classifier robustness.",
    "Write an abstract for a paper on agent-based simulation of pandemic non-pharmaceutical interventions.",
    "Compose an abstract for a study on neuromorphic spiking transformers.",
    "Write an abstract for a paper on robust regression with heavy-tailed contamination.",
    "Compose an abstract for a paper on knowledge-graph completion with hyperbolic embeddings.",
    "Write an abstract for a study on optimal-transport-based domain adaptation for medical imaging.",
    "Compose an abstract for a paper on energy-based models for Monte Carlo inference in statistical physics.",
    "Write an abstract for an econometrics paper on natural-experiment identification of school-choice effects.",
    "Compose an abstract for a paper on scalable Bayesian neural networks via Laplace approximation.",
    "Write an abstract for a study on the anti-microbial peptide landscape of marine sediments.",
    "Compose an abstract for a paper on satisfiability-modulo-theories solvers for cryptanalysis.",
    "Write an abstract for a paper on programming-by-example for spreadsheet automation.",
    "Compose an abstract for a paper on latent-action world models for embodied agents.",
    "Write an abstract for a clinical study on adjunctive psilocybin for treatment-resistant depression.",
    "Compose an abstract for a paper on noise-robust diffusion sampling for text-to-image generation.",
    "Write an abstract for a study on quantum error mitigation via virtual distillation.",
    "Compose an abstract for a paper on differentiable physics simulators for soft-body manipulation.",
    "Write an abstract for a paper on counterfactual fairness in algorithmic hiring tools.",
    "Compose an abstract for a study on mesh-free finite element methods for fracture mechanics.",
    "Write an abstract for a paper on stochastic gradient Hamiltonian Monte Carlo for Bayesian deep learning.",
    "Compose an abstract for a paper on online auctions with strategic buyers under bandit feedback.",
    "Write an abstract for a study on volcanic precursor signals identified via deep change-point detection.",
    "Compose an abstract for a paper on lifelong meta-learning of structured exploration policies.",
    "Write an abstract for a study on hyperspectral imaging for early-stage skin cancer detection.",
    "Compose an abstract for a paper on principal-agent models of AI-assisted decision making.",
    "Write an abstract for a paper on graph contrastive learning under distribution shift.",
    "Compose an abstract for a study on mineralogical constraints on early Mars hydrology.",
    "Write an abstract for a paper on certified defences against adversarial patches in object detection.",
    "Compose an abstract for a paper on equilibrium models for liquid-cooling-aware data-center scheduling.",
    "Write an abstract for a study on long-context evaluation suites for open-source LLMs.",
    "Compose an abstract for a paper on causal mediation analysis of mass-vaccination policies.",
    "Write an abstract for a paper on tensor-network compression of attention layers.",
    "Compose an abstract for a study on self-supervised acoustic embeddings for whale-song clustering.",
    "Write an abstract for a paper on randomised controlled trials of personalised UBI policies.",
]


def get_prompts(domain: str):
    """Return (concepts, prompt_fn) for a given domain."""
    if domain == "wiki":
        return WIKIPEDIA_CONCEPTS, generate_prompt
    if domain == "news":
        return NEWS_PROMPTS, lambda p: p
    if domain == "social":
        return SOCIAL_PROMPTS, lambda p: p
    if domain == "abstract":
        return ABSTRACT_PROMPTS, lambda p: p
    raise ValueError(f"Unknown domain: {domain}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens per generation (default 512 for longer sequences; "
                             "the empirical FPR-vs-length plot in Section 6 also benefits from "
                             "having long samples to subsample).")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--topology", default="soft_cycle", choices=["clockwork", "binary", "soft_cycle"],
                        help="Transition topology: clockwork, binary, or soft_cycle")
    parser.add_argument("--states", type=int, nargs="+", default=[2, 4, 5, 7, 9, 11, 15],
                        help="Which state counts to run (e.g. --states 2 4 5)")
    parser.add_argument("--overlaps", type=int, nargs="+", default=[0, 5, 10, 15],
                        help="Which overlap percentages to run (e.g. --overlaps 0 5 10 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--decoding", choices=["greedy", "sampling"], default="greedy",
                        help="Non-watermarked baseline decoding: greedy (matches watermarked default) "
                             "or sampling (T=0.7, top_p=0.9). Use 'greedy' for fair PPL comparison.")
    parser.add_argument("--domain", choices=["wiki", "news", "social", "abstract"], default="wiki",
                        help="Prompt domain. 'wiki' uses the original 173 concepts; the others draw "
                             "from cross-domain prompt pools (100 each) for cross-domain evaluation.")
    parser.add_argument("--n-prompts", type=int, default=None,
                        help="If set, cap the number of prompts at N (deterministic prefix). "
                             "Use 100 for a balanced cross-domain comparison.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for the non-watermarked baseline. Recommended: 8 for "
                             "Llama-3.2-3B on A100-80GB, 16 for gemma-1B. The watermarked path "
                             "(EnhancedMCLGenerator) currently runs prompt-by-prompt regardless.")
    parser.add_argument("--skip-non-watermarked", action="store_true", help="Skip non-watermarked generation")
    parser.add_argument("--resume-from-config", type=str, default=None, help="Resume from specific config name")
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    secret_key = "curated_wiki_dataset_2024"

    concepts, prompt_fn = get_prompts(args.domain)
    if args.n_prompts is not None and args.n_prompts > 0:
        concepts = concepts[: args.n_prompts]
    MCL_CONFIGS = build_mcl_configs(args.topology, args.states, args.overlaps)

    print("=" * 80)
    print("CURATED WIKIPEDIA MCL DATASET GENERATOR")
    print("=" * 80)
    print(f"Concepts: {len(concepts)}")
    print(f"Topology: {args.topology}")
    print(f"Configurations: {len(MCL_CONFIGS)} + non-watermarked")
    print(f"Total generations: {len(concepts) * (len(MCL_CONFIGS) + 1)}")
    print(f"States: 2, 4, 5, 7, 9, 11, 15")
    print(f"Overlaps: 0%, 5%, 10%, 15%")
    print(f"Seed: {args.seed}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"data/curated_wiki_dataset_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. GENERATE NON-WATERMARKED
    # =========================================================================
    if not args.skip_non_watermarked:
        print("\n" + "=" * 80)
        print("1. GENERATING NON-WATERMARKED BASELINE")
        print("=" * 80)
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Left-pad so each prompt's generation continues at its own real end.
        tokenizer.padding_side = "left"

        prompts = [prompt_fn(c) for c in concepts]
        non_wm_samples = []
        bs = max(1, args.batch_size)

        for start in tqdm(range(0, len(concepts), bs), desc=f"Non-watermarked (bs={bs})"):
            chunk_concepts = concepts[start:start + bs]
            chunk_prompts = prompts[start:start + bs]
            enc = tokenizer(
                chunk_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=args.max_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                if args.decoding == "greedy":
                    gen_kwargs["do_sample"] = False
                else:
                    gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
                gen_out = model.generate(**gen_kwargs)
                outputs = gen_out.sequences
                # Compute per-sample mean log-prob of the chosen token under
                # the unmasked next-token distribution -> perplexity.
                # gen_out.scores: tuple (n_steps,) of [batch, vocab] logits
                if gen_out.scores:
                    log_probs_per_step = []
                    for t, step_logits in enumerate(gen_out.scores):
                        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
                        chosen = outputs[:, enc["input_ids"].shape[1] + t]
                        chosen_lp = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
                        log_probs_per_step.append(chosen_lp)
                    lp_matrix = torch.stack(log_probs_per_step, dim=1)  # [batch, n_steps]
                    # Mask pad-token positions (sequences that hit EOS early)
                    in_len = enc["input_ids"].shape[1]
                    gen_only = outputs[:, in_len:]
                    # Build a mask that is 1 for positions up to and including
                    # the FIRST eos (if any), 0 thereafter. This counts the
                    # legitimate end-of-generation EOS in PPL but excludes
                    # subsequent right-padding eos tokens.
                    is_eos = (gen_only == tokenizer.eos_token_id)
                    # arange[None, :] <= first_eos[:, None]; if no eos, keep all.
                    arange = torch.arange(gen_only.shape[1], device=gen_only.device)
                    has_eos = is_eos.any(dim=1)
                    first_eos = torch.where(has_eos, is_eos.int().argmax(dim=1),
                                            torch.full((gen_only.shape[0],),
                                                       gen_only.shape[1] - 1,
                                                       device=gen_only.device))
                    valid = (arange[None, :] <= first_eos[:, None]).float()
                    n_valid = valid.sum(dim=1).clamp(min=1)
                    mean_lp = (lp_matrix * valid).sum(dim=1) / n_valid
                    sample_ppls = torch.exp(-mean_lp).cpu().tolist()
                    n_valid_list = n_valid.cpu().tolist()
                else:
                    sample_ppls = [None] * outputs.size(0)
                    n_valid_list = [0] * outputs.size(0)

            input_len = enc["input_ids"].shape[1]
            for i, (concept, prompt) in enumerate(zip(chunk_concepts, chunk_prompts)):
                gen_ids_t = outputs[i, input_len:]
                text = tokenizer.decode(gen_ids_t, skip_special_tokens=True)
                ppl = sample_ppls[i]
                non_wm_samples.append({
                    "concept": concept,
                    "prompt": prompt,
                    "text": text,
                    "type": "non_watermarked",
                    "decoding": args.decoding,
                    "batch_size": bs,
                    "perplexity": float(ppl) if ppl is not None and ppl == ppl else None,
                    "n_scoring_tokens": int(n_valid_list[i]),
                })
        
        # Save
        with open(output_dir / "non_watermarked.jsonl", "w") as f:
            for s in non_wm_samples:
                f.write(json.dumps(s) + "\n")
        print(f"✓ Saved {len(non_wm_samples)} non-watermarked samples")
        
        del model
        torch.cuda.empty_cache()
    
    # =========================================================================
    # 2. GENERATE WATERMARKED FOR EACH CONFIG
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"2. GENERATING WATERMARKED ({len(MCL_CONFIGS)} CONFIGURATIONS)")
    print("=" * 80)
    
    summary_stats = []
    skip_until_found = args.resume_from_config is not None
    
    for config in MCL_CONFIGS:
        config_name = config["name"]
        num_states = config["num_states"]
        chain_key = config["chain_key"]
        overlap = config["overlap"]
        
        # Handle resume
        if skip_until_found:
            if config_name == args.resume_from_config:
                skip_until_found = False
            else:
                print(f"Skipping {config_name} (resuming from {args.resume_from_config})")
                continue
        
        print(f"\n--- {config_name} ---")
        
        generator = EnhancedMCLGenerator(
            model_name=args.model,
            secret_key=secret_key,
            num_states=num_states,
            chain_key=chain_key,
            overlap_ratio=overlap,
            device=device
        )
        
        detector = EnhancedMCLDetector(
            tokenizer_name=args.model,
            secret_key=secret_key,
            num_states=num_states,
            chain_key=chain_key,
            overlap_ratio=overlap,
            detection_threshold=0.5
        )
        
        samples = []
        scores = []
        ppls = []
        detected = 0
        
        for concept in tqdm(concepts, desc=config_name):
            prompt = prompt_fn(concept)
            text, meta = generator.generate(prompt, max_new_tokens=args.max_tokens)
            result = detector.detect(text)
            
            samples.append({
                "concept": concept,
                "prompt": prompt,
                "text": text,
                "type": f"watermarked_{config_name}",
                "num_states": num_states,
                "overlap": overlap,
                "perplexity": meta["perplexity"],
                "empirical_mean_kl_nats": meta.get("empirical_mean_kl_nats"),
                "empirical_p95_kl_nats": meta.get("empirical_p95_kl_nats"),
                "empirical_max_kl_nats": meta.get("empirical_max_kl_nats"),
                "log_S_bound_nats": meta.get("log_S_bound_nats"),
                "chain_score": result.chain_score,
                "is_detected": result.is_watermarked,
            })
            
            scores.append(result.chain_score)
            ppls.append(meta["perplexity"])
            if result.is_watermarked:
                detected += 1
        
        # Save incrementally
        with open(output_dir / f"{config_name}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        avg_score = np.mean(scores)
        avg_ppl = np.mean(ppls)
        detect_rate = detected / len(concepts)
        
        summary_stats.append({
            "config": config_name,
            "num_states": num_states,
            "overlap": overlap,
            "avg_score": avg_score,
            "avg_ppl": avg_ppl,
            "detection_rate": detect_rate,
            "baseline": detector.expected_random,
        })
        
        print(f"  Score={avg_score:.4f}, PPL={avg_ppl:.2f}, Detection={detect_rate*100:.0f}%")
        
        # Save summary incrementally
        summary = {
            "timestamp": timestamp,
            "model": args.model,
            "secret_key": secret_key,
            "seed": args.seed,
            "concepts_count": len(concepts),
            "configs": MCL_CONFIGS,
            "stats": summary_stats,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        del generator
        del detector
        torch.cuda.empty_cache()
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Config':<30} {'States':>6} {'Overlap':>8} {'Score':>8} {'PPL':>8} {'Detect':>8}")
    print("-" * 80)
    for s in summary_stats:
        print(f"{s['config']:<30} {s['num_states']:>6} {s['overlap']*100:>7.0f}% {s['avg_score']:>8.4f} {s['avg_ppl']:>8.2f} {s['detection_rate']*100:>7.0f}%")
    
    print(f"\n✓ Dataset saved to: {output_dir}")
    print(f"  Files: {len(list(output_dir.glob('*.jsonl')))} JSONL + summary.json")


if __name__ == "__main__":
    main()
