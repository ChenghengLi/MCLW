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
                             "from cross-domain prompt pools (~25 each) for cross-domain evaluation.")
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
