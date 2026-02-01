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

from ltw_watermark.enhanced_mcl import EnhancedMCLGenerator, EnhancedMCLDetector
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

# MCL configurations: states 2, 4, 5, 7, 9, 11, 15 × overlaps 0%, 5%, 10%, 15%
MCL_CONFIGS = []
for num_states in [2, 4, 5, 7, 9, 11, 15]:
    for overlap_pct in [0, 5, 10, 15]:
        overlap = overlap_pct / 100.0
        MCL_CONFIGS.append({
            "name": f"states{num_states}_overlap{overlap_pct}pct",
            "num_states": num_states,
            "chain_key": "soft_cycle",
            "overlap": overlap,
        })


def generate_prompt(concept: str) -> str:
    """Create a prompt from a Wikipedia concept."""
    readable = concept.replace("_", " ").replace("(", "- ").replace(")", "")
    return f"Explain {readable} in a comprehensive way."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens per generation")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
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
    
    concepts = WIKIPEDIA_CONCEPTS
    
    print("=" * 80)
    print("CURATED WIKIPEDIA MCL DATASET GENERATOR")
    print("=" * 80)
    print(f"Concepts: {len(concepts)}")
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
        
        non_wm_samples = []
        for concept in tqdm(concepts, desc="Non-watermarked"):
            prompt = generate_prompt(concept)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            non_wm_samples.append({
                "concept": concept,
                "prompt": prompt,
                "text": text,
                "type": "non_watermarked",
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
            prompt = generate_prompt(concept)
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
