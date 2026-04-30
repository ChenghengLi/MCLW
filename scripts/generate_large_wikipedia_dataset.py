#!/usr/bin/env python
"""
Large-Scale Wikipedia MCL Dataset Generator

Generates text for 500+ Wikipedia concepts with:
- Non-watermarked baseline
- 24 MCL configurations (2-7 states × 0%, 5%, 10%, 15% overlap)

Usage:
    uv run python scripts/generate_large_wikipedia_dataset.py
    uv run python scripts/generate_large_wikipedia_dataset.py --max-concepts 50  # For testing
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcl_watermark.enhanced_mcl import EnhancedMCLGenerator, EnhancedMCLDetector
from transformers import AutoModelForCausalLM, AutoTokenizer


# All Wikipedia concepts from user
WIKIPEDIA_CONCEPTS = [
    # --- Top Articles of 2025 ---
    "Charlie_Kirk", "Deaths_in_2025", "Ed_Gein", "Donald_Trump", "Pope_Leo_XIV",
    "Elon_Musk", "Zohran_Mamdani", "Sinners_(2025_film)", "Ozzy_Osbourne", "Superman_(2025_film)",
    "Pope_Francis", "Severance_(TV_series)", "United_States", "Thunderbolts*", "Weapons_(2025_film)",
    "JD_Vance", "Adolescence_(TV_series)", "MrBeast", "Cristiano_Ronaldo", "Squid_Game_(season_2)",
    "David_Lynch", "Gene_Hackman", "Val_Kilmer", "Anora_(film)", "Captain_America:_Brave_New_World",
    "A_Minecraft_Movie", "Mickey_17", "28_Years_Later", "Nintendo_Switch_2", "Grand_Theft_Auto_VI",
    
    # --- Top Articles of 2024 ---
    "Deaths_in_2024", "Kamala_Harris", "2024_United_States_presidential_election",
    "Indian_Premier_League", "Deadpool_&_Wolverine", "Project_2025", "ChatGPT",
    "Taylor_Swift", "2024_Summer_Olympics", "UEFA_Euro_2024", "Joe_Biden",
    "Sean_Combs", "Robert_F._Kennedy_Jr.", "Dune:_Part_Two", "Liam_Payne",
    "Cillian_Murphy", "Oppenheimer_(film)", "Barbie_(film)", "Andrew_Tate",
    "Baby_Reindeer", "Bridgerton", "The_Boys_(TV_series)", "Shogun_(2024_TV_series)",
    
    # --- Perennial Most Viewed ---
    "Bible", "World_War_II", "World_War_I", "The_Beatles", "Michael_Jackson",
    "Barack_Obama", "Elizabeth_II", "YouTube", "Google", "Facebook",
    "India", "Adolf_Hitler", "Game_of_Thrones", "United_Kingdom", "Canada",
    "Russia", "China", "Japan", "Germany", "France", "Star_Wars", "Harry_Potter",
    "Albert_Einstein", "Leonardo_DiCaprio", "Abraham_Lincoln", "William_Shakespeare",
    "Isaac_Newton", "Earth", "Sun", "Moon", "Democracy", "Capitalism",
    
    # --- Tech & Science ---
    "Artificial_intelligence", "Machine_learning", "Deep_learning", "Neural_network",
    "Data_science", "Cloud_computing", "Blockchain", "Bitcoin", "Quantum_computing",
    "Nanotechnology", "Biotechnology", "Genetics", "DNA", "RNA", "CRISPR",
    "Virus", "Bacteria", "Immune_system", "Vaccine", "Cancer", "Diabetes",
    
    # --- Psychology & Sociology ---
    "Psychology", "Sociology", "Philosophy", "Religion", "Christianity", "Islam",
    "Hinduism", "Buddhism", "Judaism", "Greek_mythology", "Norse_mythology",
    
    # --- Entertainment ---
    "Music", "Literature", "Film", "Television", "Video_game", "Sport",
    "Football", "Basketball", "Tennis", "Olympics", "FIFA_World_Cup",
    "NBA", "NFL", "Premier_League", "UEFA_Champions_League", "Formula_One",
    
    # --- Video Games ---
    "Minecraft", "Fortnite", "Grand_Theft_Auto", "World_of_Warcraft",
    "League_of_Legends", "Elden_Ring", "God_of_War", "Cyberpunk_2077",
    "Pokemon", "Super_Mario", "The_Legend_of_Zelda",
    
    # --- Anime ---
    "One_Piece", "Naruto", "Dragon_Ball", "Attack_on_Titan", "Demon_Slayer",
    "Jujutsu_Kaisen", "Death_Note", "Studio_Ghibli", "Spirited_Away",
]

# MCL configurations to generate
MCL_CONFIGS = []
for num_states in [2, 3, 4, 5, 6, 7]:
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
    parser.add_argument("--max-concepts", type=int, default=100, help="Max concepts to process")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens per generation")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    secret_key = "large_wiki_dataset_2024"
    
    # Limit concepts
    concepts = WIKIPEDIA_CONCEPTS[:args.max_concepts]
    
    print("="*80)
    print("LARGE-SCALE WIKIPEDIA MCL DATASET GENERATOR")
    print("="*80)
    print(f"Concepts: {len(concepts)}")
    print(f"Configurations: {len(MCL_CONFIGS)} + non-watermarked")
    print(f"Total generations: {len(concepts) * (len(MCL_CONFIGS) + 1)}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"data/large_wiki_dataset_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. GENERATE NON-WATERMARKED
    # =========================================================================
    print("\n" + "="*80)
    print("1. GENERATING NON-WATERMARKED BASELINE")
    print("="*80)
    
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
    print("\n" + "="*80)
    print("2. GENERATING WATERMARKED (24 CONFIGURATIONS)")
    print("="*80)
    
    summary_stats = []
    
    for config in MCL_CONFIGS:
        config_name = config["name"]
        num_states = config["num_states"]
        chain_key = config["chain_key"]
        overlap = config["overlap"]
        
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
        
        # Save
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
        
        del generator
        del detector
        torch.cuda.empty_cache()
    
    # =========================================================================
    # 3. SAVE SUMMARY
    # =========================================================================
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "secret_key": secret_key,
        "concepts_count": len(concepts),
        "configs": MCL_CONFIGS,
        "stats": summary_stats,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Config':<30} {'States':>6} {'Overlap':>8} {'Score':>8} {'PPL':>8} {'Detect':>8}")
    print("-"*80)
    for s in summary_stats:
        print(f"{s['config']:<30} {s['num_states']:>6} {s['overlap']*100:>7.0f}% {s['avg_score']:>8.4f} {s['avg_ppl']:>8.2f} {s['detection_rate']*100:>7.0f}%")
    
    print(f"\n✓ Dataset saved to: {output_dir}")
    print(f"  Files: {len(list(output_dir.glob('*.jsonl')))} JSONL + summary.json")


if __name__ == "__main__":
    main()
