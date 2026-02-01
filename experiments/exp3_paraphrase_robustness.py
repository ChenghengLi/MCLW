#!/usr/bin/env python
"""
Experiment 3: Paraphrase Robustness

This experiment tests the robustness of LTW watermarks against paraphrasing attacks.
The hypothesis is that LTW should survive paraphrasing better than token-level
watermarks because it operates in semantic space.

Paraphrasing methods tested:
1. Synonym replacement
2. Sentence restructuring
3. Back-translation (simulated)
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark.watermark import LTWWatermarker
from ltw_watermark.embeddings import EmbeddingExtractor


# Sample watermarked texts (we'll paraphrase these)
ORIGINAL_TEXTS = [
    "Artificial intelligence systems are transforming modern healthcare. Diagnostic algorithms can detect diseases earlier than human doctors. Treatment recommendations are becoming more personalized. The future of medicine will be deeply integrated with AI.",
    "Climate change poses significant risks to global food security. Rising temperatures affect crop yields worldwide. Water scarcity is becoming more common. Agricultural practices must adapt to these new conditions.",
    "The stock market experienced significant volatility last quarter. Investors remained cautious amid economic uncertainty. Technology stocks led the recovery effort. Analysts predict continued fluctuations ahead.",
    "Machine learning models require substantial training data. Data quality directly impacts model performance. Preprocessing steps are crucial for success. Validation ensures models generalize properly.",
]


# Simple synonym dictionary for paraphrasing
SYNONYMS = {
    "transforming": ["changing", "revolutionizing", "altering", "reshaping"],
    "modern": ["contemporary", "current", "present-day", "today's"],
    "significant": ["substantial", "considerable", "major", "notable"],
    "detect": ["identify", "discover", "find", "recognize"],
    "future": ["coming years", "tomorrow", "prospects", "outlook"],
    "poses": ["presents", "creates", "introduces", "causes"],
    "global": ["worldwide", "international", "universal", "planetary"],
    "affect": ["impact", "influence", "alter", "change"],
    "becoming": ["getting", "growing", "turning", "evolving into"],
    "adapt": ["adjust", "modify", "change", "evolve"],
    "experienced": ["saw", "underwent", "witnessed", "faced"],
    "remained": ["stayed", "continued to be", "kept being", "persisted as"],
    "predict": ["forecast", "expect", "anticipate", "foresee"],
    "require": ["need", "demand", "necessitate", "call for"],
    "directly": ["immediately", "clearly", "obviously", "plainly"],
    "crucial": ["essential", "vital", "critical", "important"],
}


def synonym_replacement(text: str, replacement_rate: float = 0.3) -> str:
    """
    Replace words with synonyms.
    
    Args:
        text: Original text
        replacement_rate: Fraction of replaceable words to replace
        
    Returns:
        Paraphrased text
    """
    words = text.split()
    new_words = []
    
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        if word_lower in SYNONYMS and random.random() < replacement_rate:
            # Preserve capitalization
            synonym = random.choice(SYNONYMS[word_lower])
            if word[0].isupper():
                synonym = synonym.capitalize()
            # Preserve trailing punctuation
            if word[-1] in '.,!?;:':
                synonym += word[-1]
            new_words.append(synonym)
        else:
            new_words.append(word)
    
    return ' '.join(new_words)


def sentence_restructure(text: str) -> str:
    """
    Restructure sentences by reordering clauses and changing voice.
    
    This is a simplified version - real restructuring would use NLP libraries.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    restructured = []
    
    for sentence in sentences:
        # Simple restructuring: swap clauses around commas
        if ', ' in sentence and random.random() < 0.5:
            parts = sentence.split(', ', 1)
            if len(parts) == 2:
                # Swap and adjust capitalization
                new_sentence = parts[1].rstrip('.!?').capitalize() + ', ' + parts[0].lower()
                if sentence[-1] in '.!?':
                    new_sentence += sentence[-1]
                restructured.append(new_sentence)
                continue
        
        # Sometimes just keep the original
        restructured.append(sentence)
    
    return ' '.join(restructured)


def simulated_back_translation(text: str) -> str:
    """
    Simulate back-translation by applying multiple transformations.
    
    In reality, this would use translation APIs. We simulate it with:
    - Synonym replacement
    - Minor restructuring
    - Some word order changes
    """
    # Apply synonym replacement
    text = synonym_replacement(text, replacement_rate=0.4)
    
    # Apply restructuring
    text = sentence_restructure(text)
    
    # Minor word variations
    text = text.replace(" is ", " is being ")
    text = text.replace(" are ", " are currently ")
    text = text.replace("The ", "Currently, the ")
    
    return text


def calculate_semantic_similarity(extractor: EmbeddingExtractor, text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    emb1 = extractor.embed_text(text1)
    emb2 = extractor.embed_text(text2)
    return extractor.cosine_similarity(emb1, emb2)


def run_paraphrase_experiment(config: dict, results_dir: Path, quick: bool = False):
    """Run paraphrase robustness experiment."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Paraphrase Robustness")
    print("="*60)
    
    # Initialize detector
    print("\n[1/4] Initializing LTW detector...")
    detector = LTWWatermarker(
        secret_key=config['watermark']['secret_key'],
        rotation_strength=config['watermark']['rotation_strength'],
        detection_threshold=config['watermark']['detection_threshold'],
        use_differential_scoring=config['watermark']['use_differential_scoring'],
        unit=config['watermark']['unit']
    )
    
    extractor = EmbeddingExtractor()
    
    paraphrase_methods = {
        'synonym_replacement': lambda t: synonym_replacement(t, 0.3),
        'sentence_restructure': sentence_restructure,
        'back_translation': simulated_back_translation,
    }
    
    results = {method: [] for method in paraphrase_methods}
    results['original'] = []
    
    print("\n[2/4] Testing original texts...")
    for text in ORIGINAL_TEXTS:
        result = detector.detect(text)
        results['original'].append({
            'text': text[:50] + '...',
            'score': result.differential_score,
            'detected': result.is_watermarked,
        })
        print(f"  Original: score={result.differential_score:.4f}, detected={result.is_watermarked}")
    
    print("\n[3/4] Testing paraphrased versions...")
    all_results = []
    
    for method_name, method_func in paraphrase_methods.items():
        print(f"\n  Method: {method_name}")
        
        for i, original_text in enumerate(ORIGINAL_TEXTS):
            paraphrased = method_func(original_text)
            
            # Calculate semantic similarity
            similarity = calculate_semantic_similarity(extractor, original_text, paraphrased)
            
            # Detect watermark
            result = detector.detect(paraphrased)
            
            results[method_name].append({
                'original': original_text[:50] + '...',
                'paraphrased': paraphrased[:50] + '...',
                'semantic_similarity': similarity,
                'score': result.differential_score,
                'detected': result.is_watermarked,
            })
            
            all_results.append({
                'method': method_name,
                'text_id': i,
                'similarity': similarity,
                'score': result.differential_score,
                'detected': result.is_watermarked,
            })
            
            print(f"    Text {i+1}: similarity={similarity:.3f}, score={result.score:.4f}, detected={result.is_watermarked}")
    
    # Analysis
    print("\n[4/4] Analysis Results")
    print("-" * 60)
    
    # Summary by method
    summary = {}
    original_scores = [r['score'] for r in results['original']]
    original_detected = sum(1 for r in results['original'] if r['detected'])
    
    print(f"\nOriginal texts:")
    print(f"  Mean score: {np.mean(original_scores):.4f}")
    print(f"  Detection rate: {original_detected}/{len(results['original'])} = {original_detected/len(results['original']):.1%}")
    
    for method_name in paraphrase_methods:
        scores = [r['score'] for r in results[method_name]]
        similarities = [r['semantic_similarity'] for r in results[method_name]]
        detected = sum(1 for r in results[method_name] if r['detected'])
        
        score_retention = np.mean(scores) / np.mean(original_scores) if np.mean(original_scores) != 0 else 0
        
        summary[method_name] = {
            'mean_score': np.mean(scores),
            'mean_similarity': np.mean(similarities),
            'detection_rate': detected / len(results[method_name]),
            'score_retention': score_retention,
        }
        
        print(f"\n{method_name}:")
        print(f"  Mean score: {np.mean(scores):.4f} (retention: {score_retention:.1%})")
        print(f"  Mean similarity: {np.mean(similarities):.3f}")
        print(f"  Detection rate: {detected}/{len(results[method_name])} = {detected/len(results[method_name]):.1%}")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Bar chart of detection rates
    methods = ['original'] + list(paraphrase_methods.keys())
    detection_rates = [
        original_detected / len(results['original'])
    ] + [summary[m]['detection_rate'] for m in paraphrase_methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, detection_rates, color=['green'] + ['steelblue']*len(paraphrase_methods))
    ax.set_ylabel('Detection Rate', fontsize=12)
    ax.set_xlabel('Paraphrase Method', fontsize=12)
    ax.set_title('Watermark Detection Rate After Paraphrasing', fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='red', linestyle='--', label='50% baseline')
    
    # Add value labels
    for bar, rate in zip(bars, detection_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
    
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    fig.savefig(results_dir / 'exp3_detection_rates.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {results_dir / 'exp3_detection_rates.png'}")
    
    # Scatter: similarity vs score retention
    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name in paraphrase_methods:
        sims = [r['semantic_similarity'] for r in results[method_name]]
        scores = [r['score'] for r in results[method_name]]
        ax.scatter(sims, scores, label=method_name, s=100, alpha=0.7)
    
    ax.set_xlabel('Semantic Similarity to Original', fontsize=12)
    ax.set_ylabel('Differential Score', fontsize=12)
    ax.set_title('Semantic Similarity vs Watermark Score', fontsize=14)
    ax.axhline(y=config['watermark']['detection_threshold'], color='red', 
               linestyle='--', label=f'Threshold ({config["watermark"]["detection_threshold"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(results_dir / 'exp3_similarity_vs_score.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {results_dir / 'exp3_similarity_vs_score.png'}")
    
    # Save results
    with open(results_dir / 'exp3_results.json', 'w') as f:
        json.dump({
            'summary': summary,
            'detailed_results': results,
            'config': config['watermark'],
        }, f, indent=2, default=str)
    print(f"  Saved: {results_dir / 'exp3_results.json'}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Paraphrase Robustness")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'watermark': {
                'secret_key': 'test-secret-key',
                'rotation_strength': 0.3,
                'detection_threshold': 0.1,
                'use_differential_scoring': True,
                'unit': 'sentence'
            }
        }
    
    random.seed(42)  # Reproducibility
    
    results_dir = Path(__file__).parent.parent / "results" / "exp3"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_paraphrase_experiment(config, results_dir, args.quick)
    
    print("\n" + "="*60)
    print("EXPERIMENT 3 COMPLETE")
    print("="*60)
    
    # Key insight
    best_method = min(results, key=lambda m: results[m]['detection_rate'])
    print(f"\nMost challenging attack: {best_method} "
          f"(detection dropped to {results[best_method]['detection_rate']:.1%})")


if __name__ == "__main__":
    main()
