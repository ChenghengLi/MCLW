#!/usr/bin/env python
"""
LTW Watermark Demo

A quick demonstration of Latent Trajectory Watermarking.
Run: uv run python -m ltw_watermark.demo
"""

from ltw_watermark.watermark import LTWWatermarker


def main():
    print("=" * 60)
    print("Latent Trajectory Watermarking - Demo")
    print("=" * 60)
    
    # Initialize detector
    secret_key = "demo-secret-key"
    detector = LTWWatermarker(
        secret_key=secret_key,
        rotation_strength=0.3,
        detection_threshold=0.1,
        use_differential_scoring=True,
        unit="sentence"
    )
    
    # Test texts
    ai_like_text = """
    Machine learning algorithms have demonstrated remarkable capabilities in pattern recognition.
    Deep neural networks extract hierarchical features from raw data automatically.
    Training procedures optimize millions of parameters through backpropagation.
    Model evaluation requires careful validation strategies.
    """
    
    human_like_text = """
    So I tried that new AI thing everyone's talking about.
    It's... weird? Like, sometimes it nails it, other times it's totally off.
    My friend says it's 'revolutionary' but idk.
    Guess we'll see how it goes!
    """
    
    print("\n[1] AI-like Text (formal, structured):")
    print(f"    \"{ai_like_text.strip()[:80]}...\"")
    result1 = detector.detect(ai_like_text)
    print(f"    Differential Score: {result1.differential_score:.4f}")
    print(f"    Detected as Watermarked: {result1.is_watermarked}")
    print(f"    Confidence: {result1.confidence:.2%}")
    
    print("\n[2] Human-like Text (casual, variable):")
    print(f"    \"{human_like_text.strip()[:80]}...\"")
    result2 = detector.detect(human_like_text)
    print(f"    Differential Score: {result2.differential_score:.4f}")
    print(f"    Detected as Watermarked: {result2.is_watermarked}")
    print(f"    Confidence: {result2.confidence:.2%}")
    
    print("\n[3] Testing with Wrong Key:")
    wrong_detector = LTWWatermarker(
        secret_key="wrong-key",
        rotation_strength=0.3,
        detection_threshold=0.1,
    )
    result3 = wrong_detector.detect(ai_like_text)
    print(f"    Same text, wrong key:")
    print(f"    Differential Score: {result3.differential_score:.4f}")
    print(f"    Detected: {result3.is_watermarked}")
    
    print("\n" + "=" * 60)
    print("Key Insight:")
    print("- Differential scoring reduces false positives")
    print("- Score = watermark_alignment - natural_coherence")
    print("- Human text has high natural coherence → low differential")
    print("=" * 60)


if __name__ == "__main__":
    main()
