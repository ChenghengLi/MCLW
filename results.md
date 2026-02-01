# MCL Watermarking Experiment Results

## Experiment Overview

**Model:** `meta-llama/Llama-3.2-3B-Instruct`  
**Dataset:** 173 curated Wikipedia concepts  
**Seed:** 42 (for reproducibility)  
**Secret Key:** `curated_wiki_dataset_2024`  
**Date:** February 1, 2026

### Configurations Tested
- **States:** 2, 4, 5, 7, 9, 11, 15
- **Overlaps:** 0%, 5%, 10%, 15%
- **Total configurations:** 28

---

## Summary Results

### Non-Watermarked Text Baseline Scores

| States | Avg Score | Max Score | FPR | Expected Baseline |
|--------|-----------|-----------|-----|-------------------|
| 2 | 1.0000 | 1.0000 | 100% | 1.00 |
| 4 | 0.4890 | 0.6067 | 33.5% | 0.50 |
| 5 | 0.4021 | 0.5000 | **0.0%** | 0.40 |
| 7 | 0.2901 | 0.4000 | **0.0%** | 0.29 |
| 9 | 0.2213 | 0.3333 | **0.0%** | 0.22 |
| 11 | 0.1971 | 0.3133 | **0.0%** | 0.18 |
| 15 | 0.1376 | 0.2467 | **0.0%** | 0.13 |

---

## Detailed Comparison: Watermarked vs Non-Watermarked

| Config | States | Overlap | Non-WM Avg | WM Avg | Gap | WM Det% | Non-WM FPR% | PPL | Verdict |
|--------|--------|---------|------------|--------|-----|---------|-------------|-----|---------|
| states2_overlap0pct | 2 | 0% | 1.0000 | 1.0000 | 0.00 | 100% | 100% | 1.29 | ❌ POOR |
| states2_overlap5pct | 2 | 5% | 1.0000 | 1.0000 | 0.00 | 100% | 100% | 1.29 | ❌ POOR |
| states2_overlap10pct | 2 | 10% | 1.0000 | 1.0000 | 0.00 | 100% | 100% | 1.29 | ❌ POOR |
| states2_overlap15pct | 2 | 15% | 1.0000 | 1.0000 | 0.00 | 100% | 100% | 1.29 | ❌ POOR |
| **states4_overlap0pct** | 4 | 0% | 0.4890 | 0.9887 | **+0.50** | 100% | 33.5% | 4.89 | ✅ GOOD |
| states4_overlap5pct | 4 | 5% | 0.4890 | 0.8424 | +0.35 | 99.4% | 33.5% | 4.48 | ✅ GOOD |
| states4_overlap10pct | 4 | 10% | 0.4890 | 0.6959 | +0.21 | 98.8% | 33.5% | 4.00 | ⚠️ MARGINAL |
| states4_overlap15pct | 4 | 15% | 0.4890 | 0.6261 | +0.14 | 96.0% | 33.5% | 3.84 | ❌ POOR |
| **states5_overlap0pct** | 5 | 0% | 0.4021 | 0.9915 | **+0.59** | 100% | **0.0%** | 4.98 | ✅ EXCELLENT |
| states5_overlap5pct | 5 | 5% | 0.4021 | 0.7260 | +0.32 | 96.5% | 0.0% | 4.74 | ✅ GOOD |
| states5_overlap10pct | 5 | 10% | 0.4021 | 0.6391 | +0.24 | 95.4% | 0.0% | 4.86 | ⚠️ MARGINAL |
| states5_overlap15pct | 5 | 15% | 0.4021 | 0.5418 | +0.14 | 70.5% | 0.0% | 4.22 | ❌ POOR |
| **states7_overlap0pct** | 7 | 0% | 0.2901 | 0.9908 | **+0.70** | 100% | **0.0%** | 4.20 | ✅ EXCELLENT |
| states7_overlap5pct | 7 | 5% | 0.2901 | 0.6430 | +0.35 | 91.9% | 0.0% | 4.76 | ✅ GOOD |
| states7_overlap10pct | 7 | 10% | 0.2901 | 0.4771 | +0.19 | 35.3% | 0.0% | 5.32 | ⚠️ MARGINAL |
| states7_overlap15pct | 7 | 15% | 0.2901 | 0.4138 | +0.12 | 12.7% | 0.0% | 5.01 | ❌ POOR |
| **states9_overlap0pct** | 9 | 0% | 0.2213 | 0.9839 | **+0.76** | 100% | **0.0%** | 5.37 | ✅ EXCELLENT |
| states9_overlap5pct | 9 | 5% | 0.2213 | 0.5310 | +0.31 | 60.1% | 0.0% | 5.25 | ✅ GOOD |
| states9_overlap10pct | 9 | 10% | 0.2213 | 0.3794 | +0.16 | 9.8% | 0.0% | 5.56 | ⚠️ MARGINAL |
| states9_overlap15pct | 9 | 15% | 0.2213 | 0.3070 | +0.09 | 0.6% | 0.0% | 5.40 | ❌ POOR |
| **states11_overlap0pct** | 11 | 0% | 0.1971 | 0.9875 | **+0.79** | 100% | **0.0%** | 4.61 | ✅ EXCELLENT |
| states11_overlap5pct | 11 | 5% | 0.1971 | 0.4314 | +0.23 | 27.2% | 0.0% | 6.18 | ⚠️ MARGINAL |
| states11_overlap10pct | 11 | 10% | 0.1971 | 0.3149 | +0.12 | 2.9% | 0.0% | 6.86 | ❌ POOR |
| states11_overlap15pct | 11 | 15% | 0.1971 | 0.2549 | +0.06 | 0.6% | 0.0% | 6.26 | ❌ POOR |
| **states15_overlap0pct** | 15 | 0% | 0.1376 | 0.9877 | **+0.85** | 100% | **0.0%** | 5.11 | ✅ EXCELLENT |
| states15_overlap5pct | 15 | 5% | 0.1376 | 0.3465 | +0.21 | 8.1% | 0.0% | 5.89 | ⚠️ MARGINAL |
| states15_overlap10pct | 15 | 10% | 0.1376 | 0.2573 | +0.12 | 1.2% | 0.0% | 8.05 | ❌ POOR |
| states15_overlap15pct | 15 | 15% | 0.1376 | 0.2029 | +0.07 | 1.2% | 0.0% | 7.10 | ❌ POOR |

---

## Best Configurations (0% Overlap)

| States | Score Gap | WM Detection | Non-WM FPR | PPL | Recommendation |
|--------|-----------|--------------|------------|-----|----------------|
| **7** | 0.70 | 100% | 0% | **4.20** | 🏆 **BEST OVERALL** |
| 11 | 0.79 | 100% | 0% | 4.61 | ✅ Excellent |
| 5 | 0.59 | 100% | 0% | 4.98 | ✅ Excellent |
| 15 | **0.85** | 100% | 0% | 5.11 | ✅ Highest Gap |
| 9 | 0.76 | 100% | 0% | 5.37 | ✅ Excellent |

---

## Key Findings

### 1. States Matter More Than Overlap
- **0% overlap with any states ≥5** achieves 100% detection with 0% false positives
- Adding overlap severely degrades detection at higher state counts

### 2. The "Sweet Spot"
- **7 states, 0% overlap** offers the best balance:
  - Lowest PPL (4.20) among high-performing configs
  - Large score gap (0.70) for easy separation
  - 100% true positive rate, 0% false positive rate

### 3. 2-State Configuration is Useless
- Cannot distinguish watermarked from non-watermarked text
- Both get score 1.0, resulting in 100% false positive rate

### 4. Overlap Degradation Pattern
- At 5% overlap: Detection drops to 60-90%
- At 10% overlap: Detection drops to 10-35% for states ≥7
- At 15% overlap: Detection effectively fails (<15%) for states ≥7

### 5. PPL Trade-offs
- PPL increases with more states (1.29 → ~5-7)
- Overlap can reduce PPL slightly but at the cost of detection accuracy

---

## Separability Analysis (0% Overlap Configs)

| States | WM Min Score | Non-WM Max Score | Overlap? |
|--------|--------------|------------------|----------|
| 5 | 0.6067 | 0.5000 | ✅ No overlap |
| 7 | 0.7067 | 0.4000 | ✅ No overlap |
| 9 | 0.6533 | 0.3333 | ✅ No overlap |
| 11 | 0.7200 | 0.3133 | ✅ No overlap |
| 15 | 0.7868 | 0.2467 | ✅ No overlap |

All 0% overlap configurations with ≥5 states show **perfect separability** - the worst watermarked sample scores higher than the best non-watermarked sample.

---

## Recommendations

### For Production Use
1. **7 states, 0% overlap** - Best balance of low PPL and high detection
2. **5 states, 0% overlap** - If lower PPL is critical

### Avoid
- 2 states (any overlap) - Cannot distinguish WM from non-WM
- 4 states (any overlap) - High FPR (33.5%)
- Any config with ≥10% overlap for states ≥7 - Poor detection rates

---

## Data Location

All experiment data is stored in:
```
/home/LTW/data/curated_wiki_dataset_20260201_112721/
├── non_watermarked.jsonl           # 173 non-watermarked samples
├── states{N}_overlap{X}pct.jsonl   # 28 watermarked config files
├── summary.json                    # Aggregated statistics
├── detailed_comparison.json        # Per-config comparison data
└── non_watermarked_detection_results.json  # Non-WM detection scores
```
