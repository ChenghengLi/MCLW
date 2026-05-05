# Tokenizer drift — detection under non-generator tokenizers

Source dataset: `curated_wiki_dataset_20260201_112721`, config `states7_overlap0pct`, S=7 ρ=0%, threshold τ=0.50.

| Tokenizer | n_wm | n_nwm | avg φ_wm | TPR (95% CI) | avg φ_nwm | FPR (95% CI) | k/S baseline |
|-----------|-----:|------:|---------:|---------------|----------:|---------------|-------------:|
| `meta-llama/Llama-3.1-8B-Instruct` | 173 | 173 | 0.9908 | 100.0% [97.8, 100.0] | 0.2901 | 0.0% [0.0, 2.2] | 0.2857 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 173 | 173 | 0.3040 | 2.3% [0.9, 5.8] | 0.2902 | 0.0% [0.0, 2.2] | 0.2857 |
| `gpt2-xl` | 173 | 173 | 0.2972 | 1.7% [0.6, 5.0] | 0.2818 | 0.0% [0.0, 2.2] | 0.2857 |
