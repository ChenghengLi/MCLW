#!/usr/bin/env python3
"""Empirical FPR experiment: generate non-watermarked text from each model,
score under MCL and KGW detectors, compute the empirical FPR @ z>2.326.

Output: data/v7_min/exp6_fpr_<model>/records.jsonl with fields
    text, token_ids, n_tokens, ppl
    z_mcl, phi_mcl, z_kgw, phi_kgw

Usage: python3 measure_fpr.py --model <hf_id> --n-prompts 200 [--int8]
"""
from __future__ import annotations
import argparse, json, math, hashlib, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ltw_watermark.prompts_4domain import get_prompts, DOMAINS

# Constants
SECRET_KEY = "mclw_v7_min_2026"
MCL_S = 5
KGW_GAMMA = 0.5
TEMPERATURE = 0.7
ZA = 2.326

# ---- MCL detector ---------------------------------------------------------
_TOK_STATES = {}
def init_state_partition(secret_key, vocab_size, num_states):
    key = (secret_key, vocab_size, num_states)
    if key not in _TOK_STATES:
        states = np.zeros(vocab_size, dtype=np.int16)
        for t in range(vocab_size):
            d = hashlib.sha256(f"{secret_key}|{t}".encode()).digest()
            states[t] = int.from_bytes(d[:4], "little") % num_states
        _TOK_STATES[key] = states
    return _TOK_STATES[key]

def mcl_z(token_ids, secret_key, num_states, vocab_size):
    if len(token_ids) < 2: return float("nan"), float("nan")
    states = init_state_partition(secret_key, vocab_size, num_states)
    p0 = 1.0/num_states
    n_eval = len(token_ids) - 1
    n_valid = sum(1 for i in range(1, len(token_ids))
                  if int(states[int(token_ids[i])]) == (int(states[int(token_ids[i-1])]) + 1) % num_states)
    phi = n_valid/n_eval
    z = (phi - p0) / np.sqrt(p0*(1-p0)/n_eval)
    return float(phi), float(z)

# ---- KGW detector ---------------------------------------------------------
_TOK_HASH = {}
def init_kgw_hash(secret_key, vocab_size):
    key = (secret_key, vocab_size)
    if key not in _TOK_HASH:
        h = np.zeros(vocab_size, dtype=np.uint32)
        for t in range(vocab_size):
            d = hashlib.sha256(f"{secret_key}|tok|{t}".encode()).digest()
            h[t] = int.from_bytes(d[:4], "little")
        _TOK_HASH[key] = h.astype(np.int64)
    return _TOK_HASH[key]

def kgw_z(token_ids, secret_key, gamma, vocab_size):
    if len(token_ids) < 2: return float("nan"), float("nan")
    H = init_kgw_hash(secret_key, vocab_size)
    threshold = int(round(gamma * (1 << 32)))
    n_green = 0
    for i in range(1, len(token_ids)):
        prev = int(token_ids[i-1]); cur = int(token_ids[i])
        if (H[cur] + H[prev]) & 0xFFFFFFFF < threshold:
            n_green += 1
    n_eval = len(token_ids) - 1
    phi = n_green/n_eval
    z = (n_green - gamma*n_eval) / np.sqrt(n_eval * gamma * (1-gamma))
    return float(phi), float(z)

# ---- Plain LM generation --------------------------------------------------
@torch.no_grad()
def lm_generate(model, tokenizer, prompt, max_tokens, temperature, device):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]; prompt_len = input_ids.shape[1]
    log_probs = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max_tokens):
        out = model(input_ids)
        logits = out.logits[0,-1].float()
        probs = F.softmax(logits/temperature, dim=-1)
        next_token = int(torch.multinomial(probs, 1).item())
        nat_lp = F.log_softmax(logits, dim=-1)
        log_probs.append(float(nat_lp[next_token].item()))
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
        if eos_id is not None and next_token == eos_id: break
    gen = input_ids[0, prompt_len:].cpu().tolist()
    n = len(gen)
    if n == 0: return None
    ppl = float(np.exp(-np.mean(log_probs)))
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return {"token_ids": gen, "text": text, "n_tokens": n, "ppl": ppl}

def short_name(model_id): return model_id.replace("/", "_").replace(".", "-").lower().split("_",1)[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--out-root", default=str(REPO/"data/v7_min"))
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_root) / f"exp6_fpr_{short_name(args.model)}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[exp6_fpr] model={args.model} n_prompts={args.n_prompts} out={out_dir}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.int8:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size
    print(f"[exp6_fpr] model loaded, V={vocab_size}, device={device}", flush=True)
    init_state_partition(SECRET_KEY, vocab_size, MCL_S)
    init_kgw_hash(SECRET_KEY, vocab_size)

    # Pull prompts: 50 prompts × 4 domains = 200 non-watermarked samples per model
    prompts = []
    for d in DOMAINS:
        try: prompts.extend([(d, p) for p in get_prompts(d, n=args.n_prompts)])
        except Exception as e: print(f"  skip {d}: {e}", flush=True)

    rec_path = out_dir / "records.jsonl"
    n_records = 0
    with rec_path.open("w") as fh:
        for i, (dom, p) in enumerate(prompts):
            t0 = time.time()
            res = lm_generate(model, tok, p, args.max_tokens, TEMPERATURE, device)
            if res is None: continue
            phi_mcl, z_mcl_v = mcl_z(res["token_ids"], SECRET_KEY, MCL_S, vocab_size)
            phi_kgw, z_kgw_v = kgw_z(res["token_ids"], SECRET_KEY, KGW_GAMMA, vocab_size)
            rec = {"idx": i, "domain": dom, "model": args.model,
                   "n_tokens": res["n_tokens"], "ppl": res["ppl"],
                   "text": res["text"], "token_ids": res["token_ids"],
                   "phi_mcl": phi_mcl, "z_mcl": z_mcl_v,
                   "phi_kgw": phi_kgw, "z_kgw": z_kgw_v}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            n_records += 1
            if (i+1) % 25 == 0:
                print(f"  [exp6_fpr] {i+1}/{len(prompts)} dom={dom} z_mcl={z_mcl_v:.2f} z_kgw={z_kgw_v:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # Summary
    recs = [json.loads(l) for l in rec_path.read_text().splitlines() if l.strip()]
    z_mcl_all = [r["z_mcl"] for r in recs if not math.isnan(r["z_mcl"])]
    z_kgw_all = [r["z_kgw"] for r in recs if not math.isnan(r["z_kgw"])]
    fpr_mcl = sum(z>ZA for z in z_mcl_all)/len(z_mcl_all)
    fpr_kgw = sum(z>ZA for z in z_kgw_all)/len(z_kgw_all)
    summary = {"model": args.model, "n_records": n_records,
               "fpr_mcl": fpr_mcl, "fpr_kgw_sweet": fpr_kgw,
               "mean_z_mcl": float(np.mean(z_mcl_all)) if z_mcl_all else None,
               "mean_z_kgw": float(np.mean(z_kgw_all)) if z_kgw_all else None}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[exp6_fpr] DONE n={n_records}  FPR_MCL={fpr_mcl:.2%}  FPR_KGW(=SWEET)={fpr_kgw:.2%}", flush=True)

if __name__ == "__main__": main()
