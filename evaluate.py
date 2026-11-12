import csv
import json
import os
import random

import numpy as np
import torch
from rouge_score import rouge_scorer

from batching import collate_pref, collate_sft
from common import (
    generate_summaries,
    load_causal_lm,
    load_eval_posts,
    load_preference_splits,
    load_tokenizer,
    summary_mask,
    token_logprobs,
)
from config import MAX_SUMMARY_TOKENS, SEED
from train_reward import RewardModel


@torch.no_grad()
def score_and_kl(policy, ref, rm, tok, posts, gen_ids, batch_size=16):
    device = next(policy.parameters()).device
    items = [
        {"prompt_ids": p["prompt_ids"], "summary_ids": g or [tok.eos_token_id]}
        for p, g in zip(posts, gen_ids)
    ]

    rewards, kls, lengths = [], [], []
    for start in range(0, len(items), batch_size):
        batch = collate_sft(items[start : start + batch_size], pad_id=tok.pad_token_id)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask = summary_mask(attn, batch["prompt_lens"])

        rewards.extend(rm(ids, attn).cpu().tolist())
        kl = ((token_logprobs(policy, ids, attn) - token_logprobs(ref, ids, attn)) * mask).sum(1)
        kls.extend(kl.cpu().tolist())
        lengths.extend(mask.sum(1).cpu().tolist())

    return np.array(rewards), np.array(kls), np.array(lengths)


def main():
    os.makedirs("results", exist_ok=True)
    tok = load_tokenizer()
    posts = load_eval_posts(tok)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    sft = load_causal_lm("checkpoints/sft").eval()
    device = next(sft.parameters()).device
    rm = RewardModel.load("checkpoints/reward", "checkpoints/sft", device=device).eval()
    models = {
        "SFT": sft,
        "PPO": load_causal_lm("checkpoints/ppo", device=device).eval(),
        "DPO": load_causal_lm("checkpoints/dpo", device=device).eval(),
    }

    res, table = {}, {}
    for name, model in models.items():
        sums, g_ids = generate_summaries(model, tok, posts, 16, MAX_SUMMARY_TOKENS, do_sample=False)
        r, kl, lengths = score_and_kl(model, sft, rm, tok, posts, g_ids)
        rouges = [scorer.score(p["summary"].strip(), s.strip())["rougeL"].fmeasure for p, s in zip(posts, sums)]
        res[name] = {"sums": sums, "r": r, "kl": kl, "len": lengths, "rg": np.array(rouges)}

    for name in ("SFT", "PPO", "DPO"):
        d = res[name]
        win = None if name == "SFT" else float((d["r"] > res["SFT"]["r"]).mean())
        table[name] = {
            "mean_r": float(d["r"].mean()),
            "win_vs_sft": win,
            "kl_to_sft": 0.0 if name == "SFT" else float(d["kl"].mean()),
            "length": float(d["len"].mean()),
            "rouge_l": float(d["rg"].mean()),
        }
        print(
            f"{name:4s} | r: {table[name]['mean_r']:+.4f} | win: {win or 0:.3f} "
            f"| KL: {table[name]['kl_to_sft']:.4f} | len: {table[name]['length']:.2f} "
            f"| ROUGE-L: {table[name]['rouge_l']:.4f}"
        )

    with open("results/eval_summaries.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(posts):
            row = {
                "id": i,
                "prompt": p["prompt"],
                "reference": p["summary"],
                **{k.lower(): res[k]["sums"][i] for k in models},
                **{f"{k.lower()}_r": float(res[k]["r"][i]) for k in models},
            }
            f.write(json.dumps(row) + "\n")

    with open("results/table3_eval.json", "w") as f:
        json.dump(table, f, indent=2)

    # Export 30 blind posts for human evaluation
    rng = random.Random(SEED)
    with open("results/human_eval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["post_id", "prompt", "summary_A", "summary_B", "summary_C", "key_A", "key_B", "key_C", "member1_pick", "member2_pick"])
        for idx in rng.sample(range(len(posts)), 30):
            order = ["SFT", "PPO", "DPO"]
            rng.shuffle(order)
            writer.writerow([idx, posts[idx]["prompt"], *[res[k]["sums"][idx] for k in order], *order, "", ""])

    # Export 3 preference pairs misclassified by the reward model
    _, pref_val = load_preference_splits(tok)
    wrong_pairs = []
    with torch.no_grad():
        for item in pref_val:
            b = collate_pref([item], pad_id=tok.pad_token_id)
            rw = rm(b["chosen_ids"].to(device), b["chosen_mask"].to(device)).item()
            rl = rm(b["rejected_ids"].to(device), b["rejected_mask"].to(device)).item()
            if rw <= rl:
                wrong_pairs.append(
                    {"prompt": item["prompt"], "chosen": item["chosen"], "rejected": item["rejected"], "r_chosen": rw, "r_rejected": rl}
                )
            if len(wrong_pairs) == 3:
                break
    with open("results/rm_errors.json", "w", encoding="utf-8") as f:
        json.dump(wrong_pairs, f, indent=2)

    # Export 10 qualitative samples (top 5 improved + bottom 5 vs SFT)
    deltas = np.maximum(res["PPO"]["rg"], res["DPO"]["rg"]) - res["SFT"]["rg"]
    order_idx = np.argsort(deltas)
    pick_10 = list(order_idx[-5:][::-1]) + list(order_idx[:5])
    with open("results/samples.md", "w", encoding="utf-8") as f:
        f.write("# Qualitative Samples (10 Evaluation Posts)\n\n")
        for rank, i in enumerate(pick_10, 1):
            tag = "PPO/DPO Better than SFT" if rank <= 5 else "PPO/DPO Worse than SFT"
            f.write(f"## Sample {rank} (Post #{i} — {tag})\n\n**Prompt:**\n```text\n{posts[i]['prompt']}\n```\n")
            f.write(f"- **Human Reference:** {posts[i]['summary'].strip()}\n")
            for k in ("SFT", "PPO", "DPO"):
                f.write(f"- **{k}** ($r_\\phi = {res[k]['r'][i]:+.3f}$, ROUGE-L = {res[k]['rg'][i]:.3f}): {res[k]['sums'][i].strip()}\n")
            f.write("\n---\n\n")


if __name__ == "__main__":
    main()
