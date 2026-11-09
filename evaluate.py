import os, json, csv, random
import numpy as np
import torch
from rouge_score import rouge_scorer
from common import load_tokenizer, load_causal_lm, load_eval_posts, load_preference_splits, generate_summaries, token_logprobs, summary_mask
from config import SEED, MAX_SUMMARY_TOKENS
from batching import collate_sft, collate_pref
from train_reward import RewardModel


# Section 3.1: Compute Reward-model score r_phi(x, y), KL to SFT, and token Length on greedy summaries
@torch.no_grad()
def score_and_kl(policy, ref, rm, tok, posts, gen_ids, batch_size=16):
    device = next(policy.parameters()).device
    items = [{"prompt_ids": p["prompt_ids"], "summary_ids": g or [tok.eos_token_id]} for p, g in zip(posts, gen_ids)]
    rews, kls, lens = [], [], []
    for s in range(0, len(items), batch_size):
        b = collate_sft(items[s:s + batch_size], pad_id=tok.pad_token_id)
        ids, attn, sm = b["input_ids"].to(device), b["attention_mask"].to(device), summary_mask(b["attention_mask"].to(device), b["prompt_lens"])
        rews.extend(rm(ids, attn).cpu().tolist())
        kls.extend(((token_logprobs(policy, ids, attn) - token_logprobs(ref, ids, attn)) * sm).sum(1).cpu().tolist())
        lens.extend(sm.sum(1).cpu().tolist())
    return np.array(rews), np.array(kls), np.array(lens)


def main():
    os.makedirs("results", exist_ok=True)
    tok = load_tokenizer()
    posts = load_eval_posts(tok)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Task 6 (Section 2.6): Load SFT, PPO, and DPO checkpoints + frozen Reward Model
    sft = load_causal_lm("checkpoints/sft").eval()
    dev = next(sft.parameters()).device
    rm = RewardModel.load("checkpoints/reward", "checkpoints/sft", device=dev).eval()
    models = {"SFT": sft, "PPO": load_causal_lm("checkpoints/ppo", device=dev).eval(), "DPO": load_causal_lm("checkpoints/dpo", device=dev).eval()}

    # Task 6 & Section 3.1: Generate greedy summaries (at most 48 tokens) on 300 eval posts & compute metrics
    res, table = {}, {}
    for name, m in models.items():
        sums, g_ids = generate_summaries(m, tok, posts, 16, MAX_SUMMARY_TOKENS, do_sample=False)
        r, kl, l = score_and_kl(m, sft, rm, tok, posts, g_ids)
        rg = [scorer.score(p["summary"].strip(), s.strip())["rougeL"].fmeasure for p, s in zip(posts, sums)]
        res[name] = {"sums": sums, "r": r, "kl": kl, "len": l, "rg": np.array(rg)}

    # Section 4.3 (Table 3): Final comparison on 300 evaluation posts
    for name in ["SFT", "PPO", "DPO"]:
        d = res[name]
        win = None if name == "SFT" else float((d["r"] > res["SFT"]["r"]).mean())
        table[name] = {"mean_r": float(d["r"].mean()), "win_vs_sft": win,
                       "kl_to_sft": 0.0 if name == "SFT" else float(d["kl"].mean()),
                       "length": float(d["len"].mean()), "rouge_l": float(d["rg"].mean())}
        print(f"{name:4s} | r: {table[name]['mean_r']:+.4f} | win: {win if win else 0:.3f} | KL: {table[name]['kl_to_sft']:.4f} | len: {table[name]['length']:.2f} | ROUGE-L: {table[name]['rouge_l']:.4f}")

    # Task 6: Save all summaries to results/eval_summaries.jsonl
    with open("results/eval_summaries.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(posts):
            f.write(json.dumps({"id": i, "prompt": p["prompt"], "reference": p["summary"],
                                **{k.lower(): res[k]["sums"][i] for k in models},
                                **{f"{k.lower()}_r": float(res[k]["r"][i]) for k in models}}) + "\n")
    with open("results/table3_eval.json", "w") as f:
        json.dump(table, f, indent=2)

    # Section 3.2 & 4.4 (Table 4): Take 30 evaluation posts (fixed seed) in blind random order -> results/human_eval.csv
    rng = random.Random(SEED)
    with open("results/human_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["post_id", "prompt", "summary_A", "summary_B", "summary_C", "key_A", "key_B", "key_C", "member1_pick", "member2_pick"])
        for idx in rng.sample(range(len(posts)), 30):
            ord_k = ["SFT", "PPO", "DPO"]; rng.shuffle(ord_k)
            w.writerow([idx, posts[idx]["prompt"], *[res[k]["sums"][idx] for k in ord_k], *ord_k, "", ""])

    # Task 7.1: Save 3 validation preference pairs that the reward model gets wrong (r_phi(x, y_w) <= r_phi(x, y_l))
    _, pref_val = load_preference_splits(tok)
    wrong_pairs = []
    with torch.no_grad():
        for item in pref_val:
            b = collate_pref([item], pad_id=tok.pad_token_id)
            rw = rm(b["chosen_ids"].to(dev), b["chosen_mask"].to(dev)).item()
            rl = rm(b["rejected_ids"].to(dev), b["rejected_mask"].to(dev)).item()
            if rw <= rl:
                wrong_pairs.append({"prompt": item["prompt"], "chosen": item["chosen"], "rejected": item["rejected"], "r_chosen": rw, "r_rejected": rl})
            if len(wrong_pairs) == 3:
                break
    with open("results/rm_errors.json", "w", encoding="utf-8") as f:
        json.dump(wrong_pairs, f, indent=2)

    # Section 4.6: Save 10 qualitative evaluation samples (best & worst RLHF vs SFT deltas) to results/samples.md
    deltas = np.maximum(res["PPO"]["rg"], res["DPO"]["rg"]) - res["SFT"]["rg"]
    order_idx = np.argsort(deltas)
    pick_10 = list(order_idx[-5:][::-1]) + list(order_idx[:5])
    with open("results/samples.md", "w", encoding="utf-8") as f:
        f.write("# Qualitative Samples (10 Evaluation Posts)\n\n")
        for rank, i in enumerate(pick_10, 1):
            tag = "PPO/DPO Better than SFT" if rank <= 5 else "PPO/DPO Worse than SFT"
            f.write(f"## Sample {rank} (Post #{i} — {tag})\n\n**Prompt:**\n```text\n{posts[i]['prompt']}\n```\n")
            f.write(f"- **Human Reference:** {posts[i]['summary'].strip()}\n")
            for k in ["SFT", "PPO", "DPO"]:
                f.write(f"- **{k}** ($r_\\phi = {res[k]['r'][i]:+.3f}$, ROUGE-L = {res[k]['rg'][i]:.3f}): {res[k]['sums'][i].strip()}\n")
            f.write("\n---\n\n")


if __name__ == "__main__":
    main()
