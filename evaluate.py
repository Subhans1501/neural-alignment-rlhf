import os, json, csv, random
import numpy as np
import torch
from rouge_score import rouge_scorer
from common import load_tokenizer, load_causal_lm, load_eval_posts, generate_summaries, token_logprobs, summary_mask
from config import SEED, MAX_SUMMARY_TOKENS
from batching import collate_sft
from train_reward import RewardModel


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

    sft = load_causal_lm("checkpoints/sft").eval()
    dev = next(sft.parameters()).device
    rm = RewardModel.load("checkpoints/reward", "checkpoints/sft", device=dev).eval()
    models = {"SFT": sft, "PPO": load_causal_lm("checkpoints/ppo", device=dev).eval(), "DPO": load_causal_lm("checkpoints/dpo", device=dev).eval()}

    res, table = {}, {}
    for name, m in models.items():
        sums, g_ids = generate_summaries(m, tok, posts, 16, MAX_SUMMARY_TOKENS, do_sample=False)
        r, kl, l = score_and_kl(m, sft, rm, tok, posts, g_ids)
        rg = [scorer.score(p["summary"].strip(), s.strip())["rougeL"].fmeasure for p, s in zip(posts, sums)]
        res[name] = {"sums": sums, "r": r, "kl": kl, "len": l, "rg": np.array(rg)}

    for name in ["SFT", "PPO", "DPO"]:
        d = res[name]
        win = None if name == "SFT" else float((d["r"] > res["SFT"]["r"]).mean())
        table[name] = {"mean_r": float(d["r"].mean()), "win_vs_sft": win,
                       "kl_to_sft": 0.0 if name == "SFT" else float(d["kl"].mean()),
                       "length": float(d["len"].mean()), "rouge_l": float(d["rg"].mean())}
        print(f"{name:4s} | r: {table[name]['mean_r']:+.4f} | win: {win if win else 0:.3f} | KL: {table[name]['kl_to_sft']:.4f} | len: {table[name]['length']:.2f} | ROUGE-L: {table[name]['rouge_l']:.4f}")

    with open("results/eval_summaries.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(posts):
            f.write(json.dumps({"id": i, "prompt": p["prompt"], "reference": p["summary"],
                                **{k.lower(): res[k]["sums"][i] for k in models},
                                **{f"{k.lower()}_r": float(res[k]["r"][i]) for k in models}}) + "\n")
    with open("results/table3_eval.json", "w") as f:
        json.dump(table, f, indent=2)

    rng = random.Random(SEED)
    with open("results/human_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["post_id", "prompt", "summary_A", "summary_B", "summary_C", "key_A", "key_B", "key_C", "member1_pick", "member2_pick"])
        for idx in rng.sample(range(len(posts)), 30):
            ord_k = ["SFT", "PPO", "DPO"]; rng.shuffle(ord_k)
            w.writerow([idx, posts[idx]["prompt"], *[res[k]["sums"][idx] for k in ord_k], *ord_k, "", ""])


if __name__ == "__main__":
    main()
