import os, json, time, random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from common import load_tokenizer, load_causal_lm, load_sft_splits, generate_summaries, summary_mask
from config import SEED, MAX_SUMMARY_TOKENS
from batching import collate_sft
from train_reward import RewardModel


# Section 2.4 Models in memory: Policy pi_theta + new linear value head V_theta on the last hidden layer
class ActorCritic(nn.Module):
    def __init__(self, sft_path="checkpoints/sft"):
        super().__init__()
        self.policy = load_causal_lm(sft_path)
        self.value_head = nn.Linear(self.policy.config.hidden_size, 1).to(next(self.policy.parameters()).device)

    def forward_pass(self, input_ids, attention_mask):
        out = self.policy(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        log_p = F.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
        lp = log_p.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        # Task 4.2: Read V(s_t) from the same position whose logits predict a_t (just before a_t)
        val = self.value_head(out.hidden_states[-1][:, :-1, :].float()).squeeze(-1)
        ent = -(log_p.exp() * log_p).sum(dim=-1)
        return lp, val, ent


def main():
    os.makedirs("checkpoints/ppo", exist_ok=True); os.makedirs("results", exist_ok=True)
    random.seed(SEED); torch.manual_seed(SEED)
    tok = load_tokenizer()
    sft_train, _ = load_sft_splits(tok)

    # Section 2.4 & 3.3: Load trainable Policy+Value head, frozen Reference pi_ref, and frozen Reward r_phi
    ac = ActorCritic("checkpoints/sft")
    device = next(ac.parameters()).device
    ref = load_causal_lm("checkpoints/sft", device=device).eval()
    rm = RewardModel.load("checkpoints/reward", "checkpoints/sft", device=device).eval()
    for p in list(ref.parameters()) + list(rm.parameters()):
        p.requires_grad = False

    # Task 4.5 Settings: eps=0.2, beta_kl=0.05, c_v=0.1, c_h=0.01, gamma=1.0, K=4, lr=1e-5, B=16, iters=200
    eps, beta_kl, c_v, c_h, K, B = 0.2, 0.05, 0.1, 0.01, 4, 16
    opt = AdamW(ac.parameters(), lr=1e-5)
    logs, t0 = [], time.time()

    for it in range(1, 201):
        # Task 4.1 Rollout: 16 prompts, do_sample=True, temp=1.0, top_k=0, top_p=1.0, max 48 tokens, stop at first EOS
        prompts = sft_train[(it - 1) * B : it * B]
        _, gen_ids = generate_summaries(ac.policy, tok, prompts, B, MAX_SUMMARY_TOKENS,
                                        do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
        items = [{"prompt_ids": p["prompt_ids"], "summary_ids": g or [tok.eos_token_id]} for p, g in zip(prompts, gen_ids)]
        b = collate_sft(items, pad_id=tok.pad_token_id)
        ids, attn, sm = b["input_ids"].to(device), b["attention_mask"].to(device), summary_mask(b["attention_mask"].to(device), b["prompt_lens"])

        # Task 4.2 Store (no gradients): log pi_theta_old(a_t|s_t), log pi_ref(a_t|s_t), V_theta_old(s_t), and r_phi(x, y)
        with torch.no_grad():
            old_lp, old_val, _ = ac.forward_pass(ids, attn)
            ref_lp = F.log_softmax(ref(ids, attention_mask=attn).logits[:, :-1, :].float(), -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            scores = rm(ids, attn)

        # Task 4.3 Per-token reward: r_t = -beta_kl * (log pi_old - log pi_ref) + 1[t == T] * r_phi(x, y)
        # Task 4.4 Returns and advantages (gamma = 1): R_t = sum_{k>=0} r_{t+k}, A_t = R_t - V_old(s_t) (normalised)
        kl_tok = (old_lp - ref_lp) * sm
        rew, ret = -beta_kl * kl_tok, torch.zeros_like(kl_tok)
        for i in range(B):
            idx = torch.where(sm[i] == 1)[0]
            rew[i, idx[-1]] += scores[i]
            ret[i, idx] = torch.flip(torch.cumsum(torch.flip(rew[i, idx], [0]), 0), [0])

        adv = (ret - old_val) * sm
        m_bool = sm == 1
        adv[m_bool] = (adv[m_bool] - adv[m_bool].mean()) / (adv[m_bool].std() + 1e-8)

        # Task 4.5 Update for K = 4 epochs on the same rollout (processed in 2 halves to fit GPU memory)
        ac.train()
        cfs, lp_l, lv_l, le_l = [], [], [], []
        for _ in range(K):
            opt.zero_grad()
            denom = sm.sum()
            for sl in [slice(0, 8), slice(8, 16)]:
                lp, val, ent = ac.forward_pass(ids[sl], attn[sl])
                m = sm[sl]
                ratio = torch.exp(lp - old_lp[sl])
                l_ppo = -(torch.min(ratio * adv[sl], torch.clamp(ratio, 1 - eps, 1 + eps) * adv[sl]) * m).sum() / denom
                l_val = ((val - ret[sl]).pow(2) * m).sum() / denom
                l_ent = (ent * m).sum() / denom
                (l_ppo + c_v * l_val - c_h * l_ent).backward()
                cfs.append((((ratio < 1 - eps) | (ratio > 1 + eps)).float() * m).sum().item() / m.sum().item())
                lp_l.append(l_ppo.item() * 2); lv_l.append(l_val.item() * 2); le_l.append(l_ent.item() * 2)
            torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            opt.step()

        # Task 4.6 Log after every iteration: mean r_phi, mean KL, mean length, clip fraction, and 3 loss terms
        logs.append({"iter": it, "mean_r": scores.mean().item(), "mean_kl": kl_tok.sum(1).mean().item(),
                     "mean_len": sm.sum(1).mean().item(), "clip_frac": sum(cfs)/len(cfs),
                     "l_ppo": sum(lp_l)/len(lp_l), "l_val": sum(lv_l)/len(lv_l), "l_ent": sum(le_l)/len(le_l)})
        if it == 1 or it % 10 == 0:
            print(f"Iter {it:3d} | r: {logs[-1]['mean_r']:+.3f} | KL: {logs[-1]['mean_kl']:.4f} | len: {logs[-1]['mean_len']:.1f} | clip: {logs[-1]['clip_frac']:.3f}")

    # Task 4.7 Save the policy (without the value head) as checkpoints/ppo/
    ac.policy.save_pretrained("checkpoints/ppo"); tok.save_pretrained("checkpoints/ppo")
    with open("results/ppo_log.json", "w") as f:
        json.dump({"logs": logs, "time": time.time() - t0}, f, indent=2)

    # Section 4.5 (Figure 3): PPO mean r_phi, mean KL, mean length and clip fraction against iteration
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    for ax, k in zip(axs.flat, ["mean_r", "mean_kl", "mean_len", "clip_frac"]):
        ax.plot([x["iter"] for x in logs], [x[k] for x in logs]); ax.set_title(k)
    plt.tight_layout(); plt.savefig("results/ppo_curves.png", dpi=150)


if __name__ == "__main__":
    main()
