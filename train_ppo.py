import json
import os
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from batching import collate_sft
from common import (
    generate_summaries,
    load_causal_lm,
    load_sft_splits,
    load_tokenizer,
    summary_mask,
    token_logprobs,
)
from config import MAX_SUMMARY_TOKENS, SEED
from train_reward import RewardModel


class ActorCritic(nn.Module):
    def __init__(self, sft_path="checkpoints/sft"):
        super().__init__()
        self.policy = load_causal_lm(sft_path)
        device = next(self.policy.parameters()).device
        self.value_head = nn.Linear(self.policy.config.hidden_size, 1).to(device)

    def forward_pass(self, input_ids, attention_mask):
        hidden = self.policy.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.policy.lm_head(hidden)[:, :-1, :]
        # Value V(s_t) comes from the position right before token a_t
        values = self.value_head(hidden[:, :-1, :].float()).squeeze(-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        chosen_lp = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return chosen_lp, values, entropy


def main():
    os.makedirs("checkpoints/ppo", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    tok = load_tokenizer()
    sft_train, _ = load_sft_splits(tok)

    ac = ActorCritic("checkpoints/sft")
    device = next(ac.parameters()).device
    ref_model = load_causal_lm("checkpoints/sft", device=device).eval()
    rm = RewardModel.load("checkpoints/reward", "checkpoints/sft", device=device).eval()
    for p in list(ref_model.parameters()) + list(rm.parameters()):
        p.requires_grad = False

    eps = 0.2
    beta_kl = 0.05
    c_v = 0.1
    c_h = 0.01
    ppo_epochs = 4
    batch_size = 16

    optimizer = AdamW(ac.parameters(), lr=1e-5)
    logs = []
    start_time = time.time()

    for it in range(1, 201):
        prompts = sft_train[(it - 1) * batch_size : it * batch_size]
        _, gen_ids = generate_summaries(
            ac.policy,
            tok,
            prompts,
            batch_size=batch_size,
            max_new_tokens=MAX_SUMMARY_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
        )

        items = [
            {"prompt_ids": p["prompt_ids"], "summary_ids": g or [tok.eos_token_id]}
            for p, g in zip(prompts, gen_ids)
        ]
        batch = collate_sft(items, pad_id=tok.pad_token_id)
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask = summary_mask(attn, batch["prompt_lens"])

        with torch.no_grad():
            old_lp, old_val, _ = ac.forward_pass(ids, attn)
            ref_lp = token_logprobs(ref_model, ids, attn)
            scores = rm(ids, attn)

        kl_tok = (old_lp - ref_lp) * mask
        rewards = -beta_kl * kl_tok
        returns = torch.zeros_like(kl_tok)
        for i in range(batch_size):
            idx = torch.where(mask[i] == 1)[0]
            rewards[i, idx[-1]] += scores[i]
            # gamma = 1, so return is reverse cumsum over generated tokens
            returns[i, idx] = torch.flip(torch.cumsum(torch.flip(rewards[i, idx], [0]), 0), [0])

        adv = (returns - old_val) * mask
        valid = mask == 1
        adv[valid] = (adv[valid] - adv[valid].mean()) / (adv[valid].std() + 1e-8)

        ac.train()
        clip_fracs, ppo_losses, val_losses, ent_losses = [], [], [], []
        for _ in range(ppo_epochs):
            optimizer.zero_grad()
            denom = mask.sum()
            for sl in (slice(0, 8), slice(8, 16)):
                lp, val, ent = ac.forward_pass(ids[sl], attn[sl])
                m = mask[sl]
                ratio = torch.exp(lp - old_lp[sl])
                clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
                l_ppo = -(torch.min(ratio * adv[sl], clipped * adv[sl]) * m).sum() / denom
                l_val = ((val - returns[sl]).pow(2) * m).sum() / denom
                l_ent = (ent * m).sum() / denom
                (l_ppo + c_v * l_val - c_h * l_ent).backward()

                is_clipped = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float() * m
                clip_fracs.append(is_clipped.sum().item() / m.sum().item())
                ppo_losses.append(l_ppo.item() * 2)
                val_losses.append(l_val.item() * 2)
                ent_losses.append(l_ent.item() * 2)

            torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            optimizer.step()

        entry = {
            "iter": it,
            "mean_r": scores.mean().item(),
            "mean_kl": kl_tok.sum(1).mean().item(),
            "mean_len": mask.sum(1).mean().item(),
            "clip_frac": sum(clip_fracs) / len(clip_fracs),
            "l_ppo": sum(ppo_losses) / len(ppo_losses),
            "l_val": sum(val_losses) / len(val_losses),
            "l_ent": sum(ent_losses) / len(ent_losses),
        }
        logs.append(entry)
        if it == 1 or it % 10 == 0:
            print(
                f"Iter {it:3d} | r: {entry['mean_r']:+.3f} | KL: {entry['mean_kl']:.4f} "
                f"| len: {entry['mean_len']:.1f} | clip: {entry['clip_frac']:.3f}"
            )

    ac.policy.save_pretrained("checkpoints/ppo")
    tok.save_pretrained("checkpoints/ppo")

    with open("results/ppo_log.json", "w") as f:
        json.dump({"logs": logs, "time": time.time() - start_time}, f, indent=2)

    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    iters = [x["iter"] for x in logs]
    for ax, key in zip(axs.flat, ["mean_r", "mean_kl", "mean_len", "clip_frac"]):
        ax.plot(iters, [x[key] for x in logs])
        ax.set_title(key)
    plt.tight_layout()
    plt.savefig("results/ppo_curves.png", dpi=150)


if __name__ == "__main__":
    main()
