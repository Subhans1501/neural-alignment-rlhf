import json
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from batching import collate_pref
from common import (
    load_causal_lm,
    load_preference_splits,
    load_tokenizer,
    summary_mask,
    token_logprobs,
)


def seq_logprob(model, ids, mask, prompt_lens):
    lp = token_logprobs(model, ids, mask)
    smask = summary_mask(mask, prompt_lens)
    return (lp * smask).sum(dim=1)


def dpo_step(policy, ref, batch, device, beta=0.1):
    c_ids = batch["chosen_ids"].to(device)
    c_mask = batch["chosen_mask"].to(device)
    r_ids = batch["rejected_ids"].to(device)
    r_mask = batch["rejected_mask"].to(device)
    p_lens = batch["prompt_lens"].to(device)

    with torch.no_grad():
        ref_w = seq_logprob(ref, c_ids, c_mask, p_lens)
        ref_l = seq_logprob(ref, r_ids, r_mask, p_lens)

    pi_w = seq_logprob(policy, c_ids, c_mask, p_lens)
    pi_l = seq_logprob(policy, r_ids, r_mask, p_lens)

    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    loss = -F.logsigmoid(logits).mean()
    acc = (logits > 0).float().mean().item()
    return loss, acc, logits.size(0)


@torch.no_grad()
def evaluate_dpo(policy, ref, loader, device):
    policy.eval()
    correct = 0.0
    for batch in loader:
        _, acc, n = dpo_step(policy, ref, batch, device)
        correct += acc * n
    policy.train()
    return correct / len(loader.dataset)


def main():
    os.makedirs("checkpoints/dpo", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tok = load_tokenizer()
    pref_train, pref_val = load_preference_splits(tok)

    train_loader = DataLoader(
        pref_train,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(collate_pref, pad_id=tok.pad_token_id),
    )
    val_loader = DataLoader(
        pref_val,
        batch_size=8,
        collate_fn=partial(collate_pref, pad_id=tok.pad_token_id),
    )

    policy = load_causal_lm("checkpoints/sft")
    device = next(policy.parameters()).device
    ref_model = load_causal_lm("checkpoints/sft", device=device).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = AdamW(policy.parameters(), lr=2e-6)
    steps, losses, val_accs = [], [], []
    running_loss = 0.0
    start_time = time.time()

    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()
        step_loss = 0.0

        for sl in (slice(0, 4), slice(4, 8)):
            sub = {k: v[sl] for k, v in batch.items()}
            loss_half, _, _ = dpo_step(policy, ref_model, sub, device)
            (loss_half * 0.5).backward()
            step_loss += loss_half.item() * 0.5

        if step == 1:
            print(f"Step 1 DPO loss: {step_loss:.4f} (expected ~0.6931)")

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        running_loss += step_loss

        if step % 250 == 0:
            avg_loss = running_loss / 250
            acc = evaluate_dpo(policy, ref_model, val_loader, device)
            steps.append(step)
            losses.append(avg_loss)
            val_accs.append(acc)
            print(f"Step {step:4d} | loss: {avg_loss:.4f} | implicit_acc: {acc:.4f} | {time.time() - start_time:.1f}s")
            running_loss = 0.0

    policy.save_pretrained("checkpoints/dpo")
    tok.save_pretrained("checkpoints/dpo")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(steps, losses, "-o", color="C0")
    ax2.plot(steps, val_accs, "-s", color="C2")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color="C0")
    ax2.set_ylabel("Implicit Acc", color="C2")
    plt.title("DPO Training")
    plt.tight_layout()
    plt.savefig("results/dpo_curve.png", dpi=150)

    with open("results/dpo_log.json", "w") as f:
        json.dump(
            {"steps": steps, "loss": losses, "val_acc": val_accs, "time": time.time() - start_time},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
