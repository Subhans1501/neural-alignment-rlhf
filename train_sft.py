import json
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from common import (
    collate_sft,
    generate_summaries,
    load_causal_lm,
    load_eval_posts,
    load_sft_splits,
    load_tokenizer,
    summary_mask,
    token_logprobs,
)
def sft_loss(model, batch, device):
    ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    lp = token_logprobs(model, ids, attn)
    mask = summary_mask(attn, batch["prompt_lens"])
    return -(lp * mask).sum() / mask.sum()
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        total += sft_loss(model, batch, device).item()
    model.train()
    return total / len(loader)
def main():
    os.makedirs("checkpoints/sft", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    tok = load_tokenizer()
    sft_train, sft_val = load_sft_splits(tok)
    eval_posts = load_eval_posts(tok)[:5]

    train_loader = DataLoader(
        sft_train,
        batch_size=16,
        shuffle=True,
        collate_fn=partial(collate_sft, pad_id=tok.pad_token_id),
    )
    val_loader = DataLoader(
        sft_val,
        batch_size=8,
        collate_fn=partial(collate_sft, pad_id=tok.pad_token_id),
    )

    model = load_causal_lm()
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=5e-5)

    before_sums, _ = generate_summaries(model, tok, eval_posts, batch_size=5)

    steps = []
    train_losses = []
    val_losses = []
    running_loss = 0.0
    start_time = time.time()

    model.train()
    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        mask = summary_mask(attn, batch["prompt_lens"])
        total_tokens = mask.sum()
        step_loss = 0.0
        for sl in (slice(0, 8), slice(8, 16)):
            lp = token_logprobs(model, ids[sl], attn[sl])
            loss_half = -(lp * mask[sl]).sum() / total_tokens
            loss_half.backward()
            step_loss += loss_half.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += step_loss

        if step % 250 == 0:
            avg_train = running_loss / 250
            val_loss = evaluate(model, val_loader, device)
            steps.append(step)
            train_losses.append(avg_train)
            val_losses.append(val_loss)
            print(f"Step {step:4d} | train: {avg_train:.4f} | val: {val_loss:.4f} | {time.time() - start_time:.1f}s")
            running_loss = 0.0

    model.save_pretrained("checkpoints/sft")
    tok.save_pretrained("checkpoints/sft")

    after_sums, _ = generate_summaries(model, tok, eval_posts, batch_size=5)
    for i in range(5):
        print(f"\n[{i}] Before: {before_sums[i]}")
        print(f"    After : {after_sums[i]}")
        print(f"    Ref   : {eval_posts[i]['summary']}")

    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_losses, "-o", label="Train")
    plt.plot(steps, val_losses, "-s", label="Val")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("SFT Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/sft_loss.png", dpi=150)

    with open("results/sft_log.json", "w") as f:
        json.dump(
            {
                "steps": steps,
                "train": train_losses,
                "val": val_losses,
                "time": time.time() - start_time,
                "before": before_sums,
                "after": after_sums,
            },
            f,
            indent=2,
        )
if __name__ == "__main__":
    main()
