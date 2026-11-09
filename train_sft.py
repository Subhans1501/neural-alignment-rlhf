import os, json, time
from functools import partial
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from common import (load_tokenizer, load_causal_lm, load_sft_splits, load_eval_posts,
                    collate_sft, token_logprobs, summary_mask, generate_summaries)


# Task 2.2: SFT Loss = mean negative log-likelihood of the summary tokens only
def sft_loss(model, b, device):
    ids, attn = b["input_ids"].to(device), b["attention_mask"].to(device)
    lp, sm = token_logprobs(model, ids, attn), summary_mask(attn, b["prompt_lens"])
    return -(lp * sm).sum() / sm.sum()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    val = sum(sft_loss(model, b, device).item() for b in loader) / len(loader)
    model.train()
    return val


def main():
    os.makedirs("checkpoints/sft", exist_ok=True); os.makedirs("results", exist_ok=True)
    tok = load_tokenizer()
    sft_train, sft_val = load_sft_splits(tok)
    eval_posts = load_eval_posts(tok)[:5]

    tr_loader = DataLoader(sft_train, 16, shuffle=True, collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))
    va_loader = DataLoader(sft_val, 8, collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))

    # Task 2.1: Load the base model with load_causal_lm()
    model = load_causal_lm()
    device = next(model.parameters()).device

    # Task 2.3: AdamW, lr = 5e-5, batch size 16, gradient-norm clipping at 1.0 (1 epoch over 20,000 pairs)
    opt = AdamW(model.parameters(), lr=5e-5)

    # Task 2.5: Generate greedy summaries for 5 evaluation posts BEFORE SFT
    before, _ = generate_summaries(model, tok, eval_posts, batch_size=5)
    steps, tr_log, va_log, run_loss, t0 = [], [], [], 0.0, time.time()
    model.train()

    for step, b in enumerate(tr_loader, 1):
        opt.zero_grad()
        ids, attn, pl = b["input_ids"].to(device), b["attention_mask"].to(device), b["prompt_lens"]
        sm_full = summary_mask(attn, pl)
        denom = sm_full.sum()
        # Process batch of 16 in two halves of 8 to fit T4 GPU memory (exact same gradient)
        step_loss = 0.0
        for sl in [slice(0, 8), slice(8, 16)]:
            lp = token_logprobs(model, ids[sl], attn[sl])
            loss_half = -(lp * sm_full[sl]).sum() / denom
            loss_half.backward()
            step_loss += loss_half.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += step_loss

        # Task 2.4: Every 250 steps log training loss and validation loss on the 500 SFT validation pairs
        if step % 250 == 0:
            tr_l, va_l = run_loss / 250, evaluate(model, va_loader, device)
            steps.append(step); tr_log.append(tr_l); va_log.append(va_l)
            print(f"Step {step:4d} | train: {tr_l:.4f} | val: {va_l:.4f} | {time.time()-t0:.1f}s")
            run_loss = 0.0

    # Task 2.4: Save the final model as checkpoints/sft/
    model.save_pretrained("checkpoints/sft"); tok.save_pretrained("checkpoints/sft")

    # Task 2.5: Generate greedy summaries for 5 evaluation posts AFTER SFT
    after, _ = generate_summaries(model, tok, eval_posts, batch_size=5)
    for i in range(5):
        print(f"\n[{i}] Before: {before[i]}\n    After : {after[i]}\n    Ref   : {eval_posts[i]['summary']}")

    # Section 4.5 (Figure 1): SFT training and validation loss plot
    plt.figure(figsize=(6, 4))
    plt.plot(steps, tr_log, "-o", label="Train"); plt.plot(steps, va_log, "-s", label="Val")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("SFT Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("results/sft_loss.png", dpi=150)
    with open("results/sft_log.json", "w") as f:
        json.dump({"steps": steps, "train": tr_log, "val": va_log, "time": time.time()-t0, "before": before, "after": after}, f, indent=2)


if __name__ == "__main__":
    main()
