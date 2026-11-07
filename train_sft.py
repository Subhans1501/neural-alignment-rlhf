import os, json, time
from functools import partial
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from common import (load_tokenizer, load_causal_lm, load_sft_splits, load_eval_posts,
                    collate_sft, token_logprobs, summary_mask, generate_summaries)


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
    va_loader = DataLoader(sft_val, 16, collate_fn=partial(collate_sft, pad_id=tok.pad_token_id))

    model = load_causal_lm()
    device = next(model.parameters()).device
    opt = AdamW(model.parameters(), lr=5e-5)

    before, _ = generate_summaries(model, tok, eval_posts, batch_size=5)
    steps, tr_log, va_log, run_loss, t0 = [], [], [], 0.0, time.time()
    model.train()

    for step, b in enumerate(tr_loader, 1):
        opt.zero_grad()
        loss = sft_loss(model, b, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()

        if step % 250 == 0:
            tr_l, va_l = run_loss / 250, evaluate(model, va_loader, device)
            steps.append(step); tr_log.append(tr_l); va_log.append(va_l)
            print(f"Step {step:4d} | train: {tr_l:.4f} | val: {va_l:.4f}")
            run_loss = 0.0

    model.save_pretrained("checkpoints/sft"); tok.save_pretrained("checkpoints/sft")
    after, _ = generate_summaries(model, tok, eval_posts, batch_size=5)
    for i in range(5):
        print(f"\n[{i}] Before: {before[i]}\n    After : {after[i]}\n    Ref   : {eval_posts[i]['summary']}")

    plt.figure(figsize=(6, 4))
    plt.plot(steps, tr_log, "-o", label="Train"); plt.plot(steps, va_log, "-s", label="Val")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("SFT Loss"); plt.legend(); plt.tight_layout()
    plt.savefig("results/sft_loss.png", dpi=150)
    with open("results/sft_log.json", "w") as f:
        json.dump({"steps": steps, "train": tr_log, "val": va_log, "time": time.time()-t0, "before": before, "after": after}, f, indent=2)


if __name__ == "__main__":
    main()
