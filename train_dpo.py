import os, json, time
from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from common import load_tokenizer, load_causal_lm, load_preference_splits, token_logprobs, summary_mask
from batching import collate_pref


# Task 5.2: For a sequence, log pi(y | x) is the sum of token log-probabilities over summary tokens only
def seq_logprob(model, ids, mask, prompt_lens):
    return (token_logprobs(model, ids, mask) * summary_mask(mask, prompt_lens)).sum(dim=1)


# Task 5.3 & 5.5: DPO loss (Eq. 60) with beta = 0.1 and implicit reward accuracy (fraction where inside sigma(.) > 0)
def dpo_step(policy, ref, b, device, beta=0.1):
    c_ids, c_m, r_ids, r_m, p_l = b["chosen_ids"].to(device), b["chosen_mask"].to(device), b["rejected_ids"].to(device), b["rejected_mask"].to(device), b["prompt_lens"].to(device)
    with torch.no_grad():
        ref_w, ref_l = seq_logprob(ref, c_ids, c_m, p_l), seq_logprob(ref, r_ids, r_m, p_l)
    pi_w, pi_l = seq_logprob(policy, c_ids, c_m, p_l), seq_logprob(policy, r_ids, r_m, p_l)
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(logits).mean(), (logits > 0).float().mean().item(), logits.size(0)


@torch.no_grad()
def evaluate_dpo(policy, ref, loader, device):
    policy.eval()
    acc = sum(dpo_step(policy, ref, b, device)[1] * b["chosen_ids"].size(0) for b in loader) / len(loader.dataset)
    policy.train()
    return acc


def main():
    os.makedirs("checkpoints/dpo", exist_ok=True); os.makedirs("results", exist_ok=True)
    tok = load_tokenizer()
    pref_train, pref_val = load_preference_splits(tok)

    # Task 5.4: Train for 1 epoch over the same 20,000 preference pairs: AdamW, lr = 2e-6, batch size 8 pairs
    tr_loader = DataLoader(pref_train, 8, shuffle=True, collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))
    va_loader = DataLoader(pref_val, 8, collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))

    # Task 5.1: Load checkpoints/sft/ twice: trainable policy pi_theta and frozen reference pi_ref
    policy = load_causal_lm("checkpoints/sft")
    device = next(policy.parameters()).device
    ref = load_causal_lm("checkpoints/sft", device=device).eval()
    for p in ref.parameters():
        p.requires_grad = False

    opt = AdamW(policy.parameters(), lr=2e-6)
    steps, tr_log, acc_log, run_loss, t0 = [], [], [], 0.0, time.time()

    for step, b in enumerate(tr_loader, 1):
        opt.zero_grad()
        # Process batch of 8 pairs in 2 halves of 4 pairs to keep T4 VRAM low
        step_loss = 0.0
        for sl in [slice(0, 4), slice(4, 8)]:
            sub = {k: v[sl] for k, v in b.items()}
            loss_half, _, _ = dpo_step(policy, ref, sub, device)
            (loss_half * 0.5).backward()
            step_loss += loss_half.item() * 0.5
        if step == 1:
            # Section 3.3 Correctness Check: First step loss equals log 2 approx 0.6931
            print(f"Step 1 DPO loss: {step_loss:.4f} (expected 0.6931)")
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        run_loss += step_loss

        # Task 5.5: Every 250 steps log loss and implicit reward accuracy on 1,000 validation pairs
        if step % 250 == 0:
            tr_l, acc = run_loss / 250, evaluate_dpo(policy, ref, va_loader, device)
            steps.append(step); tr_log.append(tr_l); acc_log.append(acc)
            print(f"Step {step:4d} | loss: {tr_l:.4f} | implicit_acc: {acc:.4f} | {time.time()-t0:.1f}s")
            run_loss = 0.0

    # Task 5.5: Save as checkpoints/dpo/
    policy.save_pretrained("checkpoints/dpo"); tok.save_pretrained("checkpoints/dpo")

    # Section 4.5 (Figure 4): DPO training loss and validation implicit reward accuracy
    fig, ax1 = plt.subplots(figsize=(6, 4)); ax2 = ax1.twinx()
    ax1.plot(steps, tr_log, "-o", color="C0"); ax2.plot(steps, acc_log, "-s", color="C2")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss", color="C0"); ax2.set_ylabel("Implicit Acc", color="C2")
    plt.title("DPO Training"); plt.tight_layout(); plt.savefig("results/dpo_curve.png", dpi=150)
    with open("results/dpo_log.json", "w") as f:
        json.dump({"steps": steps, "loss": tr_log, "val_acc": acc_log, "time": time.time()-t0}, f, indent=2)


if __name__ == "__main__":
    main()
