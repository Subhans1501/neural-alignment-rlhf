import os, json, time
from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from common import load_tokenizer, load_causal_lm, load_preference_splits
from batching import collate_pref


# Task 3.1: Load checkpoints/sft/, take transformer body (model.model), add nn.Linear(hidden_size, 1)
# on the final hidden state of the last non-padding token of prompt + summary.
class RewardModel(nn.Module):
    def __init__(self, sft_path="checkpoints/sft"):
        super().__init__()
        base = load_causal_lm(sft_path)
        self.backbone = base.model
        self.head = nn.Linear(base.config.hidden_size, 1).to(next(base.parameters()).device)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Section 3.3 Correctness Check: Find last real token with attention_mask (never input_ids != pad_id)
        last_idx = attention_mask.sum(dim=1) - 1
        return self.head(h[torch.arange(input_ids.size(0), device=h.device), last_idx].float()).squeeze(-1)

    def save(self, path="checkpoints/reward"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "reward_model.pt"))

    @classmethod
    def load(cls, path="checkpoints/reward", sft_path="checkpoints/sft", device=None):
        rm = cls(sft_path)
        device = device or next(rm.parameters()).device
        rm.load_state_dict(torch.load(os.path.join(path, "reward_model.pt"), map_location=device))
        return rm.to(device)


# Task 3.4: Pairwise accuracy on the 1,000 validation pairs: fraction where r_phi(x, y_w) > r_phi(x, y_l)
@torch.no_grad()
def evaluate_rm(rm, loader, device):
    rm.eval()
    acc = sum((rm(b["chosen_ids"].to(device), b["chosen_mask"].to(device)) >
               rm(b["rejected_ids"].to(device), b["rejected_mask"].to(device))).sum().item() for b in loader)
    rm.train()
    return acc / len(loader.dataset)


def main():
    os.makedirs("checkpoints/reward", exist_ok=True); os.makedirs("results", exist_ok=True)
    tok = load_tokenizer()
    pref_train, pref_val = load_preference_splits(tok)

    # Task 3.3: Train for 1 epoch over 20,000 preference pairs, AdamW, lr = 1e-5, batch size 8 pairs
    tr_loader = DataLoader(pref_train, 8, shuffle=True, collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))
    va_loader = DataLoader(pref_val, 8, collate_fn=partial(collate_pref, pad_id=tok.pad_token_id))

    rm = RewardModel("checkpoints/sft")
    device = next(rm.parameters()).device
    opt = AdamW(rm.parameters(), lr=1e-5)
    steps, tr_log, acc_log, run_loss, t0 = [], [], [], 0.0, time.time()

    for step, b in enumerate(tr_loader, 1):
        opt.zero_grad()
        r_w = rm(b["chosen_ids"].to(device), b["chosen_mask"].to(device))
        r_l = rm(b["rejected_ids"].to(device), b["rejected_mask"].to(device))
        # Task 3.2: Bradley-Terry loss L_reward(phi) = -E[ log sigma(r_phi(x, y_w) - r_phi(x, y_l)) ]
        loss = -F.logsigmoid(r_w - r_l).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()

        if step % 250 == 0:
            tr_l, acc = run_loss / 250, evaluate_rm(rm, va_loader, device)
            steps.append(step); tr_log.append(tr_l); acc_log.append(acc)
            print(f"Step {step:4d} | loss: {tr_l:.4f} | val_acc: {acc:.4f}")
            run_loss = 0.0

    # Task 3.4: Save as checkpoints/reward/
    rm.save("checkpoints/reward")

    # Section 4.5 (Figure 2): Reward-model training loss and validation accuracy
    fig, ax1 = plt.subplots(figsize=(6, 4)); ax2 = ax1.twinx()
    ax1.plot(steps, tr_log, "-o", color="C0"); ax2.plot(steps, acc_log, "-s", color="C1")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss", color="C0"); ax2.set_ylabel("Val Acc", color="C1")
    plt.title("Reward Model"); plt.tight_layout(); plt.savefig("results/reward_curve.png", dpi=150)
    with open("results/reward_log.json", "w") as f:
        json.dump({"steps": steps, "loss": tr_log, "val_acc": acc_log, "time": time.time()-t0}, f, indent=2)


if __name__ == "__main__":
    main()
