import json
import os
import time
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from batching import collate_pref
from common import load_causal_lm, load_preference_splits, load_tokenizer


class RewardModel(nn.Module):
    def __init__(self, sft_path="checkpoints/sft"):
        super().__init__()
        base = load_causal_lm(sft_path)
        self.backbone = base.model
        device = next(base.parameters()).device
        self.head = nn.Linear(base.config.hidden_size, 1).to(device)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # pad_token_id == eos_token_id in SmolLM2, so use attention_mask to find the last real token
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, last_idx].float()
        return self.head(last_hidden).squeeze(-1)

    def save(self, path="checkpoints/reward"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "reward_model.pt"))

    @classmethod
    def load(cls, path="checkpoints/reward", sft_path="checkpoints/sft", device=None):
        rm = cls(sft_path)
        device = device or next(rm.parameters()).device
        state = torch.load(os.path.join(path, "reward_model.pt"), map_location=device)
        rm.load_state_dict(state)
        return rm.to(device)


@torch.no_grad()
def evaluate_rm(rm, loader, device):
    rm.eval()
    correct = 0
    for batch in loader:
        r_chosen = rm(batch["chosen_ids"].to(device), batch["chosen_mask"].to(device))
        r_rejected = rm(batch["rejected_ids"].to(device), batch["rejected_mask"].to(device))
        correct += (r_chosen > r_rejected).sum().item()
    rm.train()
    return correct / len(loader.dataset)


def main():
    os.makedirs("checkpoints/reward", exist_ok=True)
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

    rm = RewardModel("checkpoints/sft")
    device = next(rm.parameters()).device
    optimizer = AdamW(rm.parameters(), lr=1e-5)

    steps = []
    losses = []
    val_accs = []
    running_loss = 0.0
    start_time = time.time()

    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()
        step_loss = 0.0

        # Split 8 pairs into two halves of 4 pairs for T4 VRAM
        for sl in (slice(0, 4), slice(4, 8)):
            r_w = rm(batch["chosen_ids"][sl].to(device), batch["chosen_mask"][sl].to(device))
            r_l = rm(batch["rejected_ids"][sl].to(device), batch["rejected_mask"][sl].to(device))
            loss_half = -F.logsigmoid(r_w - r_l).mean() * 0.5
            loss_half.backward()
            step_loss += loss_half.item()

        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        optimizer.step()
        running_loss += step_loss

        if step % 250 == 0:
            avg_loss = running_loss / 250
            acc = evaluate_rm(rm, val_loader, device)
            steps.append(step)
            losses.append(avg_loss)
            val_accs.append(acc)
            print(f"Step {step:4d} | loss: {avg_loss:.4f} | val_acc: {acc:.4f} | {time.time() - start_time:.1f}s")
            running_loss = 0.0

    rm.save("checkpoints/reward")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(steps, losses, "-o", color="C0")
    ax2.plot(steps, val_accs, "-s", color="C1")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color="C0")
    ax2.set_ylabel("Val Acc", color="C1")
    plt.title("Reward Model")
    plt.tight_layout()
    plt.savefig("results/reward_curve.png", dpi=150)

    with open("results/reward_log.json", "w") as f:
        json.dump(
            {"steps": steps, "loss": losses, "val_acc": val_accs, "time": time.time() - start_time},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
