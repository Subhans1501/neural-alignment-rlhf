import os
import sys
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter"))

from config import MAX_SUMMARY_TOKENS
from models import load_tokenizer, load_causal_lm as _load_base_lm
from data import load_sft_splits, load_preference_splits, load_eval_posts
from batching import collate_sft, collate_prompts


def load_causal_lm(name_or_path=None, device=None):
    # SmolLM2 defaults to bfloat16, which loses small lr=1e-5 AdamW updates, so cast to float32
    model = _load_base_lm(name_or_path, device) if name_or_path else _load_base_lm(device=device)
    return model.float()


def token_logprobs(model, input_ids, attention_mask):
    # Shift logits and targets by 1 to get log P(x_t | x_<t) for t >= 1
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    return -nll.view(input_ids.size(0), -1)


def summary_mask(attention_mask, prompt_lens):
    # Mask is 1 only on summary tokens (after prompt_len and before right padding)
    _, seq_len = attention_mask.shape
    pos = torch.arange(1, seq_len, device=attention_mask.device).unsqueeze(0)
    is_summary = pos >= prompt_lens.to(attention_mask.device).unsqueeze(1)
    is_real = attention_mask[:, 1:] == 1
    return (is_summary & is_real).float()


@torch.no_grad()
def generate_summaries(
    model,
    tok,
    posts,
    batch_size=16,
    max_new_tokens=MAX_SUMMARY_TOKENS,
    do_sample=False,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        posts,
        batch_size=batch_size,
        collate_fn=partial(collate_prompts, pad_id=tok.pad_token_id),
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tok.pad_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_k": top_k, "top_p": top_p})

    summaries = []
    all_ids = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out = model.generate(ids, attention_mask=mask, **gen_kwargs)

        for row in out[:, ids.shape[1]:].tolist():
            if tok.eos_token_id in row:
                cut = row[: row.index(tok.eos_token_id) + 1]
            else:
                cut = row
            all_ids.append(cut)
            text_ids = cut[:-1] if (cut and cut[-1] == tok.eos_token_id) else cut
            summaries.append(tok.decode(text_ids))

    return summaries, all_ids


if __name__ == "__main__":
    tok = load_tokenizer()
    sft_train, sft_val = load_sft_splits(tok)
    pref_train, pref_val = load_preference_splits(tok)
    eval_posts = load_eval_posts(tok)

    p_lens = [len(x["prompt_ids"]) for x in sft_train]
    s_lens = [len(x["summary_ids"]) for x in sft_train]
    print(f"SFT train/val: {len(sft_train)} / {len(sft_val)}")
    print(f"Pref train/val: {len(pref_train)} / {len(pref_val)}")
    print(f"Eval posts: {len(eval_posts)}")
    print(f"Prompt len mean/max: {np.mean(p_lens):.2f} / {np.max(p_lens)}")
    print(f"Summary len mean/max: {np.mean(s_lens):.2f} / {np.max(s_lens)}")

    model = load_causal_lm()
    device = next(model.parameters()).device
    batch = collate_sft(sft_train[:2], pad_id=tok.pad_token_id)
    lp = token_logprobs(model, batch["input_ids"].to(device), batch["attention_mask"].to(device))
    mask = summary_mask(batch["attention_mask"].to(device), batch["prompt_lens"])
    print("Shapes:", tuple(batch["input_ids"].shape), tuple(lp.shape), tuple(mask.shape))
    for i in range(2):
        summary_tokens = batch["input_ids"][i, 1:][mask[i].cpu() == 1]
        print(f"[{i}] {repr(tok.decode(summary_tokens))}")
