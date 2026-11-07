import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "starter"))

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from config import MAX_SUMMARY_TOKENS
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits, load_preference_splits, load_eval_posts
from batching import collate_sft, collate_prompts


def token_logprobs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_p = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    return log_p.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def summary_mask(attention_mask, prompt_lens):
    B, L = attention_mask.shape
    pos = torch.arange(1, L, device=attention_mask.device).unsqueeze(0)
    return ((pos >= prompt_lens.to(attention_mask.device).unsqueeze(1)) & (attention_mask[:, 1:] == 1)).float()


@torch.no_grad()
def generate_summaries(model, tok, posts, batch_size=16, max_new_tokens=MAX_SUMMARY_TOKENS,
                       do_sample=False, temperature=1.0, top_k=0, top_p=1.0):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(posts, batch_size=batch_size, collate_fn=partial(collate_prompts, pad_id=tok.pad_token_id))
    summaries, all_ids = [], []
    for b in loader:
        ids, mask = b["input_ids"].to(device), b["attention_mask"].to(device)
        out = model.generate(ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for row in out[:, ids.shape[1]:].tolist():
            cut = row[:row.index(tok.eos_token_id) + 1] if tok.eos_token_id in row else row
            all_ids.append(cut)
            summaries.append(tok.decode(cut[:-1] if (cut and cut[-1] == tok.eos_token_id) else cut))
    return summaries, all_ids


if __name__ == "__main__":
    tok = load_tokenizer()
    sft_train, sft_val = load_sft_splits(tok)
    pref_train, pref_val = load_preference_splits(tok)
    eval_posts = load_eval_posts(tok)
    p_lens = [len(x["prompt_ids"]) for x in sft_train]
    s_lens = [len(x["summary_ids"]) for x in sft_train]
    print(f"SFT: {len(sft_train)}/{len(sft_val)} | Pref: {len(pref_train)}/{len(pref_val)} | Eval: {len(eval_posts)}")
    print(f"Prompt len (mean/max): {np.mean(p_lens):.2f}/{np.max(p_lens)} | Summary len (mean/max): {np.mean(s_lens):.2f}/{np.max(s_lens)}")

    model = load_causal_lm()
    dev = next(model.parameters()).device
    b = collate_sft(sft_train[:2], pad_id=tok.pad_token_id)
    lp = token_logprobs(model, b["input_ids"].to(dev), b["attention_mask"].to(dev))
    sm = summary_mask(b["attention_mask"].to(dev), b["prompt_lens"])
    print("Shapes:", tuple(b["input_ids"].shape), tuple(lp.shape), tuple(sm.shape))
    for i in range(2):
        print(f"Decoded [{i}]:", repr(tok.decode(b["input_ids"][i, 1:][sm[i].cpu() == 1])))
