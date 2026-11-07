import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
from starter.data import clean_post
from common import load_tokenizer, load_causal_lm, generate_summaries
from batching import collate_sft
from train_reward import RewardModel

st.set_page_config(page_title="RLHF Summarizer (SFT vs PPO vs DPO)", layout="wide")
st.title("Learning to Summarize from Human Feedback")
st.caption("Side-by-side comparison of SFT, PPO, and DPO policies on SmolLM2-135M with Reward Model scores.")


@st.cache_resource
def load_all():
    tok = load_tokenizer()
    sft = load_causal_lm("checkpoints/sft").eval()
    device = next(sft.parameters()).device
    ppo = load_causal_lm("checkpoints/ppo", device=device).eval()
    dpo = load_causal_lm("checkpoints/dpo", device=device).eval()
    rm = RewardModel.load("checkpoints/reward", sft_path="checkpoints/sft", device=device).eval()
    return tok, {"SFT": sft, "PPO": ppo, "DPO": dpo}, rm, device


DEFAULT_POST = (
    "SUBREDDIT: r/relationships\n"
    "TITLE: My [24F] roommate [25F] keeps borrowing my car without asking first.\n"
    "POST: We have lived together for six months and generally get along well. However, over the last two weeks "
    "she has taken my spare keys off the kitchen hook three times to run errands while I was asleep. When I confronted "
    "her, she said she always puts gas back in and didn't want to wake me up, so it shouldn't be a big deal."
)

raw_post = st.text_area("Paste a Reddit Post (Subreddit / Title / Body):", value=DEFAULT_POST, height=180)

if st.button("Generate Summaries", type="primary"):
    tok, models, rm, device = load_all()
    prompt = clean_post(raw_post)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    post_item = [{"prompt": prompt, "prompt_ids": p_ids}]

    cols = st.columns(3)
    for col, (name, model) in zip(cols, models.items()):
        sums, g_ids = generate_summaries(model, tok, post_item, batch_size=1, do_sample=False)
        b = collate_sft([{"prompt_ids": p_ids, "summary_ids": g_ids[0]}], pad_id=tok.pad_token_id)
        with torch.no_grad():
            score = rm(b["input_ids"].to(device), b["attention_mask"].to(device)).item()
        with col:
            st.subheader(f"{name} Policy")
            st.metric("Reward Score r_φ(x, y)", f"{score:+.4f}")
            st.info(sums[0].strip())
            st.caption(f"Length: {len(g_ids[0])} tokens")

