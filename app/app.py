import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "starter"))

import streamlit as st
import torch
from batching import collate_sft
from common import generate_summaries, load_causal_lm, load_tokenizer
from data import clean_post
from train_reward import RewardModel

st.set_page_config(
    page_title="TL;DR Summarizer — SFT vs PPO vs DPO",
    layout="wide",
)

# Optional: set your Hugging Face repo ID here (e.g. "Subhans1501/smollm2-rlhf-tldr")
# so the app downloads checkpoints automatically if checkpoints/ is not present locally.
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")

st.markdown("""
<style>
.header-card {
    background: linear-gradient(120deg, #1b2234 0%, #242b42 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.header-card h2 {
    margin: 0 0 6px 0;
    color: #f3f4f6;
}
.header-card p {
    margin: 0;
    color: #9ca3af;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


def ensure_checkpoints():
    if os.path.exists("checkpoints/sft") and os.path.exists("checkpoints/reward/reward_model.pt"):
        return
    if not HF_REPO_ID:
        st.error("Checkpoints not found in `checkpoints/`. Put your trained folders there or set `HF_REPO_ID`.")
        st.stop()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO_ID, local_dir="checkpoints")


@st.cache_resource(show_spinner="Loading SFT, PPO, DPO, and Reward checkpoints...")
def load_all():
    ensure_checkpoints()
    tok = load_tokenizer()
    sft = load_causal_lm("checkpoints/sft").eval()
    device = next(sft.parameters()).device
    ppo = load_causal_lm("checkpoints/ppo", device=device).eval()
    dpo = load_causal_lm("checkpoints/dpo", device=device).eval()
    rm = RewardModel.load("checkpoints/reward", sft_path="checkpoints/sft", device=device).eval()
    return tok, {"SFT": sft, "PPO": ppo, "DPO": dpo}, rm, device


PRESETS = {
    "Roommate borrowing car (r/relationships)": (
        "SUBREDDIT: r/relationships\n"
        "TITLE: My [24F] roommate [25F] keeps borrowing my car without asking first.\n"
        "POST: We have lived together for six months and generally get along well. However, over the last two weeks "
        "she has taken my spare keys off the kitchen hook three times to run errands while I was asleep. When I confronted "
        "her, she said she always puts gas back in and didn't want to wake me up, so it shouldn't be a big deal."
    ),
    "Unpaid PTO on notice (r/legaladvice)": (
        "SUBREDDIT: r/legaladvice\n"
        "TITLE: Can my boss take money out of my paycheck if I'm not going against my contract?\n"
        "POST: So I'm quitting my job and gave my two week notice to my boss. I offered to stay longer than two weeks but he "
        "seems to not care about when I go. In my contract, I get 8 paid days off. I have only taken 6 so far. I told my boss "
        "today that I have not used all my paid days off and would like to use the remaining two next week. His response was that "
        "he doesn't want to pay for my vacation since I'm leaving on short notice. Is this legal?"
    ),
    "Rescue dog training (r/Dogtraining)": (
        "SUBREDDIT: r/Dogtraining\n"
        "TITLE: New rescue won't go to the bathroom outside\n"
        "POST: I recently adopted a pretty abused Yorkie, he's two years old and is very timid around everyone. When I first got "
        "him, about a week ago, I was taking him out every 2-3 hours but could rarely get him to go to the bathroom. Instead, he just "
        "comes inside, walks around my apartment for 10 minutes and then goes. I've tried positive reinforcement with treats but "
        "he seems terrified outside. Any tips on how to build his outdoor confidence?"
    ),
}

st.markdown(
    '<div class="header-card">'
    '<h2>Learning to Summarize from Human Feedback</h2>'
    '<p>Side-by-side comparison of SFT, PPO, and DPO policies fine-tuned from SmolLM2-135M, scored by our Reward Model.</p>'
    '</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Eval Summary (300 Posts)")
    st.markdown(
        "| Metric | SFT | PPO | DPO |\n"
        "| :--- | :---: | :---: | :---: |\n"
        "| Mean Reward | +0.409 | **+0.434** | +0.401 |\n"
        "| Win vs SFT | - | **34.0%** | 18.3% |\n"
        "| ROUGE-L | 0.253 | 0.250 | 0.253 |\n"
        "| Avg Tokens | 31.5 | **28.1** | 31.8 |\n"
        "| Human Picks | 1-2 / 30 | **19-26 / 30** | 3-9 / 30 |"
    )
    st.caption("Agreement between two evaluators: 22 / 30 (73.3%).")

preset = st.selectbox("Sample post", ["Custom"] + list(PRESETS.keys()))
default_text = PRESETS[preset] if preset != "Custom" else next(iter(PRESETS.values()))
raw_post = st.text_area("Reddit post (Subreddit / Title / Body)", value=default_text, height=160)

if st.button("Generate Summaries", type="primary", use_container_width=True):
    with st.spinner("Running SFT, PPO, DPO and scoring..."):
        tok, models, rm, device = load_all()
        prompt = clean_post(raw_post)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        post_item = [{"prompt": prompt, "prompt_ids": p_ids}]

        outputs = {}
        for name, model in models.items():
            sums, g_ids = generate_summaries(model, tok, post_item, batch_size=1, do_sample=False)
            batch = collate_sft([{"prompt_ids": p_ids, "summary_ids": g_ids[0]}], pad_id=tok.pad_token_id)
            with torch.no_grad():
                score = rm(batch["input_ids"].to(device), batch["attention_mask"].to(device)).item()
            outputs[name] = {
                "summary": sums[0].strip(),
                "score": score,
                "length": len(g_ids[0]),
            }

    best = max(outputs, key=lambda k: outputs[k]["score"])
    cols = st.columns(3)
    labels = {
        "SFT": "Stage 1: Supervised Fine-Tuning",
        "PPO": "Stage 3: Proximal Policy Optimization",
        "DPO": "Stage 4: Direct Preference Optimization",
    }

    for col, name in zip(cols, ("SFT", "PPO", "DPO")):
        res = outputs[name]
        delta = res["score"] - outputs["SFT"]["score"]
        with col:
            with st.container(border=True):
                title = f"#### {name}" + (" (Top Score)" if name == best else "")
                st.markdown(title)
                st.caption(labels[name])
                st.info(res["summary"])
                if name == "SFT":
                    st.metric("Reward Score", f"{res['score']:+.4f}")
                else:
                    st.metric("Reward Score", f"{res['score']:+.4f}", delta=f"{delta:+.4f} vs SFT")
                st.caption(f"Length: {res['length']} tokens")
