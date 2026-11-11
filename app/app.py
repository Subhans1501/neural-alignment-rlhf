import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "starter"))

import streamlit as st
import torch
from data import clean_post
from common import load_tokenizer, load_causal_lm, generate_summaries
from batching import collate_sft
from train_reward import RewardModel

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Neural Alignment — SFT vs PPO vs DPO",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS for Modern, Premium Dark/Light Aesthetic
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .hero-header {
        background: linear-gradient(135deg, #1E1E2E 0%, #2A2A40 50%, #1B2838 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #D1D5DB;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    .policy-card {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 14px;
        padding: 1.4rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .policy-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }

    .badge-sft {
        background-color: #2563EB;
        color: white;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    .badge-ppo {
        background-color: #059669;
        color: white;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    .badge-dpo {
        background-color: #7C3AED;
        color: white;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    .winner-pill {
        background: linear-gradient(90deg, #F59E0B, #D97706);
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 6px;
        float: right;
        letter-spacing: 0.5px;
    }

    .summary-box {
        background: rgba(0, 0, 0, 0.18);
        border-left: 4px solid #6366F1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.98rem;
        line-height: 1.5;
        min-height: 100px;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model Checkpoint Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading checkpoints into memory...")
def load_all():
    tok = load_tokenizer()
    sft = load_causal_lm("checkpoints/sft").eval()
    device = next(sft.parameters()).device
    ppo = load_causal_lm("checkpoints/ppo", device=device).eval()
    dpo = load_causal_lm("checkpoints/dpo", device=device).eval()
    rm = RewardModel.load("checkpoints/reward", sft_path="checkpoints/sft", device=device).eval()
    return tok, {"SFT": sft, "PPO": ppo, "DPO": dpo}, rm, device


# -----------------------------------------------------------------------------
# Pre-Loaded Evaluation Examples
# -----------------------------------------------------------------------------
PRESETS = {
    "🚗 Roommate Car Dispute (r/relationships)": (
        "SUBREDDIT: r/relationships\n"
        "TITLE: My [24F] roommate [25F] keeps borrowing my car without asking first.\n"
        "POST: We have lived together for six months and generally get along well. However, over the last two weeks "
        "she has taken my spare keys off the kitchen hook three times to run errands while I was asleep. When I confronted "
        "her, she said she always puts gas back in and didn't want to wake me up, so it shouldn't be a big deal."
    ),
    "💼 Unpaid Notice Period (r/legaladvice)": (
        "SUBREDDIT: r/legaladvice\n"
        "TITLE: Can my boss take money out of my paycheck if I'm not going against my contract?\n"
        "POST: So I'm quitting my job and gave my two week notice to my boss. I offered to stay longer than two weeks but he "
        "seems to not care about when I go. In my contract, I get 8 paid days off. I have only taken 6 so far. I told my boss "
        "today that I have not used all my paid days off and would like to use the remaining two next week. His response was that "
        "he doesn't want to pay for my vacation since I'm leaving on short notice. Is this legal?"
    ),
    "🐕 Rescue Dog Training (r/Dogtraining)": (
        "SUBREDDIT: r/Dogtraining\n"
        "TITLE: New rescue won't go to the bathroom outside\n"
        "POST: I recently adopted a pretty abused Yorkie, he's two years old and is very timid around everyone. When I first got "
        "him, about a week ago, I was taking him out every 2-3 hours but could rarely get him to go to the bathroom. Instead, he just "
        "comes inside, walks around my apartment for 10 minutes and then goes. I've tried positive reinforcement with treats but "
        "he seems terrified outside. Any tips on how to build his outdoor confidence?"
    ),
}

# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">⚖️ Learning to Summarize from Human Feedback</div>
    <div class="hero-subtitle">Interactive comparison of <b>Supervised Fine-Tuning (SFT)</b>, <b>Proximal Policy Optimization (PPO)</b>, and <b>Direct Preference Optimization (DPO)</b> on <i>SmolLM2-135M</i> scored by a Bradley–Terry Reward Model.</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Alignment Benchmark Overview
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📊 Empirical Benchmark (300 Posts)")
    st.markdown("""
    | Metric | SFT | PPO | DPO |
    | :--- | :---: | :---: | :---: |
    | **Reward Score** | `+0.409` | **`+0.434`** | `+0.401` |
    | **Win Rate vs SFT** | — | **`34.0%`** | `18.3%` |
    | **ROUGE-L** | `0.253` | `0.250` | `0.253` |
    | **Avg Length** | `31.5` tok | **`28.1` tok** | `31.8` tok |
    | **Human Picks** | 2 / 30 | **26 / 30** | 3 / 30 |
    """)
    st.caption("Human agreement rate: **73.3%** across 30 blind evaluation posts.")
    st.markdown("---")
    st.markdown("### 🛠️ Architecture")
    st.markdown("""
    - **Base LM:** `SmolLM2-135M`
    - **Dataset:** OpenAI Reddit TL;DR
    - **RLHF Recipe:** Stiennon et al. (2020)
    - **Pure PyTorch:** No `trl` / `Trainer`
    """)
    st.markdown("---")
    st.markdown("[🔗 GitHub Repository](https://github.com/Subhans1501/neural-alignment-rlhf)")

# -----------------------------------------------------------------------------
# Main Input Section
# -----------------------------------------------------------------------------
col_sel, col_empty = st.columns([2, 1])
with col_sel:
    preset_choice = st.selectbox("🎯 Choose a preset example or enter custom text below:", ["Custom Post"] + list(PRESETS.keys()))

if preset_choice != "Custom Post":
    initial_text = PRESETS[preset_choice]
else:
    initial_text = PRESETS[list(PRESETS.keys())[0]]

raw_post = st.text_area("Reddit Post (Subreddit / Title / Body):", value=initial_text, height=160)

btn_generate = st.button("✨ Align & Generate Summaries", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Inference & Side-by-Side Results
# -----------------------------------------------------------------------------
if btn_generate:
    with st.spinner("Generating summaries across all 3 policies and scoring via Reward Model..."):
        tok, models, rm, device = load_all()
        prompt = clean_post(raw_post)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        post_item = [{"prompt": prompt, "prompt_ids": p_ids}]

        results = {}
        for name, model in models.items():
            sums, g_ids = generate_summaries(model, tok, post_item, batch_size=1, do_sample=False)
            b = collate_sft([{"prompt_ids": p_ids, "summary_ids": g_ids[0]}], pad_id=tok.pad_token_id)
            with torch.no_grad():
                score = rm(b["input_ids"].to(device), b["attention_mask"].to(device)).item()
            results[name] = {
                "summary": sums[0].strip(),
                "score": score,
                "length": len(g_ids[0]),
            }

        # Find best scoring policy
        best_policy = max(results, key=lambda k: results[k]["score"])

    st.markdown("### 📑 Policy Comparison & Alignment Scores")
    cols = st.columns(3)

    badges = {
        "SFT": ("badge-sft", "Stage 1: Supervised Baseline"),
        "PPO": ("badge-ppo", "Stage 3: Online RLHF"),
        "DPO": ("badge-dpo", "Stage 4: Direct Preference"),
    }

    for col, (name, d) in zip(cols, results.items()):
        badge_cls, badge_desc = badges[name]
        is_winner = (name == best_policy)
        winner_html = '<span class="winner-pill">🏆 Top Reward</span>' if is_winner else ""
        border_color = "#10B981" if is_winner else "rgba(255, 255, 255, 0.1)"

        delta_sft = d["score"] - results["SFT"]["score"]
        delta_str = f"({delta_sft:+.3f} vs SFT)" if name != "SFT" else "(baseline)"

        with col:
            st.markdown(f"""
            <div class="policy-card" style="border: 2px solid {border_color};">
                <div>
                    <div>
                        <span class="{badge_cls}">{name} Policy</span>
                        {winner_html}
                    </div>
                    <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 4px;">{badge_desc}</div>
                    <div class="summary-box">
                        "{d['summary']}"
                    </div>
                </div>
                <div>
                    <div class="metric-row">
                        <span style="color: #9CA3AF;">Reward Score $r_\\phi(x, y)$:</span>
                        <span style="font-weight: 700; font-size: 1.05rem; color: {'#34D399' if d['score'] >= results['SFT']['score'] else '#F87171'};">
                            {d['score']:+.4f} <span style="font-size: 0.75rem; font-weight: normal;">{delta_str}</span>
                        </span>
                    </div>
                    <div class="metric-row">
                        <span style="color: #9CA3AF;">Length:</span>
                        <span><b>{d['length']}</b> tokens</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Deep Dive Explanation Tabs
# -----------------------------------------------------------------------------
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🔬 Methodology & Math", "📈 Empirical Insights", "💬 Viva Discussion Points"])

with tab1:
    st.markdown(r"""
    #### The 4-Stage Alignment Flow
    1. **SFT ($\pi_{\text{SFT}}$):** Base model trained with masked cross-entropy exclusively on summary tokens:
       $$L_{\text{SFT}}(\theta) = -\frac{1}{\#\text{tokens}} \sum_{t \in \text{summary}} \log \pi_\theta(x_t \mid x_{<t})$$
    2. **Reward Model ($r_\phi$):** Linear scalar head on the final hidden state of the last non-padding token, trained with the Bradley–Terry loss:
       $$L_{\text{reward}}(\phi) = -\mathbb{E}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$
    3. **PPO ($\pi_{\text{PPO}}$):** Policy with value head $V_\theta(s_t)$ optimizing per-token reward constrained by KL drift:
       $$r_t = -\beta_{\text{KL}} \big(\log \pi_{\theta} - \log \pi_{\text{ref}}\big) + \mathbf{1}[t = T] r_\phi(x, y)$$
    4. **DPO ($\pi_{\text{DPO}}$):** Closed-form substitution eliminating the reward model and partition function $Z(x)$:
       $$L_{\text{DPO}}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$
    """)

with tab2:
    st.markdown("""
    #### Key Observations from 300 Evaluation Posts
    - **PPO achieved the highest mean reward score (+0.4344 vs +0.4093 SFT)**, proving that reinforcement learning successfully guided the policy toward higher human preference regions.
    - **Conciseness Bonus:** PPO summaries averaged **28.09 tokens**, significantly punchier than SFT (31.46 tokens), avoiding trailing cutoffs.
    - **Human Evaluation Agreement:** In our 30-post blind test, human evaluators overwhelmingly selected **PPO (26 picks by Member 1, 19 picks by Member 2)** with an inter-annotator agreement rate of **73.3%**.
    """)

with tab3:
    st.markdown("""
    #### Questions to Expect in the Viva
    - **Why does ROUGE-L slightly drop for PPO?**  
      Because of *Goodhart's Law*: PPO optimizes the reward model's criteria (conciseness, punchiness, clarity) rather than exact n-gram overlap with the reference summary.
    - **Why is $\\beta_{\\text{KL}}$ crucial?**  
      If $\\beta_{\\text{KL}} = 0$, the policy exploits flaws in $r_\\phi$ (reward hacking), causing language degeneration and repetitive gibberish.
    """)
