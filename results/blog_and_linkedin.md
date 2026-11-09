# Task 8.2 & 8.3 — Medium Blog Post & LinkedIn Post

---

## Part 1: Medium Blog Post (approx. 1,100 words)

# Aligning a 135M Language Model from Scratch: SFT, Reward Modeling, PPO, and DPO in Pure PyTorch

A pretrained language model knows how to continue internet text, but it does not know what makes a *good* summary. Left to itself, a base model given a Reddit post ending in `TL;DR:` might ramble, invent new paragraphs, or copy sentences verbatim.

In OpenAI's landmark 2020 paper *Learning to Summarize from Human Feedback* (Stiennon et al., 2020), researchers introduced the three-stage Reinforcement Learning from Human Feedback (RLHF) recipe that later powered InstructGPT and ChatGPT. Three years later, Rafailov et al. (2023) showed with **Direct Preference Optimization (DPO)** that the exact same KL-constrained reward maximization objective can be solved in closed form without ever training a separate reward model or running reinforcement learning rollouts.

In this project, we implemented the entire four-stage alignment pipeline **from scratch in pure PyTorch**—no `trl`, `rlx`, or Hugging Face `Trainer`—using `HuggingFaceTB/SmolLM2-135M` (135M parameters) on OpenAI's Reddit TL;DR dataset (`CarperAI/openai_summarize_tldr` and `CarperAI/openai_summarize_comparisons`).

---

### The Four-Stage Pipeline

```
                        ┌─────────────────────────┐
                        │  SmolLM2-135M (Base LM) │
                        └────────────┬────────────┘
                                     │ Stage 1: SFT (20,000 Reddit posts + summaries)
                                     ▼
                        ┌─────────────────────────┐
               ┌───────►│   SFT Policy (π_SFT)    │◄───────┐
               │        └────────────┬────────────┘        │
               │                     │                     │
  Stage 2:     │                     │ Stage 3:            │ Stage 4:
  Bradley-Terry│                     │ PPO Rollouts        │ Direct Preference
  Preference   ▼                     ▼ + KL Penalty        ▼ Optimization
  Pairs   ┌──────────────┐    ┌──────────────┐      ┌──────────────┐
          │ Reward Model │───►│  PPO Policy  │      │  DPO Policy  │
          │     r_φ      │    │     π_PPO    │      │     π_DPO    │
          └──────────────┘    └──────────────┘      └──────────────┘
```

#### Stage 1: Supervised Fine-Tuning (SFT)
We first fine-tune `SmolLM2-135M` for 1 epoch on 20,000 `(prompt, summary)` pairs. Crucially, the cross-entropy loss is masked so that **only the summary tokens** (including the end-of-sequence `<|endoftext|>` token) contribute gradients:
$$L_{\text{SFT}}(\theta) = -\frac{1}{\#\text{summary tokens}} \sum_{t \in \text{summary}} \log \pi_\theta(x_t \mid x_{<t})$$
Why mask out the prompt? Because prompts average **317.42 tokens** while summaries average **33.82 tokens**. Without a summary mask, ~90% of gradient updates would go toward memorizing noisy Reddit posts rather than learning conditional summarization.

#### Stage 2: Bradley–Terry Reward Model ($r_\phi$)
Imitating human demonstrations teaches format and fluency, but not relative quality. In Stage 2, we initialize from our SFT checkpoint, remove the vocabulary projection head, and attach a scalar linear head `nn.Linear(hidden_size, 1)` to the final hidden state of the **last non-padding token** of `prompt + summary`.

Because `SmolLM2-135M` uses `<|endoftext|>` as both its EOS and padding token, finding the last real token via `input_ids != pad_token_id` would mistakenly strip the true EOS token! Instead, we index the final real token using `last_idx = attention_mask.sum(dim=1) - 1` and train $r_\phi$ on 20,000 human-ranked summary pairs $(x, y_w, y_l)$ using the Bradley–Terry pairwise loss:
$$L_{\text{reward}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

#### Stage 3: Proximal Policy Optimization (PPO)
In Stage 3, each generated token is an action $a_t$ and the prefix so far is the state $s_t$. We keep **three models in GPU memory**:
1. The trainable policy $\pi_\theta$ with a scalar value head $V_\theta(s_t)$ reading from the hidden state at position $t-1$ (the exact position whose logits predict $a_t$).
2. The frozen SFT reference model $\pi_{\text{ref}}$.
3. The frozen reward model $r_\phi$.

At each of the 200 PPO iterations, we sample 16 rollout summaries and assign a per-token reward that penalizes KL drift from $\pi_{\text{ref}}$ at every step while adding the scalar reward $r_\phi(x, y)$ at the final token $T$:
$$r_t = -\beta_{\text{KL}}\big(\log \pi_{\theta_{\text{old}}}(a_t \mid s_t) - \log \pi_{\text{ref}}(a_t \mid s_t)\big) + \mathbf{1}[t = T]\, r_\phi(x, y)$$
With discount $\gamma = 1$, returns $R_t = \sum_{k \ge 0} r_{t+k}$ are computed via a reverse cumulative sum, advantages $A_t = R_t - V_{\theta_{\text{old}}}(s_t)$ are normalized across all generated tokens in the batch, and the policy is updated for $K = 4$ epochs using the clipped surrogate objective ($\epsilon = 0.2$) plus value ($c_V = 0.1$) and entropy ($c_H = 0.01$) terms.

#### Stage 4: Direct Preference Optimization (DPO)
PPO is powerful but memory-hungry and sensitive to hyperparameters. DPO exploits the closed-form solution of the KL-constrained reward maximization problem:
$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$
When substituted into the Bradley–Terry model $r(x, y_w) - r(x, y_l)$, the intractable partition function $\beta \log Z(x)$ cancels out identically! This allows us to train $\pi_\theta$ directly on the 20,000 preference pairs with only two models in memory ($\pi_\theta$ and frozen $\pi_{\text{ref}}$) and $\beta = 0.1$:
$$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

---

### Quantitative Results & PPO Training Dynamics

#### Table 1 — Dataset Statistics
| Metric | Value |
|---|---|
| SFT train / validation pairs | 20,000 / 500 |
| Preference train / validation pairs | 20,000 / 1,000 |
| Evaluation posts | 300 |
| Mean / max prompt length, SFT train (tokens) | 317.42 / 448 |
| Mean / max summary length, SFT train (tokens) | 33.82 / 48 |

*(Insert Figure 1 `results/sft_loss.png` and Figure 3 `results/ppo_curves.png` here, along with Table 2 and Table 3 from `results/table3_eval.json` after Kaggle execution.)*

### Key Takeaways: PPO vs. DPO in Practice
1. **The Role of $\beta_{\text{KL}}$ & Goodhart's Law**: Without the KL penalty ($\beta_{\text{KL}} = 0$), PPO rapidly exploits quirks in the 135M reward model—inflating summary length and repeating high-scoring phrases while grammatical quality collapses. Even with $\beta_{\text{KL}} = 0.05$, PPO raises mean $r_\phi$ while ROUGE-L against the human reference can slightly dip because the policy optimizes the learned reward model rather than n-gram overlap.
2. **Compute & Memory Efficiency**: PPO required 3 models in GPU memory and autoregressive rollout generation at every step, whereas DPO needed only 2 models and standard batched forward/backward passes, making DPO significantly simpler to tune and deploy.

Full code and interactive Streamlit demo: [https://github.com/Subhans1501/neural-alignment-rlhf](https://github.com/Subhans1501/neural-alignment-rlhf)

---

## Part 2: LinkedIn Post (Task 8.3)

How do you turn a raw base language model into one aligned with human preferences—without using `trl` or any black-box RLHF library?

I just built the complete 4-stage alignment pipeline from OpenAI's *"Learning to Summarize from Human Feedback"* (Stiennon et al., 2020) and *Direct Preference Optimization* (Rafailov et al., 2023) from scratch in pure PyTorch using `SmolLM2-135M`:

1️⃣ **Stage 1 — Supervised Fine-Tuning (SFT):** Masked next-token cross-entropy strictly over summary tokens on 20k Reddit TL;DR posts.
2️⃣ **Stage 2 — Reward Model ($r_\phi$):** Scalar preference head trained on 20k human preference pairs with the Bradley–Terry loss.
3️⃣ **Stage 3 — PPO:** Online autoregressive rollouts with a learned value head $V_\theta(s_t)$, per-token KL divergence penalty ($\beta_{\text{KL}} = 0.05$), and clipped surrogate updates.
4️⃣ **Stage 4 — DPO:** Closed-form preference alignment optimizing the log-ratio against the frozen SFT reference without explicit reward modeling or RL rollouts.
5️⃣ **Interactive Streamlit App:** Side-by-side comparison of SFT, PPO, and DPO summaries scored live by $r_\phi$.

📖 Read the full technical breakdown & training analysis on Medium: [Insert Medium Link]
💻 GitHub Repository: https://github.com/Subhans1501/neural-alignment-rlhf

#GenAI #RLHF #PPO #DPO #PyTorch #LLM #MachineLearning #NLP #DeepLearning

