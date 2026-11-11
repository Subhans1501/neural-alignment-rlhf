# Learning to Summarize from Human Feedback (SFT, Reward Model, PPO & DPO)

From-scratch PyTorch reproduction of *Learning to Summarize from Human Feedback* (Stiennon et al., 2020) using `HuggingFaceTB/SmolLM2-135M`, implemented without any RLHF libraries (`trl`, `peft`, `Trainer`, etc.).

## Repository Structure

```text
starter/          Given data loading, cleaning, tokenisation, batching, and model loading
common.py         Task 1: token_logprobs, summary_mask, and generation helper
train_sft.py      Task 2: Stage 1 Supervised Fine-Tuning (SFT)
train_reward.py   Task 3: Stage 2 Bradley-Terry Reward Model (r_phi)
train_ppo.py      Task 4: Stage 3 Proximal Policy Optimization (PPO)
train_dpo.py      Task 5: Stage 4 Direct Preference Optimization (DPO)
evaluate.py       Task 6: Automatic evaluation (r_phi, Win rate, KL, Length, ROUGE-L) & human eval sheet
results/          Training curves, logs, tables, eval_summaries.jsonl, human_eval.csv, analysis.md
app/app.py        Task 8: Streamlit web interface comparing SFT, PPO, and DPO side by side
```

## How to Run

```bash
# 1. Install dependencies
pip install torch transformers datasets rouge-score matplotlib streamlit

# 2. Task 1: Verify starter pipeline and common.py helpers
python starter/check_starter.py
python common.py

# 3. Tasks 2-5: Train all 4 stages sequentially
python train_sft.py
python train_reward.py
python train_ppo.py
python train_dpo.py

# 4. Task 6: Evaluate SFT, PPO, and DPO on the 300 held-out Reddit posts
python evaluate.py

# 5. Task 8: Launch the interactive web interface
streamlit run app/app.py
```

## Correctness Checks (Section 3.3)

Verified output from our implementation:

```text
1. SFT mask check: ' Short summary.<|endoftext|>'
2. Frozen check -> ref: True rm: True
3. PPO iter 1 KL: 0.000000
4. PPO epoch 0 ratio max diff from 1: 0.000000e+00, clip_frac: 0.0000
5. DPO step 1 loss: 0.693147 (log 2 = 0.693147)
6. Padding check -> last real token indices via attention_mask: [27, 28] actual lengths: [27, 28]
```

## Results

### Table 1 — Data
| Metric | Value |
|---|---|
| SFT train / validation pairs | 20,000 / 500 |
| Preference train / validation pairs | 20,000 / 1,000 |
| Evaluation posts | 300 |
| Mean / max prompt length, SFT train (tokens) | 317.42 / 448 |
| Mean / max summary length, SFT train (tokens) | 33.82 / 48 |

### Table 2 — Training
*(Populated from `results/*_log.json` after GPU training)*

| Stage | Final train loss | Validation metric | Time / GPU |
|---|---|---|---|
| SFT | 2.4167 | val. loss: 2.4492 | 72.6 min / T4 |
| Reward model | 0.6921 | pairwise acc.: 52.0% | 130.0 min / T4 |
| PPO | -0.0010 | final mean $r_\phi$: +0.461 | 67.9 min / T4 |
| DPO | 0.6919 | implicit reward acc.: 49.4% | 232.9 min / T4 |

### Table 3 — Final Comparison (300 Evaluation Posts)
*(Populated by `python evaluate.py` -> `results/table3_eval.json`)*

| Policy | Mean $r_\phi$ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
|---|---|---|---|---|---|
| SFT | +0.4093 | — | 0.0000 | 31.46 | 0.2534 |
| PPO | +0.4344 | 34.0% | -0.4398 | 28.09 | 0.2501 |
| DPO | +0.4010 | 18.3% | 0.0311 | 31.83 | 0.2533 |

### Table 4 — Human Evaluation (30 Posts)
*(From blind randomized human evaluation sheet `results/human_eval.csv`)*

| Evaluator | SFT picked | PPO picked | DPO picked |
|---|---|---|---|
| **Member 1** | 1 | 26 | 3 |
| **Member 2** | 2 | 19 | 9 |

**Agreement between members:** 22 / 30 (**73.3%**) — consistent with human annotator agreement reported in Stiennon et al. (2020).

## Training Figures & Curves

1. **Figure 1 — SFT Loss:** `results/sft_loss.png`
2. **Figure 2 — Reward Model Curve:** `results/reward_curve.png`
3. **Figure 3 — PPO Training Dynamics:** `results/ppo_curves.png`
4. **Figure 4 — DPO Training Curve:** `results/dpo_curve.png`

## Qualitative Samples & Analysis
- **Task 4.6 (10 Qualitative Samples):** Complete side-by-side prompt and generation breakdown in [`results/samples.md`](results/samples.md).
- **Task 7 (Analysis Questions 7.1–7.6):** Detailed mathematical answers, failure analysis, and viva reference in [`results/analysis.md`](results/analysis.md).
- **Task 8.2 & 8.3 (Blog & LinkedIn Post):** Technical article and announcement post in [`results/blog_and_linkedin.md`](results/blog_and_linkedin.md).


