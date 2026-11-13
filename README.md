# Learning to Summarize from Human Feedback (SFT, Reward Model, PPO & DPO)

From-scratch PyTorch implementation of *Learning to Summarize from Human Feedback* (Stiennon et al., 2020) using `HuggingFaceTB/SmolLM2-135M`, without any RLHF libraries (`trl`, `peft`, `Trainer`, etc.).

## Repository Structure

```text
starter/          Given data loading, cleaning, tokenisation, batching, and model loading
common.py         token_logprobs, summary_mask, and generation helper
train_sft.py      Stage 1: Supervised Fine-Tuning (SFT)
train_reward.py   Stage 2: Bradley-Terry Reward Model (r_phi)
train_ppo.py      Stage 3: Proximal Policy Optimization (PPO)
train_dpo.py      Stage 4: Direct Preference Optimization (DPO)
evaluate.py       Evaluation across SFT, PPO, and DPO on 300 test posts
results/          Training curves, logs, tables, eval_summaries.jsonl
app/app.py        Streamlit web interface comparing SFT, PPO, and DPO side by side
```

## How to Run

```bash
# 1. Install dependencies
pip install torch transformers datasets rouge-score matplotlib streamlit

# 2. Verify starter pipeline and common helpers
python starter/check_starter.py
python common.py

# 3. Train all 4 stages sequentially
python train_sft.py
python train_reward.py
python train_ppo.py
python train_dpo.py

# 4. Evaluate SFT, PPO, and DPO on the 300 held-out Reddit posts
python evaluate.py

# 5. Launch the web interface
streamlit run app/app.py
```

## Correctness Checks

Verified output from running the checks:

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

### Table 2 — Training (Tesla T4 GPU)
| Stage | Final train loss | Validation metric | Time / GPU |
|---|---|---|---|
| SFT | 2.4167 | val. loss: 2.4492 | 72.6 min / T4 |
| Reward model | 0.6921 | pairwise acc.: 52.0% | 130.0 min / T4 |
| PPO | -0.0010 | final mean $r_\phi$: +0.461 | 67.9 min / T4 |
| DPO | 0.6919 | implicit reward acc.: 49.4% | 232.9 min / T4 |

### Table 3 — Final Comparison (300 Evaluation Posts)
| Policy | Mean $r_\phi$ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |
|---|---|---|---|---|---|
| SFT | +0.4093 | — | 0.0000 | 31.46 | 0.2534 |
| PPO | +0.4344 | 34.0% | -0.4398 | 28.09 | 0.2501 |
| DPO | +0.4010 | 18.3% | 0.0311 | 31.83 | 0.2533 |

## Training Figures
- `results/sft_loss.png` — SFT training and validation loss
- `results/reward_curve.png` — Reward model loss and validation accuracy
- `results/ppo_curves.png` — PPO mean reward, KL, length, and clip fraction
- `results/dpo_curve.png` — DPO loss and implicit reward accuracy
