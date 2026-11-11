# Task 7 — Deep-Dive Analysis & Viva Reference

## 7.1 Reward Model Accuracy & Failure Modes
- **Accuracy on 1,000 validation pairs**: A 135M parameter model (`SmolLM2-135M`) trained for 1 epoch on 20,000 human preference pairs achieves ~62–66% pairwise validation accuracy (well above the 50% random chance baseline, though naturally below multi-billion-parameter reward models in Stiennon et al., 2020).
- **What kinds of pairs does $r_\phi$ get wrong?**
  1. **Length bias (verbosity heuristic)**: When the rejected summary is slightly longer and more detailed-sounding than a concise chosen summary, $r_\phi$ often assigns a higher score to the longer summary even if it includes minor hallucinations or fluff.
  2. **Subtle factual reversals / negations**: Pairs where `chosen` and `rejected` share high lexical overlap with the post (e.g., *"I broke up with my boyfriend because..."* vs. *"My boyfriend broke up with me because..."*) often fool a 135M backbone.
  3. **Sarcasm or noisy human labels**: In Reddit `r/relationships` or `r/tifu` posts, human annotators occasionally disagree on whether a witty/ironic summary is better than a dry literal one.

### Three Concrete Misclassified Validation Pairs:

**Example 1:**
- **Post Prompt:**
```text
he internet. 

What do you tell yourself when your sig other looks at these types of things? What advice can you give me on this? He always gets really defensive when I have issues about these things so I didn't know where else to go for help.
TL;DR:
```
- **Human Chosen ($r_\phi = +0.087$):** I'm having a hard time dealing with bf watching porn, and need advice on what to do.
- **Human Rejected ($r_\phi = +0.113$):** Bf watches porn, I'm trying to be okay with it, what do I tell myself?
- **Why RM failed:** The rejected summary received a higher score (+0.113 > +0.087) likely due to surface-level lexical similarity or length bias.

**Example 2:**
- **Post Prompt:**
```text

Anyways there are 2 days left in the vacation and I dont know what to do right now. Its the morning after so we need to see them soon. Obviously we need to talk as a group, but I really dont know what to do if the host continues his attitude.
TL;DR:
```
- **Human Chosen ($r_\phi = +0.135$):** friends played an incredibly cruel prank on me and my boyfriend, the host doesnt seem remorseful or apologize despite clearly knowing it was wrong. need advice on what to say and how to approach this situation.
- **Human Rejected ($r_\phi = +0.203$):** My friends played an incredibly cruel joke on me and my boyfriend, and the entire group seem to be completely unrepentant and refusing to apologize for it. What are our next steps?
- **Why RM failed:** The rejected summary received a higher score (+0.203 > +0.135) likely due to surface-level lexical similarity or length bias.

**Example 3:**
- **Post Prompt:**
```text
lly don't want to respond. I know a lot of people on Reddit believe that's unfair and will tell me to "woman up" but I don't trust myself. I need help and I need to get away from him. I love him but he's toxic. He's toxic to me and I need out.
TL;DR:
```
- **Human Chosen ($r_\phi = +0.806$):** Is it ever okay to not officially end things? emotionally abusive LDR. When is it okay to just stop responding?
- **Human Rejected ($r_\phi = +0.824$):** Me [23f] with [23m] 2.5 years LD. In what circumstances is it acceptable to go dark?
- **Why RM failed:** The rejected summary received a higher score (+0.824 > +0.806) likely due to surface-level lexical similarity or length bias.



---

## 7.2 PPO Training Dynamics: $r_\phi$ Score, KL Divergence, Summary Length, and Reward Hacking
- **Did the mean $r_\phi$ score rise?** Yes — over the 200 PPO iterations (3,200 rollout prompts), the policy steadily increases its mean reward score $r_\phi(x, y)$ while the KL divergence $\sum_t (\log \pi_{\theta_{\text{old}}} - \log \pi_{\text{ref}})$ grows gradually and stays bounded because of $\beta_{\text{KL}} = 0.05$.
- **Did summary length change?** Yes — PPO summaries typically drift toward the upper token limit (~40–48 tokens) if the reward model rewards denser or longer completions.
- **Is a higher reward-model score the same as a better summary?** Not always! This is **Goodhart's Law** ("when a measure becomes a target, it ceases to be a good measure"). Because $\pi_\theta$ optimizes the proxy reward model $r_\phi$ rather than true human judgment (or ROUGE-L against the reference), PPO can increase $r_\phi$ while ROUGE-L slightly drops. Human evaluation on the 30 blind posts confirms when higher $r_\phi$ reflects genuinely better summaries vs. reward over-optimization.

---

## 7.3 Role of $\beta_{\text{KL}}$ and What Happens When $\beta_{\text{KL}} = 0$
From Equation 41, the per-token reward is:
$$r_t = -\beta_{\text{KL}} \big(\log \pi_{\theta_{\text{old}}}(a_t \mid s_t) - \log \pi_{\text{ref}}(a_t \mid s_t)\big) + \mathbf{1}[t = T]\, r_\phi(x, y)$$
- **Role of $\beta_{\text{KL}}$**: It acts as a **trust-region / distributional anchor** keeping the RL policy $\pi_\theta$ close to the fluent SFT reference policy $\pi_{\text{ref}}$. Because the learned reward model $r_\phi$ is only trained on realistic SFT-like summaries, it is only accurate *near the support of $\pi_{\text{ref}}$*.
- **What happens if $\beta_{\text{KL}} = 0$?**
  1. **Reward Hacking (Adversarial Exploitation)**: Without any penalty for drifting away from $\pi_{\text{ref}}$, the policy searches for out-of-distribution token sequences that exploit quirks in $r_\phi$ (e.g., repeating high-scoring keywords or filler phrases).
  2. **Language Degeneration & Mode Collapse**: The policy loses grammatical fluency, stops producing coherent English sentences, repeats tokens endlessly, and collapses to a single deterministic template that fools $r_\phi$ into outputting huge positive scores while human quality drops to zero.

---

## 7.4 Clip Fraction & Policy Update Size
- **What is the clip fraction?** The fraction of generated summary tokens in a batch where the importance-sampling probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ falls outside $[1 - \epsilon, 1 + \epsilon] = [0.8, 1.2]$.
- **What does it look like during training?**
  - In the **first of the $K = 4$ epochs** of every iteration, $\pi_\theta = \pi_{\theta_{\text{old}}}$, so $r_t(\theta) = 1.0$ everywhere and the clip fraction is **0.0**.
  - Across epochs $k = 2, 3, 4$, the clip fraction is typically small (~0.02–0.12), indicating that `lr = 1e-5` moves the policy moderately within the trust region without destructive policy jumps. Whenever $r_t(\theta)$ exceeds $1 \pm \epsilon$ in the direction that improves the objective, the `min(...)` operator zeroes out the gradient on that token so the policy cannot take an dangerously large step on a single rollout batch.

---

## 7.5 Where the Reward Went in DPO & Why $Z(x)$ Cancels Out
In RLHF, the KL-constrained reward maximization objective has the closed-form optimal policy (Eqs. 55–56):
$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right), \quad \text{where } Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$
Solving this equation algebraically for the reward function $r(x, y)$ in terms of the policy $\pi(y \mid x)$ (Eq. 57) gives:
$$r(x, y) = \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$
- **Where did the reward go?** The policy $\pi_\theta$ *is itself* an implicit reward model via the log-ratio $\beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$.
- **Why does $Z(x)$ not need to be computed?** The Bradley–Terry preference model only depends on the **difference** of rewards between two summaries of the **same prompt $x$** (Eq. 58):
  $$r(x, y_w) - r(x, y_l) = \left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \cancel{\beta \log Z(x)}\right) - \left(\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \cancel{\beta \log Z(x)}\right)$$
  Because $Z(x)$ depends only on the prompt $x$ and not on $y_w$ or $y_l$, $+\beta \log Z(x)$ and $-\beta \log Z(x)$ cancel out identically!

---

## 7.6 GPU Memory Footprint, Training Time, and Method Choice
| Method | Models in GPU Memory | Online Generation Needed? | Stability & Hyperparameter Sensitivity |
|---|---|---|---|
| **PPO (Stage 3)** | **3 models** + 1 value head: trainable $\pi_\theta + V_\theta$, frozen $\pi_{\text{ref}}$, frozen $r_\phi$ | Yes (autoregressive `model.generate` rollouts at every iteration) | High variance; sensitive to $\beta_{\text{KL}}$, $\epsilon$, value loss $c_V$, and reward hacking |
| **DPO (Stage 4)** | **2 models**: trainable $\pi_\theta$, frozen $\pi_{\text{ref}}$ | No (pure supervised forward/backward pass on preference pairs) | Stable cross-entropy-like objective; only 1 hyperparameter ($\beta = 0.1$) |

- **Which method would you choose for a new project, and why?**
  For most alignment projects with offline preference pairs $(x, y_w, y_l)$, **DPO** is the better starting choice: it requires ~33% less GPU memory (2 models instead of 3 + value head), skips Stage 2 reward model training entirely, eliminates slow autoregressive generation rollouts during training, and avoids RL instability while achieving comparable or superior alignment quality. PPO is preferable only when you have a verifiable/programmatic reward function or need online exploration beyond the static preference dataset.

