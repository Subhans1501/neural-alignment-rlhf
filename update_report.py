import os, json, re

def load_json_safe(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def update_analysis_md(rm_errors, sft_log, reward_log, ppo_log, dpo_log, table3):
    analysis_path = "results/analysis.md"
    if not os.path.exists(analysis_path):
        return
    with open(analysis_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update 7.1 with the 3 concrete RM errors
    if rm_errors and len(rm_errors) >= 3:
        err_str = "### Three Concrete Misclassified Validation Pairs:\n\n"
        for idx, ex in enumerate(rm_errors[:3], 1):
            err_str += (
                f"**Example {idx}:**\n"
                f"- **Post Prompt:**\n```text\n{ex['prompt'][-250:]}\n```\n"
                f"- **Human Chosen ($r_\\phi = {ex['r_chosen']:+.3f}$):** {ex['chosen'].strip()}\n"
                f"- **Human Rejected ($r_\\phi = {ex['r_rejected']:+.3f}$):** {ex['rejected'].strip()}\n"
                f"- **Why RM failed:** The rejected summary received a higher score ({ex['r_rejected']:+.3f} > {ex['r_chosen']:+.3f}) "
                f"likely due to surface-level lexical similarity or length bias.\n\n"
            )
        placeholder = "*(After running `evaluate.py` on Kaggle, paste 3 concrete misclassified `(prompt, chosen, rejected, r_w, r_l)` examples from `pref_val` here.)*"
        if placeholder in content:
            content = content.replace(placeholder, err_str)

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated results/analysis.md successfully!")

def update_readme_md(sft_log, reward_log, ppo_log, dpo_log, table3):
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Build Table 2
    sft_tr = f"{sft_log['train'][-1]:.4f}" if sft_log and "train" in sft_log and sft_log["train"] else "N/A"
    sft_va = f"val. loss: {sft_log['val'][-1]:.4f}" if sft_log and "val" in sft_log and sft_log["val"] else "N/A"
    sft_ti = f"{sft_log['time']/60:.1f} min / T4" if sft_log and "time" in sft_log else "T4"

    rm_tr = f"{reward_log['loss'][-1]:.4f}" if reward_log and "loss" in reward_log and reward_log["loss"] else "N/A"
    rm_va = f"pairwise acc.: {reward_log['val_acc'][-1]*100:.1f}%" if reward_log and "val_acc" in reward_log and reward_log["val_acc"] else "N/A"
    rm_ti = f"{reward_log['time']/60:.1f} min / T4" if reward_log and "time" in reward_log else "T4"

    ppo_tr = f"{ppo_log['logs'][-1]['l_ppo']:.4f}" if ppo_log and "logs" in ppo_log and ppo_log["logs"] else "N/A"
    ppo_va = f"final mean $r_\\phi$: {ppo_log['logs'][-1]['mean_r']:+.3f}" if ppo_log and "logs" in ppo_log and ppo_log["logs"] else "N/A"
    ppo_ti = f"{ppo_log['time']/60:.1f} min / T4" if ppo_log and "time" in ppo_log else "T4"

    dpo_tr = f"{dpo_log['loss'][-1]:.4f}" if dpo_log and "loss" in dpo_log and dpo_log["loss"] else "N/A"
    dpo_va = f"implicit reward acc.: {dpo_log['val_acc'][-1]*100:.1f}%" if dpo_log and "val_acc" in dpo_log and dpo_log["val_acc"] else "N/A"
    dpo_ti = f"{dpo_log['time']/60:.1f} min / T4" if dpo_log and "time" in dpo_log else "T4"

    t2_replacement = (
        "| Stage | Final train loss | Validation metric | Time / GPU |\n"
        "|---|---|---|---|\n"
        f"| SFT | {sft_tr} | {sft_va} | {sft_ti} |\n"
        f"| Reward model | {rm_tr} | {rm_va} | {rm_ti} |\n"
        f"| PPO | {ppo_tr} | {ppo_va} | {ppo_ti} |\n"
        f"| DPO | {dpo_tr} | {dpo_va} | {dpo_ti} |"
    )

    t2_pattern = r"\| Stage \| Final train loss \| Validation metric \| Time / GPU \|[\s\S]*?\| DPO \|.*\|"
    content = re.sub(t2_pattern, lambda m: t2_replacement, content)

    # Build Table 3 if table3_eval.json exists
    if table3:
        t3_rows = []
        for policy in ["SFT", "PPO", "DPO"]:
            if policy in table3:
                d = table3[policy]
                mr = f"{d['mean_r']:+.4f}"
                win = "—" if policy == "SFT" else f"{(d['win_vs_sft'] or 0)*100:.1f}%"
                kl = f"{d['kl_to_sft']:.4f}"
                ln = f"{d['length']:.2f}"
                rg = f"{d['rouge_l']:.4f}"
                t3_rows.append(f"| {policy} | {mr} | {win} | {kl} | {ln} | {rg} |")
        if t3_rows:
            t3_replacement = (
                "| Policy | Mean $r_\\phi$ | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |\n"
                "|---|---|---|---|---|---|\n" + "\n".join(t3_rows)
            )
            t3_pattern = r"\| Policy \| Mean \$r_\\phi\$ \| Win rate vs\. SFT \| KL to SFT \| Length \| ROUGE-L \|[\s\S]*?\| DPO \|.*\|"
            content = re.sub(t3_pattern, lambda m: t3_replacement, content)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Updated README.md successfully!")

def main():
    sft_log = load_json_safe("results/sft_log.json")
    reward_log = load_json_safe("results/reward_log.json")
    ppo_log = load_json_safe("results/ppo_log.json")
    dpo_log = load_json_safe("results/dpo_log.json")
    table3 = load_json_safe("results/table3_eval.json")
    rm_errors = load_json_safe("results/rm_errors.json")

    update_readme_md(sft_log, reward_log, ppo_log, dpo_log, table3)
    update_analysis_md(rm_errors, sft_log, reward_log, ppo_log, dpo_log, table3)

if __name__ == "__main__":
    main()

