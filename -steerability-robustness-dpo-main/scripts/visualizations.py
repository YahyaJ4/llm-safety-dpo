#!/usr/bin/env python3
"""
Publication-Quality Visualizations for Safety-Capability Tradeoff Paper
EXPANDED EVALUATION RESULTS (520 AdvBench, 1000 MMLU, 500 GSM8K)

Creates clean, insightful plots for the DPO safety alignment study.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Color palette
COLORS = {
    "llama": "#2E86AB",
    "qwen": "#A23B72",
    "llama_light": "#7FB3D3",
    "qwen_light": "#D4899B",
    "safety": "#28965A",
    "capability": "#F18F01",
}

# EXPANDED DATA (LLM Judge for AdvBench, 520/1000/500 samples)
llama_data = {
    "stage": ["SFT", "β=0.01", "β=0.05", "β=0.1", "β=0.5", "β=1.0"],
    "beta": [None, 0.01, 0.05, 0.1, 0.5, 1.0],
    "refusal": [22.31, 85.19, 79.04, 78.08, 68.46, 66.15],  # LLM Judge (520 samples)
    "mmlu": [50.9, 49.3, 49.9, 50.2, 50.8, 50.2],  # 1000 samples
    "gsm8k": [11.8, 12.4, 11.8, 11.4, 11.8, 11.6],  # 500 samples
}

qwen_data = {
    "stage": ["SFT", "β=0.01", "β=0.05", "β=0.1", "β=0.5", "β=1.0"],
    "beta": [None, 0.01, 0.05, 0.1, 0.5, 1.0],
    "refusal": [53.27, 99.81, 99.81, 100.0, 96.15, 93.46],  # LLM Judge (520 samples)
    "mmlu": [68.0, 66.6, 67.5, 68.3, 68.1, 68.1],  # 1000 samples
    "gsm8k": [78.8, 25.2, 26.2, 28.6, 44.4, 47.6],  # 500 samples
}

output_dir = Path(
    "/Users/mahmoudzahran/Desktop/MS at KAUST/Fall 25/CS 394E/group_project/kumail/expanded_results_backup/figures"
)
output_dir.mkdir(parents=True, exist_ok=True)


def plot_1_safety_improvement():
    """Figure 1: DPO Dramatically Improves Safety"""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(2)
    width = 0.35

    sft_values = [22.31, 53.27]  # Llama, Qwen
    dpo_values = [85.19, 100.0]  # Best DPO for each (β=0.01 for Llama, β=0.1 for Qwen)

    # Use consistent colors for SFT (light gray) and DPO (dark gray/green)
    bars1 = ax.bar(
        x - width / 2,
        sft_values,
        width,
        label="After SFT",
        color="#CCCCCC",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        dpo_values,
        width,
        label="After DPO (best β)",
        color="#28965A",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar, val in zip(bars1, sft_values):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    for bar, val, beta in zip(bars2, dpo_values, ["β=0.01", "β=0.1"]):
        ax.annotate(
            f"{val:.1f}%\n({beta})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add improvement annotations (positioned between bars with arrow)
    for i, (sft, dpo) in enumerate(zip(sft_values, dpo_values)):
        improvement = dpo - sft
        # Draw arrow from SFT bar to DPO bar
        ax.annotate(
            "",
            xy=(i + width / 2 - 0.02, dpo - 5),  # arrow end (DPO bar)
            xytext=(i - width / 2 + 0.02, sft + 5),  # arrow start (SFT bar)
            arrowprops=dict(arrowstyle="->", color="#28965A", lw=1.5),
        )
        # Add text label to the left of bars
        ax.annotate(
            f"+{improvement:.0f}pp",
            xy=(i - width - 0.05, (sft + dpo) / 2),
            fontsize=10,
            ha="right",
            va="center",
            color="#28965A",
            fontweight="bold",
        )

    ax.set_ylabel("Refusal Rate (%)")
    ax.set_title(
        "DPO Dramatically Improves Safety Alignment\n(520 AdvBench samples, GPT-4o Judge)",
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Llama 3.1-8B", "Qwen 2.5-7B"])
    ax.legend(loc="upper left")
    ax.set_ylim(0, 120)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_safety_improvement.pdf")
    plt.savefig(output_dir / "fig1_safety_improvement.png")
    plt.close()
    print("Created: fig1_safety_improvement.pdf")


def plot_2_safety_capability_tradeoff():
    """Figure 2: Safety vs Capability Tradeoff (Scatter plot)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    llama_refusal = llama_data["refusal"]
    llama_mmlu = llama_data["mmlu"]

    qwen_refusal = qwen_data["refusal"]
    qwen_mmlu = qwen_data["mmlu"]

    ax.scatter(
        llama_refusal,
        llama_mmlu,
        c=COLORS["llama"],
        s=120,
        label="Llama 3.1-8B",
        marker="o",
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        qwen_refusal,
        qwen_mmlu,
        c=COLORS["qwen"],
        s=120,
        label="Qwen 2.5-7B",
        marker="s",
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )

    ax.plot(
        llama_refusal, llama_mmlu, c=COLORS["llama"], alpha=0.3, linewidth=1.5, zorder=1
    )
    ax.plot(
        qwen_refusal, qwen_mmlu, c=COLORS["qwen"], alpha=0.3, linewidth=1.5, zorder=1
    )

    # Label key points
    ax.annotate(
        "SFT",
        (llama_refusal[0], llama_mmlu[0]),
        xytext=(-15, -15),
        textcoords="offset points",
        fontsize=9,
        color=COLORS["llama"],
    )
    ax.annotate(
        "β=0.01\n(best safety)",
        (llama_refusal[1], llama_mmlu[1]),
        xytext=(10, 5),
        textcoords="offset points",
        fontsize=9,
        color=COLORS["llama"],
        fontweight="bold",
    )

    ax.annotate(
        "SFT",
        (qwen_refusal[0], qwen_mmlu[0]),
        xytext=(-25, -15),
        textcoords="offset points",
        fontsize=9,
        color=COLORS["qwen"],
    )
    ax.annotate(
        "β=0.1\n(100%!)",
        (qwen_refusal[3], qwen_mmlu[3]),
        xytext=(5, -20),
        textcoords="offset points",
        fontsize=9,
        color=COLORS["qwen"],
        fontweight="bold",
    )

    ax.axvspan(80, 105, alpha=0.1, color=COLORS["safety"], zorder=0)
    ax.axhspan(65, 75, alpha=0.1, color=COLORS["capability"], zorder=0)
    ax.annotate(
        "High Safety\nRegion",
        (92, 52),
        fontsize=9,
        ha="center",
        color="#333333",
        style="italic",
    )

    ax.set_xlabel("Safety (Refusal Rate %)", fontweight="bold")
    ax.set_ylabel("Capability (MMLU Accuracy %)", fontweight="bold")
    ax.set_title(
        "Safety-Capability Tradeoff Across DPO Configurations\n(1000 MMLU samples)",
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower left")
    ax.set_xlim(10, 110)
    ax.set_ylim(45, 75)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_safety_capability_tradeoff.pdf")
    plt.savefig(output_dir / "fig2_safety_capability_tradeoff.png")
    plt.close()
    print("Created: fig2_safety_capability_tradeoff.pdf")


def plot_3_beta_analysis():
    """Figure 3: Effect of Beta on Safety and Capability"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    betas = ["0.01", "0.05", "0.1", "0.5", "1.0"]

    llama_refusal_dpo = llama_data["refusal"][1:]
    llama_mmlu_dpo = llama_data["mmlu"][1:]
    llama_sft_refusal = llama_data["refusal"][0]
    llama_sft_mmlu = llama_data["mmlu"][0]

    qwen_refusal_dpo = qwen_data["refusal"][1:]
    qwen_mmlu_dpo = qwen_data["mmlu"][1:]
    qwen_sft_refusal = qwen_data["refusal"][0]
    qwen_sft_mmlu = qwen_data["mmlu"][0]

    # Left plot: Llama
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.plot(
        betas,
        llama_refusal_dpo,
        "o-",
        color=COLORS["safety"],
        linewidth=2,
        markersize=8,
        label="Refusal Rate",
    )
    ax1.axhline(
        y=llama_sft_refusal,
        color=COLORS["safety"],
        linestyle="--",
        alpha=0.5,
        label="SFT Baseline",
    )

    ax1_twin.plot(
        betas,
        llama_mmlu_dpo,
        "s-",
        color=COLORS["capability"],
        linewidth=2,
        markersize=8,
        label="MMLU",
    )
    ax1_twin.axhline(
        y=llama_sft_mmlu, color=COLORS["capability"], linestyle="--", alpha=0.5
    )

    ax1.set_xlabel("DPO Beta (β)", fontweight="bold")
    ax1.set_ylabel("Refusal Rate (%)", color=COLORS["safety"], fontweight="bold")
    ax1_twin.set_ylabel(
        "MMLU Accuracy (%)", color=COLORS["capability"], fontweight="bold"
    )
    ax1.set_title("Llama 3.1-8B", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1_twin.set_ylim(45, 55)
    ax1.tick_params(axis="y", labelcolor=COLORS["safety"])
    ax1_twin.tick_params(axis="y", labelcolor=COLORS["capability"])

    # Right plot: Qwen
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.plot(
        betas,
        qwen_refusal_dpo,
        "o-",
        color=COLORS["safety"],
        linewidth=2,
        markersize=8,
        label="Refusal Rate",
    )
    ax2.axhline(y=qwen_sft_refusal, color=COLORS["safety"], linestyle="--", alpha=0.5)

    ax2_twin.plot(
        betas,
        qwen_mmlu_dpo,
        "s-",
        color=COLORS["capability"],
        linewidth=2,
        markersize=8,
        label="MMLU",
    )
    ax2_twin.axhline(
        y=qwen_sft_mmlu, color=COLORS["capability"], linestyle="--", alpha=0.5
    )

    ax2.set_xlabel("DPO Beta (β)", fontweight="bold")
    ax2.set_ylabel("Refusal Rate (%)", color=COLORS["safety"], fontweight="bold")
    ax2_twin.set_ylabel(
        "MMLU Accuracy (%)", color=COLORS["capability"], fontweight="bold"
    )
    ax2.set_title("Qwen 2.5-7B", fontweight="bold")
    ax2.set_ylim(50, 105)
    ax2_twin.set_ylim(64, 72)
    ax2.tick_params(axis="y", labelcolor=COLORS["safety"])
    ax2_twin.tick_params(axis="y", labelcolor=COLORS["capability"])

    safety_patch = mpatches.Patch(color=COLORS["safety"], label="Refusal Rate (Safety)")
    cap_patch = mpatches.Patch(color=COLORS["capability"], label="MMLU (Capability)")
    fig.legend(
        handles=[safety_patch, cap_patch],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_beta_analysis.pdf")
    plt.savefig(output_dir / "fig3_beta_analysis.png")
    plt.close()
    print("Created: fig3_beta_analysis.pdf")


def plot_4_gsm8k_capability_tradeoff():
    """Figure 4: GSM8K Capability Degradation - NEW KEY FINDING"""
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = ["SFT", "β=0.01", "β=0.05", "β=0.1", "β=0.5", "β=1.0"]

    qwen_gsm8k = qwen_data["gsm8k"]
    qwen_refusal = qwen_data["refusal"]

    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        qwen_gsm8k,
        width,
        label="GSM8K Accuracy",
        color=COLORS["capability"],
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        qwen_refusal,
        width,
        label="Refusal Rate",
        color=COLORS["safety"],
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, val in zip(bars1, qwen_gsm8k):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, qwen_refusal):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Highlight the dramatic drop
    ax.annotate(
        "",
        xy=(0.175, 78.8),
        xytext=(1.175, 25.2),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ax.annotate(
        "-53.6pp!",
        xy=(0.7, 55),
        fontsize=12,
        ha="center",
        color="red",
        fontweight="bold",
    )

    # Highlight recovery
    ax.annotate(
        "Recovery\nwith higher β",
        xy=(4.5, 50),
        fontsize=10,
        ha="center",
        color="darkgreen",
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title(
        "Qwen 2.5-7B: Safety Alignment Hurts Math Reasoning\n(500 GSM8K samples)",
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend(loc="center right")
    ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_gsm8k_capability_tradeoff.pdf")
    plt.savefig(output_dir / "fig4_gsm8k_capability_tradeoff.png")
    plt.close()
    print("Created: fig4_gsm8k_capability_tradeoff.pdf")


def plot_5_base_model_comparison():
    """Figure 5: Base Model Matters"""
    fig, ax = plt.subplots(figsize=(10, 6))

    stages = ["SFT\n(Starting)", "DPO β=0.01", "DPO β=0.1", "DPO β=1.0"]

    llama_refusal = [22.31, 85.19, 78.08, 66.15]
    qwen_refusal = [53.27, 99.81, 100.0, 93.46]

    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        llama_refusal,
        width,
        label="Llama 3.1-8B",
        color=COLORS["llama"],
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        qwen_refusal,
        width,
        label="Qwen 2.5-7B",
        color=COLORS["qwen"],
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, val in zip(bars1, llama_refusal):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar, val in zip(bars2, qwen_refusal):
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.annotate(
        "",
        xy=(0 - width / 2, 22.31),
        xytext=(0 + width / 2, 53.27),
        arrowprops=dict(arrowstyle="<->", color="red", lw=2),
    )
    ax.annotate(
        "2.4x gap!",
        xy=(0, 38),
        fontsize=11,
        ha="center",
        color="red",
        fontweight="bold",
    )

    ax.set_ylabel("Refusal Rate (%)", fontweight="bold")
    ax.set_title(
        "Base Model Safety: Qwen Starts Much Safer Than Llama",
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(output_dir / "fig5_base_model_comparison.pdf")
    plt.savefig(output_dir / "fig5_base_model_comparison.png")
    plt.close()
    print("Created: fig5_base_model_comparison.pdf")


def plot_6_comprehensive_heatmap():
    """Figure 6: Comprehensive Results Heatmap"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    llama_matrix = np.array(
        [
            [22.31, 50.9, 11.8],  # SFT
            [85.19, 49.3, 12.4],  # β=0.01
            [79.04, 49.9, 11.8],  # β=0.05
            [78.08, 50.2, 11.4],  # β=0.1
            [68.46, 50.8, 11.8],  # β=0.5
            [66.15, 50.2, 11.6],  # β=1.0
        ]
    )

    qwen_matrix = np.array(
        [
            [53.27, 68.0, 78.8],  # SFT
            [99.81, 66.6, 25.2],  # β=0.01
            [99.81, 67.5, 26.2],  # β=0.05
            [100.0, 68.3, 28.6],  # β=0.1
            [96.15, 68.1, 44.4],  # β=0.5
            [93.46, 68.1, 47.6],  # β=1.0
        ]
    )

    row_labels = ["SFT", "β=0.01", "β=0.05", "β=0.1", "β=0.5", "β=1.0"]
    col_labels = ["Refusal %", "MMLU %", "GSM8K %"]

    def normalize_column(matrix):
        normalized = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            normalized[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-8)
        return normalized

    # Llama
    ax1 = axes[0]
    im1 = ax1.imshow(
        normalize_column(llama_matrix), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
    )
    ax1.set_xticks(np.arange(len(col_labels)))
    ax1.set_yticks(np.arange(len(row_labels)))
    ax1.set_xticklabels(col_labels)
    ax1.set_yticklabels(row_labels)
    ax1.set_title("Llama 3.1-8B", fontweight="bold")

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax1.text(
                j,
                i,
                f"{llama_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    # Qwen
    ax2 = axes[1]
    im2 = ax2.imshow(
        normalize_column(qwen_matrix), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1
    )
    ax2.set_xticks(np.arange(len(col_labels)))
    ax2.set_yticks(np.arange(len(row_labels)))
    ax2.set_xticklabels(col_labels)
    ax2.set_yticklabels(row_labels)
    ax2.set_title("Qwen 2.5-7B", fontweight="bold")

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax2.text(
                j,
                i,
                f"{qwen_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.suptitle(
        "Complete Results Overview - Expanded Evaluation\n(Green = Better within column)",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "fig6_comprehensive_heatmap.pdf")
    plt.savefig(output_dir / "fig6_comprehensive_heatmap.png")
    plt.close()
    print("Created: fig6_comprehensive_heatmap.pdf")


def plot_7_main_result_figure():
    """Figure 7: Main Result - Publication Hero Figure"""
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1], wspace=0.3)

    # Left: Safety improvement (matches Figure 1 exactly)
    ax1 = fig.add_subplot(gs[0])
    models = ["Llama\n3.1-8B", "Qwen\n2.5-7B"]
    sft_safety = [22.31, 53.27]
    dpo_safety = [85.19, 100.0]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        sft_safety,
        width,
        label="After SFT",
        color="#CCCCCC",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax1.bar(
        x + width / 2,
        dpo_safety,
        width,
        label="After DPO (best β)",
        color="#28965A",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bar, val in zip(bars1, sft_safety):
        ax1.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, val, beta in zip(bars2, dpo_safety, ["β=0.01", "β=0.1"]):
        ax1.annotate(
            f"{val:.1f}%\n({beta})",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    # Add improvement annotations with arrows
    for i, (sft, dpo) in enumerate(zip(sft_safety, dpo_safety)):
        improvement = dpo - sft
        ax1.annotate(
            "",
            xy=(i + width / 2 - 0.02, dpo - 5),
            xytext=(i - width / 2 + 0.02, sft + 5),
            arrowprops=dict(arrowstyle="->", color="#28965A", lw=1.5),
        )
        ax1.annotate(
            f"+{improvement:.0f}pp",
            xy=(i - width - 0.05, (sft + dpo) / 2),
            fontsize=8,
            ha="right",
            va="center",
            color="#28965A",
            fontweight="bold",
        )

    ax1.set_ylabel("Refusal Rate (%)")
    ax1.set_title("(a) Safety Alignment", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(0, 120)

    # Middle: Safety vs GSM8K tradeoff for Qwen
    ax2 = fig.add_subplot(gs[1])

    qwen_refusal = qwen_data["refusal"]
    qwen_gsm8k = qwen_data["gsm8k"]

    ax2.scatter(
        qwen_refusal,
        qwen_gsm8k,
        c=COLORS["qwen"],
        s=120,
        marker="s",
        edgecolors="black",
        linewidth=0.5,
    )
    ax2.plot(qwen_refusal, qwen_gsm8k, c=COLORS["qwen"], alpha=0.3, linewidth=1.5)

    # Label points
    labels = ["SFT", "β=0.01", "β=0.05", "β=0.1", "β=0.5", "β=1.0"]
    for i, (x, y, label) in enumerate(zip(qwen_refusal, qwen_gsm8k, labels)):
        offset = (5, 5) if i != 0 else (-20, 5)
        ax2.annotate(
            label, (x, y), xytext=offset, textcoords="offset points", fontsize=8
        )

    ax2.set_xlabel("Safety (Refusal %)")
    ax2.set_ylabel("Math Capability (GSM8K %)")
    ax2.set_title("(b) Qwen Safety-Capability Tradeoff", fontweight="bold")
    ax2.set_xlim(45, 105)
    ax2.set_ylim(20, 85)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Right: Beta effect on both models
    ax3 = fig.add_subplot(gs[2])

    betas = ["0.01", "0.05", "0.1", "0.5", "1.0"]
    llama_ref = llama_data["refusal"][1:]
    qwen_ref = qwen_data["refusal"][1:]

    ax3.plot(
        betas,
        llama_ref,
        "o-",
        color=COLORS["llama"],
        linewidth=2,
        markersize=8,
        label="Llama",
    )
    ax3.plot(
        betas,
        qwen_ref,
        "s-",
        color=COLORS["qwen"],
        linewidth=2,
        markersize=8,
        label="Qwen",
    )

    ax3.set_xlabel("DPO Beta (β)")
    ax3.set_ylabel("Refusal Rate (%)")
    ax3.set_title("(c) Effect of β on Safety", fontweight="bold")
    ax3.legend(loc="center right")
    ax3.set_ylim(60, 105)

    # Annotate trend
    ax3.annotate(
        "Lower β =\nHigher Safety",
        xy=(0.5, 68),
        fontsize=9,
        ha="center",
        color="darkgreen",
        style="italic",
    )

    plt.suptitle(
        "DPO Achieves Strong Safety but with Capability Tradeoffs",
        fontweight="bold",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig7_main_result.pdf")
    plt.savefig(output_dir / "fig7_main_result.png")
    plt.close()
    print("Created: fig7_main_result.pdf")


def create_latex_table():
    """Generate LaTeX table for paper"""
    latex = r"""
\begin{table}[t]
\centering
\caption{Safety-Capability Tradeoff Results (Expanded Evaluation). Refusal rate measured on AdvBench (520 samples) using GPT-4o judge. Capability measured on MMLU (1000 samples) and GSM8K (500 samples). Best results per metric in \textbf{bold}.}
\label{tab:main_results_expanded}
\begin{tabular}{llccc}
\toprule
\textbf{Model} & \textbf{Stage} & \textbf{Refusal \%} $\uparrow$ & \textbf{MMLU \%} $\uparrow$ & \textbf{GSM8K \%} $\uparrow$ \\
\midrule
\multirow{6}{*}{Llama 3.1-8B}
 & SFT & 22.3 & \textbf{50.9} & \textbf{12.4} \\
 & DPO $\beta$=0.01 & \textbf{85.2} & 49.3 & 12.4 \\
 & DPO $\beta$=0.05 & 79.0 & 49.9 & 11.8 \\
 & DPO $\beta$=0.1 & 78.1 & 50.2 & 11.4 \\
 & DPO $\beta$=0.5 & 68.5 & 50.8 & 11.8 \\
 & DPO $\beta$=1.0 & 66.2 & 50.2 & 11.6 \\
\midrule
\multirow{6}{*}{Qwen 2.5-7B}
 & SFT & 53.3 & 68.0 & \textbf{78.8} \\
 & DPO $\beta$=0.01 & 99.8 & 66.6 & 25.2 \\
 & DPO $\beta$=0.05 & 99.8 & 67.5 & 26.2 \\
 & DPO $\beta$=0.1 & \textbf{100.0} & \textbf{68.3} & 28.6 \\
 & DPO $\beta$=0.5 & 96.2 & 68.1 & 44.4 \\
 & DPO $\beta$=1.0 & 93.5 & 68.1 & 47.6 \\
\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / "table_main_results_expanded.tex", "w") as f:
        f.write(latex)
    print("Created: table_main_results_expanded.tex")


if __name__ == "__main__":
    print("Generating publication-quality figures for EXPANDED results...")
    print("=" * 60)

    plot_1_safety_improvement()
    plot_2_safety_capability_tradeoff()
    plot_3_beta_analysis()
    plot_4_gsm8k_capability_tradeoff()
    plot_5_base_model_comparison()
    plot_6_comprehensive_heatmap()
    plot_7_main_result_figure()
    create_latex_table()

    print("=" * 60)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nKey differences from small-scale:")
    print("  - Sample sizes: 520 AdvBench, 1000 MMLU, 500 GSM8K")
    print("  - LLM Judge (GPT-4o) for all AdvBench evaluations")
    print("  - NEW finding: Dramatic GSM8K degradation for Qwen (78.8% -> 25%)")
    print("  - Qwen reaches 100% refusal at β=0.1")
