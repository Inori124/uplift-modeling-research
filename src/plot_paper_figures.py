"""Create publication-style figures from the frozen experiment JSON files.

The script never trains a model or reads the multi-gigabyte data files.  It
only consumes committed, small JSON summaries under ``results/`` and writes
PNG (200 dpi) and vector PDF figures under ``reports/figures/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [2027, 2028, 2029, 2030, 2031]
MODEL_KEYS = ["t_learner_hgb", "s_learner_hgb", "x_learner_hgb_crossfit"]
LABELS = {
    "random_baseline": "Random draw",
    "t_learner_hgb": "T-learner HGB",
    "s_learner_hgb": "S-learner HGB",
    "x_learner_hgb_crossfit": "X-learner HGB + CF",
}
COLORS = {
    "random_baseline": "#6b7280",
    "t_learner_hgb": "#2563a6",
    "s_learner_hgb": "#c23b3b",
    "x_learner_hgb_crossfit": "#2f855a",
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
    }
)


def read_json(relative: str) -> dict:
    """Read one frozen result file; fail loudly if it is missing."""
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f"Required frozen result not found: {path}")
    return json.loads(path.read_text())


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure in the two formats used by the paper."""
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def sci_thousands(ax: plt.Axes, axis: str = "y") -> None:
    """Show probabilities in units of 10^-3 without a forest of zeroes."""
    scale = 1e3
    formatter = FuncFormatter(lambda value, _: f"{value * scale:.1f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel("Value (×10⁻³)")
    else:
        ax.xaxis.set_major_formatter(formatter)


def add_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.008, text, ha="left", va="bottom", fontsize=8, color="#4b5563")


def plot_data_overview() -> None:
    baseline = read_json("results/baseline_stats.json")
    n_control = baseline["groups"]["control"]["n"]
    n_treat = baseline["groups"]["treatment"]["n"]
    r_control = baseline["groups"]["control"]["conversion_rate"]
    r_treat = baseline["groups"]["treatment"]["conversion_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.35), gridspec_kw={"wspace": 0.32})
    axes[0].bar(["Control", "Treatment"], [n_control, n_treat], color=[COLORS["random_baseline"], "#4778a8"])
    axes[0].set_title("Treatment allocation")
    axes[0].set_ylabel("Rows")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    axes[0].text(0, n_control, f"{n_control:,}", ha="center", va="bottom", fontsize=8)
    axes[0].text(1, n_treat, f"{n_treat:,}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(["Control", "Treatment"], [r_control, r_treat], color=[COLORS["random_baseline"], "#4778a8"])
    axes[1].set_title("Observed conversion rate")
    axes[1].set_ylabel("Rate")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    axes[1].text(0, r_control, f"{r_control:.3%}", ha="center", va="bottom", fontsize=8)
    axes[1].text(1, r_treat, f"{r_treat:.3%}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Criteo randomized experiment: full-data overview", y=1.02, fontsize=13, fontweight="bold")
    add_note(fig, "Conversion rates use the full Criteo file; the unadjusted difference is descriptive, not an individual uplift estimate.")
    save_figure(fig, "paper_data_overview")


def plot_validation_qini() -> None:
    main = read_json("results/v2_main_seed2027.json")
    fig, ax = plt.subplots(figsize=(7.25, 4.1))
    for key in ["random_baseline"] + MODEL_KEYS:
        curve = main["curves"][key]
        ax.plot(
            curve["fraction"],
            curve["qini"],
            label=LABELS[key],
            color=COLORS[key],
            linewidth=2.0 if key != "random_baseline" else 1.2,
            linestyle="--" if key == "random_baseline" else "-",
        )
    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.set(xlabel="Fraction targeted", ylabel="Qini gain (×10⁻³)", title="Development validation Qini curves (seed 2027)")
    sci_thousands(ax)
    ax.legend(ncol=2)
    add_note(fig, "Qini(q) = G(q) − qG(1); the horizontal zero line is the theoretical random-selection reference. Tied scores use the evaluation routine's random-expectation convention.")
    save_figure(fig, "paper_qini_validation")


def plot_validation_summary() -> None:
    runs = [read_json(f"results/v2_main_seed{seed}.json") for seed in SEEDS]
    values = {key: np.array([run["metrics"][key]["qini_area"] for run in runs]) for key in MODEL_KEYS}
    means = np.array([values[key].mean() for key in MODEL_KEYS])
    sds = np.array([values[key].std(ddof=1) for key in MODEL_KEYS])
    x = np.arange(len(MODEL_KEYS))
    fig, ax = plt.subplots(figsize=(7.3, 4.15))
    for i, key in enumerate(MODEL_KEYS):
        jitter = np.linspace(-0.09, 0.09, len(SEEDS))
        ax.scatter(np.full(len(SEEDS), x[i]) + jitter, values[key], color=COLORS[key], s=28, alpha=0.8, zorder=3)
        ax.errorbar(x[i], means[i], yerr=sds[i], fmt="o", color="#111827", markerfacecolor="white", markersize=7, capsize=4, linewidth=1.5, zorder=4)
    ax.axhline(0, color="#374151", linewidth=0.9, label="Random expectation = 0")
    ax.set_xticks(x, [LABELS[key] for key in MODEL_KEYS])
    ax.set_ylabel("Qini area (×10⁻³)")
    ax.set_title("Validation Qini area across five model seeds", fontweight="bold")
    sci_thousands(ax)
    ax.legend(loc="upper left")
    add_note(fig, "Colored points: individual seeds (2027–2031). Black marker and whisker: mean ± sample SD. Zero is the theoretical random expectation.")
    save_figure(fig, "paper_validation_summary")


def plot_holdout_qini_ci() -> None:
    hold = read_json("results/final_holdout.json")
    keys = MODEL_KEYS
    fig, ax = plt.subplots(figsize=(7.25, 4.1))
    x = np.arange(len(keys))
    means = np.array([hold["metrics"][key]["qini_area"] for key in keys])
    lo = np.array([hold["bootstrap"]["ci"][key]["qini_area"][0] for key in keys])
    hi = np.array([hold["bootstrap"]["ci"][key]["qini_area"][1] for key in keys])
    err = np.vstack([means - lo, hi - means])
    ax.errorbar(x, means, yerr=err, fmt="o", ms=7, capsize=5, lw=1.6, color="#244b7a")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_xticks(x, [LABELS[key] for key in keys])
    ax.set_ylabel("Qini area (×10⁻³)")
    ax.set_title("Final holdout Qini area", fontweight="bold")
    sci_thousands(ax)
    add_note(fig, f"Whiskers: conditional stratified paired percentile 95% CI (B={hold['bootstrap']['n_bootstrap']}); models were not refit during bootstrap.")
    save_figure(fig, "paper_holdout_qini_ci")


def plot_holdout_qini() -> None:
    hold = read_json("results/final_holdout.json")
    fig, ax = plt.subplots(figsize=(7.25, 4.1))
    for key in ["random_baseline"] + MODEL_KEYS:
        curve = hold["curves"][key]
        ax.plot(curve["fraction"], curve["qini"], label=LABELS[key], color=COLORS[key], linewidth=2 if key != "random_baseline" else 1.2, linestyle="--" if key == "random_baseline" else "-")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set(xlabel="Fraction targeted", ylabel="Qini gain (×10⁻³)", title="Qini curves on untouched final holdout")
    sci_thousands(ax)
    ax.legend(ncol=2)
    add_note(fig, "Qini gain is centered at qG(1), so this panel is distinct from the raw cumulative policy gain G(q) shown in Figure 7. Tied scores use the random-expectation convention.")
    save_figure(fig, "paper_qini_holdout")


def plot_holdout_pairwise() -> None:
    hold = read_json("results/final_holdout.json")
    pairs = [
        ("t_learner_hgb", "s_learner_hgb", "T − S"),
        ("t_learner_hgb", "x_learner_hgb_crossfit", "T − X"),
        ("s_learner_hgb", "x_learner_hgb_crossfit", "S − X"),
    ]
    point = np.array([hold["metrics"][a]["qini_area"] - hold["metrics"][b]["qini_area"] for a, b, _ in pairs])
    intervals = np.array([hold["bootstrap"]["pairwise_difference_ci"][f"{a}-{b}"]["qini_area"] for a, b, _ in pairs])
    y = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(7.15, 3.7))
    lower = point - intervals[:, 0]
    upper = intervals[:, 1] - point
    ax.errorbar(point, y, xerr=np.vstack([lower, upper]), fmt="o", color="#244b7a", capsize=5, lw=1.7, markersize=7)
    ax.axvline(0, color="#374151", lw=0.9)
    ax.set_yticks(y, [label for _, _, label in pairs])
    ax.invert_yaxis()
    ax.set_xlabel("Paired Qini-area difference (×10⁻³)")
    ax.set_title("Pairwise model differences on final holdout", fontweight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1e3:.1f}"))
    ax.set_xlim(min(intervals[:, 0].min(), point.min()) * 1.15, max(intervals[:, 1].max(), point.max()) * 1.15)
    add_note(fig, f"Point = first model − second model. Whiskers: conditional paired percentile 95% CI (B={hold['bootstrap']['n_bootstrap']}); intervals crossing zero do not establish a reliable ranking.")
    fig.subplots_adjust(bottom=0.26, left=0.16, right=0.98, top=0.86)
    save_figure(fig, "paper_holdout_pairwise")


def plot_policy_gain_holdout() -> None:
    hold = read_json("results/final_holdout.json")
    q = np.asarray(hold["curves"]["s_learner_hgb"]["fraction"], dtype=float)
    ate = float(hold["metrics"]["s_learner_hgb"]["overall_ate"])
    fig, ax = plt.subplots(figsize=(7.25, 4.1))
    for key in ["random_baseline"] + MODEL_KEYS:
        curve = hold["curves"][key]
        ax.plot(curve["fraction"], curve["gain"], label=LABELS[key], color=COLORS[key], linewidth=1.25 if key == "random_baseline" else 2.0, linestyle="--" if key == "random_baseline" else "-")
    ax.plot(q, q * ate, color="#111827", linewidth=1.1, linestyle=":", label="Theoretical random: q × ATE")
    ax.set(xlabel="Fraction targeted, q", ylabel="Cumulative policy gain G(q) (×10⁻³)", title="Raw policy gain on final holdout")
    sci_thousands(ax)
    ax.legend(ncol=2)
    add_note(fig, "G(q) is the IPW cumulative incremental contribution per overall holdout row; dotted line is q × overall ATE. This is not the centered Qini curve. Tied scores use the random-expectation convention.")
    fig.subplots_adjust(bottom=0.24, left=0.14, right=0.98, top=0.86)
    save_figure(fig, "paper_policy_gain_holdout")


def plot_treatment_fraction() -> None:
    fractions = [("0.85", "085"), ("0.50", "050"), ("0.33", "033"), ("0.20", "020")]
    all_values = {key: [] for key in MODEL_KEYS}
    all_sds = {key: [] for key in MODEL_KEYS}
    for label, code in fractions:
        summary = read_json(f"results/v2_fixed_n120k_f{code}_summary.json")["summary"]
        for key in MODEL_KEYS:
            all_values[key].append(summary[key]["qini_area"]["mean"])
            all_sds[key].append(summary[key]["qini_area"]["std"])
    x = np.arange(len(fractions))
    fig, ax = plt.subplots(figsize=(7.35, 4.2))
    for key in MODEL_KEYS:
        means = np.asarray(all_values[key])
        sds = np.asarray(all_sds[key])
        ax.plot(x, means, color=COLORS[key], linewidth=2, marker="o", label=LABELS[key])
        ax.errorbar(x, means, yerr=sds, color=COLORS[key], alpha=0.75, capsize=3, linewidth=1.1)
        jitter = np.linspace(-0.055, 0.055, len(SEEDS))
        for i, (_, code) in enumerate(fractions):
            run_values = np.array([read_json(f"results/v2_fixed_n120k_f{code}_seed{seed}.json")["metrics"][key]["qini_area"] for seed in SEEDS])
            ax.scatter(np.full(len(run_values), x[i]) + jitter, run_values, color=COLORS[key], s=14, alpha=0.35, zorder=2)
    ax.set_xticks(x, [label for label, _ in fractions])
    ax.set_xlabel("Treated fraction in fixed 120k training set")
    ax.set_ylabel("Qini area (×10⁻³)")
    ax.set_title("Treatment-composition sensitivity across three learners", fontweight="bold")
    sci_thousands(ax)
    ax.legend(ncol=3, fontsize=9)
    add_note(fig, "Colored points: five seeds per fraction. Lines and whiskers: mean ± seed SD. Total training size is held at 120,000; only treatment composition changes.")
    fig.subplots_adjust(bottom=0.24, left=0.13, right=0.98, top=0.85)
    save_figure(fig, "paper_treatment_fraction")


def plot_study_design() -> None:
    baseline = read_json("results/baseline_stats.json")
    main = read_json("results/v2_main_seed2027.json")
    hold = read_json("results/final_holdout.json")
    fixed = read_json("results/v2_fixed_n120k_f085_summary.json")
    total = baseline["rows"]
    dev_n = main["n"]
    train_n = main["train_n"]
    validation_n = main["validation_n"]
    holdout_n = hold["holdout_n"]
    fixed_n = fixed["runs"][0]["train_n"]

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#1f2937", linewidth=0.8, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", fontsize=8.5, transform=ax.transAxes)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#374151"})

    box(0.02, 0.62, 0.19, 0.20, "Full Criteo file", f"{total:,} rows\nrandomized treatment", "#e5eef8")
    box(0.27, 0.62, 0.19, 0.20, "Development sample", f"{dev_n:,} rows\nfixed random seed", "#e8f5e9")
    box(0.53, 0.70, 0.17, 0.18, "Train pool", f"{train_n:,} rows", "#fff4d6")
    box(0.53, 0.39, 0.17, 0.18, "Validation", f"{validation_n:,} rows\nmodel comparison", "#fde8e8")
    box(0.78, 0.62, 0.20, 0.20, "Final holdout", f"{holdout_n:,} rows\nuntouched once", "#eee6ff")
    box(0.27, 0.10, 0.19, 0.18, "Fixed-size sensitivity", f"{fixed_n:,} rows\nfractions 0.85–0.20", "#f3f4f6")
    # Main split: the 1M development sample is split into a 750k train pool and 250k validation set.
    arrow(0.21, 0.72, 0.27, 0.72)
    arrow(0.46, 0.76, 0.53, 0.79)
    arrow(0.46, 0.67, 0.53, 0.48)
    # Fixed-size robustness samples are drawn from the train pool.
    arrow(0.61, 0.70, 0.46, 0.27)
    ax.text(0.52, 0.40, "from train pool", ha="center", va="center", fontsize=8, color="#4b5563", transform=ax.transAxes)
    # The holdout is row-disjoint from development; the final model is refit on all 1M dev rows.
    ax.plot([0.115, 0.115, 0.88, 0.88], [0.82, 0.95, 0.95, 0.82], color="#6b7280", linestyle="--", linewidth=1.0, transform=ax.transAxes)
    ax.annotate("", xy=(0.88, 0.82), xytext=(0.88, 0.95), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#6b7280"})
    ax.text(0.50, 0.965, "row-disjoint holdout sample from full file", ha="center", va="bottom", fontsize=8, color="#4b5563", transform=ax.transAxes)
    arrow(0.46, 0.63, 0.78, 0.63)
    ax.text(0.625, 0.57, "freeze configuration; refit on all 1M development rows", ha="center", va="center", fontsize=8, color="#4b5563", transform=ax.transAxes)
    fig.suptitle("Study design and sample separation", y=0.98, fontsize=13, fontweight="bold")
    add_note(fig, "The 300k holdout is sampled from rows excluded from the 1M development sample; it is evaluated only after the v2.1 configuration is frozen.")
    save_figure(fig, "paper_study_design")


def main() -> None:
    plot_data_overview()
    plot_validation_qini()
    plot_validation_summary()
    plot_holdout_qini_ci()
    plot_holdout_qini()
    plot_holdout_pairwise()
    plot_policy_gain_holdout()
    plot_treatment_fraction()
    plot_study_design()
    print(f"Saved {len(list(OUT.glob('paper_*.png')))} PNG figures and matching PDFs to {OUT}")


if __name__ == "__main__":
    main()
