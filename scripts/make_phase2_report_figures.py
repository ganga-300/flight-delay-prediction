"""Draw the figures of the Phase 2 report from the saved results (no model is run).

Every value is read from reports/results/phase2_* and models/phase2/member1, the same files the
decision log (D-302 to D-313) and the Phase 2 deck use. Output: report_phase2/figures/*.png.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports/results"
OUT = ROOT / "report_phase2/figures"

# Same warm palette as the Phase 2 deck; no blue.
INK, INK2, SAND, ORANGE = "#2B2420", "#5C5148", "#C4B49C", "#B84A1C"
CARD, RULE, PAPER = "#EEE7DA", "#D8CFC0", "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

NAMES = {
    "lr": "Phase 1 baseline (LR)",
    "gru": "M2 GRU",
    "mlp": "M1 simple network",
    "embedding": "M3 embedding network",
    "plain_xgboost": "M4 plain XGBoost",
    "hybrid_dl_xgboost": "M4 hybrid",
}


def signed(v: float) -> str:
    """+0.039 / −0.006, with a true minus sign as in the report and deck."""
    return f"{v:+.3f}".replace("-", "\u2212")


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", name)


def comparison() -> pd.DataFrame:
    return pd.read_csv(RES / "phase2_member5/comparison_table.csv").set_index("model")


def bootstrap(path: str) -> pd.DataFrame:
    b = pd.read_csv(RES / path)
    b["comparison"] = b["comparison"].str.replace(" ", "")
    return b.set_index("comparison")


def fig_timeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    parts = [
        (0, 7, "Learn\nJan–Jul", CARD, INK),
        (7, 1, "Tune\nAug", SAND, INK),
        (8, 2, "Compare\nSep–Oct", ORANGE, PAPER),
        (10, 2, "Final test\nNov–Dec", INK, PAPER),
    ]
    for start, length, label, fill, color in parts:
        ax.add_patch(
            FancyBboxPatch(
                (start + 0.05, 0.1),
                length - 0.1,
                0.8,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc=fill,
                ec="none",
            )
        )
        ax.text(
            start + length / 2,
            0.5,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=9 if length < 2 else 10,
            fontweight="bold",
        )
    months = "JFMAMJJASOND"
    for i, m in enumerate(months):
        ax.text(i + 0.5, -0.12, m, ha="center", va="top", color=INK2, fontsize=8)
    ax.text(
        12,
        1.05,
        "opened once, after the model was frozen",
        ha="right",
        va="bottom",
        color=INK2,
        fontsize=8,
        style="italic",
    )
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.4, 1.3)
    ax.axis("off")
    save(fig, "fig1_timeline.png")


def fig_overall() -> None:
    c = comparison()
    order = ["hybrid_dl_xgboost", "plain_xgboost", "embedding", "mlp", "lr", "gru"]
    vals = [c.loc[m, "sepoct_pr_auc_mean"] for m in order]
    sds = [c.loc[m, "sepoct_pr_auc_sd"] for m in order]
    colors = [ORANGE if m == "gru" else INK if m == "lr" else SAND for m in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.barh(
        [NAMES[m] for m in order], vals, xerr=sds, color=colors, ecolor=INK2, capsize=3, height=0.6
    )
    for i, (v, s) in enumerate(zip(vals, sds, strict=True)):
        ax.text(v + s + 0.004, i, f"{v:.3f}", va="center", color=INK, fontsize=9)
    rate = pd.read_parquet(
        RES / "phase2_member4/sep_oct_predictions.parquet", columns=["is_delayed"]
    )
    rand = rate["is_delayed"].mean()
    ax.axvline(rand, color=INK2, ls=":", lw=1)
    ax.text(
        rand + 0.003, 5.45, f"random guessing = {rand:.3f}", color=INK2, fontsize=8, va="center"
    )
    ax.set_xlim(0, 0.34)
    ax.set_xlabel("Ranking score on September–October (higher is better)")
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig2_overall.png")


def fig_differences() -> None:
    b = bootstrap("phase2_member5/bootstrap_differences.csv")
    b2 = bootstrap("phase2_member2/bootstrap_pr_auc_differences.csv")
    rows = [
        ("M4 hybrid vs LR", b.loc["hybrid_dl_xgboost-lr"]),
        ("M4 plain XGBoost vs LR", b.loc["plain_xgboost-lr"]),
        ("M3 embedding network vs LR", b.loc["embedding-lr"]),
        ("M1 simple network vs LR", b.loc["mlp-lr"]),
        ("M2 GRU vs LR", b.loc["gru-lr"]),
        ("M2 GRU vs the same days, unordered", b2.loc["gru-control"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, (_, r) in enumerate(rows):
        real = bool(r["excludes_zero"])
        color = ORANGE if real else SAND
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color, lw=4, solid_capstyle="round")
        ax.plot(r["difference"], i, "o", color=INK, ms=6)
        ax.text(
            max(r["ci_high"], 0) + 0.003,
            i,
            signed(r["difference"]),
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Difference in ranking score, with the range it could plausibly take")
    ax.set_xlim(-0.06, 0.09)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        "Orange = a real difference (its range does not cross 0)",
        loc="right",
        color=ORANGE,
        fontsize=8.5,
    )
    save(fig, "fig3_differences.png")


def fig_mlp_curve() -> None:
    h = json.loads((ROOT / "models/phase2/member1/training_history.json").read_text())
    a = h["stage_a_jan_jul_vs_august"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(a["epoch"], a["valid_pr_auc"], "-o", color=ORANGE, lw=2)
    for e, v in zip(a["epoch"], a["valid_pr_auc"], strict=True):
        ax.text(e, v + 0.002, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xlabel("Passes through the training data")
    ax.set_ylabel("August ranking score")
    ax.set_ylim(0.33, 0.39)
    ax.grid(axis="y", color=RULE, lw=0.6)
    save(fig, "fig4_mlp_learning.png")


def fig_gru_idea() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    for i in range(14):
        ax.add_patch(
            FancyBboxPatch(
                (0.1 + i * 0.4, 0.55),
                0.32,
                0.5,
                boxstyle="round,pad=0,rounding_size=0.06",
                fc=ORANGE if i == 13 else CARD,
                ec="none",
            )
        )
    ax.text(0.1, 0.42, "14 days ago", fontsize=8, color=INK2, va="top")
    ax.text(
        0.1 + 13 * 0.4 + 0.32, 0.42, "yesterday", fontsize=8, color=ORANGE, va="top", ha="right"
    )
    ax.text(
        2.86,
        1.25,
        "each day: how delayed the airport, airline and route were; how busy it was",
        fontsize=8.5,
        color=INK2,
        ha="center",
    )

    def box(x, y, w, h, label, fill, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fill, ec="none"
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=8.5,
            fontweight="bold",
        )

    box(6.2, 0.55, 2.0, 0.55, "reads the days\nin order (GRU)", CARD, INK)
    box(6.2, -0.1, 2.0, 0.55, "the flight's\nown details", CARD, INK)
    box(8.85, 0.2, 1.55, 0.6, "chance of\ndelay", INK, PAPER)
    for y0 in (0.82, 0.17):
        ax.add_patch(
            FancyArrowPatch(
                (8.25, y0), (8.8, 0.5), arrowstyle="-|>", mutation_scale=12, color=ORANGE, lw=1.5
            )
        )
    ax.add_patch(
        FancyArrowPatch(
            (5.65, 0.8), (6.15, 0.8), arrowstyle="-|>", mutation_scale=12, color=ORANGE, lw=1.5
        )
    )
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.15, 1.4)
    ax.axis("off")
    save(fig, "fig5_gru_idea.png")


def fig_ablation() -> None:
    b = bootstrap("phase2_member5/ablation/bootstrap_differences.csv")
    rows = [
        ("Dropout and weight decay", b.loc["ablation_no_regularization-embedding"]),
        ("Embeddings (one-hot instead)", b.loc["ablation_one_hot-embedding"]),
        ("29 past-delay features", b.loc["ablation_no_history-embedding"]),
        ("4 timetable-busyness features", b.loc["ablation_no_congestion-embedding"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, (_, r) in enumerate(rows):
        real = bool(r["excludes_zero"])
        ax.plot(
            [r["ci_low"], r["ci_high"]],
            [i, i],
            color=ORANGE if real else SAND,
            lw=4,
            solid_capstyle="round",
        )
        ax.plot(r["difference"], i, "o", color=INK, ms=6)
        ax.text(
            max(r["ci_high"], 0) + 0.002,
            i,
            signed(r["difference"]),
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Change in ranking score when this ingredient is removed")
    ax.set_xlim(-0.05, 0.02)
    ax.grid(axis="x", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig6_ablation.png")


def fig_calibration() -> None:
    c = comparison()
    order = ["gru", "mlp", "lr", "embedding", "plain_xgboost", "hybrid_dl_xgboost"]
    raw = [c.loc[m, "oct_raw_ece_10_mean"] for m in order]
    cal = [c.loc[m, "oct_sigmoid_ece_10_mean"] for m in order]
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar([i - 0.2 for i in x], raw, width=0.38, color=SAND, label="before correction")
    ax.bar([i + 0.2 for i in x], cal, width=0.38, color=ORANGE, label="after September correction")
    for i, (r, k) in enumerate(zip(raw, cal, strict=True)):
        ax.text(i - 0.2, r + 0.004, f"{r:.3f}", ha="center", fontsize=7.5, color=INK2)
        ax.text(i + 0.2, k + 0.004, f"{k:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(
        list(x),
        [
            "M2\nGRU",
            "M1 simple\nnetwork",
            "Phase 1\nLR",
            "M3 embedding\nnetwork",
            "M4 plain\nXGBoost",
            "M4\nhybrid",
        ],
        fontsize=8,
    )
    ax.set_ylabel("Gap between predicted\nand actual chance (October)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=RULE, lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig7_calibration.png")


def fig_test() -> None:
    t = json.loads((RES / "phase2_member5/test_nov_dec/test_metrics.json").read_text())
    gru_raw, gru_cal = t["metrics"][0], t["metrics"][1]
    lr_raw, lr_cal = t["logistic_regression_d205"]["metrics"][:2]
    panels = [
        ("Ranking score (higher is better)", "pr_auc", gru_raw, lr_raw),
        ("Gap after correction (lower is better)", "ece_10", gru_cal, lr_cal),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, (title, key, g, lr) in zip(axes, panels, strict=True):
        vals = [g[key], lr[key]]
        ax.bar(["M2 GRU", "Phase 1 LR"], vals, color=[ORANGE, INK], width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f"{v:.3f}", ha="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", color=RULE, lw=0.6)
        ax.set_axisbelow(True)
    save(fig, "fig8_final_test.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_timeline()
    fig_overall()
    fig_differences()
    fig_mlp_curve()
    fig_gru_idea()
    fig_ablation()
    fig_calibration()
    fig_test()


if __name__ == "__main__":
    main()
