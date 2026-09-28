"""Report figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clearpath.config import DATA_PROCESSED, FIGURES_DIR, METRICS_DIR, SHAP_DIR


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = ["Stage 1\nItemKNN candidates", "Stage 2\nLambdaMART ranker", "Stage 3\nTreeSHAP explain"]
    for i, text in enumerate(boxes):
        ax.add_patch(plt.Rectangle((i * 3.2, 0.3), 2.8, 1.2, fill=False, linewidth=2))
        ax.text(i * 3.2 + 1.4, 0.9, text, ha="center", va="center", fontsize=11)
        if i < 2:
            ax.annotate("", xy=((i + 1) * 3.2, 0.9), xytext=(i * 3.2 + 2.8, 0.9),
                        arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_xlim(-0.2, 9.5)
    ax.set_ylim(0, 2)
    ax.set_title("ClearPath cascade hybrid architecture")
    _save(fig, "fig1_architecture.png")


def fig2_funnel() -> None:
    funnel = pd.read_csv(DATA_PROCESSED / "funnel_counts.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(funnel))
    ax.barh(y - 0.2, funnel["rows"], height=0.35, color="#2F5D50", label="event rows")
    if "n_enrolments" in funnel.columns:
        ax.barh(y + 0.2, funnel["n_enrolments"], height=0.35, color="#C47B4A", label="enrolments")
    ax.set_yticks(y)
    ax.set_yticklabels(funnel["step"])
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("OULAD preprocessing funnel (rows vs enrolments)")
    ax.legend()
    _save(fig, "fig2_preprocessing_funnel.png")


def fig3_splits() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].set_title("Split A: cohort holdout")
    axes[0].bar(["Train\n2013", "Val\n15% of 2013", "Test\n2014"], [0.85, 0.15, 1.0], color=["#4C7C6B", "#7A9E8E", "#C47B4A"])
    axes[0].set_ylim(0, 1.4)
    axes[1].set_title("Split B: temporal + disjoint users")
    axes[1].bar(["Train 70%", "Val 15%", "Test 15%"], [0.7, 0.15, 0.15], color=["#4C7C6B", "#7A9E8E", "#C47B4A"])
    axes[1].set_ylim(0, 1.0)
    for ax in axes:
        ax.set_ylabel("Relative mass")
    fig.suptitle("Evaluation split designs")
    _save(fig, "fig3_split_designs.png")


def fig4_ndcg_bars() -> None:
    frames = []
    for split in ("split_a", "split_b"):
        path = METRICS_DIR / f"metrics_{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    pivot = df.pivot(index="variant", columns="split", values="ndcg@10")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax, color=["#2F5D50", "#C47B4A"])
    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 by variant and split")
    ax.legend(title="Split")
    _save(fig, "fig4_ndcg_by_variant.png")


def fig5_coverage_scatter() -> None:
    frames = []
    for split in ("split_a", "split_b"):
        path = METRICS_DIR / f"metrics_{split}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["split"] = split
            frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for split, g in df.groupby("split"):
        ax.scatter(g["ndcg@10"], g["coverage"], label=split, s=60)
        for _, row in g.iterrows():
            ax.annotate(row["variant"], (row["ndcg@10"], row["coverage"]), fontsize=8)
    ax.set_xlabel("NDCG@10")
    ax.set_ylabel("Catalogue coverage")
    ax.set_title("Accuracy vs coverage")
    ax.legend()
    _save(fig, "fig5_coverage_vs_ndcg.png")


def fig6_shap_beeswarm() -> None:
    cache = SHAP_DIR / "shap_demo_v4.parquet"
    meta_path = SHAP_DIR / "shap_demo_v4_meta.json"
    if not cache.exists():
        cache = SHAP_DIR / "shap_split_b_v4.parquet"
        meta_path = SHAP_DIR / "shap_split_b_v4_meta.json"
    if not cache.exists():
        return
    df = pd.read_parquet(cache)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    feat_cols = meta["features"]
    shap_cols = [f"shap_{c}" for c in feat_cols if f"shap_{c}" in df.columns]
    means = df[shap_cols].abs().mean().sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([c.replace("shap_", "") for c in means.index], means.values, color="#2F5D50")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("TreeSHAP feature importance summary (demo model)")
    _save(fig, "fig6_shap_summary.png")


def fig7_waterfall() -> None:
    cache = SHAP_DIR / "shap_demo_v4.parquet"
    meta_path = SHAP_DIR / "shap_demo_v4_meta.json"
    if not cache.exists():
        cache = SHAP_DIR / "shap_split_b_v4.parquet"
        meta_path = SHAP_DIR / "shap_split_b_v4_meta.json"
    if not cache.exists():
        return
    df = pd.read_parquet(cache)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    row = df.iloc[0]
    feat_cols = meta["features"]
    values = np.array([row[f"shap_{c}"] for c in feat_cols])
    order = np.argsort(-np.abs(values))[:8]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([feat_cols[i] for i in order][::-1], values[order][::-1], color="#C47B4A")
    ax.axvline(0, color="black", lw=1)
    title = row.get("label", f"site {row['id_site']}")
    ax.set_title(f"SHAP contributions - {title}")
    ax.set_xlabel("SHAP value")
    top3 = json.loads(row["top3"])
    ax.text(0.01, -0.22, " | ".join(top3), transform=ax.transAxes, fontsize=7, wrap=True)
    _save(fig, "fig7_shap_waterfall.png")


def fig8_faithfulness() -> None:
    path = SHAP_DIR / "faithfulness.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["ks"], data["top_k_drop"], marker="o", label="Top-k SHAP", color="#2F5D50")
    ax.plot(data["ks"], data["random_k_drop"], marker="s", label="Random-k", color="#C47B4A")
    ax.set_xlabel("k features removed")
    ax.set_ylabel("Mean score drop")
    ax.set_title(f"Faithfulness deletion (AUC gap={data['auc_gap']:.3f})")
    ax.legend()
    _save(fig, "fig8_faithfulness.png")


def fig9_stability() -> None:
    path = SHAP_DIR / "stability.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mat = np.array(data["matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(data["seeds"])))
    ax.set_yticks(range(len(data["seeds"])))
    ax.set_xticklabels(data["seeds"])
    ax.set_yticklabels(data["seeds"])
    ax.set_title(f"Attribution stability (mean ρ={data['mean_off_diagonal_spearman']:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "fig9_stability_heatmap.png")


def fig10_demo_from_live_data() -> None:
    """Render fig10 from the live Flask API."""
    from clearpath.app import app
    from clearpath.demo_showcase import STUDENTS

    client = app.test_client()
    eid = STUDENTS[0].enrolment_id if STUDENTS else None
    resp = client.get(f"/api/student/{STUDENTS[0].id}") if STUDENTS else None
    if resp is None or resp.status_code != 200:
        if not eid:
            print("Skipping fig10: no showcase students")
            return
        resp = client.get(f"/api/recommend/{eid}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    label = (data.get("student") or {}).get("name") or data.get("enrolment_id") or eid

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.95, "ClearPath demo (live API render)", fontsize=14, fontweight="bold", va="top")
    ax.text(
        0.02,
        0.90,
        f"Student: {label}  |  server {data.get('server_ms', 0):.1f} ms",
        fontsize=9,
        va="top",
    )
    y = 0.82
    for row in data["recommendations"][:5]:
        title = row.get("title") or row.get("label")
        ax.text(0.04, y, f"{row['rank']}. {title}  (score {row['score']:.3f})", fontsize=10, va="top")
        y -= 0.04
        for exp in row["explanations"][:2]:
            ax.text(0.08, y, f"• {exp}", fontsize=8, va="top", color="#333333")
            y -= 0.035
        y -= 0.02
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.88, fill=False, lw=1.2, edgecolor="#2F5D50"))
    _save(fig, "fig10_demo_screenshot.png")


def main() -> None:
    fig1_architecture()
    fig2_funnel()
    fig3_splits()
    fig4_ndcg_bars()
    fig5_coverage_scatter()
    fig6_shap_beeswarm()
    fig7_waterfall()
    fig8_faithfulness()
    fig9_stability()
    fig10_demo_from_live_data()


if __name__ == "__main__":
    main()
