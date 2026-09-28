"""Offline metrics, ablations, Wilcoxon tests, table export."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from clearpath.config import DATA_PROCESSED, METRICS_DIR, MODELS_DIR, RANDOM_SEED, TOP_N
from clearpath.ranker import RANK_FEATURES, predict_ranking


def relevant_items(observed: pd.DataFrame, heldout: pd.DataFrame) -> dict[str, set[int]]:
    seen = observed.groupby("enrolment_id")["id_site"].apply(lambda s: set(int(x) for x in s))
    out: dict[str, set[int]] = {}
    for eid, g in heldout.groupby("enrolment_id"):
        out[eid] = set(int(x) for x in g["id_site"]) - set(seen.get(eid, set()))
    return out


def ndcg_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return float(dcg / ideal) if ideal > 0 else 0.0


def recall_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_items[:k]) & relevant) / len(relevant)


def ap_at_k(ranked_items: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits, sum_prec = 0, 0.0
    for i, item in enumerate(ranked_items[:k], start=1):
        if item in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(relevant), k)


def _boot_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = RANDOM_SEED) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_rankings(
    rankings: pd.DataFrame,
    relevance: dict[str, set[int]],
    all_items_by_presentation: dict[tuple[str, str], set[int]] | None,
    popularity_rank: dict[int, float],
    enrolments: pd.DataFrame,
    k: int = TOP_N,
) -> tuple[dict, pd.DataFrame]:
    enr_meta = enrolments.set_index("enrolment_id")
    per_user = []
    recommended_items: set[int] = set()
    tail_hits = total_recs = 0
    if popularity_rank:
        thresh = np.quantile(list(popularity_rank.values()), 0.8)
        head_items = {i for i, p in popularity_rank.items() if p >= thresh}
    else:
        head_items = set()

    for eid, g in rankings.groupby("enrolment_id"):
        ranked = g.sort_values("rank")["id_site"].astype(int).tolist()
        rel = relevance.get(eid, set())
        per_user.append(
            {
                "enrolment_id": eid,
                "ndcg": ndcg_at_k(ranked, rel, k),
                "recall": recall_at_k(ranked, rel, k),
                "map": ap_at_k(ranked, rel, k),
            }
        )
        recommended_items.update(ranked[:k])
        for item in ranked[:k]:
            total_recs += 1
            if item not in head_items:
                tail_hits += 1

    user_df = pd.DataFrame(per_user)
    catalogue = float("nan")
    if all_items_by_presentation is not None and len(user_df):
        universe: set[int] = set()
        for eid in user_df["enrolment_id"]:
            if eid not in enr_meta.index:
                continue
            row = enr_meta.loc[eid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            universe |= all_items_by_presentation.get(
                (row["code_module"], row["code_presentation"]), set()
            )
        catalogue = len(recommended_items) / len(universe) if universe else 0.0

    ndcg = user_df["ndcg"].to_numpy() if len(user_df) else np.array([])
    ndcg_ci = _boot_mean_ci(ndcg)
    summary = {
        "ndcg@10": float(ndcg.mean()) if len(ndcg) else 0.0,
        # Bootstrap CI of mean NDCG@10.
        "ndcg@10_ci95_low": ndcg_ci[0],
        "ndcg@10_ci95_high": ndcg_ci[1],
        "recall@10": float(user_df["recall"].mean()) if len(user_df) else 0.0,
        "map@10": float(user_df["map"].mean()) if len(user_df) else 0.0,
        "coverage": float(catalogue) if catalogue == catalogue else float("nan"),
        "tail_exposure": float(tail_hits / total_recs) if total_recs else 0.0,
        "n_users": int(len(user_df)),
    }
    return summary, user_df


def load_baselines(path: Path, variant: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.loc[df["variant"] == variant].copy()


def joblib_features(model_path: Path) -> list[str]:
    import joblib

    feat_path = model_path.with_name(model_path.stem + "_features.joblib")
    if feat_path.exists():
        return joblib.load(feat_path)
    meta_path = model_path.with_name(model_path.stem + "_meta.json")
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)["features"]
    return [c for c in RANK_FEATURES]


def rank_from_model(model_path: Path, features_path: Path, variant: str) -> pd.DataFrame:
    feats = pd.read_parquet(features_path)
    booster = lgb.Booster(model_file=str(model_path))
    feature_cols = [c for c in joblib_features(model_path) if c in feats.columns]
    ranked = predict_ranking(booster, feats, feature_cols, top_n=TOP_N)
    ranked["variant"] = variant
    return ranked


def bootstrap_mean_diff_ci(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = RANDOM_SEED
) -> dict:
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    diff = (paired["a"] - paired["b"]).to_numpy()
    if len(diff) < 10:
        return {"mean_diff": None, "ci95_low": None, "ci95_high": None, "n": int(len(diff))}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        boots[i] = sample.mean()
    return {
        "mean_diff": float(diff.mean()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n": int(len(diff)),
    }


def wilcoxon_safe(a: pd.Series, b: pd.Series) -> dict:
    """Wilcoxon signed-rank with matched-pairs rank-biserial effect size."""
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    diff = paired["a"] - paired["b"]
    nonzero = diff[diff != 0]
    n = int(len(paired))
    if len(nonzero) < 10:
        return {
            "stat": None,
            "pvalue": None,
            "effect_size_rbc": None,
            "n": n,
            **bootstrap_mean_diff_ci(a, b),
        }
    stat, p = wilcoxon(paired["a"], paired["b"], zero_method="wilcox")
    # Rank-biserial from positive Wilcoxon ranks.
    ranks = nonzero.abs().rank()
    w_plus = float(ranks[nonzero > 0].sum())
    n_nz = len(nonzero)
    r_bc = (2.0 * w_plus) / (n_nz * (n_nz + 1) / 2.0) - 1.0
    out = {
        "stat": float(stat),
        "pvalue": float(p),
        "effect_size_rbc": float(r_bc),
        "n": n,
    }
    out.update(bootstrap_mean_diff_ci(a, b))
    return out


def run_eval_bundle(
    split_label: str,
    observed_path: Path,
    heldout_path: Path,
    baselines_path: Path,
    model_prefix: str,
    feature_prefix: str,
    save_demo_rankings: bool = False,
) -> None:
    enrolments = pd.read_parquet(DATA_PROCESSED / "enrolments.parquet")
    items = pd.read_parquet(DATA_PROCESSED / "items.parquet")
    observed = pd.read_parquet(observed_path)
    heldout = pd.read_parquet(heldout_path)
    relevance = relevant_items(observed, heldout)

    pool = {
        (m, p): set(int(x) for x in g["id_site"])
        for (m, p), g in items.groupby(["code_module", "code_presentation"])
    }
    pop = observed.groupby("id_site")["sum_click"].sum().astype(float).to_dict()

    results = []
    user_scores = {}

    for variant, key in [
        ("V0_WeekPop", "V0"),
        ("V1_MostPopular", "V1"),
        ("V2_ItemKNN", "V2"),
    ]:
        rankings = load_baselines(baselines_path, variant)
        summary, user_df = evaluate_rankings(rankings, relevance, pool, pop, enrolments)
        summary.update({"split": split_label, "variant": key})
        results.append(summary)
        user_scores[key] = user_df.set_index("enrolment_id")["ndcg"]

    for variant, key in [("v3", "V3"), ("v4", "V4")]:
        model_path = MODELS_DIR / f"lambdamart_{model_prefix}_{variant}.txt"
        feats_path = DATA_PROCESSED / f"features_{feature_prefix}_v4.parquet"
        rankings = rank_from_model(model_path, feats_path, key)
        summary, user_df = evaluate_rankings(rankings, relevance, pool, pop, enrolments)
        summary.update({"split": split_label, "variant": key})
        results.append(summary)
        user_scores[key] = user_df.set_index("enrolment_id")["ndcg"]
        if save_demo_rankings and variant == "v4":
            rankings.to_parquet(DATA_PROCESSED / "rankings_demo_v4.parquet", index=False)

    tests = {}
    for other in ("V0", "V1", "V2", "V3"):
        aligned = pd.concat([user_scores["V4"], user_scores[other]], axis=1, join="inner")
        aligned.columns = ["V4", other]
        tests[f"V4_vs_{other}"] = wilcoxon_safe(aligned["V4"], aligned[other])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(METRICS_DIR / f"metrics_{split_label}.csv", index=False)
    md_lines = [
        "| " + " | ".join(res_df.columns) + " |",
        "| " + " | ".join(["---"] * len(res_df.columns)) + " |",
    ]
    for _, row in res_df.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in res_df.columns) + " |")
    (METRICS_DIR / f"metrics_{split_label}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    with open(METRICS_DIR / f"wilcoxon_{split_label}.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)
    (METRICS_DIR / f"metrics_{split_label}.tex").write_text(
        res_df.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )
    print(res_df.to_string(index=False))
    print(json.dumps(tests, indent=2))


def main() -> None:
    run_eval_bundle(
        split_label="split_b",
        observed_path=DATA_PROCESSED / "split_b_test_observed.parquet",
        heldout_path=DATA_PROCESSED / "split_b_test_heldout.parquet",
        baselines_path=DATA_PROCESSED / "baselines_split_b_test.parquet",
        model_prefix="split_b",
        feature_prefix="split_b_test",
        save_demo_rankings=False,
    )
    # Split A powers the demo rankings.
    run_eval_bundle(
        split_label="split_a",
        observed_path=DATA_PROCESSED / "split_a_test_observed.parquet",
        heldout_path=DATA_PROCESSED / "split_a_test_heldout.parquet",
        baselines_path=DATA_PROCESSED / "baselines_split_a_test.parquet",
        model_prefix="split_a",
        feature_prefix="split_a_test",
        save_demo_rankings=True,
    )


if __name__ == "__main__":
    main()
