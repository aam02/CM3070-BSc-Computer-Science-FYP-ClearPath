"""TreeSHAP cache, plain-language reasons, faithfulness and stability checks."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from clearpath.config import DATA_PROCESSED, MODELS_DIR, RANDOM_SEED, SHAP_DIR
from clearpath.ranker import RANK_FEATURES, train_ranker


PLAIN = {
    "cf_score": "similarity to activities you already used",
    "cf_score_user_norm": "relative collaborative match among candidates",
    "global_pop": "how widely other learners used this activity",
    "log_global_pop": "how widely other learners used this activity",
    "week_position": "when this activity usually appears in the module",
    "abs_week_distance": "how close this activity is to your current week",
    "engagement_band": "your observed engagement level so far",
    "type_match": "match to the activity types you use most",
    "item_duration_proxy": "typical click depth on this activity (from observed data)",
}


def plain_language(feature: str, value: float) -> str:
    if feature.startswith("activity_"):
        act = feature.replace("activity_", "")
        verb = "favoured" if value >= 0 else "penalised"
        return f"activity type '{act}' {verb} this recommendation"
    if feature.startswith("module_"):
        mod = feature.replace("module_", "")
        verb = "favoured" if value >= 0 else "penalised"
        return f"module {mod} {verb} this recommendation"
    name = PLAIN.get(feature, feature.replace("_", " "))
    verb = "increased" if value >= 0 else "decreased"
    return f"{name} {verb} the ranking score"


def top3_statements(feature_names: list[str], shap_values: np.ndarray) -> list[str]:
    order = np.argsort(-np.abs(shap_values))[:3]
    return [plain_language(feature_names[i], float(shap_values[i])) for i in order]


def compute_shap_cache(
    model_path: Path,
    features_path: Path,
    cache_name: str,
    items: pd.DataFrame | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(model_path))
    feat_cols = joblib.load(model_path.with_name(model_path.stem + "_features.joblib"))
    df = pd.read_parquet(features_path)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_SEED)

    activity_map = {}
    if items is not None:
        activity_map = items.drop_duplicates("id_site").set_index("id_site")["activity_type"].to_dict()
    if "activity_type" not in df.columns:
        df["activity_type"] = df["id_site"].map(activity_map).fillna("activity")

    x = df[feat_cols]
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x)
    expected = float(np.array(explainer.expected_value).reshape(-1)[0])
    scores = booster.predict(x)

    records = []
    for i in range(len(df)):
        sv = np.asarray(shap_values[i]).ravel()
        act = str(df.iloc[i].get("activity_type", "activity"))
        records.append(
            {
                "enrolment_id": df.iloc[i]["enrolment_id"],
                "id_site": int(df.iloc[i]["id_site"]),
                "activity_type": act,
                "label": f"{act} (site {int(df.iloc[i]['id_site'])})",
                "score": float(scores[i]),
                "base_value": expected,
                "additivity_error": abs(float(sv.sum() + expected - scores[i])),
                "top3": json.dumps(top3_statements(feat_cols, sv)),
                **{f"shap_{c}": float(sv[j]) for j, c in enumerate(feat_cols)},
            }
        )
    out = pd.DataFrame.from_records(records)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    path = SHAP_DIR / f"{cache_name}.parquet"
    out.to_parquet(path, index=False)
    meta = {
        "model": str(model_path),
        "features": feat_cols,
        "expected_value": expected,
        "n_rows": len(out),
        "max_additivity_error": float(out["additivity_error"].max()),
    }
    with open(SHAP_DIR / f"{cache_name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote SHAP cache {path} (max additivity error={meta['max_additivity_error']:.2e})")
    return out


def faithfulness_experiment(
    model_path: Path,
    features_path: Path,
    ks: list[int] | None = None,
    n_samples: int = 500,
) -> dict:
    """Delete top SHAP features (mean-masked) vs random features; compare score drop."""
    ks = ks or [1, 2, 3, 5, 8]
    booster = lgb.Booster(model_file=str(model_path))
    feat_cols = joblib.load(model_path.with_name(model_path.stem + "_features.joblib"))
    df = pd.read_parquet(features_path)
    df = df.sample(min(n_samples, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
    means = df[feat_cols].mean().to_numpy(dtype=float)
    x = df[feat_cols].to_numpy(dtype=float)
    explainer = shap.TreeExplainer(booster)
    sv = np.asarray(explainer.shap_values(df[feat_cols]))
    base_scores = booster.predict(df[feat_cols])

    rng = np.random.default_rng(RANDOM_SEED)
    top_drops, rand_drops = [], []
    for k in ks:
        top_score, rand_score = [], []
        for i in range(len(df)):
            order = np.argsort(-sv[i])
            x_top = x[i].copy()
            x_top[order[:k]] = means[order[:k]]
            x_rand = x[i].copy()
            rand_idx = rng.choice(len(feat_cols), size=k, replace=False)
            x_rand[rand_idx] = means[rand_idx]
            top_score.append(booster.predict(x_top.reshape(1, -1))[0])
            rand_score.append(booster.predict(x_rand.reshape(1, -1))[0])
        top_drops.append(float(np.mean(base_scores - np.asarray(top_score))))
        rand_drops.append(float(np.mean(base_scores - np.asarray(rand_score))))

    gap = np.asarray(top_drops) - np.asarray(rand_drops)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc_gap = float(trapz(gap, ks))
    result = {
        "ks": ks,
        "top_k_drop": top_drops,
        "random_k_drop": rand_drops,
        "auc_gap": auc_gap,
        "masking": "train_mean",
        "ordering": "signed_shap_descending",
    }
    with open(SHAP_DIR / "faithfulness.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Faithfulness AUC gap={auc_gap:.4f} (mean-mask, signed order)")
    return result


def stability_experiment(
    features_train_path: Path,
    features_eval_path: Path,
    seeds: list[int] | None = None,
    n_eval: int = 300,
) -> dict:
    seeds = seeds or [0, 1, 2, 3, 4]
    train_df = pd.read_parquet(features_train_path)
    eval_df = pd.read_parquet(features_eval_path).sample(
        min(n_eval, len(pd.read_parquet(features_eval_path))),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)
    feat_cols = [c for c in RANK_FEATURES if c in train_df.columns]
    attr_orders = []
    for seed in seeds:
        users = train_df["enrolment_id"].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(users)
        n_val = max(1, int(0.15 * len(users)))
        val_users = set(users[:n_val])
        tr = train_df.loc[~train_df["enrolment_id"].isin(val_users)]
        va = train_df.loc[train_df["enrolment_id"].isin(val_users)]
        booster = train_ranker(tr, va, feat_cols, seed=seed)
        explainer = shap.TreeExplainer(booster)
        sv = np.asarray(explainer.shap_values(eval_df[feat_cols]))
        mean_abs = np.abs(sv).mean(axis=0)
        attr_orders.append(np.argsort(-mean_abs))

    n = len(feat_cols)
    corr = np.zeros((len(seeds), len(seeds)))
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            rank_i = np.empty(n)
            rank_j = np.empty(n)
            rank_i[attr_orders[i]] = np.arange(n)
            rank_j[attr_orders[j]] = np.arange(n)
            corr[i, j] = spearmanr(rank_i, rank_j).correlation

    result = {
        "seeds": seeds,
        "mean_off_diagonal_spearman": float(
            (corr.sum() - np.trace(corr)) / (corr.size - len(seeds))
        ),
        "matrix": corr.tolist(),
        "features": feat_cols,
    }
    with open(SHAP_DIR / "stability.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Stability mean Spearman={result['mean_off_diagonal_spearman']:.4f}")
    return result


def main() -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(DATA_PROCESSED / "items.parquet")
    # Demo uses Split A (cohort transfer) model.
    model_path = MODELS_DIR / "lambdamart_split_a_v4.txt"
    features_path = DATA_PROCESSED / "features_split_a_test_v4.parquet"
    rankings_path = DATA_PROCESSED / "rankings_demo_v4.parquet"
    feats = pd.read_parquet(features_path)
    if rankings_path.exists():
        rankings = pd.read_parquet(rankings_path)
        keys = rankings[["enrolment_id", "id_site"]].drop_duplicates()
        ranked_feats = feats.merge(keys, on=["enrolment_id", "id_site"], how="inner")
        tmp = DATA_PROCESSED / "_shap_input_demo.parquet"
        ranked_feats.to_parquet(tmp, index=False)
        compute_shap_cache(model_path, tmp, cache_name="shap_demo_v4", items=items, max_rows=None)
        tmp.unlink(missing_ok=True)
    else:
        compute_shap_cache(model_path, features_path, cache_name="shap_demo_v4", items=items, max_rows=8000)

    faithfulness_experiment(model_path, features_path)
    stability_experiment(
        DATA_PROCESSED / "features_split_a_train_v4.parquet",
        features_path,
    )


if __name__ == "__main__":
    main()
