"""LightGBM LambdaMART training."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from clearpath.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    RANDOM_SEED,
    RANKER_PARAM_GRID,
    RANKER_SEEDS,
)
from clearpath.features import FEATURE_COLUMNS

RANK_FEATURES = FEATURE_COLUMNS + ["cf_score_user_norm"]
LGB_DROP = {"activity_type"}


def _feature_cols(df: pd.DataFrame, variant: str) -> list[str]:
    cols = [c for c in RANK_FEATURES if c in df.columns and c not in LGB_DROP]
    if variant == "v3":
        cols = [c for c in cols if c not in ("cf_score", "cf_score_user_norm")]
    return cols


def _prepare_lgb_data(df: pd.DataFrame, feature_cols: list[str]) -> lgb.Dataset:
    df = df.sort_values("enrolment_id").reset_index(drop=True)
    group_sizes = df.groupby("enrolment_id", sort=False).size().to_numpy()
    return lgb.Dataset(
        df[feature_cols],
        label=df["label"].to_numpy(),
        group=group_sizes,
        feature_name=feature_cols,
        free_raw_data=False,
    )


def train_ranker(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    feature_cols: list[str],
    seed: int = RANDOM_SEED,
    params_extra: dict | None = None,
    num_boost_round: int = 300,
) -> lgb.Booster:
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambdarank_truncation_level": 30,
        "verbosity": -1,
        "seed": seed,
    }
    if params_extra:
        params.update(params_extra)
    dtrain = _prepare_lgb_data(train_df, feature_cols)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=0)]
    if valid_df is not None and len(valid_df) > 0:
        dvalid = _prepare_lgb_data(valid_df, feature_cols)
        valid_sets.append(dvalid)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=40, verbose=False))
    return lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def predict_ranking(
    booster: lgb.Booster, df: pd.DataFrame, feature_cols: list[str], top_n: int = 10
) -> pd.DataFrame:
    scores = booster.predict(df[feature_cols])
    out = df[["enrolment_id", "id_site"]].copy()
    if "activity_type" in df.columns:
        out["activity_type"] = df["activity_type"].values
    out["score"] = scores
    out = out.sort_values(["enrolment_id", "score"], ascending=[True, False])
    out["rank"] = out.groupby("enrolment_id").cumcount() + 1
    return out.loc[out["rank"] <= top_n].copy()


def _val_ndcg(booster: lgb.Booster, valid_df: pd.DataFrame, feature_cols: list[str]) -> float:
    def _ndcg(ranked: list[int], relevant: set[int], k: int = 10) -> float:
        if not relevant:
            return 0.0
        dcg = sum(1.0 / np.log2(i + 1) for i, item in enumerate(ranked[:k], 1) if item in relevant)
        ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
        return float(dcg / ideal) if ideal else 0.0

    ranked = predict_ranking(booster, valid_df, feature_cols, top_n=10)
    labels = (
        valid_df.loc[valid_df["label"] == 1]
        .groupby("enrolment_id")["id_site"]
        .apply(lambda s: set(int(x) for x in s))
        .to_dict()
    )
    scores = []
    for eid, g in ranked.groupby("enrolment_id"):
        scores.append(
            _ndcg(g.sort_values("rank")["id_site"].astype(int).tolist(), labels.get(eid, set()), 10)
        )
    return float(np.mean(scores)) if scores else 0.0


def tune_and_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    variant: str,
    split_name: str,
) -> Path:
    feature_cols = _feature_cols(train_df, variant)
    best_score = -1.0
    best_params: dict = dict(RANKER_PARAM_GRID[0])
    print(f"[{split_name}/{variant}] tuning over {len(RANKER_PARAM_GRID)} configs...")
    for cfg in RANKER_PARAM_GRID:
        booster = train_ranker(train_df, valid_df, feature_cols, seed=RANDOM_SEED, params_extra=cfg)
        score = _val_ndcg(booster, valid_df, feature_cols)
        print(f"  {cfg} -> val NDCG@10={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = dict(cfg)

    best_booster = None
    best_seed = RANKER_SEEDS[0]
    seed_scores = {}
    for seed in RANKER_SEEDS:
        booster = train_ranker(
            train_df, valid_df, feature_cols, seed=seed, params_extra=best_params
        )
        score = _val_ndcg(booster, valid_df, feature_cols)
        seed_scores[str(seed)] = score
        if best_booster is None or score > best_score:
            best_score = score
            best_booster = booster
            best_seed = seed

    assert best_booster is not None
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"lambdamart_{split_name}_{variant}.txt"
    best_booster.save_model(str(path))
    meta = {
        "variant": variant,
        "split": split_name,
        "features": feature_cols,
        "best_params": best_params,
        "best_seed": best_seed,
        "val_ndcg_by_seed": seed_scores,
        "best_val_ndcg": best_score,
        "best_iteration": best_booster.best_iteration,
    }
    with open(MODELS_DIR / f"lambdamart_{split_name}_{variant}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(feature_cols, MODELS_DIR / f"lambdamart_{split_name}_{variant}_features.joblib")
    print(f"Saved {path} (seed={best_seed}, val NDCG@10={best_score:.4f})")
    return path


def train_split_b(variant: str = "v4") -> Path:
    train_df = pd.read_parquet(DATA_PROCESSED / "features_split_b_train_v4.parquet")
    valid_df = pd.read_parquet(DATA_PROCESSED / "features_split_b_val_v4.parquet")
    return tune_and_train(train_df, valid_df, variant, "split_b")


def train_split_a(variant: str = "v4") -> Path:
    feats = pd.read_parquet(DATA_PROCESSED / "features_split_a_train_v4.parquet")
    with open(DATA_PROCESSED / "split_a_val_enrolments.json", encoding="utf-8") as f:
        val_users = set(json.load(f))
    train_df = feats.loc[~feats["enrolment_id"].isin(val_users)]
    valid_df = feats.loc[feats["enrolment_id"].isin(val_users)]
    return tune_and_train(train_df, valid_df, variant, "split_a")


def main() -> None:
    for variant in ("v3", "v4"):
        train_split_b(variant)
        train_split_a(variant)


if __name__ == "__main__":
    main()
