"""Feature matrix for LambdaMART. Engagement and duration use observed events only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from clearpath.config import DATA_PROCESSED, TOP_ACTIVITY_TYPES


FEATURE_COLUMNS = [
    "cf_score",
    "global_pop",
    "log_global_pop",
    "week_position",
    "abs_week_distance",
    "engagement_band",
    "type_match",
    "item_duration_proxy",
    "activity_resource",
    "activity_oucontent",
    "activity_forumng",
    "activity_quiz",
    "activity_url",
    "activity_subpage",
    "activity_glossary",
    "activity_oucollaborate",
    "activity_other",
    "module_BBB",
    "module_DDD",
    "module_FFF",
]


def _dominant_activity(observed: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    merged = observed.merge(items[["id_site", "activity_type"]], on="id_site", how="left")
    counts = merged.groupby(["enrolment_id", "activity_type"])["sum_click"].sum().reset_index()
    idx = counts.groupby("enrolment_id")["sum_click"].idxmax()
    return counts.loc[idx].set_index("enrolment_id")["activity_type"]


def _engagement_band_from_observed(observed: pd.DataFrame) -> pd.Series:
    totals = observed.groupby("enrolment_id")["sum_click"].sum()
    return pd.qcut(totals.rank(method="first"), q=4, labels=[0, 1, 2, 3]).astype(int)


def _item_duration_from_observed(observed: pd.DataFrame) -> pd.Series:
    return observed.groupby("id_site")["sum_click"].median()


def _student_week_position(observed: pd.DataFrame) -> pd.Series:
    g = observed.groupby("enrolment_id").agg(
        max_date=("date", "max"),
        length=("module_presentation_length", "first"),
    )
    length = g["length"].replace(0, np.nan).fillna(270.0)
    return (g["max_date"] / length).clip(0, 1)


def build_feature_table(
    candidates: pd.DataFrame,
    observed_events: pd.DataFrame,
    enrolments: pd.DataFrame,
    items: pd.DataFrame,
    popularity: pd.DataFrame,
    heldout_events: pd.DataFrame | None = None,
    include_cf: bool = True,
) -> pd.DataFrame:
    item_cols = ["id_site", "activity_type", "week_position", "code_module", "code_presentation"]
    items_small = items[item_cols].drop_duplicates("id_site")

    obs = observed_events.copy()
    if "module_presentation_length" not in obs.columns:
        length_map = enrolments.set_index("enrolment_id")["module_presentation_length"]
        obs["module_presentation_length"] = obs["enrolment_id"].map(length_map)

    engagement = _engagement_band_from_observed(obs)
    duration = _item_duration_from_observed(obs)
    student_week = _student_week_position(obs)
    dominant = _dominant_activity(obs, items_small)

    feats = candidates.merge(items_small, on="id_site", how="inner", suffixes=("", "_item"))
    feats = feats.merge(
        enrolments[["enrolment_id", "code_module"]],
        on="enrolment_id",
        how="inner",
        suffixes=("", "_enr"),
    )
    if "code_module_enr" in feats.columns:
        feats["code_module"] = feats["code_module_enr"]

    feats = feats.merge(popularity, on="id_site", how="left")
    feats["global_pop"] = feats["global_pop"].fillna(0.0)
    feats["log_global_pop"] = np.log1p(feats["global_pop"])

    feats["engagement_band"] = feats["enrolment_id"].map(engagement).fillna(0).astype(int)
    feats["item_duration_proxy"] = feats["id_site"].map(duration).fillna(1.0).astype(float)
    feats["dominant_type"] = feats["enrolment_id"].map(dominant).fillna("other")
    feats["student_week"] = feats["enrolment_id"].map(student_week).fillna(0.5)
    feats["week_position"] = feats["week_position"].fillna(0.5).astype(float)
    feats["abs_week_distance"] = (feats["week_position"] - feats["student_week"]).abs()
    feats["type_match"] = (feats["activity_type"] == feats["dominant_type"]).astype(int)

    if not include_cf:
        feats["cf_score"] = 0.0
    feats["cf_score"] = feats["cf_score"].fillna(0.0).astype(float)

    act = feats["activity_type"].fillna("other").astype(str)
    for t in TOP_ACTIVITY_TYPES:
        feats[f"activity_{t}"] = (act == t).astype(int)
    feats["activity_other"] = (~act.isin(TOP_ACTIVITY_TYPES)).astype(int)
    for mod in ("BBB", "DDD", "FFF"):
        feats[f"module_{mod}"] = (feats["code_module"] == mod).astype(int)

    if heldout_events is not None and len(heldout_events):
        seen = obs.groupby("enrolment_id")["id_site"].apply(lambda s: set(int(x) for x in s)).to_dict()
        hold_items = (
            heldout_events.groupby("enrolment_id")["id_site"]
            .apply(lambda s: set(int(x) for x in s))
            .to_dict()
        )
        feats["label"] = [
            1 if int(item) in (hold_items.get(eid, set()) - seen.get(eid, set())) else 0
            for eid, item in zip(feats["enrolment_id"], feats["id_site"])
        ]
    else:
        feats["label"] = 0

    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi <= lo:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - lo) / (hi - lo)

    feats["cf_score_user_norm"] = (
        feats.groupby("enrolment_id")["cf_score"].transform(_norm) if include_cf else 0.0
    )

    keep = ["enrolment_id", "id_site", "label", "activity_type"] + FEATURE_COLUMNS + ["cf_score_user_norm"]
    return feats[[c for c in keep if c in feats.columns]].copy()


def run_split(tag: str, observed_name: str, heldout_name: str, candidates_name: str, pop_name: str) -> None:
    enrolments = pd.read_parquet(DATA_PROCESSED / "enrolments.parquet")
    items = pd.read_parquet(DATA_PROCESSED / "items.parquet")
    observed = pd.read_parquet(DATA_PROCESSED / observed_name)
    heldout = pd.read_parquet(DATA_PROCESSED / heldout_name)
    candidates = pd.read_parquet(DATA_PROCESSED / candidates_name)
    popularity = pd.read_parquet(DATA_PROCESSED / pop_name)

    print(f"[{tag}] building features for {candidates['enrolment_id'].nunique()} enrolments...")
    feats_full = build_feature_table(
        candidates, observed, enrolments, items, popularity, heldout, include_cf=True
    )
    feats_full.to_parquet(DATA_PROCESSED / f"features_{tag}_v4.parquet", index=False)
    print(
        f"[{tag}] rows={len(feats_full)} positives={int(feats_full['label'].sum())} "
        f"-> features_{tag}_v4.parquet"
    )


def main() -> None:
    run_split(
        "split_b_train",
        "split_b_train_observed.parquet",
        "split_b_train_heldout.parquet",
        "candidates_split_b_train.parquet",
        "item_popularity_split_b_train.parquet",
    )
    run_split(
        "split_b_val",
        "split_b_val_observed.parquet",
        "split_b_val_heldout.parquet",
        "candidates_split_b_val.parquet",
        "item_popularity_split_b_train.parquet",  # train pop only
    )
    run_split(
        "split_b_test",
        "split_b_test_observed.parquet",
        "split_b_test_heldout.parquet",
        "candidates_split_b_test.parquet",
        "item_popularity_split_b_train.parquet",
    )
    run_split(
        "split_a_train",
        "split_a_train_observed.parquet",
        "split_a_train_heldout.parquet",
        "candidates_split_a_train.parquet",
        "item_popularity_split_a_train.parquet",
    )
    # Split A test items are 2014-only; use 2014 observed popularity (no held-out leakage).
    run_split(
        "split_a_test",
        "split_a_test_observed.parquet",
        "split_a_test_heldout.parquet",
        "candidates_split_a_test.parquet",
        "item_popularity_split_a_test.parquet",
    )
    with open(DATA_PROCESSED / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS + ["cf_score_user_norm"], f, indent=2)


if __name__ == "__main__":
    main()
