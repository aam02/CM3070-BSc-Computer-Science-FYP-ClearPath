"""Train/test split indices for ClearPath evaluation protocols."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from clearpath.config import (
    DATA_PROCESSED,
    RANDOM_SEED,
    SPLIT_A_VAL_FRAC,
    SPLIT_B_TRAIN_FRAC,
    SPLIT_B_VAL_FRAC,
    TEMPORAL_HOLDOUT_FRAC,
    TEST_PRESENTATIONS,
    TRAIN_PRESENTATIONS,
)


def _load_events() -> pd.DataFrame:
    path = DATA_PROCESSED / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m clearpath.data_prep first")
    return pd.read_parquet(path)


def _temporal_split_enrolment(group: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = group.sort_values(["date", "id_site"])
    n = len(group)
    if n < 5:
        return group, group.iloc[0:0].copy()
    cut = max(1, int(np.floor(n * (1.0 - frac))))
    cut = min(cut, n - 1)
    return group.iloc[:cut].copy(), group.iloc[cut:].copy()


def _temporal_split_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_parts, hold_parts = [], []
    for _, g in df.groupby("enrolment_id", sort=False):
        obs, hold = _temporal_split_enrolment(g, TEMPORAL_HOLDOUT_FRAC)
        if len(hold) == 0 or len(obs) == 0:
            continue
        obs_parts.append(obs)
        hold_parts.append(hold)
    if not obs_parts:
        empty = df.iloc[0:0].copy()
        return empty, empty
    return pd.concat(obs_parts, ignore_index=True), pd.concat(hold_parts, ignore_index=True)


def build_split_b(events: pd.DataFrame) -> dict:
    """Temporal holdout with disjoint train/val/test enrolments."""
    enrolments = np.array(sorted(events["enrolment_id"].unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(enrolments)
    n = len(enrolments)
    n_train = int(n * SPLIT_B_TRAIN_FRAC)
    n_val = int(n * SPLIT_B_VAL_FRAC)
    train_ids = set(enrolments[:n_train])
    val_ids = set(enrolments[n_train : n_train + n_val])
    test_ids = set(enrolments[n_train + n_val :])

    def _slice(ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return _temporal_split_all(events.loc[events["enrolment_id"].isin(ids)])

    train_obs, train_hold = _slice(train_ids)
    val_obs, val_hold = _slice(val_ids)
    test_obs, test_hold = _slice(test_ids)

    return {
        "train_observed": train_obs,
        "train_heldout": train_hold,
        "val_observed": val_obs,
        "val_heldout": val_hold,
        "test_observed": test_obs,
        "test_heldout": test_hold,
        "train_enrolments": sorted(train_ids),
        "val_enrolments": sorted(val_ids),
        "test_enrolments": sorted(test_ids),
    }


def build_split_a(events: pd.DataFrame) -> dict:
    """Cohort holdout: train 2013, test 2014; val is a slice of 2013."""
    train_events = events.loc[events["code_presentation"].isin(TRAIN_PRESENTATIONS)].copy()
    test_events = events.loc[events["code_presentation"].isin(TEST_PRESENTATIONS)].copy()

    train_obs, train_hold = _temporal_split_all(train_events)
    test_obs, test_hold = _temporal_split_all(test_events)

    train_users = np.array(sorted(train_obs["enrolment_id"].unique()))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(train_users)
    n_val = max(1, int(len(train_users) * SPLIT_A_VAL_FRAC))
    val_enrolments = sorted(train_users[:n_val].tolist())

    return {
        "train_observed": train_obs,
        "train_heldout": train_hold,
        "test_observed": test_obs,
        "test_heldout": test_hold,
        "val_enrolments": val_enrolments,
    }


def assert_no_leakage(observed: pd.DataFrame, heldout: pd.DataFrame, label: str) -> None:
    if len(heldout) == 0:
        return
    max_obs = observed.groupby("enrolment_id")["date"].max()
    merged = heldout[["enrolment_id", "date"]].copy()
    merged["max_obs"] = merged["enrolment_id"].map(max_obs)
    if merged["max_obs"].isna().any():
        raise AssertionError(f"{label}: heldout enrolments missing from observed")
    bad = merged["date"] < merged["max_obs"]
    if bad.any():
        raise AssertionError(
            f"{label}: {int(bad.sum())} heldout events earlier than max observed date"
        )


def save_splits() -> None:
    events = _load_events()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    split_b = build_split_b(events)
    for key in ("train", "val", "test"):
        assert_no_leakage(split_b[f"{key}_observed"], split_b[f"{key}_heldout"], f"split_b_{key}")
        split_b[f"{key}_observed"].to_parquet(
            DATA_PROCESSED / f"split_b_{key}_observed.parquet", index=False
        )
        split_b[f"{key}_heldout"].to_parquet(
            DATA_PROCESSED / f"split_b_{key}_heldout.parquet", index=False
        )
    with open(DATA_PROCESSED / "split_b_partitions.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_enrolments": split_b["train_enrolments"],
                "val_enrolments": split_b["val_enrolments"],
                "test_enrolments": split_b["test_enrolments"],
            },
            f,
        )

    split_a = build_split_a(events)
    assert_no_leakage(split_a["train_observed"], split_a["train_heldout"], "split_a_train")
    assert_no_leakage(split_a["test_observed"], split_a["test_heldout"], "split_a_test")
    split_a["train_observed"].to_parquet(DATA_PROCESSED / "split_a_train_observed.parquet", index=False)
    split_a["train_heldout"].to_parquet(DATA_PROCESSED / "split_a_train_heldout.parquet", index=False)
    split_a["test_observed"].to_parquet(DATA_PROCESSED / "split_a_test_observed.parquet", index=False)
    split_a["test_heldout"].to_parquet(DATA_PROCESSED / "split_a_test_heldout.parquet", index=False)
    with open(DATA_PROCESSED / "split_a_val_enrolments.json", "w", encoding="utf-8") as f:
        json.dump(split_a["val_enrolments"], f)

    tr, va, te = (
        set(split_b["train_enrolments"]),
        set(split_b["val_enrolments"]),
        set(split_b["test_enrolments"]),
    )
    assert not (tr & va) and not (tr & te) and not (va & te)

    summary = {
        "split_b": {
            "n_train_enrolments": len(tr),
            "n_val_enrolments": len(va),
            "n_test_enrolments": len(te),
            "n_test_observed_events": int(len(split_b["test_observed"])),
            "n_test_heldout_events": int(len(split_b["test_heldout"])),
            "note": "Evaluation uses test enrolments only; train/val are disjoint.",
        },
        "split_a": {
            "n_train_enrolments": int(split_a["train_observed"]["enrolment_id"].nunique()),
            "n_test_enrolments": int(split_a["test_observed"]["enrolment_id"].nunique()),
            "n_val_enrolments": len(split_a["val_enrolments"]),
            "n_test_heldout_events": int(len(split_a["test_heldout"])),
        },
    }
    with open(DATA_PROCESSED / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    save_splits()


if __name__ == "__main__":
    main()
