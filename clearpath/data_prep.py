"""Load and filter OULAD; write processed artefacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from clearpath.config import (
    DATA_PROCESSED,
    DATA_RAW,
    MIN_ITEMS_PER_STUDENT,
    MIN_STUDENTS_PER_ITEM,
    MODULES,
    PRESENTATIONS,
)


def _enrolment_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["id_student"].astype(str)
        + "|"
        + df["code_module"].astype(str)
        + "|"
        + df["code_presentation"].astype(str)
    )


def _funnel_row(step: str, sv: pd.DataFrame) -> dict:
    return {
        "step": step,
        "rows": int(len(sv)),
        "n_students": int(sv["id_student"].nunique()) if len(sv) else 0,
        "n_enrolments": int(sv["enrolment_id"].nunique()) if "enrolment_id" in sv.columns and len(sv) else 0,
        "n_items": int(sv["id_site"].nunique()) if len(sv) else 0,
    }


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "student_vle": pd.read_csv(DATA_RAW / "studentVle.csv"),
        "vle": pd.read_csv(DATA_RAW / "vle.csv"),
        "student_info": pd.read_csv(DATA_RAW / "studentInfo.csv"),
        "student_registration": pd.read_csv(DATA_RAW / "studentRegistration.csv"),
        "courses": pd.read_csv(DATA_RAW / "courses.csv"),
    }


def prepare(raw: dict[str, pd.DataFrame] | None = None) -> dict:
    raw = raw or load_raw()
    funnel: list[dict] = []

    sv = raw["student_vle"].copy()
    funnel.append(
        {
            "step": "raw_studentVle",
            "rows": int(len(sv)),
            "n_students": int(sv["id_student"].nunique()),
            "n_enrolments": 0,
            "n_items": int(sv["id_site"].nunique()),
        }
    )

    mask = sv["code_module"].isin(MODULES) & sv["code_presentation"].isin(PRESENTATIONS)
    sv = sv.loc[mask].copy()
    sv["enrolment_id"] = _enrolment_key(sv)
    funnel.append(_funnel_row("subset_modules_presentations", sv))

    courses = raw["courses"].copy()
    courses = courses.loc[
        courses["code_module"].isin(MODULES) & courses["code_presentation"].isin(PRESENTATIONS)
    ]
    sv = sv.merge(courses, on=["code_module", "code_presentation"], how="inner")
    in_window = (sv["date"] >= 0) & (sv["date"] <= sv["module_presentation_length"])
    sv = sv.loc[in_window].copy()
    funnel.append(_funnel_row("drop_out_of_window", sv))

    reg = raw["student_registration"].copy()
    reg = reg.loc[
        reg["code_module"].isin(MODULES) & reg["code_presentation"].isin(PRESENTATIONS)
    ].copy()
    reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
    keep_reg = reg["date_unregistration"].isna() | (reg["date_unregistration"] >= 0)
    reg = reg.loc[keep_reg, ["id_student", "code_module", "code_presentation"]]
    sv = sv.merge(reg, on=["id_student", "code_module", "code_presentation"], how="inner")
    funnel.append(_funnel_row("drop_early_unregister", sv))

    item_counts = sv.groupby("enrolment_id")["id_site"].nunique()
    keep_users = item_counts[item_counts >= MIN_ITEMS_PER_STUDENT].index
    sv = sv.loc[sv["enrolment_id"].isin(keep_users)].copy()
    funnel.append(_funnel_row(f"min_{MIN_ITEMS_PER_STUDENT}_items_per_enrolment", sv))

    user_counts = sv.groupby("id_site")["enrolment_id"].nunique()
    keep_items = user_counts[user_counts >= MIN_STUDENTS_PER_ITEM].index
    sv = sv.loc[sv["id_site"].isin(keep_items)].copy()
    item_counts = sv.groupby("enrolment_id")["id_site"].nunique()
    keep_users = item_counts[item_counts >= MIN_ITEMS_PER_STUDENT].index
    sv = sv.loc[sv["enrolment_id"].isin(keep_users)].copy()
    funnel.append(_funnel_row(f"min_{MIN_STUDENTS_PER_ITEM}_enrolments_per_item", sv))

    # Item metadata only; click proxies are built later from observed events.
    vle = raw["vle"].copy()
    vle = vle.loc[vle["id_site"].isin(sv["id_site"].unique())].copy()
    vle = vle.merge(courses, on=["code_module", "code_presentation"], how="left")
    week_from = pd.to_numeric(vle["week_from"], errors="coerce")
    length_weeks = (vle["module_presentation_length"] / 7.0).clip(lower=1.0)
    vle["week_position"] = (week_from.fillna(length_weeks / 2.0) / length_weeks).clip(0, 1)

    info = raw["student_info"].copy()
    info = info.loc[
        info["code_module"].isin(MODULES) & info["code_presentation"].isin(PRESENTATIONS)
    ].copy()
    info["enrolment_id"] = _enrolment_key(info)
    enrolments = (
        sv.groupby("enrolment_id", as_index=False)
        .agg(
            id_student=("id_student", "first"),
            code_module=("code_module", "first"),
            code_presentation=("code_presentation", "first"),
            module_presentation_length=("module_presentation_length", "first"),
            n_events=("sum_click", "size"),
            n_items=("id_site", "nunique"),
        )
    )
    enrolments = enrolments.merge(
        info[["enrolment_id", "final_result"]],
        on="enrolment_id",
        how="left",
    )

    agg = (
        sv.groupby(["enrolment_id", "id_site"], as_index=False)
        .agg(weight=("sum_click", "sum"))
    )
    user_ids = sorted(enrolments["enrolment_id"].unique())
    item_ids = sorted(sv["id_site"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {i: j for j, i in enumerate(item_ids)}
    rows = agg["enrolment_id"].map(user_to_idx).to_numpy()
    cols = agg["id_site"].map(item_to_idx).to_numpy()
    data = agg["weight"].to_numpy(dtype=np.float32)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    density = float(matrix.nnz) / float(matrix.shape[0] * matrix.shape[1])
    funnel.append(
        {
            "step": "final_matrix",
            "rows": int(matrix.nnz),
            "n_students": int(enrolments["id_student"].nunique()),
            "n_enrolments": int(matrix.shape[0]),
            "n_items": int(matrix.shape[1]),
            "density": density,
        }
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    events = sv[
        [
            "enrolment_id",
            "id_student",
            "code_module",
            "code_presentation",
            "id_site",
            "date",
            "sum_click",
            "module_presentation_length",
        ]
    ].copy()
    events.to_parquet(DATA_PROCESSED / "events.parquet", index=False)
    vle.to_parquet(DATA_PROCESSED / "items.parquet", index=False)
    enrolments.to_parquet(DATA_PROCESSED / "enrolments.parquet", index=False)
    pd.DataFrame(funnel).to_csv(DATA_PROCESSED / "funnel_counts.csv", index=False)

    print("Preprocessing complete.")
    print(pd.DataFrame(funnel).to_string(index=False))
    print(f"Density: {density:.6f}")
    return {
        "events": events,
        "items": vle,
        "enrolments": enrolments,
        "matrix": matrix,
        "funnel": funnel,
        "density": density,
    }


def main() -> None:
    import pyarrow  # noqa: F401  # fail fast if missing

    prepare()


if __name__ == "__main__":
    main()
