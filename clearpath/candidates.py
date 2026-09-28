"""ItemKNN candidates and V0/V1/V2 baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from clearpath.config import CANDIDATE_K, DATA_PROCESSED, TOP_N


def aggregate_weights(events: pd.DataFrame) -> pd.DataFrame:
    return events.groupby(
        ["enrolment_id", "id_site", "code_module", "code_presentation"], as_index=False
    ).agg(weight=("sum_click", "sum"))


def build_user_item_matrix(
    agg: pd.DataFrame,
) -> tuple[sparse.csr_matrix, dict[str, int], dict[int, int], list[str], list[int]]:
    user_ids = sorted(agg["enrolment_id"].unique())
    item_ids = sorted(int(x) for x in agg["id_site"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {i: j for j, i in enumerate(item_ids)}
    rows = agg["enrolment_id"].map(user_to_idx).to_numpy()
    cols = agg["id_site"].map(item_to_idx).to_numpy()
    data = np.log1p(agg["weight"].to_numpy(dtype=np.float64))
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return mat, user_to_idx, item_to_idx, user_ids, item_ids


def item_cosine_similarity(matrix: sparse.csr_matrix, top_k: int = 100) -> sparse.csr_matrix:
    item_vectors = normalize(matrix.T.tocsr(), norm="l2", axis=1)
    n_items = item_vectors.shape[0]
    block = 256
    rows_i, cols_j, vals = [], [], []
    for start in range(0, n_items, block):
        end = min(start + block, n_items)
        sims = (item_vectors[start:end] @ item_vectors.T).toarray()
        for local_i, global_i in enumerate(range(start, end)):
            row = sims[local_i]
            row[global_i] = -np.inf
            take = min(top_k, n_items - 1)
            nn = np.argpartition(-row, take - 1)[:take]
            for j in nn:
                s = float(row[j])
                if s > 0:
                    rows_i.append(global_i)
                    cols_j.append(int(j))
                    vals.append(s)
    return sparse.csr_matrix((vals, (rows_i, cols_j)), shape=(n_items, n_items))


def generate_candidates_for_users(
    observed_agg: pd.DataFrame,
    enrolments: pd.DataFrame,
    items: pd.DataFrame,
    sim: sparse.csr_matrix,
    item_ids: list[int],
    k: int = CANDIDATE_K,
) -> pd.DataFrame:
    item_index = {int(i): j for j, i in enumerate(item_ids)}
    idx_to_item = np.asarray(item_ids, dtype=np.int64)
    pool_idx: dict[tuple[str, str], np.ndarray] = {}
    for (mod, pres), g in items.groupby(["code_module", "code_presentation"]):
        idxs = [item_index[int(x)] for x in g["id_site"] if int(x) in item_index]
        pool_idx[(mod, pres)] = np.asarray(idxs, dtype=np.int64)

    enr_meta = enrolments.set_index("enrolment_id")[["code_module", "code_presentation"]]
    seen_map = observed_agg.groupby("enrolment_id")["id_site"].apply(lambda s: set(int(x) for x in s))

    user_ids = sorted(observed_agg["enrolment_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    rows, cols, data = [], [], []
    for _, row in observed_agg.iterrows():
        item = int(row["id_site"])
        if item not in item_index:
            continue
        rows.append(user_to_idx[row["enrolment_id"]])
        cols.append(item_index[item])
        data.append(np.log1p(float(row["weight"])))
    user_mat = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(item_ids)), dtype=np.float64
    )
    scores = user_mat @ sim

    records: list[dict] = []
    for u, ui in user_to_idx.items():
        if u not in enr_meta.index:
            continue
        meta = enr_meta.loc[u]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        mod, pres = meta["code_module"], meta["code_presentation"]
        allowed = pool_idx.get((mod, pres))
        if allowed is None or len(allowed) == 0:
            continue
        seen_u = seen_map.get(u, set())
        row = scores.getrow(ui).toarray().ravel()
        cand_idx, cand_scores = [], []
        for j in allowed:
            item = int(idx_to_item[j])
            if item in seen_u:
                continue
            cand_idx.append(item)
            cand_scores.append(row[j])
        if not cand_scores:
            continue
        arr = np.asarray(cand_scores, dtype=float)
        take = min(k, len(arr))
        top = np.argpartition(-arr, take - 1)[:take]
        top = top[np.argsort(-arr[top])]
        for rank, t in enumerate(top, start=1):
            records.append(
                {
                    "enrolment_id": u,
                    "id_site": int(cand_idx[t]),
                    "cf_score": float(arr[t]),
                    "candidate_rank": rank,
                    "code_module": mod,
                    "code_presentation": pres,
                }
            )
    return pd.DataFrame.from_records(records)


def most_popular_rankings(observed_agg: pd.DataFrame, enrolments: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    pop = observed_agg.groupby(["code_module", "code_presentation", "id_site"], as_index=False)["weight"].sum()
    seen = observed_agg.groupby("enrolment_id")["id_site"].apply(lambda s: set(int(x) for x in s))
    pop_groups = {
        key: g.sort_values("weight", ascending=False)
        for key, g in pop.groupby(["code_module", "code_presentation"])
    }
    records = []
    for _, enr in enrolments.iterrows():
        u = enr["enrolment_id"]
        pool = pop_groups.get((enr["code_module"], enr["code_presentation"]))
        if pool is None:
            continue
        picked = pool[~pool["id_site"].isin(seen.get(u, set()))].head(top_n)
        for rank, (_, row) in enumerate(picked.iterrows(), start=1):
            records.append(
                {
                    "enrolment_id": u,
                    "id_site": int(row["id_site"]),
                    "score": float(row["weight"]),
                    "rank": rank,
                    "variant": "V1_MostPopular",
                }
            )
    return pd.DataFrame.from_records(records)


def week_popularity_rankings(
    candidates: pd.DataFrame,
    observed_events: pd.DataFrame,
    items: pd.DataFrame,
    popularity: pd.DataFrame,
    enrolments: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """V0: prefer items near the student's current week, then by popularity."""
    from clearpath.features import _student_week_position

    if candidates is None or len(candidates) == 0 or "enrolment_id" not in candidates.columns:
        return pd.DataFrame(columns=["enrolment_id", "id_site", "score", "rank", "variant"])

    obs = observed_events.copy()
    if "module_presentation_length" not in obs.columns:
        obs["module_presentation_length"] = obs["enrolment_id"].map(
            enrolments.set_index("enrolment_id")["module_presentation_length"]
        )
    student_week = _student_week_position(obs)
    week_pos = items.drop_duplicates("id_site").set_index("id_site")["week_position"]
    pop = popularity.set_index("id_site")["global_pop"]

    df = candidates.copy()
    df["student_week"] = df["enrolment_id"].map(student_week).fillna(0.5)
    df["week_position"] = df["id_site"].map(week_pos).fillna(0.5)
    df["abs_week_distance"] = (df["week_position"] - df["student_week"]).abs()
    df["log_pop"] = np.log1p(df["id_site"].map(pop).fillna(0.0))
    df["score"] = df["log_pop"] - df["abs_week_distance"]
    df = df.sort_values(["enrolment_id", "score"], ascending=[True, False])
    df["rank"] = df.groupby("enrolment_id").cumcount() + 1
    out = df.loc[df["rank"] <= top_n, ["enrolment_id", "id_site", "score", "rank"]].copy()
    out["variant"] = "V0_WeekPop"
    return out


def itemknn_rankings(candidates: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    out = candidates.sort_values(["enrolment_id", "cf_score"], ascending=[True, False])
    out["rank"] = out.groupby("enrolment_id").cumcount() + 1
    out = out.loc[out["rank"] <= top_n].copy()
    out["score"] = out["cf_score"]
    out["variant"] = "V2_ItemKNN"
    return out[["enrolment_id", "id_site", "score", "rank", "variant"]]


def _fit_sim(train_observed: pd.DataFrame) -> tuple[sparse.csr_matrix, list[int]]:
    agg = aggregate_weights(train_observed)
    mat, _, _, _, item_ids = build_user_item_matrix(agg)
    print(f"  CF train matrix {mat.shape}, nnz={mat.nnz}")
    return item_cosine_similarity(mat, top_k=100), item_ids


def _write_candidates(
    name: str,
    observed: pd.DataFrame,
    enrolments: pd.DataFrame,
    items: pd.DataFrame,
    sim: sparse.csr_matrix,
    item_ids: list[int],
) -> pd.DataFrame:
    users = observed["enrolment_id"].unique()
    enr = enrolments.loc[enrolments["enrolment_id"].isin(users)].copy()
    agg = aggregate_weights(observed)
    candidates = generate_candidates_for_users(agg, enr, items, sim, item_ids)
    path = DATA_PROCESSED / f"candidates_{name}.parquet"
    candidates.to_parquet(path, index=False)
    print(f"[{name}] wrote {len(candidates)} candidates")
    return candidates


def _write_baselines(
    name: str,
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    enrolments: pd.DataFrame,
    items: pd.DataFrame,
    pop: pd.DataFrame,
) -> None:
    users = observed["enrolment_id"].unique()
    enr = enrolments.loc[enrolments["enrolment_id"].isin(users)].copy()
    agg = aggregate_weights(observed)
    v0 = week_popularity_rankings(candidates, observed, items, pop, enrolments)
    v1 = most_popular_rankings(agg, enr)
    v2 = itemknn_rankings(candidates)
    baselines = pd.concat([v0, v1, v2], ignore_index=True)
    path = DATA_PROCESSED / f"baselines_{name}.parquet"
    baselines.to_parquet(path, index=False)
    print(f"[{name}] wrote baselines (V0/V1/V2) -> {path}")


def main() -> None:
    enrolments = pd.read_parquet(DATA_PROCESSED / "enrolments.parquet")
    items = pd.read_parquet(DATA_PROCESSED / "items.parquet")

    train_obs = pd.read_parquet(DATA_PROCESSED / "split_b_train_observed.parquet")
    sim_b, item_ids_b = _fit_sim(train_obs)
    pop_b = (
        aggregate_weights(train_obs)
        .groupby("id_site", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "global_pop"})
    )
    pop_b.to_parquet(DATA_PROCESSED / "item_popularity_split_b_train.parquet", index=False)

    for part in ("train", "val", "test"):
        obs = pd.read_parquet(DATA_PROCESSED / f"split_b_{part}_observed.parquet")
        cand = _write_candidates(f"split_b_{part}", obs, enrolments, items, sim_b, item_ids_b)
        if part == "test":
            _write_baselines(f"split_b_{part}", obs, cand, enrolments, items, pop_b)

    # Split A: fit CF per cohort (id_site is presentation-local).
    train_a = pd.read_parquet(DATA_PROCESSED / "split_a_train_observed.parquet")
    sim_a_train, item_ids_a_train = _fit_sim(train_a)
    pop_a = (
        aggregate_weights(train_a)
        .groupby("id_site", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "global_pop"})
    )
    pop_a.to_parquet(DATA_PROCESSED / "item_popularity_split_a_train.parquet", index=False)
    cand_train = _write_candidates(
        "split_a_train", train_a, enrolments, items, sim_a_train, item_ids_a_train
    )

    test_a = pd.read_parquet(DATA_PROCESSED / "split_a_test_observed.parquet")
    sim_a_test, item_ids_a_test = _fit_sim(test_a)
    # Test baselines: popularity from test observed only (no held-out leakage).
    pop_a_test = (
        aggregate_weights(test_a)
        .groupby("id_site", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "global_pop"})
    )
    pop_a_test.to_parquet(DATA_PROCESSED / "item_popularity_split_a_test.parquet", index=False)
    cand_test = _write_candidates(
        "split_a_test", test_a, enrolments, items, sim_a_test, item_ids_a_test
    )
    _write_baselines("split_a_test", test_a, cand_test, enrolments, items, pop_a_test)

    with open(DATA_PROCESSED / "candidates_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate_k": CANDIDATE_K,
                "baselines": ["V0_WeekPop", "V1_MostPopular", "V2_ItemKNN"],
                "note": "ItemKNN similarity fit on train observed only.",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
