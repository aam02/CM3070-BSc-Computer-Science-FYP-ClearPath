"""Requirement and methodology checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SHAP = ROOT / "artifacts" / "shap"
METRICS = ROOT / "artifacts" / "metrics"


@pytest.mark.skipif(not (PROCESSED / "rankings_demo_v4.parquet").exists(), reason="need evaluate")
def test_t01_ranked_list_max_ten():
    df = pd.read_parquet(PROCESSED / "rankings_demo_v4.parquet")
    counts = df.groupby("enrolment_id").size()
    assert (counts <= 10).all()
    assert counts.min() >= 1


@pytest.mark.skipif(not (PROCESSED / "split_b_test_observed.parquet").exists(), reason="need splits")
def test_t02_candidates_exclude_seen():
    observed = pd.read_parquet(PROCESSED / "split_b_test_observed.parquet")
    cand = pd.read_parquet(PROCESSED / "candidates_split_b_test.parquet")
    seen = observed.groupby("enrolment_id")["id_site"].apply(lambda s: set(int(x) for x in s))
    for eid, g in cand.groupby("enrolment_id"):
        assert not (set(int(x) for x in g["id_site"]) & seen.get(eid, set()))


@pytest.mark.skipif(not (PROCESSED / "candidates_split_a_test.parquet").exists(), reason="need candidates")
def test_t03_same_presentation_only():
    items = pd.read_parquet(PROCESSED / "items.parquet")
    enrolments = pd.read_parquet(PROCESSED / "enrolments.parquet")
    cand = pd.read_parquet(PROCESSED / "candidates_split_a_test.parquet")
    item_pres = items.set_index("id_site")[["code_module", "code_presentation"]]
    enr = enrolments.set_index("enrolment_id")[["code_module", "code_presentation"]]
    sample = cand.sample(min(400, len(cand)), random_state=0)
    for _, row in sample.iterrows():
        e = enr.loc[row["enrolment_id"]]
        if isinstance(e, pd.DataFrame):
            e = e.iloc[0]
        it = item_pres.loc[int(row["id_site"])]
        if isinstance(it, pd.DataFrame):
            it = it.iloc[0]
        assert e["code_module"] == it["code_module"]
        assert e["code_presentation"] == it["code_presentation"]


@pytest.mark.skipif(not (SHAP / "shap_demo_v4_meta.json").exists(), reason="need shap")
def test_t04_attribution_additivity():
    meta = json.loads((SHAP / "shap_demo_v4_meta.json").read_text())
    assert meta["max_additivity_error"] < 1e-4


@pytest.mark.skipif(not (SHAP / "shap_demo_v4.parquet").exists(), reason="need shap")
def test_t05_explanations_human_readable():
    df = pd.read_parquet(SHAP / "shap_demo_v4.parquet")
    sample = json.loads(df.iloc[0]["top3"])
    assert len(sample) == 3
    joined = " ".join(sample)
    assert "+0." not in joined and "-0." not in joined
    assert "label" in df.columns
    assert "(" in str(df.iloc[0]["label"])


@pytest.mark.skipif(not (PROCESSED / "funnel_counts.csv").exists(), reason="need prep")
def test_funnel_tracks_students_and_enrolments():
    funnel = pd.read_csv(PROCESSED / "funnel_counts.csv")
    assert "n_students" in funnel.columns
    assert "n_enrolments" in funnel.columns
    assert funnel["rows"].iloc[0] >= funnel["rows"].iloc[-1]


@pytest.mark.skipif(not (PROCESSED / "split_b_partitions.json").exists(), reason="need splits")
def test_split_b_users_disjoint():
    parts = json.loads((PROCESSED / "split_b_partitions.json").read_text())
    tr, va, te = set(parts["train_enrolments"]), set(parts["val_enrolments"]), set(parts["test_enrolments"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(te) > 0


@pytest.mark.skipif(not (PROCESSED / "features_split_b_test_v4.parquet").exists(), reason="need features")
def test_engagement_band_from_observed_only():
    from clearpath.features import _engagement_band_from_observed

    obs = pd.read_parquet(PROCESSED / "split_b_test_observed.parquet")
    feats = pd.read_parquet(PROCESSED / "features_split_b_test_v4.parquet")
    bands = _engagement_band_from_observed(obs)
    sample = feats.sample(min(200, len(feats)), random_state=0)
    for _, row in sample.iterrows():
        assert int(row["engagement_band"]) == int(bands.loc[row["enrolment_id"]])


@pytest.mark.skipif(not (METRICS / "metrics_split_b.csv").exists(), reason="need evaluate")
def test_metrics_include_v0_and_cis():
    df = pd.read_csv(METRICS / "metrics_split_b.csv")
    assert set(df["variant"]) >= {"V0", "V1", "V2", "V3", "V4"}
    assert "ndcg@10_ci95_low" in df.columns
    assert "ndcg@10_ci95_high" in df.columns


@pytest.mark.skipif(not (METRICS / "wilcoxon_split_a.json").exists(), reason="need evaluate")
def test_wilcoxon_effect_size_finite():
    data = json.loads((METRICS / "wilcoxon_split_a.json").read_text())
    for key, payload in data.items():
        r = payload.get("effect_size_rbc")
        if r is not None:
            assert abs(r) <= 1.0 + 1e-6
            assert r == r
        assert "ci95_low" in payload


@pytest.mark.skipif(not (SHAP / "faithfulness.json").exists(), reason="need explain")
def test_faithfulness_uses_mean_mask_and_signed_order():
    data = json.loads((SHAP / "faithfulness.json").read_text())
    assert data.get("masking") == "train_mean"
    assert data.get("ordering") == "signed_shap_descending"
    assert "auc_gap" in data


def test_unknown_enrolment_returns_404():
    from clearpath.app import app

    if not (PROCESSED / "rankings_demo_v4.parquet").exists():
        pytest.skip("demo rankings missing")
    import clearpath.app as a

    a.RANKINGS = None
    a.SHAP_DF = None
    a.ENROLMENT_META = None
    a.ITEMS_META = None
    a.SEEN_BY_ENROLMENT = None
    a.POPULAR_BASELINE = None
    client = app.test_client()
    resp = client.get("/api/recommend/NOT_A_REAL_ENROLMENT")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_config_imports():
    from clearpath.config import MODULES, PRESENTATIONS, SPLIT_B_TRAIN_FRAC

    assert MODULES == ("BBB", "DDD", "FFF")
    assert "2014J" in PRESENTATIONS
    assert SPLIT_B_TRAIN_FRAC == 0.70
