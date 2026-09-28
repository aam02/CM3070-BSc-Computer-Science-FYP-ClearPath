"""Flask demo with campus showcase overlay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from clearpath.config import DATA_PROCESSED, METRICS_DIR, SHAP_DIR, TOP_N
from clearpath.demo_showcase import (
    activity_title,
    course_by_id,
    showcase_payload,
    student_by_enrolment,
    student_by_id,
)

app = Flask(__name__)

ACTIVITY_NAMES = {
    "forumng": "Forum",
    "resource": "Reading",
    "oucontent": "Lesson",
    "quiz": "Quiz",
    "externalquiz": "External quiz",
    "subpage": "Section",
    "url": "Link",
    "glossary": "Glossary",
    "oucollaborate": "Live session",
    "ouwiki": "Wiki",
    "page": "Page",
    "questionnaire": "Survey",
    "dataplus": "Data lab",
    "homepage": "Homepage",
}

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ClearPath - Campus Demo</title>
  <style>
    :root {
      --ink: #12201a;
      --muted: #5b6f66;
      --leaf: #1a6b52;
      --leaf-deep: #0e4334;
      --wash: #eef4f0;
      --panel: #ffffff;
      --line: rgba(18, 32, 26, 0.1);
      --warn: #8a4b12;
      --ok: #1a6b52;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(65% 45% at 0% 0%, rgba(26,107,82,0.10), transparent 55%),
        linear-gradient(180deg, #f8fbf9 0%, var(--wash) 100%);
      font-size: 16.5px; line-height: 1.45;
    }
    .shell { width: min(1040px, calc(100% - 2.4rem)); margin: 0 auto; padding: 1.8rem 0 3rem; }

    header { margin-bottom: 1.4rem; animation: rise .45s ease both; }
    .brand {
      font-family: "Iowan Old Style", Palatino, Georgia, serif;
      font-size: clamp(2.4rem, 5vw, 3.2rem);
      letter-spacing: -0.03em; line-height: 1; margin: 0;
    }
    .brand span { color: var(--leaf); }
    .lede { margin: 0.5rem 0 0; max-width: 38rem; color: var(--muted); font-size: 1.05rem; }
    .fine { margin: 0.55rem 0 0; color: var(--muted); font-size: 0.86rem; }

    .courses {
      display: flex; flex-wrap: wrap; gap: 0.5rem;
      margin: 1.2rem 0 1rem; animation: rise .45s ease .04s both;
    }
    .course-btn {
      border: 1px solid var(--line); background: var(--panel);
      border-radius: 14px; padding: 0.7rem 0.95rem; cursor: pointer;
      text-align: left; font: inherit; min-width: 11rem;
      transition: border-color .15s, background .15s;
    }
    .course-btn:hover { border-color: rgba(26,107,82,.4); }
    .course-btn.active { border-color: var(--leaf); background: #eaf6f0; }
    .course-btn .code { font-weight: 750; font-size: 0.92rem; }
    .course-btn .name { color: var(--muted); font-size: 0.86rem; margin-top: 0.12rem; }

    .layout {
      display: grid; grid-template-columns: 280px 1fr; gap: 1rem;
      animation: rise .5s ease .08s both;
    }
    @media (max-width: 860px) { .layout { grid-template-columns: 1fr; } }

    .panel {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 18px; padding: 1.05rem 1.15rem 1.2rem;
    }
    .panel h2 {
      font-family: "Iowan Old Style", Palatino, Georgia, serif;
      font-size: 1.2rem; margin: 0 0 0.75rem; letter-spacing: -0.02em;
    }

    .students { display: grid; gap: 0.55rem; }
    .student {
      display: grid; grid-template-columns: auto 1fr; gap: 0.7rem;
      align-items: center; width: 100%; text-align: left;
      border: 1px solid var(--line); background: #fff;
      border-radius: 14px; padding: 0.7rem 0.75rem; cursor: pointer;
      font: inherit; color: inherit; transition: border-color .15s, background .15s;
    }
    .student:hover { border-color: rgba(26,107,82,.35); }
    .student.active { border-color: var(--leaf); background: #eaf6f0; }
    .avatar {
      width: 2.4rem; height: 2.4rem; border-radius: 999px;
      display: grid; place-items: center; font-weight: 750; font-size: 0.85rem;
      background: #dff0e7; color: var(--leaf-deep);
    }
    .student .nm { font-weight: 700; }
    .student .sub { color: var(--muted); font-size: 0.84rem; margin-top: 0.1rem; }
    .badge {
      display: inline-block; margin-top: 0.28rem;
      font-size: 0.72rem; font-weight: 700; letter-spacing: .02em;
      padding: 0.12rem 0.45rem; border-radius: 999px;
      background: var(--wash); color: var(--muted);
    }
    .badge.risk { background: #f8e8d8; color: var(--warn); }
    .badge.ok { background: #dff0e7; color: var(--ok); }

    .toolbar {
      display: flex; flex-wrap: wrap; justify-content: space-between;
      gap: 0.75rem; align-items: start; margin-bottom: 0.7rem;
    }
    .who { margin: 0; }
    .who .name {
      font-family: "Iowan Old Style", Palatino, Georgia, serif;
      font-size: 1.45rem; letter-spacing: -0.02em;
    }
    .who .meta { color: var(--muted); margin-top: 0.2rem; font-size: 0.95rem; }
    .seg {
      display: inline-flex; padding: 0.18rem; border-radius: 999px;
      background: var(--wash); border: 1px solid var(--line);
    }
    .seg button {
      border: 0; background: transparent; font: inherit; font-weight: 650;
      padding: 0.42rem 0.8rem; border-radius: 999px; cursor: pointer; color: var(--muted);
    }
    .seg button.active {
      background: #fff; color: var(--ink);
      box-shadow: 0 1px 2px rgba(18,32,26,.08);
    }

    .note { margin: 0 0 0.85rem; color: var(--muted); font-size: 0.93rem; }
    .seen { margin: 0 0 1rem; color: var(--muted); font-size: 0.9rem; }
    .seen b { color: var(--ink); }
    .err { color: #8b2e2e; margin: 0 0 0.8rem; }
    .empty { color: var(--muted); padding: 1.2rem 0; }

    ol.recs { list-style: none; margin: 0; padding: 0; }
    .rec { border-top: 1px solid var(--line); animation: rise .35s ease both; }
    .rec:first-child { border-top: 0; }
    .rec summary {
      list-style: none; cursor: pointer;
      display: grid; grid-template-columns: 2.1rem 1fr auto;
      gap: 0.8rem; align-items: baseline; padding: 0.95rem 0.1rem;
    }
    .rec summary::-webkit-details-marker { display: none; }
    .rank { font-variant-numeric: tabular-nums; font-weight: 700; color: var(--leaf-deep); }
    .rec-title { font-weight: 700; font-size: 1.05rem; }
    .rec-meta { margin-top: 0.15rem; color: var(--muted); font-size: 0.86rem; }
    .score { font-variant-numeric: tabular-nums; font-weight: 650; color: var(--leaf-deep); }
    .bar {
      height: 3px; margin: 0 0 0.8rem 2.9rem; border-radius: 99px;
      background: rgba(18,32,26,.07); overflow: hidden;
    }
    .bar > i {
      display: block; height: 100%; background: var(--leaf);
      transform-origin: left; animation: grow .4s ease both;
    }
    .why { margin: 0 0 0.95rem; padding: 0 0 0 2.9rem; color: var(--muted); font-size: 0.93rem; }
    .why li { margin: 0.25rem 0; }

    @keyframes rise {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: none; }
    }
    @keyframes grow { from { transform: scaleX(0); } to { transform: none; } }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1 class="brand">Clear<span>Path</span></h1>
      <p class="lede">A small campus sample: pick a course and student, then see what to open next, with reasons.</p>
      <p class="fine">Illustrative names on top of real ClearPath rankings & TreeSHAP · NDCG@10 <b>{{ metrics.ndcg }}</b> ({{ metrics.lift }} vs baseline)</p>
    </header>

    <div class="courses" id="courses"></div>

    <div class="layout">
      <aside class="panel">
        <h2>Students</h2>
        <div class="students" id="students"></div>
      </aside>

      <section class="panel">
        <div class="toolbar">
          <div class="who" id="who">
            <div class="name">Select a student</div>
            <div class="meta">Recommendations appear here</div>
          </div>
          <div class="seg">
            <button type="button" class="active" id="mode-clearpath">ClearPath</button>
            <button type="button" id="mode-popular">Most popular</button>
          </div>
        </div>
        <p class="note">Next activities inside this course (forums, readings, quizzes), not other courses.</p>
        <p class="seen" id="seen" hidden></p>
        <p class="note" id="baseline-note" hidden>Most-popular baseline for the same student (no personalisation).</p>
        <p class="err" id="err" hidden></p>
        <div id="empty" class="empty">Choose a student on the left.</div>
        <ol class="recs" id="results"></ol>
      </section>
    </div>
  </div>

<script>
const SHOWCASE = {{ showcase_json|safe }};
const coursesEl = document.getElementById('courses');
const studentsEl = document.getElementById('students');
const who = document.getElementById('who');
const seenEl = document.getElementById('seen');
const errEl = document.getElementById('err');
const empty = document.getElementById('empty');
const results = document.getElementById('results');
const baselineNote = document.getElementById('baseline-note');
const modeClear = document.getElementById('mode-clearpath');
const modePop = document.getElementById('mode-popular');

let courseId = SHOWCASE.courses[0]?.id || '';
let studentId = '';
let mode = 'clearpath';

function badgeClass(status) {
  const s = (status || '').toLowerCase();
  if (s.includes('risk')) return 'badge risk';
  if (s.includes('track') || s.includes('distinction') || s.includes('strong')) return 'badge ok';
  return 'badge';
}

function renderCourses() {
  coursesEl.innerHTML = '';
  for (const c of SHOWCASE.courses) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'course-btn' + (c.id === courseId ? ' active' : '');
    btn.innerHTML = `<div class="code">${c.code} · ${c.term}</div><div class="name">${c.name}</div>`;
    btn.addEventListener('click', () => {
      courseId = c.id;
      renderCourses();
      renderStudents();
      const first = SHOWCASE.students.find(s => s.course_id === courseId);
      if (first) pickStudent(first.id);
    });
    coursesEl.appendChild(btn);
  }
}

function renderStudents() {
  studentsEl.innerHTML = '';
  const rows = SHOWCASE.students.filter(s => s.course_id === courseId);
  for (const s of rows) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'student' + (s.id === studentId ? ' active' : '');
    btn.innerHTML = `
      <div class="avatar">${s.initials}</div>
      <div>
        <div class="nm">${s.name}</div>
        <div class="sub">${s.year}</div>
        <span class="${badgeClass(s.status)}">${s.status}</span>
      </div>`;
    btn.addEventListener('click', () => pickStudent(s.id));
    studentsEl.appendChild(btn);
  }
}

function setMode(next) {
  mode = next;
  modeClear.classList.toggle('active', mode === 'clearpath');
  modePop.classList.toggle('active', mode === 'popular');
  if (studentId) load();
}

async function pickStudent(id) {
  studentId = id;
  renderStudents();
  await load();
}

async function load() {
  const student = SHOWCASE.students.find(s => s.id === studentId);
  if (!student) return;
  errEl.hidden = true;
  const url = '/api/student/' + encodeURIComponent(student.id) + '?source=' + encodeURIComponent(mode);
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) {
    errEl.hidden = false;
    errEl.textContent = data.error || ('HTTP ' + res.status);
    results.innerHTML = '';
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  who.innerHTML = `
    <div class="name">${data.student.name}</div>
    <div class="meta">${data.course.code} · ${data.course.name} · ${data.student.status}<br/>${data.student.blurb}</div>`;
  if (data.seen && data.seen.length) {
    seenEl.hidden = false;
    seenEl.innerHTML = '<b>Already opened:</b> ' + data.seen.slice(0,4).map(s => s.label).join(' · ');
  } else {
    seenEl.hidden = true;
  }
  baselineNote.hidden = data.source !== 'popular';
  results.innerHTML = '';
  const maxScore = Math.max(...data.recommendations.map(r => Math.abs(r.score)), 1e-6);
  data.recommendations.forEach((row, idx) => {
    const det = document.createElement('details');
    det.className = 'rec';
    det.style.animationDelay = (idx * 0.03) + 's';
    if (idx === 0) det.open = true;
    const summary = document.createElement('summary');
    const timing = row.timing ? row.timing + ' · ' : '';
    summary.innerHTML = `
      <div class="rank">${String(row.rank).padStart(2,'0')}</div>
      <div>
        <div class="rec-title">${row.title}</div>
        <div class="rec-meta">${row.type_label} · ${timing}${row.week_label || 'this module'}</div>
      </div>
      <div class="score">${row.score.toFixed(2)}</div>`;
    det.appendChild(summary);
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.innerHTML = `<i style="width:${Math.max(6, 100 * Math.abs(row.score) / maxScore)}%"></i>`;
    det.appendChild(bar);
    const ul = document.createElement('ul');
    ul.className = 'why';
    for (const s of row.explanations) {
      const li = document.createElement('li');
      li.textContent = s;
      ul.appendChild(li);
    }
    det.appendChild(ul);
    results.appendChild(det);
  });
}

modeClear.addEventListener('click', () => setMode('clearpath'));
modePop.addEventListener('click', () => setMode('popular'));
renderCourses();
renderStudents();
const first = SHOWCASE.students.find(s => s.course_id === courseId) || SHOWCASE.students[0];
if (first) pickStudent(first.id);
</script>
</body>
</html>
"""


def _activity_display_name(activity_type: str) -> str:
    key = str(activity_type or "activity")
    if key in ACTIVITY_NAMES:
        return ACTIVITY_NAMES[key]
    return key.replace("_", " ").strip().title() or "Activity"


def _timing_phrase(week_from, week_position) -> str | None:
    try:
        if week_from is not None and str(week_from).isdigit():
            return f"~week {int(week_from)}"
    except (TypeError, ValueError):
        pass
    try:
        wp = float(week_position)
    except (TypeError, ValueError):
        return None
    if wp != wp:
        return None
    if wp < 0.34:
        return "early in module"
    if wp > 0.66:
        return "later in module"
    return "mid-module"


def _load_metrics_blurb() -> dict[str, str]:
    out = {"ndcg": "-", "recall": "-", "lift": "-", "faith": "-"}
    metrics_path = METRICS_DIR / "metrics_split_b.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        v4 = df.loc[df["variant"] == "V4"]
        v0 = df.loc[df["variant"] == "V0"]
        if len(v4):
            ndcg = float(v4.iloc[0]["ndcg@10"])
            recall = float(v4.iloc[0]["recall@10"])
            out["ndcg"] = f"{ndcg:.3f}"
            out["recall"] = f"{recall:.3f}"
            if len(v0):
                base = float(v0.iloc[0]["ndcg@10"])
                if base > 0:
                    out["lift"] = f"+{(ndcg / base - 1.0) * 100:.0f}%"
    faith_path = SHAP_DIR / "faithfulness.json"
    if faith_path.exists():
        try:
            payload = json.loads(Path(faith_path).read_text())
            gap = payload.get("auc_gap")
            if gap is not None:
                out["faith"] = f"{float(gap):.3f}"
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return out


def _load_demo_data():
    rankings_path = DATA_PROCESSED / "rankings_demo_v4.parquet"
    shap_path = SHAP_DIR / "shap_demo_v4.parquet"
    if not rankings_path.exists():
        raise FileNotFoundError("Missing rankings_demo_v4.parquet; run evaluate.py first")
    rankings = pd.read_parquet(rankings_path)
    if "variant" in rankings.columns:
        rankings = rankings.loc[rankings["variant"].isin(["V4", "v4"])].copy()
    shap_df = pd.read_parquet(shap_path) if shap_path.exists() else None

    items_meta: dict[int, dict] = {}
    items_path = DATA_PROCESSED / "items.parquet"
    if items_path.exists():
        items = pd.read_parquet(items_path).drop_duplicates("id_site")
        for _, row in items.iterrows():
            items_meta[int(row["id_site"])] = {
                "activity_type": str(row.get("activity_type", "activity")),
                "week_from": row.get("week_from"),
                "week_position": row.get("week_position"),
            }
        if "activity_type" not in rankings.columns:
            rankings["activity_type"] = rankings["id_site"].map(
                lambda i: items_meta.get(int(i), {}).get("activity_type", "activity")
            )
    elif "activity_type" not in rankings.columns:
        rankings["activity_type"] = "activity"

    seen_by_enrolment: dict[str, list[dict]] = {}
    obs_path = DATA_PROCESSED / "split_a_test_observed.parquet"
    if obs_path.exists():
        obs = pd.read_parquet(obs_path, columns=["enrolment_id", "id_site", "sum_click"])
        if items_meta:
            obs = obs.copy()
            obs["activity_type"] = obs["id_site"].map(
                lambda i: items_meta.get(int(i), {}).get("activity_type", "activity")
            )
        else:
            obs["activity_type"] = "activity"
        grouped = (
            obs.groupby(["enrolment_id", "activity_type"], as_index=False)["sum_click"]
            .sum()
            .sort_values(["enrolment_id", "sum_click"], ascending=[True, False])
        )
        for eid, g in grouped.groupby("enrolment_id"):
            top = g.head(5)
            seen_by_enrolment[str(eid)] = [
                {
                    "activity_type": str(row["activity_type"]),
                    "label": _activity_display_name(str(row["activity_type"])),
                    "count": int(row["sum_click"]),
                }
                for _, row in top.iterrows()
            ]

    popular = None
    base_path = DATA_PROCESSED / "baselines_split_a_test.parquet"
    if base_path.exists():
        popular = pd.read_parquet(base_path)
        popular = popular.loc[popular["variant"] == "V1_MostPopular"].copy()
        if "activity_type" not in popular.columns:
            popular["activity_type"] = popular["id_site"].map(
                lambda i: items_meta.get(int(i), {}).get("activity_type", "activity")
            )

    return rankings, shap_df, items_meta, seen_by_enrolment, popular


RANKINGS = None
SHAP_DF = None
ITEMS_META = None
SEEN_BY_ENROLMENT = None
POPULAR_BASELINE = None
# Alias kept for older tests.
ENROLMENT_META = None


def get_data():
    global RANKINGS, SHAP_DF, ITEMS_META, SEEN_BY_ENROLMENT, POPULAR_BASELINE
    if RANKINGS is None:
        RANKINGS, SHAP_DF, ITEMS_META, SEEN_BY_ENROLMENT, POPULAR_BASELINE = _load_demo_data()
    return RANKINGS, SHAP_DF, ITEMS_META, SEEN_BY_ENROLMENT, POPULAR_BASELINE


def _build_recs_from_rows(
    rows: pd.DataFrame,
    enrolment_id: str,
    shap_df: pd.DataFrame | None,
    items_meta: dict,
    with_shap: bool,
) -> list[dict]:
    recs = []
    for _, row in rows.iterrows():
        site = int(row["id_site"])
        meta = items_meta.get(site, {})
        act = str(row.get("activity_type") or meta.get("activity_type") or "activity")
        timing = _timing_phrase(meta.get("week_from"), meta.get("week_position"))
        title = activity_title(act, site)
        explanations: list[str] = []
        if with_shap and shap_df is not None:
            hit = shap_df.loc[
                (shap_df["enrolment_id"] == enrolment_id) & (shap_df["id_site"] == site)
            ]
            if len(hit):
                explanations = json.loads(hit.iloc[0]["top3"])
        if not explanations:
            if with_shap:
                explanations = ["No cached explanation for this item."]
            else:
                explanations = ["Popular with peers on this course, not personalised."]
        recs.append(
            {
                "id_site": site,
                "activity_type": act,
                "type_label": _activity_display_name(act),
                "title": title,
                "label": title,
                "timing": timing,
                "week_label": timing,
                "score": float(row.get("score", 0.0)),
                "rank": int(row.get("rank", 0)),
                "explanations": explanations,
            }
        )
    return recs


def _recommend_for_enrolment(enrolment_id: str, source: str) -> tuple[dict, int]:
    rankings, shap_df, items_meta, seen_by_enrolment, popular = get_data()
    use_popular = source in {"popular", "v1", "mostpopular", "baseline"}
    if use_popular:
        if popular is None or popular.empty:
            return {"error": "Most-popular baseline unavailable; run candidates.py"}, 503
        rows = popular.loc[popular["enrolment_id"] == enrolment_id].sort_values("rank").head(TOP_N)
        if rows.empty:
            return {"error": f"Unknown enrolment_id: {enrolment_id}"}, 404
        recs = _build_recs_from_rows(rows, enrolment_id, shap_df, items_meta, with_shap=False)
        source_name = "popular"
    else:
        rows = rankings.loc[rankings["enrolment_id"] == enrolment_id].sort_values("rank").head(TOP_N)
        if rows.empty:
            return {"error": f"Unknown enrolment_id: {enrolment_id}"}, 404
        recs = _build_recs_from_rows(rows, enrolment_id, shap_df, items_meta, with_shap=True)
        source_name = "clearpath"

    student = student_by_enrolment(enrolment_id)
    course = course_by_id(student.course_id) if student else None
    payload = {
        "enrolment_id": enrolment_id,
        "source": source_name,
        "recommendations": recs,
        "seen": seen_by_enrolment.get(enrolment_id, []),
        "student": None,
        "course": None,
    }
    if student and course:
        payload["student"] = {
            "id": student.id,
            "name": student.name,
            "year": student.year,
            "status": student.status,
            "blurb": student.blurb,
        }
        payload["course"] = {
            "id": course.id,
            "name": course.name,
            "code": course.code,
            "term": course.term,
            "dept": course.dept,
        }
    return payload, 200


@app.route("/")
def index():
    get_data()
    return render_template_string(
        INDEX_HTML,
        showcase_json=json.dumps(showcase_payload()),
        metrics=_load_metrics_blurb(),
        top_n=TOP_N,
    )


@app.route("/api/showcase")
def api_showcase():
    return jsonify(showcase_payload())


@app.route("/api/student/<student_id>")
def api_student(student_id: str):
    t0 = time.perf_counter()
    student = student_by_id(student_id)
    if student is None:
        return jsonify({"error": f"Unknown student: {student_id}"}), 404
    source = (request.args.get("source") or "clearpath").lower()
    payload, status = _recommend_for_enrolment(student.enrolment_id, source)
    if status != 200:
        return jsonify(payload), status
    payload["server_ms"] = (time.perf_counter() - t0) * 1000.0
    return jsonify(payload)


@app.route("/api/recommend/<path:enrolment_id>")
def recommend(enrolment_id: str):
    """Enrolment API used by tests and fig10."""
    t0 = time.perf_counter()
    source = (request.args.get("source") or "clearpath").lower()
    payload, status = _recommend_for_enrolment(enrolment_id, source)
    if status != 200:
        return jsonify(payload), status
    if "meta" not in payload:
        student = payload.get("student") or {}
        course = payload.get("course") or {}
        payload["meta"] = {
            "module": course.get("code") or "",
            "presentation": course.get("term") or "",
            "final_result": student.get("status") or "",
        }
    payload["server_ms"] = (time.perf_counter() - t0) * 1000.0
    return jsonify(payload)


def main() -> None:
    print("ClearPath campus demo → http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
