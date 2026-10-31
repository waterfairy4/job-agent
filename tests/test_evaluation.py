import csv
import json

import pytest

from jobagent.evaluation import (append_history, cohen_kappa, evaluate, export_labels, history_markdown,
                                 read_csv, report_markdown, run_settings)


def _job(jobs_dir, job_id, company, verdicts):
    results = [{"text": f"item {i}", "type": t, "verdict": v, "reason": f"r{i}", "cited": [], "tip": ""}
               for i, (t, v) in enumerate(verdicts)]
    (jobs_dir / f"{job_id}.json").write_text(json.dumps(
        {"id": job_id, "analysis": {"company": company, "title": "Eval Engineer", "requirements": []},
         "match": {"score": 0, "results": results}}), encoding="utf-8")


def _fill(path, labels: dict, encoding="utf-8-sig"):
    rows = read_csv(path)
    for r in rows:
        r["your_label"] = labels.get(r["key"], "")
    with open(path, "w", encoding=encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def jobs(tmp_path):
    d = tmp_path / "jobs"
    d.mkdir()
    _job(d, "aaa", "Huawei", [("duty", "strong"), ("must", "partial"), ("nice", "missing")])
    _job(d, "bbb", "", [("must", "strong"), ("must", "missing")])
    (d / "unmatched.json").write_text(json.dumps({"id": "unmatched", "analysis": {}}), encoding="utf-8")
    return d


def test_export_is_blind_and_keeps_labels(jobs, tmp_path):
    out = tmp_path / "eval" / "labels.csv"
    assert export_labels(jobs, out) == (5, 5)
    text = out.read_text(encoding="utf-8-sig")
    assert "strong" not in text and "partial" not in text  # judge verdicts hidden
    assert "Unknown company" in text

    _fill(out, {"aaa:0": "s"})
    assert export_labels(jobs, out) == (5, 0)  # re-export keeps labels, adds nothing new
    assert read_csv(out)[0]["your_label"] == "s"


def test_metrics(jobs, tmp_path):
    out = tmp_path / "labels.csv"
    export_labels(jobs, out)
    # judge: aaa = strong, partial, missing ; bbb = strong, missing
    _fill(out, {"aaa:0": "s", "aaa:1": "m", "aaa:2": "missing", "bbb:0": "p", "bbb:1": "x"}, encoding="gbk")
    res = evaluate(jobs, out)
    assert res["n"] == 4 and res["skipped"] == ["bbb:1"]
    assert res["agreement"] == 0.5
    assert res["too_generous"] == 0.5 and res["too_strict"] == 0.0
    assert res["confusion"]["missing"]["partial"] == 1 and res["confusion"]["partial"]["strong"] == 1
    assert res["by_type"]["must"] == {"n": 2, "agreement": 0.0}
    # only job aaa is fully labeled: judge (1.5 + 1 + 0)/4.5 = 56%, you (1.5)/4.5 = 33%
    assert res["jobs"] == [{"name": "Huawei – Eval Engineer", "judge_score": 56, "your_score": 33}]
    md = report_markdown(res)
    assert "Agreement 50%" in md and "Disagreements" in md and "Skipped 1" in md


def test_kappa():
    assert cohen_kappa([("strong", "strong"), ("missing", "missing")]) == 1.0
    assert cohen_kappa([("strong", "missing"), ("missing", "strong")]) == -1.0
    assert abs(cohen_kappa([("strong", "strong"), ("strong", "partial"),
                            ("partial", "partial"), ("missing", "partial")]) - 0.2727) < 1e-3


def test_empty_report(jobs, tmp_path):
    out = tmp_path / "labels.csv"
    export_labels(jobs, out)
    assert "No labels yet" in report_markdown(evaluate(jobs, out))


def test_history_and_settings(jobs, tmp_path):
    out = tmp_path / "labels.csv"
    export_labels(jobs, out)
    _fill(out, {"aaa:0": "s", "aaa:1": "m"})
    res = evaluate(jobs, out)
    hist = tmp_path / "eval" / "history.csv"
    assert run_settings(jobs) == "v1 judge, top-3, no expansion"
    append_history(hist, "v1", res, run_settings(jobs), "2026-10-19")
    append_history(hist, "v2", res, "strict judge, top-5, expansion on", "2026-10-19")
    rows = append_history(hist, "v1", res, "rerun", "2026-10-20")   # same name replaces
    assert [r["version"] for r in rows] == ["v2", "v1"] and rows[1]["settings"] == "rerun"
    md = history_markdown(rows)
    assert "| v2 | 2 | 50% | 0.33 | 50% | 0% |" in md  # kappa = (0.5-0.25)/(1-0.25)

    job = json.loads((jobs / "aaa.json").read_text(encoding="utf-8"))
    job["match"]["settings"] = {"expand": True, "top_k": 5}
    (jobs / "aaa.json").write_text(json.dumps(job), encoding="utf-8")
    assert run_settings(jobs) == "mixed settings: run rematch.py"
