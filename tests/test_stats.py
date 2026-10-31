import csv

from jobagent.agent_eval import compare_routing, load_routing_run
from jobagent.evaluation import compare_runs, load_run
from jobagent.stats import bootstrap_diff, mcnemar_exact


def test_mcnemar_and_bootstrap():
    assert mcnemar_exact([True] * 10, [True] * 9 + [False])["p"] == 1.0          # one change: chance
    r = mcnemar_exact([False] * 8 + [True] * 2, [True] * 10)
    assert (r["b_only"], r["a_only"]) == (8, 0) and abs(r["p"] - 2 / 256) < 1e-12   # 8 fixes, 0 breaks
    items = [{"a": 0, "b": 1}] * 8 + [{"a": 1, "b": 1}] * 2
    ci = bootstrap_diff(items, lambda rows, k: sum(r[k] for r in rows) / len(rows), "a", "b")
    assert ci["diff"] == 0.8 and 0 < ci["low"] <= 0.8 <= ci["high"]
    assert ci == bootstrap_diff(items, lambda rows, k: sum(r[k] for r in rows) / len(rows), "a", "b")  # seeded


def test_routing_versions_rebuilt_from_reports(tmp_path):
    with open(tmp_path / "routing_cases.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["message", "expected", "note"], ["帮我分析一下这个岗位", "analyze_job", ""],
                                 ["Hi", "none", ""]])
    (tmp_path / "routing-v1.md").write_text("# R\n\n## Wrong routes\n\n| Message | Expected | Got |\n| --- | --- | --- |\n"
                                            "| 帮我分析一下这个岗位 | analyze_job | none |\n", encoding="utf-8")
    (tmp_path / "routing-v2.md").write_text("# R\n\nall right\n", encoding="utf-8")
    v1, v2 = load_routing_run(tmp_path, "v1"), load_routing_run(tmp_path, "v2")
    assert v1 == {"帮我分析一下这个岗位": False, "Hi": True} and all(v2.values())
    md = compare_routing(v1, v2, "v1", "v2")
    assert "fixed 1, broke 0; p = 1.000" in md and "Not statistically significant" in md


def test_judge_versions_rebuilt_from_reports(tmp_path):
    rows = [["key", "company", "title", "type", "item", "your_label", "notes"]]
    rows += [[f"j:{i}", "Huawei", "Eval Engineer", "must", f"item {i}", "s", ""] for i in range(10)]
    with open(tmp_path / "labels.csv", "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)
    table = "## Disagreements (for error analysis)\n\n| Job | Item | You | Judge | Judge's reason | Your note |\n| --- |\n"
    (tmp_path / "report-v1.md").write_text(
        "# J\n\n" + table + "".join(f"| Huawei – Eval Engineer | item {i} | strong | partial | r |  |\n"
                                    for i in range(8)), encoding="utf-8")
    (tmp_path / "report-v3.md").write_text("# J\n\nNo disagreements.\n", encoding="utf-8")
    v1, v3 = load_run(tmp_path, "v1"), load_run(tmp_path, "v3")
    assert sum(r["verdict"] == "partial" for r in v1.values()) == 8 and all(r["verdict"] == "strong" for r in v3.values())
    md = compare_runs(v1, v3, "v1", "v3")
    assert "8 items right only in v3, 0 right only in v1; p = 0.008" in md and "statistically significant" in md
