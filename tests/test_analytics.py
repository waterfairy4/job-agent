import sqlite3

from jobagent.analytics import cv_ab_test, fisher_exact, funnel, report, segments
from jobagent.store import JobStore


def job(jid, texts, score=50, company="Co"):
    return {"id": jid, "analysis": {"company": company, "title": f"Role {jid}",
                                    "requirements": [{"text": t, "type": "must"} for t in texts]},
            "match": {"score": score}}


def test_stage_is_kept_through_rejection_and_funnel(tmp_path):
    store = JobStore(str(tmp_path))
    a, b, c = job("a", ["Python"]), job("b", ["Python"]), job("c", ["Python"])
    store.track(a, "applied", cv_version="tailored", channel="campus site")
    store.track(a, "interview")
    store.track(a, "rejected")                       # rejected after the interview
    store.track(b, "applied", cv_version="original")
    store.track(c, "saved")
    rows = {r["id"]: r for r in store.tracker()}
    assert rows["a"]["stage"] == "interview" and rows["a"]["status"] == "rejected"
    assert rows["a"]["cv_version"] == "tailored" and rows["a"]["channel"] == "campus site"   # kept on updates
    f = {x["stage"]: x for x in funnel(store.tracker())}
    assert (f["saved"]["count"], f["applied"]["count"], f["interview"]["count"], f["offer"]["count"]) == (3, 2, 1, 0)
    assert f["applied"]["conversion"] == 2 / 3 and f["interview"]["conversion"] == 0.5


def test_fisher_and_ab_test():
    assert abs(fisher_exact(3, 1, 1, 3) - 0.4857) < 1e-3          # textbook value
    assert abs(fisher_exact(10, 0, 0, 10) - 1.0825e-05) < 1e-7
    rows = ([{"stage": "interview", "cv_version": "tailored"}] * 3 + [{"stage": "applied", "cv_version": "tailored"}]
            + [{"stage": "interview", "cv_version": "original"}] + [{"stage": "applied", "cv_version": "original"}] * 3)
    ab = cv_ab_test(rows)
    assert ab["tailored"] == {"group": "tailored", "sent": 4, "responded": 3} and abs(ab["p"] - 0.4857) < 1e-3


def test_segments_separate_two_kinds_of_jobs():
    evals = [job(f"e{i}", ["LLM evaluation benchmark design", "evaluation metrics", "Python"], 70) for i in range(4)]
    infra = [job(f"i{i}", ["Kubernetes cluster operations", "Linux networking", "Go"], 30) for i in range(4)]
    segs = segments(evals + infra, k=2)
    groups = sorted(sorted(j["id"][0] for j in s["jobs"]) for s in segs)
    assert groups == [["e"] * 4, ["i"] * 4] and segs[0]["avg_score"] == 70 and "evaluation" in segs[0]["label"]


def test_report_and_old_database_columns(tmp_path):
    con = sqlite3.connect(tmp_path / "tracker.sqlite")              # a tracker made by the previous version
    con.execute("CREATE TABLE applications (id TEXT PRIMARY KEY, company TEXT, title TEXT, location TEXT, "
                "deadline TEXT, score INTEGER, url TEXT, status TEXT NOT NULL, saved_on TEXT, applied_on TEXT, updated TEXT)")
    con.execute("INSERT INTO applications (id, company, title, status, applied_on) VALUES ('x', 'Co', 'T', 'applied', '2026-10-01')")
    con.commit()
    con.close()
    store = JobStore(str(tmp_path))
    store.track(job("x", ["Python"]), "interview", cv_version="tailored")
    row = store.tracker()[0]
    assert row["stage"] == "interview" and row["applied_on"] == "2026-10-01"
    text = report(store.tracker(), [job("x", ["Python evaluation"]), job("y", ["Kubernetes"])])
    assert "## Funnel" in text and "## A/B test" in text and "Not enough data yet" in text and "## Job segments" in text
