import csv

from jobagent.cv_eval import cv_pairs, export_sheet, human_scores, judge, report
from jobagent.schemas import CVDocument, CVEntry, CVScore, CVSection
from jobagent.stats import sign_test, weighted_kappa
from jobagent.store import JobStore
from tests.fakes import FakeLLM


def test_ordinal_stats():
    assert weighted_kappa([(1, 1), (3, 3), (5, 5)]) == 1.0
    assert weighted_kappa([(1, 5), (5, 1)]) < 0
    assert sign_test(6, 0) == 2 / 64 and sign_test(1, 1) == 1.0


def test_blind_sheet_judge_and_report(tmp_path):
    store = JobStore(str(tmp_path / "data"))
    for jid in ("j1", "j2", "j3"):
        store.put({"id": jid, "analysis": {"company": "Co", "title": f"Eval {jid}",
                                           "requirements": [{"text": "LLM evaluation", "type": "must"}]}})
        cv = CVDocument(name="X", sections=[CVSection(title="项目经历", entries=[CVEntry(title=f"Eval work {jid}")])])
        (store.reports_dir / f"{jid}-cv.json").write_text(cv.model_dump_json(), encoding="utf-8")
    pairs = cv_pairs(store, "original CV text")
    assert [p["job_id"] for p in pairs] == ["j1", "j2", "j3"] and "Eval work j1" in pairs[0]["tailored"]

    eval_dir = tmp_path / "eval"
    assert export_sheet(pairs, eval_dir) == (3, 3)
    key = {r["job_id"]: r["tailored"] for r in csv.DictReader(open(eval_dir / "cv_blind" / "key.csv", encoding="utf-8"))}
    for jid, letter in key.items():                 # the blind files really hold the right texts
        assert "Eval work" in (eval_dir / "cv_blind" / f"{jid}-{letter}.txt").read_text(encoding="utf-8")

    rows = list(csv.DictReader(open(eval_dir / "cv_ratings.csv", encoding="utf-8-sig")))
    for r in rows:                                  # you: tailored 4, original 3, except j3 (worse)
        t = 2 if r["job_id"] == "j3" else 4
        r["a_relevance"], r["b_relevance"] = (t, 3) if key[r["job_id"]] == "A" else (3, t)
    with open(eval_dir / "cv_ratings.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert export_sheet(pairs, eval_dir) == (3, 0)  # re-export keeps the ratings
    human = human_scores(eval_dir)
    assert human[("j1", "tailored")] == 4 and human[("j3", "original")] == 3

    llm = FakeLLM([], {CVScore: lambda p: CVScore(relevance=4 if "Eval work" in p else 2, clarity=3, reason="ok")})
    scored = judge(pairs, llm)
    assert len(scored) == 6 and len(llm.prompts) == 6
    md = report(scored, human, {"j3": "Co – Eval j3"})
    assert "| LLM judge | 3 | 2.00 | 4.00 | 3 / 0 / 0 |" in md and "| You (blind) | 3 |" in md
    assert "quadratic-weighted kappa" in md and "Co – Eval j3" in md     # j3 is a badcase
