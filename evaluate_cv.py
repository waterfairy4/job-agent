"""Evaluate the generated CVs (AIGC quality evaluation).

  python evaluate_cv.py export      -> eval/cv_ratings.csv + eval/cv_blind/<job>-A.txt / -B.txt
                                       Read each job file and both CVs, give each a relevance score 1-5.
                                       You don't know which is the tailored one: that is the point.
  python evaluate_cv.py report v1   -> LLM judge scores (cached in eval/runs/cvjudge-v1.csv) + your ratings
                                       -> eval/cv-report-v1.md

Needs tailored CVs: ask the app "make me a CV for <job>" for a few jobs first (made after this
update, which also saves each CV as data/reports/<job>-cv.json).
"""
import sys
from pathlib import Path

from jobagent.cv_eval import cv_pairs, export_sheet, human_scores, judge, load_judge, report, save_judge
from jobagent.cvdoc import cv_text, find_cv
from jobagent.store import JobStore

EVAL = Path("eval")


def pairs():
    path = find_cv()
    if not path:
        sys.exit("Could not find your CV in profile/ (name it with cv/resume/简历, or set CV_FILE in .env).")
    found = cv_pairs(JobStore(), cv_text(path), original_pdf=path)
    if not found:
        sys.exit("No tailored CVs yet: ask the app to make a CV for a few jobs first.")
    return found


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        total, new = export_sheet(pairs(), EVAL)
        print(f"eval/cv_ratings.csv: {total} jobs ({new} new). Rate a_relevance and b_relevance 1-5, "
              f"then run: python evaluate_cv.py report v1")
    elif cmd == "report":
        version = sys.argv[2] if len(sys.argv) > 2 else "latest"
        cache = EVAL / "runs" / f"cvjudge-{version}.csv"
        found = pairs()
        if cache.exists():
            rows = load_judge(cache)
        else:
            from jobagent.llm import get_llm

            print(f"Judging {len(found)} jobs x 2 CVs...")
            rows = judge(found, get_llm())
            save_judge(cache, rows)
        text = report(rows, human_scores(EVAL), {p["job_id"]: f"{p['company']} – {p['title']}" for p in found})
        (EVAL / f"cv-report-{version}.md").write_text(text, encoding="utf-8")
        print(text)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
