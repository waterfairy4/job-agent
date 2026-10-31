"""Evaluate the LLM judge against your own labels.

  python evaluate.py export          -> writes eval/labels.csv (blind: judge verdicts hidden)
                                        fill `your_label` with s / p / m (strong / partial / missing)
  python evaluate.py report v1       -> compares your labels with the judge
                                        -> eval/report-v1.md, and adds a row to eval/history.csv
  python evaluate.py compare v1 v3   -> is v3 really better? paired McNemar test + bootstrap CIs

Typical loop: report v1 -> change the system -> python rematch.py -> report v2 -> compare history.
Re-running `export` after matching more jobs adds new rows and keeps your existing labels.
"""
import sys
from datetime import date
from pathlib import Path

from jobagent import config
from jobagent.evaluation import (append_history, compare_runs, evaluate, export_labels, history_markdown,
                                 load_run, report_markdown, run_settings, save_run)

JOBS = Path(config.DATA_DIR) / "jobs"
EVAL = Path("eval")
LABELS = EVAL / "labels.csv"
HISTORY = EVAL / "history.csv"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "export":
        total, new = export_labels(JOBS, LABELS)
        print(f"{LABELS}: {total} items ({new} new). Fill in `your_label` with s / p / m, then run: "
              f"python evaluate.py report v1")
    elif cmd == "report":
        if not LABELS.exists():
            sys.exit("No label sheet yet. Run: python evaluate.py export")
        version = sys.argv[2] if len(sys.argv) > 2 else "latest"
        res = evaluate(JOBS, LABELS)
        text = report_markdown(res)
        out = EVAL / f"report-{version}.md"
        out.write_text(text, encoding="utf-8")
        print(text)
        if res["n"]:
            save_run(EVAL / "runs" / f"judge-{version}.csv", res, LABELS, JOBS)
            rows = append_history(HISTORY, version, res, run_settings(JOBS), date.today().isoformat())
            print("## History\n\n" + history_markdown(rows))
        print(f"Saved to {out}" + (f" and {HISTORY}" if res["n"] else ""))
    elif cmd == "compare" and len(sys.argv) > 3:
        a, b = sys.argv[2], sys.argv[3]
        text = compare_runs(load_run(EVAL, a), load_run(EVAL, b), a, b)
        (EVAL / f"compare-{a}-{b}.md").write_text(text, encoding="utf-8")
        print(text)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
