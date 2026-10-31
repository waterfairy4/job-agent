"""Re-run matching on every analyzed job with the current judge, without pasting the jobs again.

The job items stay the same, so your evaluation labels stay valid.

  python rematch.py                          # current settings (query expansion on, top-5 evidence)
  python rematch.py --no-expansion --top-k 3 # retrieval as in v1, only the new judge rubric

Then: python evaluate.py report <version-name>
"""
import argparse

from jobagent.llm import get_llm
from jobagent.rag import ProfileIndex
from jobagent.store import JobStore
from jobagent.tools import company_name, run_match


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-expansion", action="store_true", help="search with the job item text only")
    parser.add_argument("--top-k", type=int, default=None, help="evidence chunks per item (default from config)")
    args = parser.parse_args()

    store, index, llm = JobStore(), ProfileIndex(), get_llm()
    jobs = [j for j in (store.get(p.stem) for p in sorted(store.jobs_dir.glob("*.json"))) if j and j.get("analysis")]
    if not jobs:
        print("No analyzed jobs yet. Paste a job in the app first.")
        return
    tracked = {r["id"] for r in store.tracker()}
    for job in jobs:
        old = job.get("match", {}).get("score")
        pct, _, _ = run_match(job, index, store, llm, expand=not args.no_expansion, top_k=args.top_k)
        name = f"{company_name(job['analysis'])} – {job['analysis']['title']}"
        print(f"{name}: {old}% -> {pct}%")
        if job["id"] in tracked:  # keep the tracker's score in sync, dates and status unchanged
            row = next(r for r in store.tracker() if r["id"] == job["id"])
            store.track(job, row["status"])
    print(f"Re-matched {len(jobs)} jobs. Now run: python evaluate.py report <version-name>")


if __name__ == "__main__":
    main()
