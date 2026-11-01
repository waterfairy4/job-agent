"""Files on disk: job drafts (analysis, match, CV suggestions) and the tracker of saved jobs.

A job's id is a hash of its text, so pasting the same job twice gives the same id (no duplicates).
"""
import hashlib
import json
from datetime import date
from pathlib import Path

from jobagent import config

STATUSES = ["saved", "applied", "interview", "offer", "rejected"]


class JobStore:
    def __init__(self, root: str | None = None):
        self.root = Path(root or config.DATA_DIR)
        self.jobs_dir = self.root / "jobs"
        self.reports_dir = self.root / "reports"
        self.tracker_file = self.root / "tracker.json"
        for d in (self.jobs_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def job_id(text: str) -> str:
        return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:8]

    # ---- job drafts ----
    def put(self, job: dict):
        (self.jobs_dir / f"{job['id']}.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, job_id: str) -> dict | None:
        path = self.jobs_dir / f"{job_id.strip()}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_report(self, job_id: str, name: str, text: str) -> Path:
        path = self.reports_dir / f"{job_id}-{name}.md"
        path.write_text(text, encoding="utf-8")
        return path

    # ---- tracker ----
    def tracker(self) -> list[dict]:
        if self.tracker_file.exists():
            return json.loads(self.tracker_file.read_text(encoding="utf-8"))
        return []

    def track(self, job: dict, status: str = "saved") -> dict:
        rows = [r for r in self.tracker() if r["id"] != job["id"]]
        row = {
            "id": job["id"],
            "company": job["analysis"].get("company", ""),
            "title": job["analysis"].get("title", ""),
            "location": job["analysis"].get("location", ""),
            "deadline": job["analysis"].get("deadline", ""),
            "score": job.get("match", {}).get("score"),
            "status": status,
            "updated": date.today().isoformat(),
        }
        rows.append(row)
        self.tracker_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return row
