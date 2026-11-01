"""On disk: job drafts (analysis, match, CV suggestions) as JSON files, and the tracker of saved jobs
in a SQLite database (data/tracker.sqlite).

A job's id is a hash of its text, so pasting the same job twice gives the same id (no duplicates).
"""
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from jobagent import config

STATUSES = ["saved", "applied", "interview", "offer", "rejected"]
STAGES = ["saved", "applied", "interview", "offer"]  # the funnel; "rejected" ends it at the stage reached


class JobStore:
    def __init__(self, root: str | None = None):
        self.root = Path(root or config.DATA_DIR)
        self.jobs_dir = self.root / "jobs"
        self.reports_dir = self.root / "reports"
        self.tracker_file = self.root / "tracker.json"   # older versions; migrated into the database
        self.tracker_db = self.root / "tracker.sqlite"
        for d in (self.jobs_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- the conversation the app had open (so a restart reopens it) ----
    def last_thread(self) -> str | None:
        path = self.root / "session.json"
        return json.loads(path.read_text(encoding="utf-8")).get("thread") if path.exists() else None

    def set_last_thread(self, thread_id: str):
        (self.root / "session.json").write_text(json.dumps({"thread": thread_id}), encoding="utf-8")

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

    def all_jobs(self) -> list[dict]:
        """Every analyzed job (drafts included, not only tracked ones)."""
        jobs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(self.jobs_dir.glob("*.json"))]
        return [j for j in jobs if j.get("analysis", {}).get("requirements")]

    def write_report(self, job_id: str, name: str, text: str) -> Path:
        path = self.reports_dir / f"{job_id}-{name}.md"
        path.write_text(text, encoding="utf-8")
        return path

    # ---- tracker (SQLite: one row per saved job) ----
    COLUMNS = ("id", "company", "title", "location", "deadline", "score", "url", "status",
               "saved_on", "applied_on", "updated", "stage", "cv_version", "channel")

    def _db(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.tracker_db)
        con.row_factory = sqlite3.Row
        con.execute("""CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY, company TEXT, title TEXT, location TEXT, deadline TEXT, score INTEGER,
            url TEXT, status TEXT NOT NULL, saved_on TEXT, applied_on TEXT, updated TEXT)""")
        have = {r[1] for r in con.execute("PRAGMA table_info(applications)")}
        for col in ("stage", "cv_version", "channel"):  # columns added after the first SQLite version
            if col not in have:
                con.execute(f"ALTER TABLE applications ADD COLUMN {col} TEXT")
        self._migrate(con)
        return con

    def _migrate(self, con: sqlite3.Connection):
        """Rows from older versions (tracker.json) move into SQLite once; the file is kept as .bak."""
        if not self.tracker_file.exists():
            return
        rows = json.loads(self.tracker_file.read_text(encoding="utf-8") or "[]")
        with con:
            for r in rows:
                con.execute(f"INSERT OR IGNORE INTO applications ({', '.join(self.COLUMNS)}) "
                            f"VALUES ({', '.join('?' * len(self.COLUMNS))})", [r.get(c) for c in self.COLUMNS])
        self.tracker_file.replace(self.tracker_file.with_suffix(".json.bak"))

    def tracker(self) -> list[dict]:
        with closing(self._db()) as con:
            rows = con.execute("SELECT * FROM applications ORDER BY rowid").fetchall()
        return [{k: r[k] for k in r.keys() if r[k] is not None or k == "score"} for r in rows]

    def untrack(self, job_id: str) -> bool:
        """Remove a job from the tracker. Its analysis and reports stay on disk."""
        with closing(self._db()) as con, con:
            return con.execute("DELETE FROM applications WHERE id = ?", (job_id,)).rowcount > 0

    def track(self, job: dict, status: str = "saved", today: str | None = None,
              cv_version: str = "", channel: str = "") -> dict:
        """Add or update a job. First-saved and applied dates are kept when the status changes later.
        `stage` is the furthest funnel stage ever reached (a rejection keeps it), for the funnel analysis.
        cv_version: "original" or "tailored" (the CV sent); channel: how you applied (campus site, BOSS…)."""
        today = today or date.today().isoformat()
        previous = next((r for r in self.tracker() if r["id"] == job["id"]), {})
        applied_on = previous.get("applied_on", "")
        if status in ("applied", "interview", "offer") and not applied_on:
            applied_on = today
        old_stage = previous.get("stage") or (previous.get("status") if previous.get("status") in STAGES else "saved")
        stage = max(old_stage, status if status in STAGES else old_stage, key=STAGES.index)
        row = {
            "id": job["id"],
            "company": job["analysis"].get("company", "").strip() or "Unknown company",
            "title": job["analysis"].get("title", ""),
            "location": job["analysis"].get("location", ""),
            "deadline": job["analysis"].get("deadline", ""),
            "score": job.get("match", {}).get("score"),
            "url": job.get("url", ""),
            "status": status,
            "saved_on": previous.get("saved_on") or previous.get("updated") or today,
            "applied_on": applied_on,
            "updated": today,
            "stage": stage,
            "cv_version": cv_version or previous.get("cv_version", ""),
            "channel": channel or previous.get("channel", ""),
        }
        cols = ", ".join(self.COLUMNS)
        updates = ", ".join(f"{c} = excluded.{c}" for c in self.COLUMNS if c != "id")
        with closing(self._db()) as con, con:
            con.execute(f"INSERT INTO applications ({cols}) VALUES ({', '.join('?' * len(self.COLUMNS))}) "
                        f"ON CONFLICT(id) DO UPDATE SET {updates}", [row[c] for c in self.COLUMNS])
        return row
