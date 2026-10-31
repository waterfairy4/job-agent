from jobagent.store import JobStore

JOB = {"id": "abc12345", "url": "https://example.com/job",
       "analysis": {"company": "", "title": "Eval Engineer", "deadline": "2026-11-30"},
       "match": {"score": 66}}


def test_dates_are_kept_across_status_changes(tmp_path):
    store = JobStore(str(tmp_path))
    row = store.track(JOB, "saved", today="2026-10-19")
    assert (row["saved_on"], row["applied_on"], row["company"]) == ("2026-10-19", "", "Unknown company")

    row = store.track(JOB, "applied", today="2026-10-21")
    assert (row["saved_on"], row["applied_on"]) == ("2026-10-19", "2026-10-21")

    row = store.track(JOB, "interview", today="2026-11-02")
    assert (row["saved_on"], row["applied_on"], row["updated"]) == ("2026-10-19", "2026-10-21", "2026-11-02")
    assert len(store.tracker()) == 1 and row["url"] == "https://example.com/job"


def test_applied_directly_and_old_rows(tmp_path):
    store = JobStore(str(tmp_path))
    row = store.track(JOB, "applied", today="2026-10-19")
    assert row["saved_on"] == row["applied_on"] == "2026-10-19"

    # an older version kept the tracker in tracker.json, and its rows have only "updated"
    old = JobStore(str(tmp_path / "old"))
    old.tracker_file.write_text('[{"id": "abc12345", "status": "saved", "updated": "2026-10-01"}]', encoding="utf-8")
    assert old.track(JOB, "applied", today="2026-10-20")["saved_on"] == "2026-10-01"   # migrated, date kept
    assert not old.tracker_file.exists() and old.tracker_file.with_suffix(".json.bak").exists()
    assert [r["status"] for r in JobStore(str(tmp_path / "old")).tracker()] == ["applied"]


def test_untrack(tmp_path):
    store = JobStore(str(tmp_path))
    store.track(JOB, "saved")
    store.track({**JOB, "id": "other"}, "saved")
    assert store.untrack("abc12345") is True
    assert [r["id"] for r in store.tracker()] == ["other"]
    assert store.untrack("abc12345") is False
