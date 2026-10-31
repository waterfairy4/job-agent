"""Job-search analytics:  python analytics.py
Funnel (saved -> applied -> interview -> offer), response rate per channel, an A/B test of tailored vs
original CVs (Fisher's exact test) and job segments (k-means clustering of the analyzed postings).
Writes data/reports/analytics-<date>.md and prints it.
"""
from datetime import date

from jobagent.analytics import report
from jobagent.store import JobStore

if __name__ == "__main__":
    store = JobStore()
    text = report(store.tracker(), store.all_jobs())
    path = store.write_report(f"analytics-{date.today().isoformat()}", "report", text)
    print(text)
    print(f"Saved to {path}")
