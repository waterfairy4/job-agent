"""Judge evaluation: how often does the LLM judge agree with YOUR verdicts?

1. export_labels(): writes a BLIND label sheet (the judge's verdicts are hidden, so they can't
   influence you). You fill in `your_label` with strong / partial / missing (or s / p / m).
2. evaluate(): compares your labels with the judge and reports agreement, Cohen's kappa,
   a confusion matrix, whether the judge is too generous or too strict, results per item type,
   the score difference per job, and every disagreement for error analysis.
"""
import csv
import json
from pathlib import Path

LABELS = ["strong", "partial", "missing"]
RANK = {"missing": 0, "partial": 1, "strong": 2}
ALIASES = {"s": "strong", "p": "partial", "m": "missing", "strong": "strong", "partial": "partial",
           "missing": "missing", "强": "strong", "部分": "partial", "无": "missing"}
FIELDS = ["key", "company", "title", "type", "item", "your_label", "notes"]
POINTS = {"strong": 1.0, "partial": 0.5, "missing": 0.0}
WEIGHT = {"must": 2.0, "duty": 1.5, "nice": 1.0}


def matched_jobs(jobs_dir: Path) -> list[dict]:
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(jobs_dir).glob("*.json"))]
    return [j for j in jobs if j.get("match", {}).get("results")]


def read_csv(path: Path) -> list[dict]:
    """Excel on Chinese Windows may save as GBK instead of UTF-8, so try both."""
    for enc in ("utf-8-sig", "gbk"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {path}: save it as 'CSV UTF-8' in Excel.")


def export_labels(jobs_dir: Path, out: Path) -> tuple[int, int]:
    """Write the blind label sheet. Labels you already filled in are kept. Returns (rows, new rows)."""
    existing = {r["key"]: r for r in read_csv(out)} if out.exists() else {}
    rows, new = [], 0
    for job in matched_jobs(jobs_dir):
        a = job["analysis"]
        for i, r in enumerate(job["match"]["results"]):
            key = f"{job['id']}:{i}"
            old = existing.get(key, {})
            new += key not in existing
            rows.append({"key": key, "company": a.get("company") or "Unknown company", "title": a["title"],
                         "type": r["type"], "item": r["text"],
                         "your_label": old.get("your_label", ""), "notes": old.get("notes", "")})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel shows Chinese correctly
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows), new


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Agreement corrected for chance: 1 = perfect, 0 = no better than chance."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in LABELS)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def _score(results: list[dict], key: str) -> int:
    total = sum(WEIGHT[r["type"]] for r in results)
    return round(100 * sum(WEIGHT[r["type"]] * POINTS[r[key]] for r in results) / total) if total else 0


def evaluate(jobs_dir: Path, labels_path: Path) -> dict:
    judge = {}
    for job in matched_jobs(jobs_dir):
        for i, r in enumerate(job["match"]["results"]):
            judge[f"{job['id']}:{i}"] = {**r, "job": job["id"],
                                         "name": f"{job['analysis'].get('company') or 'Unknown company'} – "
                                                 f"{job['analysis']['title']}"}
    rows, skipped = [], []
    for row in read_csv(labels_path):
        raw = (row.get("your_label") or "").strip().lower()
        if not raw:
            continue
        label = ALIASES.get(raw)
        if label is None or row["key"] not in judge:
            skipped.append(row["key"])
            continue
        rows.append({**judge[row["key"]], "human": label, "notes": row.get("notes", "")})

    pairs = [(r["human"], r["verdict"]) for r in rows]
    n = len(pairs)
    result = {"n": n, "skipped": skipped}
    if n == 0:
        return result
    result["agreement"] = sum(a == b for a, b in pairs) / n
    result["kappa"] = cohen_kappa(pairs)
    result["too_generous"] = sum(RANK[b] > RANK[a] for a, b in pairs) / n
    result["too_strict"] = sum(RANK[b] < RANK[a] for a, b in pairs) / n
    result["confusion"] = {h: {j: sum(a == h and b == j for a, b in pairs) for j in LABELS} for h in LABELS}
    result["by_type"] = {}
    for t in ("duty", "must", "nice"):
        sub = [(a, b) for (a, b), r in zip(pairs, rows) if r["type"] == t]
        if sub:
            result["by_type"][t] = {"n": len(sub), "agreement": sum(a == b for a, b in sub) / len(sub)}
    result["jobs"] = []
    for job_id in dict.fromkeys(r["job"] for r in rows):
        items = [r for r in rows if r["job"] == job_id]
        total = sum(1 for k in judge if k.startswith(job_id + ":"))
        if len(items) == total:  # only compare scores for fully labeled jobs
            result["jobs"].append({"name": items[0]["name"], "judge_score": _score(items, "verdict"),
                                   "your_score": _score(items, "human")})
    result["disagreements"] = [r for r in rows if r["human"] != r["verdict"]]
    return result


HISTORY_FIELDS = ["version", "date", "items", "agreement", "kappa", "too_generous", "too_strict", "settings"]


def append_history(path: Path, version: str, res: dict, settings: str, today: str) -> list[dict]:
    """Record one evaluation run (replacing an earlier run with the same version name)."""
    rows = [r for r in read_csv(path) if r["version"] != version] if path.exists() else []
    rows.append({"version": version, "date": today, "items": res["n"],
                 "agreement": f"{res['agreement']:.3f}", "kappa": f"{res['kappa']:.3f}",
                 "too_generous": f"{res['too_generous']:.3f}", "too_strict": f"{res['too_strict']:.3f}",
                 "settings": settings})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def history_markdown(rows: list[dict]) -> str:
    pct = lambda x: f"{100 * float(x):.0f}%"
    lines = ["| Version | Items | Agreement | Kappa | Too generous | Too strict | Settings |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['version']} | {r['items']} | {pct(r['agreement'])} | {float(r['kappa']):.2f} | "
              f"{pct(r['too_generous'])} | {pct(r['too_strict'])} | {r['settings']} |" for r in rows]
    return "\n".join(lines) + "\n"


def run_settings(jobs_dir: Path) -> str:
    """Describe how the current judge results were produced (v1 results have no settings stored)."""
    found = {json.dumps(j["match"].get("settings", {}), sort_keys=True) for j in matched_jobs(jobs_dir)}
    if found == {"{}"}:
        return "v1 judge, top-3, no expansion"
    if len(found) > 1:
        return "mixed settings: run rematch.py"
    s = json.loads(found.pop())
    return f"strict judge, top-{s['top_k']}, expansion {'on' if s['expand'] else 'off'}"


def report_markdown(res: dict) -> str:
    if res["n"] == 0:
        return "# Judge evaluation\n\nNo labels yet. Fill in `your_label` in the label sheet first.\n"
    pct = lambda x: f"{100 * x:.0f}%"
    lines = ["# Judge evaluation", "",
             f"**{res['n']} items labeled.** Agreement {pct(res['agreement'])}, "
             f"Cohen's kappa {res['kappa']:.2f}.", "",
             f"- Judge more generous than you: {pct(res['too_generous'])}",
             f"- Judge stricter than you: {pct(res['too_strict'])}", "",
             "## Confusion matrix (rows = you, columns = judge)", "",
             "| you \\ judge | strong | partial | missing |", "| --- | --- | --- | --- |"]
    lines += [f"| {h} | " + " | ".join(str(res["confusion"][h][j]) for j in LABELS) + " |" for h in LABELS]
    lines += ["", "## By item type", "", "| Type | Items | Agreement |", "| --- | --- | --- |"]
    lines += [f"| {t} | {v['n']} | {pct(v['agreement'])} |" for t, v in res["by_type"].items()]
    if res["jobs"]:
        lines += ["", "## Match score: judge vs you (fully labeled jobs)", "",
                  "| Job | Judge score | Your score |", "| --- | --- | --- |"]
        lines += [f"| {j['name']} | {j['judge_score']}% | {j['your_score']}% |" for j in res["jobs"]]
    if res["disagreements"]:
        lines += ["", "## Disagreements (for error analysis)", "",
                  "| Job | Item | You | Judge | Judge's reason | Your note |", "| --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {r['name']} | {r['text']} | {r['human']} | {r['verdict']} | {r['reason']} | {r['notes']} |"
                  for r in res["disagreements"]]
    if res["skipped"]:
        lines += ["", f"_Skipped {len(res['skipped'])} rows with an unknown label or key._"]
    return "\n".join(lines) + "\n"


# ---------- comparing versions (paired: the same items) ----------
RUN_FIELDS = ["key", "name", "item", "type", "human", "verdict"]


def save_run(path: Path, res: dict, labels_path: Path, jobs_dir: Path):
    """Keep every labeled item's verdict for this version, so later versions can be compared item by item."""
    rows = []
    judge = {f"{j['id']}:{i}": r for j in matched_jobs(jobs_dir) for i, r in enumerate(j["match"]["results"])}
    for row in read_csv(labels_path):
        human = ALIASES.get((row.get("your_label") or "").strip().lower())
        if human and row["key"] in judge:
            rows.append({"key": row["key"], "name": f"{row['company']} – {row['title']}", "item": row["item"],
                         "type": row["type"], "human": human, "verdict": judge[row["key"]]["verdict"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_run(eval_dir: Path, version: str) -> dict[str, dict]:
    """{item key: {human, verdict}} for one version: from eval/runs/judge-<v>.csv, or rebuilt from an
    older report (its disagreement table + the label sheet: every other labeled item agreed)."""
    snap = eval_dir / "runs" / f"judge-{version}.csv"
    if snap.exists():
        return {r["key"]: r for r in read_csv(snap)}
    report = eval_dir / f"report-{version}.md"
    if not report.exists():
        raise FileNotFoundError(f"No results for version {version}: run `python evaluate.py report {version}`")
    wrong = {}
    in_table = False
    for line in report.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Disagreements"):
            in_table = True
        elif line.startswith("## "):
            in_table = False
        elif in_table and line.startswith("| ") and not line.startswith("| Job") and not line.startswith("| ---"):
            cells = [c.strip() for c in line.strip("|").split(" | ")]
            if len(cells) >= 4:
                wrong[(cells[0], cells[1])] = cells[3]
    out = {}
    for row in read_csv(eval_dir / "labels.csv"):
        human = ALIASES.get((row.get("your_label") or "").strip().lower())
        if human:
            name = f"{row['company'] or 'Unknown company'} – {row['title']}"
            out[row["key"]] = {"key": row["key"], "name": name, "item": row["item"], "type": row["type"],
                               "human": human, "verdict": wrong.get((name, row["item"]), human)}
    return out


def compare_runs(a: dict[str, dict], b: dict[str, dict], name_a: str, name_b: str) -> str:
    from jobagent.stats import bootstrap_diff, mcnemar_exact, verdict_text

    keys = sorted(set(a) & set(b))
    items = [{"human": a[k]["human"], "a": a[k]["verdict"], "b": b[k]["verdict"]} for k in keys]
    acc = lambda rows, v: sum(r["human"] == r[v] for r in rows) / len(rows)
    kap = lambda rows, v: cohen_kappa([(r["human"], r[v]) for r in rows])
    test = mcnemar_exact([r["human"] == r["a"] for r in items], [r["human"] == r["b"] for r in items])
    ci_acc = bootstrap_diff(items, acc, "a", "b")
    ci_kap = bootstrap_diff(items, kap, "a", "b")
    pct = lambda x: f"{100 * x:+.1f} pts" if x else "0"
    return "\n".join([
        f"# Judge {name_a} vs {name_b} (paired, {len(items)} items)", "",
        f"| Metric | {name_a} | {name_b} | Difference | 95% CI |", "| --- | --- | --- | --- | --- |",
        f"| Agreement | {acc(items, 'a'):.1%} | {acc(items, 'b'):.1%} | {pct(ci_acc['diff'])} | "
        f"{pct(ci_acc['low'])} to {pct(ci_acc['high'])} |",
        f"| Cohen's kappa | {kap(items, 'a'):.3f} | {kap(items, 'b'):.3f} | {ci_kap['diff']:+.3f} | "
        f"{ci_kap['low']:+.3f} to {ci_kap['high']:+.3f} |", "",
        f"McNemar exact test: {test['b_only']} items right only in {name_b}, {test['a_only']} right only in "
        f"{name_a}; p = {test['p']:.3f}.", "", verdict_text(test["p"], ci_acc["low"], ci_acc["high"]), ""])
