"""Quality evaluation of the generated (tailored) CVs: AIGC evaluation with an A/B design.

For every job with a tailored CV, the ORIGINAL CV and the TAILORED CV are both scored for that job:
- by an LLM judge with a rubric (relevance and clarity, 1-5), and
- by you, blind: the two versions are saved as A and B in random order, so you don't know which is which.
Then: does tailoring really help (sign test on wins vs losses), does the judge agree with you
(exact agreement, within one point, quadratic-weighted kappa), and the badcases to read.
Faithfulness is not judged here: the code checks in build_cv already guarantee it word for word.
"""
import csv
import hashlib
import shutil
from pathlib import Path

from jobagent.cvdoc import cv_plain_text
from jobagent.llm import structured
from jobagent.schemas import CVDocument, CVScore
from jobagent.stats import sign_test, weighted_kappa

JUDGE_PROMPT = """You are a strict recruiter. Score this CV for the job below, using only the CV text.
relevance 1-5: 5 = the job's main duties and must-haves are clearly backed by specific evidence near the top;
3 = relevant evidence exists but is buried or vague; 1 = little visible fit.
clarity 1-5: 5 = most relevant first, concise, easy to scan; 1 = cluttered or badly ordered.
Do not reward keywords without evidence. Give one sentence naming the strongest and weakest point.

Job: {title} at {company}
Duties and must-haves:
{items}

CV:
{cv}"""

SHEET_FIELDS = ["job_id", "company", "title", "a_file", "b_file", "a_relevance", "b_relevance", "notes"]


def cv_pairs(store, original_text: str, original_pdf: Path | None = None) -> list[dict]:
    """Jobs that have a tailored CV, with the job items, both CV texts and (if there) both PDFs."""
    out = []
    for path in sorted(store.reports_dir.glob("*-cv.json")):
        job = store.get(path.name[:-len("-cv.json")])
        if not job:
            continue
        a = job["analysis"]
        tailored = cv_plain_text(CVDocument.model_validate_json(path.read_text(encoding="utf-8")))
        out.append({"job_id": job["id"], "company": a.get("company") or "Unknown company", "title": a["title"],
                    "items": [r["text"] for r in a["requirements"] if r["type"] in ("duty", "must")],
                    "original": original_text, "tailored": tailored,
                    "original_pdf": original_pdf if original_pdf and original_pdf.suffix.lower() == ".pdf" else None,
                    "tailored_pdf": p if (p := store.reports_dir / f"{job['id']}-cv.pdf").exists() else None})
    return out


def _tailored_is_a(job_id: str) -> bool:
    return int(hashlib.sha1(job_id.encode()).hexdigest(), 16) % 2 == 0  # stable "coin flip" per job


def export_sheet(pairs: list[dict], eval_dir: Path) -> tuple[int, int]:
    """Blind rating sheet eval/cv_ratings.csv + the CV texts as eval/cv_blind/<job>-A.txt / -B.txt.
    The answer key (which is tailored) is in eval/cv_blind/key.csv: don't open it before rating."""
    blind = eval_dir / "cv_blind"
    blind.mkdir(parents=True, exist_ok=True)
    sheet = eval_dir / "cv_ratings.csv"
    old = {}
    if sheet.exists():
        with open(sheet, encoding="utf-8-sig", newline="") as f:
            old = {r["job_id"]: r for r in csv.DictReader(f)}
    rows, key, new = [], [], 0
    for p in pairs:
        t_is_a = _tailored_is_a(p["job_id"])
        texts = {"A": p["tailored"] if t_is_a else p["original"], "B": p["original"] if t_is_a else p["tailored"]}
        job_text = f"{p['company']} – {p['title']}\n\nDuties and must-haves:\n" + "\n".join(f"- {i}" for i in p["items"])
        (blind / f"{p['job_id']}-job.txt").write_text(job_text, encoding="utf-8")
        for v, text in texts.items():
            (blind / f"{p['job_id']}-{v}.txt").write_text(text, encoding="utf-8")
        # Rate the PDFs when both exist: same layout, so the look does not give away which one is tailored
        # (the .txt of the original comes from PDF extraction and looks messier).
        pdfs = {"A": p.get("tailored_pdf") if t_is_a else p.get("original_pdf"),
                "B": p.get("original_pdf") if t_is_a else p.get("tailored_pdf")}
        use_pdf = all(pdfs.values())
        if use_pdf:
            for v, src in pdfs.items():
                shutil.copyfile(src, blind / f"{p['job_id']}-{v}.pdf")
        prev = old.get(p["job_id"], {})
        new += p["job_id"] not in old
        rows.append({"job_id": p["job_id"], "company": p["company"], "title": p["title"],
                     "a_file": f"cv_blind/{p['job_id']}-A.{'pdf' if use_pdf else 'txt'}",
                     "b_file": f"cv_blind/{p['job_id']}-B.{'pdf' if use_pdf else 'txt'}",
                     "a_relevance": prev.get("a_relevance", ""), "b_relevance": prev.get("b_relevance", ""),
                     "notes": prev.get("notes", "")})
        key.append({"job_id": p["job_id"], "tailored": "A" if t_is_a else "B"})
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(blind / "key.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "tailored"])
        w.writeheader()
        w.writerows(key)
    return len(rows), new


def judge(pairs: list[dict], llm) -> list[dict]:
    """LLM rubric scores for the original and the tailored CV of every job (2 calls per job)."""
    scorer = structured(llm, CVScore)
    rows = []
    for p in pairs:
        for version in ("original", "tailored"):
            s: CVScore = scorer.invoke(JUDGE_PROMPT.format(
                title=p["title"], company=p["company"], items="\n".join(f"- {i}" for i in p["items"]),
                cv=p[version]))
            rows.append({"job_id": p["job_id"], "version": version, "relevance": s.relevance,
                         "clarity": s.clarity, "reason": s.reason})
    return rows


def human_scores(eval_dir: Path) -> dict[tuple[str, str], int]:
    """{(job_id, "original"|"tailored"): your relevance score}, decoded with the blind key."""
    sheet, key_file = eval_dir / "cv_ratings.csv", eval_dir / "cv_blind" / "key.csv"
    if not (sheet.exists() and key_file.exists()):
        return {}
    with open(key_file, encoding="utf-8", newline="") as f:
        key = {r["job_id"]: r["tailored"] for r in csv.DictReader(f)}
    out = {}
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["job_id"] not in key:
                continue
            for v in ("A", "B"):
                raw = (r.get(f"{v.lower()}_relevance") or "").strip()
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    out[(r["job_id"], "tailored" if key[r["job_id"]] == v else "original")] = int(raw)
    return out


def _ab(scores: dict[tuple[str, str], int], jobs: list[str]) -> dict:
    both = [j for j in jobs if (j, "original") in scores and (j, "tailored") in scores]
    wins = sum(scores[(j, "tailored")] > scores[(j, "original")] for j in both)
    losses = sum(scores[(j, "tailored")] < scores[(j, "original")] for j in both)
    mean = lambda v: sum(scores[(j, v)] for j in both) / len(both) if both else None
    return {"n": len(both), "original": mean("original"), "tailored": mean("tailored"),
            "wins": wins, "losses": losses, "ties": len(both) - wins - losses, "p": sign_test(wins, losses),
            "worse": [j for j in both if scores[(j, "tailored")] < scores[(j, "original")]]}


def report(judge_rows: list[dict], human: dict[tuple[str, str], int], titles: dict[str, str]) -> str:
    j_scores = {(r["job_id"], r["version"]): int(r["relevance"]) for r in judge_rows}
    reasons = {(r["job_id"], r["version"]): r["reason"] for r in judge_rows}
    jobs = sorted({r["job_id"] for r in judge_rows})
    fmt = lambda x: "–" if x is None else f"{x:.2f}"
    lines = ["# Generated CV quality (AIGC evaluation)", "",
             f"{len(jobs)} jobs with a tailored CV. Relevance for the job, 1-5.", "",
             "## Does tailoring help? (A/B: original vs tailored CV)", "",
             "| Rater | Jobs | Original | Tailored | Tailored better / worse / tie | Sign test p |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, scores in (("LLM judge", j_scores), ("You (blind)", human)):
        ab = _ab(scores, jobs)
        if ab["n"]:
            lines.append(f"| {name} | {ab['n']} | {fmt(ab['original'])} | {fmt(ab['tailored'])} | "
                         f"{ab['wins']} / {ab['losses']} / {ab['ties']} | {ab['p']:.3f} |")
    if not human:
        lines += ["", "No blind ratings from you yet: run `python evaluate_cv.py export`, rate eval/cv_ratings.csv."]

    shared = [k for k in j_scores if k in human]
    if shared:
        pairs = [(human[k], j_scores[k]) for k in shared]
        exact = sum(a == b for a, b in pairs) / len(pairs)
        near = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
        lines += ["", "## Does the judge agree with you?", "",
                  f"{len(pairs)} scores: exact agreement {exact:.0%}, within one point {near:.0%}, "
                  f"quadratic-weighted kappa {weighted_kappa(pairs):.2f}."]

    bad = []
    for j in jobs:
        for name, scores in (("judge", j_scores), ("you", human)):
            if (j, "tailored") in scores and (j, "original") in scores and scores[(j, "tailored")] < scores[(j, "original")]:
                bad.append((j, f"{name}: tailored {scores[(j, 'tailored')]} < original {scores[(j, 'original')]}",
                            reasons.get((j, "tailored"), "")))
        for v in ("original", "tailored"):
            if (j, v) in human and (j, v) in j_scores and abs(human[(j, v)] - j_scores[(j, v)]) >= 2:
                bad.append((j, f"{v}: you {human[(j, v)]}, judge {j_scores[(j, v)]}", reasons.get((j, v), "")))
    if bad:
        lines += ["", "## Badcases (read these first)", "", "| Job | What | Judge's reason |", "| --- | --- | --- |"]
        lines += [f"| {titles.get(j, j)} | {what} | {why.replace('|', '/')} |" for j, what, why in bad]
    return "\n".join(lines) + "\n"


def save_judge(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job_id", "version", "relevance", "clarity", "reason"])
        w.writeheader()
        w.writerows(rows)


def load_judge(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
