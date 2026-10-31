"""Job-search analytics, with the same methods as marketing analytics:

- Funnel (用户转化漏斗): saved -> applied -> interview -> offer, with conversion between stages,
  overall and per channel (campus site, BOSS直聘, referral...).
- A/B test (A/B 测试): response rate (reached interview) of applications sent with a TAILORED CV vs the
  ORIGINAL CV, with Fisher's exact test (right for small counts).
- Segmentation (用户分层 / 聚类): the analyzed job postings are clustered by their requirements
  (k-means on TF-IDF vectors), each segment is named by its top terms and compared on match score
  and response rate: which kind of job fits you, and which answers.
"""
import math
import random
from collections import Counter
from datetime import date

from jobagent.rag import tokenize
from jobagent.store import STAGES

RESPONDED = ("interview", "offer")


# ---------- funnel ----------
def funnel(rows: list[dict]) -> list[dict]:
    """Count of applications that reached each stage, and conversion from the previous stage."""
    reached = [sum(STAGES.index(_stage(r)) >= i for r in rows) for i in range(len(STAGES))]
    out = []
    for i, stage in enumerate(STAGES):
        prev = reached[i - 1] if i else None
        out.append({"stage": stage, "count": reached[i],
                    "conversion": (reached[i] / prev) if prev else None})
    return out


def _stage(row: dict) -> str:
    stage = row.get("stage") or row.get("status")
    return stage if stage in STAGES else "applied" if row.get("applied_on") else "saved"


def response_rate(rows: list[dict]) -> tuple[int, int]:
    """(responded, applied) among applications that were sent."""
    sent = [r for r in rows if STAGES.index(_stage(r)) >= STAGES.index("applied")]
    return sum(_stage(r) in RESPONDED for r in sent), len(sent)


def by_group(rows: list[dict], key: str) -> list[dict]:
    groups = {}
    for r in rows:
        groups.setdefault(r.get(key) or "not recorded", []).append(r)
    out = []
    for name, rs in sorted(groups.items()):
        responded, sent = response_rate(rs)
        out.append({"group": name, "sent": sent, "responded": responded})
    return out


# ---------- A/B test ----------
def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]] (rows: A/B, columns: yes/no)."""
    n, row1, col1 = a + b + c + d, a + b, a + c

    def prob(x):
        return math.comb(col1, x) * math.comb(n - col1, row1 - x) / math.comb(n, row1)

    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    observed = prob(a)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= observed * (1 + 1e-9)))


def cv_ab_test(rows: list[dict]) -> dict:
    groups = {g["group"]: g for g in by_group(rows, "cv_version")}
    t = groups.get("tailored", {"sent": 0, "responded": 0})
    o = groups.get("original", {"sent": 0, "responded": 0})
    out = {"tailored": t, "original": o, "p": None}
    if t["sent"] and o["sent"]:
        out["p"] = fisher_exact(t["responded"], t["sent"] - t["responded"], o["responded"], o["sent"] - o["responded"])
    return out


# ---------- segmentation ----------
def _vectors(docs: list[list[str]]) -> tuple[list[dict], dict]:
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
    vecs = []
    for d in docs:
        tf = Counter(d)
        v = {t: tf[t] * idf[t] for t in tf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs, idf


def _cos(a: dict, b: dict) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def _mean(vs: list[dict]) -> dict:
    out = Counter()
    for v in vs:
        out.update(v)
    norm = math.sqrt(sum(x * x for x in out.values())) or 1.0
    return {t: x / norm for t, x in out.items()}


STOP = {"and", "or", "the", "of", "in", "with", "to", "for", "a", "an", "on", "at", "as", "is", "be", "such",
        "experience", "skills", "ability", "strong", "good", "familiar", "knowledge", "related", "using", "e.g",
        "including", "etc", "years", "work", "team", "degree", "relevant", "least", "one", "more", "like"}


def segments(jobs: list[dict], k: int | None = None, seed: int = 0) -> list[dict]:
    """Cluster analyzed jobs by their duty + must-have texts (spherical k-means, k-means++ start)."""
    docs = [[t for r in j["analysis"]["requirements"] if r["type"] in ("duty", "must")
             for t in tokenize(r["text"]) if t not in STOP and len(t) > 1] for j in jobs]
    keep = [i for i, d in enumerate(docs) if d]
    if len(keep) < 2:
        return []
    jobs, docs = [jobs[i] for i in keep], [docs[i] for i in keep]
    k = k or max(2, min(5, round(math.sqrt(len(jobs) / 2))))
    k = min(k, len(jobs))
    vecs, _ = _vectors(docs)
    rng = random.Random(seed)
    centers = [vecs[rng.randrange(len(vecs))]]
    while len(centers) < k:  # k-means++: far points are more likely to start a new cluster
        dist = [min(1 - _cos(v, c) for c in centers) for v in vecs]
        total = sum(dist)
        if total <= 0:
            break
        pick, acc = rng.random() * total, 0.0
        for v, dd in zip(vecs, dist):
            acc += dd
            if acc >= pick:
                centers.append(v)
                break
    assign = [0] * len(vecs)
    for _ in range(20):
        new = [max(range(len(centers)), key=lambda c: _cos(v, centers[c])) for v in vecs]
        if new == assign and _ > 0:
            break
        assign = new
        centers = [_mean([v for v, a in zip(vecs, assign) if a == c]) or centers[c] for c in range(len(centers))]
    out = []
    for c in range(len(centers)):
        members = [j for j, a in zip(jobs, assign) if a == c]
        if not members:
            continue
        top = [t for t, _ in sorted(centers[c].items(), key=lambda kv: -kv[1])[:4]]
        scores = [j["match"]["score"] for j in members if j.get("match", {}).get("score") is not None]
        out.append({"label": " · ".join(top), "jobs": members,
                    "avg_score": round(sum(scores) / len(scores)) if scores else None})
    return sorted(out, key=lambda s: -(s["avg_score"] or 0))


# ---------- report ----------
def report(rows: list[dict], jobs: list[dict]) -> str:
    pct = lambda x: "–" if x is None else f"{x:.0%}"
    lines = [f"# Job-search analytics ({date.today().isoformat()})", "",
             f"{len(rows)} jobs in the tracker, {len(jobs)} analyzed postings.", "",
             "## Funnel (转化漏斗)", "", "| Stage | Jobs | Conversion from previous |", "| --- | --- | --- |"]
    lines += [f"| {f['stage']} | {f['count']} | {pct(f['conversion'])} |" for f in funnel(rows)]

    lines += ["", "## Response rate by channel", "", "| Channel | Sent | Interview or offer | Rate |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {g['group']} | {g['sent']} | {g['responded']} | {pct(g['responded'] / g['sent'] if g['sent'] else None)} |"
              for g in by_group(rows, "channel") if g["sent"]]

    ab = cv_ab_test(rows)
    t, o = ab["tailored"], ab["original"]
    lines += ["", "## A/B test: tailored CV vs original CV", "",
              "| CV | Sent | Interview or offer | Rate |", "| --- | --- | --- | --- |",
              f"| tailored | {t['sent']} | {t['responded']} | {pct(t['responded'] / t['sent'] if t['sent'] else None)} |",
              f"| original | {o['sent']} | {o['responded']} | {pct(o['responded'] / o['sent'] if o['sent'] else None)} |", ""]
    if ab["p"] is None:
        lines.append("Not enough data yet: record `cv_version` for applications sent with each CV "
                     "(e.g. \"I applied to <job> with the tailored CV\").")
    else:
        lines.append(f"Fisher's exact test: p = {ab['p']:.3f}. " + (
            "The difference is statistically significant." if ab["p"] < 0.05 else
            "Not significant: with this few applications the difference could be chance."))
        lines.append("Caveat: not a randomized test: you chose which jobs got a tailored CV, so job fit may differ.")

    segs = segments(jobs)
    if segs:
        tracked = {r["id"]: r for r in rows}
        lines += ["", "## Job segments (聚类)", "",
                  "Analyzed postings clustered by their duties and must-have requirements.", "",
                  "| Segment (top terms) | Jobs | Avg. match | Applied | Interview or offer | Examples |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for s in segs:
            rs = [tracked[j["id"]] for j in s["jobs"] if j["id"] in tracked]
            responded, sent = response_rate(rs)
            examples = "; ".join(f"{j['analysis'].get('company') or '?'} {j['analysis']['title']}"[:40]
                                 for j in s["jobs"][:2])
            lines.append(f"| {s['label']} | {len(s['jobs'])} | {'–' if s['avg_score'] is None else str(s['avg_score']) + '%'} "
                         f"| {sent} | {responded} | {examples} |")
    return "\n".join(lines) + "\n"
