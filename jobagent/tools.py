"""The agent's tools. The LLM decides WHEN to call them; this code does the work.

Grounding rules enforced in code (not just in prompts):
- every verdict and CV bullet must cite evidence passages that really exist
- a "strong" verdict without valid evidence is downgraded to "partial"
- a CV bullet without valid evidence is dropped (no invented experience)
- save_job pauses the graph and waits for the user's approval
"""
import json
from typing import Literal

from langchain_core.tools import tool
from langgraph.types import interrupt

from jobagent import config
from jobagent.llm import structured
from jobagent.rag import ProfileIndex, format_hits
from jobagent.schemas import CVDraft, JobAnalysis, Verdict
from jobagent.store import JobStore

POINTS = {"strong": 1.0, "partial": 0.5, "missing": 0.0}
WEIGHT = {"must": 2.0, "nice": 1.0}

ANALYZE_PROMPT = """Extract the structured facts from this job description (English or Chinese).
Requirements: one skill or experience per item, short, in English, at most 12.
"must" = required qualifications; "nice" = preferred or bonus (优先, preferred, plus).
Skip generic soft skills unless the posting stresses them.

Job description:
{job}"""

JUDGE_PROMPT = """Does the candidate meet this ONE job requirement?
Use ONLY the numbered evidence from the candidate's own documents. Do not assume anything else.
strong = clearly demonstrated; partial = related or weaker; missing = no support.

Requirement: {req}

Evidence:
{evidence}"""

TAILOR_PROMPT = """Suggest 3-6 CV bullets that present the candidate for this job.
Rules:
- Use ONLY facts from the numbered evidence. Never invent projects, numbers or skills.
- Reuse the job's own vocabulary where it is honest to do so.
- Cite the evidence numbers each bullet is based on.
- Write bullets in the same language as the job description.

Job: {title} at {company}
Requirements: {reqs}

Evidence:
{evidence}"""


def _valid_refs(refs, n: int) -> list[int]:
    return sorted({r for r in refs if isinstance(r, int) and 1 <= r <= n})


def _cite(hits, refs) -> list[str]:
    return [f"{hits[r - 1].metadata['source']} p.{hits[r - 1].metadata['page']}" for r in refs]


def score(results: list[dict]) -> int:
    total = sum(WEIGHT[r["type"]] for r in results)
    got = sum(WEIGHT[r["type"]] * POINTS[r["verdict"]] for r in results)
    return round(100 * got / total) if total else 0


def build_tools(index: ProfileIndex, store: JobStore, llm) -> list:
    """Tools are built with closures so tests can pass in fake models and temp folders."""

    @tool
    def search_profile(query: str) -> str:
        """Search the user's CV, papers and notes. Use for any question about the user's
        own skills, projects or experience. Returns numbered passages with source and page."""
        hits = index.search(query, k=config.TOP_K)
        return format_hits(hits) if hits else "No matching passages."

    @tool
    def analyze_job(job_text: str) -> str:
        """Analyze a job description: company, title, location, deadline and requirements.
        Call this first whenever the user gives a job description. Returns a job_id for the other tools."""
        job_id = store.job_id(job_text)
        existing = store.get(job_id)
        if existing:  # same job pasted again: reuse, no extra LLM cost
            analysis = existing["analysis"]
        else:
            result: JobAnalysis = structured(llm, JobAnalysis).invoke(ANALYZE_PROMPT.format(job=job_text))
            result.requirements = [r for r in result.requirements if r.text.strip()][:12]
            analysis = result.model_dump()
            store.put({"id": job_id, "text": job_text, "analysis": analysis})
        return json.dumps({"job_id": job_id, **analysis}, ensure_ascii=False)

    @tool
    def match_job(job_id: str) -> str:
        """Check each requirement of an analyzed job against the user's profile, with cited evidence.
        Returns a match score, verdicts and gaps, and saves a report."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        judge = structured(llm, Verdict)
        results = []
        for req in job["analysis"]["requirements"]:
            hits = index.search(req["text"], k=config.MATCH_TOP_K)
            v: Verdict = judge.invoke(JUDGE_PROMPT.format(req=req["text"], evidence=format_hits(hits)))
            refs = _valid_refs(v.evidence, len(hits))
            verdict = v.verdict
            if verdict == "strong" and not refs:
                verdict = "partial"  # no citation, no "strong"
            results.append({**req, "verdict": verdict, "cited": _cite(hits, refs),
                            "reason": v.reason, "tip": "" if verdict == "strong" else v.tip})

        pct = score(results)
        job["match"] = {"score": pct, "results": results}
        store.put(job)
        a = job["analysis"]
        lines = [f"# Match: {a['company']} – {a['title']}", "", f"**Score: {pct}%** (must-have counts double)", "",
                 "| Requirement | Type | Verdict | Evidence | Why |", "| --- | --- | --- | --- | --- |"]
        for r in results:
            lines.append(f"| {r['text']} | {r['type']} | {r['verdict']} | {', '.join(r['cited']) or '–'} | {r['reason']} |")
        gaps = [r for r in results if r["verdict"] != "strong"]
        if gaps:
            lines += ["", "## Gaps", ""] + [f"- **{r['text']}** ({r['verdict']}): {r['tip']}" for r in gaps]
        path = store.write_report(job_id, "match", "\n".join(lines) + "\n")

        summary = [f"Score {pct}%. Report: {path}"]
        summary += [f"- [{r['verdict']}] {r['text']} ({r['type']}) {', '.join(r['cited'])} — {r['reason']}"
                    for r in results]
        return "\n".join(summary)

    @tool
    def tailor_cv(job_id: str) -> str:
        """Draft CV bullets tailored to an analyzed job, based only on evidence from the user's profile.
        Use when the user wants to prepare an application or improve their CV for this job."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        a = job["analysis"]
        seen, hits = set(), []
        for req in a["requirements"]:
            for h in index.search(req["text"], k=2):
                if h.metadata["chunk_id"] not in seen:
                    seen.add(h.metadata["chunk_id"])
                    hits.append(h)
        hits = hits[:12]
        draft: CVDraft = structured(llm, CVDraft).invoke(TAILOR_PROMPT.format(
            title=a["title"], company=a["company"],
            reqs="; ".join(r["text"] for r in a["requirements"]), evidence=format_hits(hits)))

        kept, dropped = [], 0
        for b in draft.bullets:
            refs = _valid_refs(b.evidence, len(hits))
            if refs:
                kept.append(f"- {b.bullet}\n  (for: {b.targets}; based on: {', '.join(_cite(hits, refs))})")
            else:
                dropped += 1  # not grounded in the profile: never show invented experience
        text = "\n".join(kept) or "No grounded suggestions could be made."
        if draft.keywords:
            text += "\n\nKeywords from the posting: " + ", ".join(draft.keywords)
        if dropped:
            text += f"\n\n({dropped} suggestion(s) removed because they had no supporting evidence.)"
        path = store.write_report(job_id, "cv", f"# CV suggestions: {a['company']} – {a['title']}\n\n{text}\n")
        return f"{text}\n\nSaved to {path}"

    @tool
    def save_job(job_id: str, status: Literal["saved", "applied"] = "saved") -> str:
        """Add an analyzed job to the user's application tracker. The user is asked to approve first."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        a = job["analysis"]
        # Pause the graph. The UI shows this request and resumes with the user's answer.
        decision = interrupt({
            "action": "save_job", "job_id": job_id, "status": status,
            "company": a["company"], "title": a["title"],
            "score": job.get("match", {}).get("score"),
        })
        if decision is True or (isinstance(decision, dict) and decision.get("approved")):
            row = store.track(job, status)
            return f"Saved to tracker: {row['company']} – {row['title']} ({status})."
        return "The user did not approve. Nothing was saved."

    return [search_profile, analyze_job, match_job, tailor_cv, save_job]
