"""The agent's tools. The LLM decides WHEN to call them; this code does the work.

Grounding rules enforced in code (not just in prompts):
- every verdict and CV bullet must cite evidence passages that really exist
- a "strong" verdict without valid evidence is downgraded to "partial";
  a "partial" without valid evidence becomes "missing"
- a CV bullet without valid evidence, or not aimed at a real job item, is dropped
- save_job pauses the graph and waits for the user's approval
"""
import json
import re
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool
from langgraph.types import interrupt

from jobagent import config
from jobagent.llm import structured
from jobagent.rag import ProfileIndex, format_hits
from jobagent import web
from jobagent.cvdoc import (CJK, OBJECTIVE, added_runs, cv_plain_text, cv_text, enforce_faithful, find_cv,
                            find_images, normalize_layout, restore_languages,
                            missing_languages, render_docx, unknown_numbers, visual_length)
from jobagent.cvpdf import render_pdf
from jobagent.schemas import (ATSTerms, CompanyBrief, CVCritique, CVDocument, CVDraft, GapPlan, JobAnalysis,
                              JobCandidates, SearchPlan, SearchTerms, Verdict)
from jobagent.store import JobStore

ANALYSIS_VERSION = 2  # bump when the analysis format changes, so cached analyses are redone
POINTS = {"strong": 1.0, "partial": 0.5, "missing": 0.0}
WEIGHT = {"must": 2.0, "duty": 1.5, "nice": 1.0}
TYPE_ORDER = {"duty": 0, "must": 1, "nice": 2}
SECTION = {"duty": "Responsibilities: can you do this job?",
           "must": "Required qualifications",
           "nice": "Preferred (bonus)"}
UNKNOWN_COMPANY = "Unknown company"

ANALYZE_PROMPT = """Extract the structured facts from this job posting (English or Chinese).

Items (each short, specific, written in English):
- "duty": the main responsibilities, i.e. what the person will DO (at most 6)
- "must": required qualifications and skills (at most 8)
- "nice": preferred or bonus items (优先, preferred, plus) (at most 5)
Split combined sentences into separate items: "do A, B and C" becomes three items when A, B and C
are different skills or tasks. Skip generic soft skills unless the posting stresses them.
Leave company empty if the posting does not name it; never guess.
Title and location: copy them exactly as the posting writes them (do not translate).

Job posting:
{job}"""

JUDGE_PROMPT = """Does the candidate's background fit this ONE job item?
For a responsibility: has the candidate done this kind of work? For a qualification: do they have it?
Use ONLY the numbered evidence from the candidate's own documents. Do not assume anything else.
- strong: the evidence directly shows this item.
- partial: the evidence shows a closely related skill, or clearly covers part of a combined item.
  You must cite that evidence.
- missing: no evidence, or only loosely related evidence. If your reason says the evidence does
  not show this item, the verdict is missing, not partial.

Job item ({kind}): {req}

Evidence:
{evidence}"""

EXPAND_PROMPT = """Turn this job item into search terms for finding evidence in a CV.
Give 4-8 short, concrete terms that a CV would literally contain: tool names, technique names,
typical section words. Include both English and Chinese terms (the CV may be in either).
Example: "familiar with deep learning frameworks" -> PyTorch, TensorFlow, JAX, 深度学习框架, 机器学习

Job item: {item}"""

TAILOR_PROMPT = """Write CV bullets that present the candidate for this job.
Rules:
- Write every bullet and keyword in {language}.
- Each bullet targets exactly ONE numbered job item below; prefer responsibilities (duty) and must-haves.
- Rephrase the candidate's evidence in the posting's own vocabulary where that stays true.
- Use ONLY facts from the numbered evidence. Never invent projects, numbers or skills.
- Do not repeat the same fact in two bullets. 3-6 bullets.

Job: {title} at {company}

Job items:
{items}

Evidence:
{evidence}"""


CV_RULES = """Rules:
- Copy every bullet WORD FOR WORD from the CV. Do not add words, keywords or brackets to a bullet,
  and do not merge bullets. You may only choose which bullets to keep and in what order.
- Only the objective line (求职意向/Objective) may be shortened or reworded, using the CV's own words.
- Never add any fact, skill, tool, number, date, title, goal or name that is not in the CV.
- Never remove the Languages or Honors items. Keep every number exactly as written.
- Copy the name and contact details exactly.
- Copy every date exactly as written (keep "2026 年"; do not shorten it to "2026")."""

BUILD_CV_PROMPT = """Tailor the candidate's CV for this job, in {language}.
You may: reorder sections, entries and bullets so the most relevant come first; shorten the objective;
leave out project/research bullets that are irrelevant to this job, if needed to fit one page.
{rules}
{preferences}
Keep the CV's own structure and section types (e.g. 求职意向/Objective, Education, Papers, Projects,
Internship, Skills as a table, Languages & Honors). Entries: title + date on line 1, subtitle + meta
(e.g. GPA) on line 2, then bullets. Use layout "table" for skills and "text" for the objective and
Languages & Honors (plain paragraphs; an entry title like "语言：" is a bold label).
The objective may be reworded for this job (e.g. internship instead of full-time) but only with facts
from the CV; keep its sentences complete.
Length: at most about {budget} characters (Chinese characters count double), so it fits on ONE page.
The CV text was extracted from a PDF, so its line order may be jumbled; reconstruct the structure.
{context}
Job: {title} at {company}
Most important for this job: {items}

CV text:
{cv}"""

CRITIQUE_PROMPT = """You are a strict reviewer of a CV tailored for a job. Compare the draft with the
candidate's ORIGINAL CV. Report concrete issues only:
- unsupported: a fact, goal, skill, number or claim not in the original CV
- stretched: wording that changes what a project or role actually was
- irrelevant: content that wastes space for this job
- too_long: if the draft is over the length budget, say what to cut (least relevant first)
- ordering: the most relevant entries for this job are not first
Do not suggest adding keywords or any words to bullets: bullets must stay word for word.
Return an empty list if it is good.

Job: {title} at {company}. Most important: {items}
Length: draft {length} / budget {budget}.

ORIGINAL CV:
{cv}

DRAFT (JSON):
{draft}"""

REVISE_PROMPT = """Revise the tailored CV draft to fix every issue below. Same language ({language}).
{rules}
{preferences}
Length: at most about {budget} characters (Chinese characters count double).

Issues:
{issues}

ORIGINAL CV:
{cv}

DRAFT (JSON):
{draft}"""

SEARCH_PLAN_PROMPT = """Plan 2-3 web searches to learn about this employer and team for a job applicant:
what the team builds, its products and technical focus, and recent news. Mix Chinese and English
queries when the company is Chinese. Short queries, like a person would type.

Company: {company}
Job title: {title}
Job focus: {items}"""

JOB_SEARCH_PROMPT = """Plan 3 web searches that find individual, currently open job postings for this
request. The searches run on job boards, so don't name sites. Use words that appear ON a posting page
(岗位职责, 任职要求, 校招, 2027届, 27届, 招聘) and avoid words of discussion posts and news (经验, 面经,
总结, 形势). Mostly Chinese for jobs in China. Use different role names across the 3 queries (e.g.
大模型评测, 大模型算法, 模型评估, 数据策略) when the request allows. Put the city in at most one query.
Short queries (under 30 characters).

Request: {request}"""

JOB_FILTER_PROMPT = """These are web search results. The user is looking for: {request}
For each result decide:
- is_posting: is it ONE specific job (a job-detail page, also on job boards like 牛客/实习僧/BOSS)?
  Blogs, news, salary pages, forum threads and pages listing many companies are not.
- is_open: false only if the result says it is closed or clearly old.
- relevant: does the ROLE fit? City and graduation year are preferences only: never reject for them,
  set fits_city_year instead. fits_city_year is also false when the user wants campus / new-graduate
  roles (校招, 应届, 2027届) but the result asks for years of work experience (e.g. 3-5年). When the snippet is too short to tell, prefer true.
- reason: a few words. Company and title only if the result states them. Today is {today}.

{results}"""

BRIEF_PROMPT = """Write a short research brief for a candidate applying to this job, using ONLY the
numbered search results. Every overview/recent point must cite the result numbers it comes from.
Skip anything the results do not support. "emphasize": topics from the results that connect to the job
(not claims about the candidate). "questions": 3 smart questions to ask in an interview. Write in {language}.

Company: {company}. Job: {title}. Job focus: {items}

Search results:
{results}"""

GAP_PROMPT = """The candidate wants this job. Below are the job items where their profile shows a gap.
For each gap, propose ONE action:
- add_evidence: they may already have this but it is not written down -> what to add to their profile
- reframe: existing experience covers it but is presented poorly -> how to present it
- build: a small, realistic project that creates proof (prefer projects that reuse their existing work)
- learn: something to study well enough to discuss in an interview
Be concrete and realistic; prefer the cheapest action that creates real proof. Most important first.
Honesty rules (the candidate will copy your wording into applications):
- add_evidence and reframe may only reword what the background excerpts show. Never upgrade it: no
  "internal launch", "pilot", "production", "users", "team lead" or any number, duration or result that
  the excerpts do not state. A portfolio demo stays a demo; a course project stays a course project.
- If an action needs a number the candidate has not measured, say "measure X first", never give the number.
- Never claim the candidate already did a build or learn action; those are suggestions for future work.

Job: {title} at {company}

Candidate's background (excerpts):
{background}

Gaps:
{gaps}"""


ATS_PROMPT = """List the key terms an applicant tracking system (ATS) would search for in CVs for this job.
Copy each term EXACTLY as the posting writes it (same language and characters): skills, tools, methods and
task words (e.g. 大模型评测, 评测指标, Python, 数据分析). 8-20 terms, most important first.
Leave out degrees, majors, years of experience, soft skills, company or team names, locations and dates.

Job posting:
{job}"""


def ats_contains(text: str, term: str) -> bool:
    """Is `term` in `text`? English terms match whole words ("AI" is not in "OpenAI"); Chinese terms ignore spaces."""
    term = term.strip().lower()
    if not term:
        return False
    if term.isascii():
        norm = " ".join(text.lower().split())
        pattern = r"(?<![a-z0-9])" + re.escape(" ".join(term.split())) + r"(?![a-z0-9])"
        return re.search(pattern, norm) is not None
    return "".join(term.split()) in "".join(text.lower().split())


def ats_terms(job: dict, llm) -> list[str]:
    """The posting's key terms, each checked to really occur in the posting (cached in the job file)."""
    if "ats_terms" not in job:
        raw = structured(llm, ATSTerms).invoke(ATS_PROMPT.format(job=job["text"][:6000])).terms
        seen, kept = set(), []
        for term in raw:
            term = term.strip(" ，,。；;、:：")
            if len(term) >= 2 and term.lower() not in seen and ats_contains(job["text"], term):
                seen.add(term.lower())
                kept.append(term)
        job["ats_terms"] = kept[:20]
    return job["ats_terms"]


def ats_check(terms: list[str], cv_text: str, chunks) -> dict:
    """Which terms the CV has; which missing ones the user's documents show (term -> source); which neither."""
    present = [t for t in terms if ats_contains(cv_text, t)]
    supported, unsupported = {}, []
    for term in terms:
        if term in present:
            continue
        src = next((c.metadata.get("source", "profile") for c in chunks if ats_contains(c.page_content, term)), None)
        if src:
            supported[term] = src
        else:
            unsupported.append(term)
    return {"terms": terms, "present": present, "supported": supported, "unsupported": unsupported}


def add_keywords(cv, terms: list[str]) -> str:
    """Add the terms as one row of the skills table (a new table section if the CV has none). Returns the label."""
    from jobagent.schemas import CVEntry, CVSection

    chinese = CJK.search(cv_plain_text(cv)) is not None
    label, sep = ("关键词", "、") if chinese else ("Keywords", ", ")
    entry = CVEntry(title=label, bullets=[sep.join(terms)])
    table = next((sec for sec in cv.sections if sec.layout == "table"), None)
    if table:
        table.entries.append(entry)
    else:
        cv.sections.append(CVSection(title="技能" if chinese else "Skills", layout="table", entries=[entry]))
    return label


def _valid_refs(refs, n: int) -> list[int]:
    return sorted({r for r in refs if isinstance(r, int) and 1 <= r <= n})


def _cite(hits, refs) -> list[str]:
    return [f"{hits[r - 1].metadata['source']} p.{hits[r - 1].metadata['page']}" for r in refs]


def company_name(analysis: dict) -> str:
    return analysis.get("company", "").strip() or UNKNOWN_COMPANY


def score(results: list[dict]) -> int:
    total = sum(WEIGHT[r["type"]] for r in results)
    got = sum(WEIGHT[r["type"]] * POINTS[r["verdict"]] for r in results)
    return round(100 * got / total) if total else 0


def search_query(job: dict, item: str, llm, expand: bool) -> str:
    """The retrieval query for one job item: the item itself, plus LLM-made search terms.
    Terms are cached in the job file, so re-matching costs no extra calls and stays reproducible."""
    if not expand:
        return item
    cache = job.setdefault("search_terms", {})
    if item not in cache:
        cache[item] = structured(llm, SearchTerms).invoke(EXPAND_PROMPT.format(item=item)).terms[:8]
    return item + " " + " ".join(cache[item])


# Degree / major items: the general search ranks skill lists that mention "统计" or "computer" above the
# education section, so the match said "only a software-engineering bachelor's". Such items also get the
# education chunks, found by a query in the words an education section is written in.
EDUCATION_ITEM = re.compile(r"degree|bachelor|master|ph\.?d|major|graduat|学历|学位|本科|硕士|博士|专业|毕业",
                            re.IGNORECASE)
EDUCATION_QUERY = "教育背景 教育经历 Education 大学 University 硕士 学士 Master Bachelor 专业 GPA"


def evidence_hits(index: ProfileIndex, query: str, item: str, k: int) -> list:
    hits = index.search(query, k=k)
    if EDUCATION_ITEM.search(item):
        seen = {h.metadata.get("chunk_id") for h in hits}
        extra = [h for h in index.search(EDUCATION_QUERY, k=2, mode="bm25")
                 if h.metadata.get("chunk_id") not in seen]
        hits = extra + hits[:max(k - len(extra), 1)]
    return hits


def run_match(job: dict, index: ProfileIndex, store: JobStore, llm,
              expand: bool | None = None, top_k: int | None = None):
    """Judge every item of an analyzed job; save results and a report. Returns (score, report path, results)."""
    expand = config.QUERY_EXPANSION if expand is None else expand
    top_k = top_k or config.MATCH_TOP_K
    judge = structured(llm, Verdict)
    results = []
    for req in job["analysis"]["requirements"]:
        hits = evidence_hits(index, search_query(job, req["text"], llm, expand), req["text"], top_k)
        kind = "responsibility" if req["type"] == "duty" else "qualification"
        v: Verdict = judge.invoke(JUDGE_PROMPT.format(kind=kind, req=req["text"], evidence=format_hits(hits)))
        refs = _valid_refs(v.evidence, len(hits))
        verdict = v.verdict
        if not refs and verdict == "strong":
            verdict = "partial"   # no citation, no "strong"
        elif not refs and verdict == "partial":
            verdict = "missing"   # "related" must point to something real
        results.append({**req, "verdict": verdict, "cited": _cite(hits, refs),
                        "reason": v.reason, "tip": "" if verdict == "strong" else v.tip})

    pct = score(results)
    job["match"] = {"score": pct, "results": results, "settings": {"expand": expand, "top_k": top_k}}
    store.put(job)
    a = job["analysis"]
    lines = [f"# Match: {company_name(a)} – {a['title']}", "",
             f"**Score: {pct}%** (weights: must-have ×2, responsibility ×1.5, preferred ×1)"]
    for t in ("duty", "must", "nice"):
        rows = [r for r in results if r["type"] == t]
        if rows:
            lines += ["", f"## {SECTION[t]}", "", "| Item | Verdict | Evidence | Why |", "| --- | --- | --- | --- |"]
            lines += [f"| {r['text']} | {r['verdict']} | {', '.join(r['cited']) or '–'} | {r['reason']} |"
                      for r in rows]
    gaps = [r for r in results if r["verdict"] != "strong"]
    if gaps:
        lines += ["", "## Gaps", ""] + [f"- **{r['text']}** ({r['type']}, {r['verdict']}): {r['tip']}"
                                        for r in gaps]
    path = store.write_report(job["id"], "match", "\n".join(lines) + "\n")
    return pct, path, results


def run_tailor(job: dict, index: ProfileIndex, store: JobStore, llm):
    """Draft grounded CV bullets; keeps them in the job file for build_cv. Returns (text, report path)."""
    a = job["analysis"]
    items = [r for r in a["requirements"] if r["type"] in ("duty", "must")] or a["requirements"]
    seen, hits = set(), []
    for req in items:
        for h in index.search(search_query(job, req["text"], llm, config.QUERY_EXPANSION), k=2):
            if h.metadata["chunk_id"] not in seen:
                seen.add(h.metadata["chunk_id"])
                hits.append(h)
    hits = hits[:12]
    draft: CVDraft = structured(llm, CVDraft).invoke(TAILOR_PROMPT.format(
        language=a.get("language") or "the language of the job posting",
        title=a["title"], company=company_name(a),
        items="\n".join(f"{i}. ({r['type']}) {r['text']}" for i, r in enumerate(items, start=1)),
        evidence=format_hits(hits)))

    kept, bullets, dropped = [], [], 0
    for b in draft.bullets:
        refs = _valid_refs(b.evidence, len(hits))
        if refs and 1 <= b.target <= len(items):
            target = items[b.target - 1]
            bullets.append(b.bullet)
            kept.append(f"- {b.bullet}\n  (for {target['type']}: {target['text']}; "
                        f"based on: {', '.join(_cite(hits, refs))})")
        else:
            dropped += 1  # ungrounded or not aimed at a real job item
    text = "\n".join(kept) or "No grounded suggestions could be made."
    if draft.keywords:
        text += "\n\nKeywords from the posting: " + ", ".join(draft.keywords)
    if dropped:
        text += f"\n\n({dropped} suggestion(s) removed: no supporting evidence or no matching job item.)"
    job["cv_bullets"] = bullets
    store.put(job)
    path = store.write_report(job["id"], "cv", f"# CV suggestions: {company_name(a)} – {a['title']}\n\n{text}\n")
    return text, path


def _brief_context(job: dict) -> str:
    r = job.get("research")
    if not r or not r.get("emphasize"):
        return ""
    return ("Employer research (use only to decide what to emphasize; never write these as the "
            "candidate's experience): " + "; ".join(r["emphasize"]) + "\n")


def _preferences() -> str:
    path = Path(config.CV_PREFERENCES_FILE)
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return "The candidate's own standing rules (always follow them):\n" + text
    return ""


def _cjk_share(text: str) -> float:
    letters = [c for c in text if not c.isspace() and not c.isdigit()]
    return len(CJK.findall("".join(letters))) / max(len(letters), 1)


def _is_chinese(text: str) -> bool:
    """Mostly Chinese? An English text with a few Chinese names stays far below 0.2; a Chinese CV with many
    English technical terms stays far above it."""
    return _cjk_share(text) >= 0.2


NOW = ("至今", "现在", "present", "now", "current")


def _newest(date: str) -> int | None:
    """Latest year in a CV date ('2026 年春 – 至今' -> ongoing = 9999), None without a year."""
    if any(w in date.lower() for w in NOW):
        return 9999
    years = [int(y) for y in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", date)]
    return max(years) if years else None


def sort_by_date(cv) -> list[str]:
    """Newest first inside each section (like the original CV). Sections where an entry has no date
    are left as they are. Returns the titles of sections whose order changed."""
    moved = []
    for sec in cv.sections:
        keys = [_newest(e.date) for e in sec.entries]
        if len(keys) < 2 or None in keys:
            continue
        order = sorted(range(len(keys)), key=lambda i: -keys[i])   # stable: same year keeps its order
        if order != list(range(len(keys))):
            sec.entries = [sec.entries[i] for i in order]
            moved.append(sec.title)
    return moved


def objective_status(cv, source: str) -> str:
    """'same' if the objective uses only the CV's own wording, 'reworded' if not, '' if there is none."""
    for sec in cv.sections:
        if any(k in sec.title.lower() for k in OBJECTIVE):
            text = "\n".join(p for e in sec.entries for p in (e.title, e.subtitle, *e.bullets) if p)
            return "reworded" if added_runs(text, source) else "same"
    return ""


def run_build_cv(job: dict, cv_source: str, llm, language: str = "") -> dict:
    """Draft -> critique -> revise -> code checks. Returns a dict with the CV and what was checked."""
    a = job["analysis"]
    language = language or a.get("language") or "the language of the job posting"
    budget = int(visual_length(cv_source) * 1.05)  # the original is one page, so stay close to it
    items = "; ".join(r["text"] for r in a["requirements"] if r["type"] in ("duty", "must"))
    prefs = _preferences()
    common = dict(title=a["title"], company=company_name(a), items=items, budget=budget)
    to_cv = structured(llm, CVDocument)

    draft: CVDocument = to_cv.invoke(BUILD_CV_PROMPT.format(
        language=language, rules=CV_RULES, preferences=prefs, context=_brief_context(job),
        cv=cv_source, **common))

    def deterministic(cv) -> list[str]:
        found = []
        n = unknown_numbers(cv_plain_text(cv), cv_source)
        if n:
            found.append(f"unsupported: these numbers are not in the original CV: {', '.join(n)}")
        lost = missing_languages(cv_plain_text(cv), cv_source)
        if lost:
            found.append(f"unsupported removal: keep these languages from the original CV: {', '.join(lost)}")
        length = visual_length(cv_plain_text(cv))
        if length > budget:
            found.append(f"too_long: {length} > budget {budget}; leave out the least relevant project bullets")
        return found

    try:
        critique: CVCritique = structured(llm, CVCritique).invoke(CRITIQUE_PROMPT.format(
            cv=cv_source, draft=draft.model_dump_json(), length=visual_length(cv_plain_text(draft)), **common))
        review = [f"{i.kind} ({i.where}): {i.problem} -> {i.fix}" for i in critique.issues]
    except Exception:  # a failed review must not block the CV: the code checks below still run
        review = []
    issues = review + deterministic(draft)
    final = draft
    if issues:
        try:
            final = to_cv.invoke(REVISE_PROMPT.format(
                language=language, rules=CV_RULES, preferences=prefs, budget=budget, cv=cv_source,
                issues="\n".join(f"- {i}" for i in issues), draft=draft.model_dump_json()))
        except Exception:  # keep the draft; the code checks below still clean it
            final = draft

    # Code-level guarantee: lines copied from the CV keep exactly the CV's words.
    # (Skipped for a translated CV, where word-for-word comparison makes no sense.)
    translated = abs(_cjk_share(cv_plain_text(final)) - _cjk_share(cv_source)) > 0.3
    changes = [] if translated else enforce_faithful(final, cv_source)
    if not translated:
        changes += [f"put back the language {lang}, which the draft had dropped" for lang in restore_languages(final, cv_source)]
    resorted = sort_by_date(final)
    normalize_layout(final)
    return {"cv": final, "issues": issues, "changes": changes, "translated": translated, "resorted": resorted,
            "objective": "" if translated else objective_status(final, cv_source),
            "numbers": unknown_numbers(cv_plain_text(final), cv_source),
            "languages": missing_languages(cv_plain_text(final), cv_source),
            "length": visual_length(cv_plain_text(final)), "budget": budget}


def run_research(job: dict, llm, store: JobStore, language: str = "") -> tuple[str, Path]:
    """Search the web about the employer/team and write a cited brief. Returns (text, report path)."""
    a = job["analysis"]
    items = "; ".join(r["text"] for r in a["requirements"] if r["type"] == "duty")[:400]
    plan: SearchPlan = structured(llm, SearchPlan).invoke(SEARCH_PLAN_PROMPT.format(
        company=company_name(a), title=a["title"], items=items))
    results, seen = [], set()
    for q in plan.queries[:3]:
        for r in web.search(q, count=5):
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)
    if not results:
        return "The web search returned no results.", None
    numbered = "\n\n".join(f"[{i}] {r['title']} ({r['date'] or 'no date'})\n{r['url']}\n{r['snippet'][:700]}"
                            for i, r in enumerate(results, start=1))
    brief: CompanyBrief = structured(llm, CompanyBrief).invoke(BRIEF_PROMPT.format(
        language=language or a.get("language") or "English", company=company_name(a),
        title=a["title"], items=items, results=numbered))

    def cited(points):
        out = []
        for p in points:
            refs = _valid_refs(p.sources, len(results))
            if refs:  # a point without a real source is dropped
                out.append(f"- {p.text} " + " ".join(f"[{n}]" for n in refs))
        return out

    overview, recent = cited(brief.overview), cited(brief.recent)
    lines = [f"# Research: {company_name(a)} – {a['title']}", "", "## Overview", *(overview or ["- (nothing sourced)"]),
             "", "## Recent", *(recent or ["- (nothing sourced)"]),
             "", "## Worth emphasizing", *[f"- {e}" for e in brief.emphasize],
             "", "## Questions to ask", *[f"- {q}" for q in brief.questions],
             "", "## Sources", *[f"[{i}] [{r['title']}]({r['url']}) {r['date']}" for i, r in enumerate(results, 1)]]
    text = "\n".join(lines)
    job["research"] = {"emphasize": brief.emphasize, "queries": plan.queries,
                       "sources": [r["url"] for r in results]}
    store.put(job)
    return text, store.write_report(job["id"], "research", text + "\n")


CLOSED = ("已截止", "已结束", "停止招聘", "职位已关闭", "已下线", "position closed", "no longer accepting",
          "expired")


SNIPPET_ENOUGH = 600  # characters of search summary that usually hold duties + requirements


def run_find_jobs(request: str, llm, limit: int, analyze_and_store, index, store, today: str):
    """Search -> filter -> read -> match. Returns (shortlist rows, notes)."""
    plan: SearchPlan = structured(llm, SearchPlan).invoke(JOB_SEARCH_PROMPT.format(request=request))
    queries = plan.queries[:3]
    results, seen = [], set()

    def add(found):
        for r in found:
            if r["url"] and r["url"] not in seen:
                seen.add(r["url"])
                results.append(r)

    # job boards first (one posting per page); the whole web only if they return little
    for q in queries:
        add(web.search(q, count=10, sites=config.JOB_SITES or None, recency=config.JOB_RECENCY))
    where = "job sites" if config.JOB_SITES else "the web"
    if config.JOB_SITES and len(results) < 5 and queries:
        add(web.search(queries[0], count=10, recency=config.JOB_RECENCY))
        where = "job sites + the web"
    notes = [f"{len(results)} search results from {len(queries)} searches on {where}"]
    if not results:
        return [], notes
    numbered = "\n\n".join(f"[{i}] {r['title']}\n{r['url']}\n{r['snippet'][:400]}"
                            for i, r in enumerate(results, start=1))
    verdicts: JobCandidates = structured(llm, JobCandidates).invoke(
        JOB_FILTER_PROMPT.format(request=request, today=today, results=numbered))
    known = {j.get("url") for j in (store.get(p.stem) for p in store.jobs_dir.glob("*.json")) if j}
    picked, rejected, decided = [], [], set()
    for v in sorted(verdicts.results, key=lambda v: not v.fits_city_year):
        if not (1 <= v.index <= len(results)) or v.index in decided:
            continue
        decided.add(v.index)
        r = results[v.index - 1]
        why = None
        if not v.is_posting:
            why = "not a single posting"
        elif not v.is_open or any(c in (r["title"] + r["snippet"]).lower() for c in CLOSED):
            why = "closed"
        elif not v.relevant:
            why = "role does not fit"
        elif r["url"] in known:
            why = "already analyzed"
        if why:
            rejected.append((r, f"{why}: {v.reason}" if v.reason else why))
        else:
            picked.append({**r, "company_hint": v.company.strip()})
    rejected += [(r, "not judged") for i, r in enumerate(results, start=1) if i not in decided]
    notes.append(f"{len(picked)} look like open, relevant postings")

    rows = []
    for r in picked:
        if len(rows) >= limit:
            break
        try:
            text = web.fetch_page(r["url"])
        except web.WebError as e:
            # Baidu sometimes returns most of the posting in its summary: use that instead of giving up
            if len(r.get("snippet", "")) >= SNIPPET_ENOUGH and not web.looks_blocked(r["snippet"]):
                text = f"{r['title']}\n{r['snippet']}"
                r = {**r, "from_snippet": True}
            else:
                rejected.append((r, "could not read: " + str(e).split(".")[0]))
                continue
        if any(c in text.lower() for c in CLOSED):
            rejected.append((r, "closed (says so on the page)"))
            continue
        job_id = json.loads(analyze_and_store(text, url=r["url"]).split("\n")[-1])["job_id"]
        job = store.get(job_id)
        if not job["analysis"].get("title", "").strip() or not job["analysis"].get("requirements"):
            rejected.append((r, "no job description on the page (list, redirect or empty page)"))
            continue
        if not job["analysis"].get("company", "").strip() and r.get("company_hint"):
            job["analysis"]["company"] = r["company_hint"]  # the search result named it; the page text did not
            store.put(job)
        deadline = job["analysis"].get("deadline", "")
        if deadline and deadline < today:
            rejected.append((r, f"deadline passed ({deadline})"))
            continue
        pct, _, _ = run_match(job, index, store, llm)
        a = job["analysis"]
        rows.append({"job_id": job_id, "score": pct, "company": company_name(a), "title": a["title"],
                     "location": a.get("location", ""), "deadline": deadline, "url": r["url"],
                     "from_snippet": r.get("from_snippet", False)})
    rows.sort(key=lambda x: -x["score"])
    notes.append({"rejected": rejected})
    return rows, notes


def _split_notes(notes):
    rejected = next((n["rejected"] for n in notes if isinstance(n, dict)), [])
    return [n for n in notes if not isinstance(n, dict)], rejected


def build_tools(index: ProfileIndex, store: JobStore, llm, cv_llm=None) -> list:
    """Tools are built with closures so tests can pass in fake models and temp folders.
    cv_llm: an optional stronger model used only for writing the full CV."""
    cv_llm = cv_llm or llm

    def _analyze(job_text: str) -> dict:
        result: JobAnalysis = structured(llm, JobAnalysis).invoke(ANALYZE_PROMPT.format(job=job_text))
        limits = {"duty": 6, "must": 8, "nice": 5}
        items, counts = [], {"duty": 0, "must": 0, "nice": 0}
        for r in result.requirements:
            if r.text.strip() and counts[r.type] < limits[r.type]:
                counts[r.type] += 1
                items.append({"text": r.text.strip(), "type": r.type})
        analysis = result.model_dump()
        analysis["requirements"] = sorted(items, key=lambda r: TYPE_ORDER[r["type"]])
        return analysis

    @tool
    def search_profile(query: str) -> str:
        """Search the user's CV, papers and notes. Use for any question about the user's
        own skills, projects or experience. Returns numbered passages with source and page."""
        hits = index.search(query, k=config.TOP_K)
        return format_hits(hits) if hits else "No matching passages."

    def _analyze_and_store(job_text: str, url: str = "") -> str:
        job_id = store.job_id(job_text)
        job = store.get(job_id)
        if job and job.get("version") == ANALYSIS_VERSION:  # same posting again: no extra LLM cost
            analysis = job["analysis"]
            if url and not job.get("url"):
                job["url"] = url
                store.put(job)
        else:
            analysis = _analyze(job_text)
            new = {"id": job_id, "version": ANALYSIS_VERSION, "text": job_text, "analysis": analysis}
            if url:
                new["url"] = url
            store.put(new)
        note = "" if analysis["company"].strip() else "NOTE: The posting does not name the company; ask the user.\n"
        return note + json.dumps({"job_id": job_id, **analysis}, ensure_ascii=False)

    @tool
    def analyze_job(job_text: str) -> str:
        """Analyze a job posting: company, title, location, deadline, language, responsibilities and requirements.
        Call this first whenever the user pastes a job posting. Returns a job_id for the other tools."""
        return _analyze_and_store(job_text)

    @tool
    def fetch_job(url: str) -> str:
        """Read a job posting from a web link and analyze it (like analyze_job). Use when the user gives
        a link to a job posting instead of the text. The link is saved with the job."""
        try:
            text = web.fetch_page(url)
        except web.WebError as e:
            return f"Could not read the posting: {e}"
        return f"Read {len(text)} characters from {url}.\n" + _analyze_and_store(text, url=url)

    @tool
    def find_jobs(request: str, limit: int = 3) -> str:
        """Search the web for open job postings that fit the request (e.g. "LLM evaluation, Shanghai,
        2027 campus recruitment"), read the best ones and match each against the user's profile.
        Returns a ranked shortlist with job_ids and links. Nothing is saved; the user chooses.
        `limit`: how many postings to read and match (1-5; each costs several model calls)."""
        from datetime import date

        limit = max(1, min(int(limit), 5))
        try:
            rows, notes = run_find_jobs(request, llm, limit, _analyze_and_store, index, store,
                                        date.today().isoformat())
        except web.WebError as e:
            return f"Job search not possible: {e}"
        notes, rejected = _split_notes(notes)
        lines = [f"# Job search: {request}", ""]
        if rows:
            lines += ["| # | Match | Company | Title | Location | Deadline | Link | job_id |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            lines += [f"| {i} | {r['score']}%{'*' if r.get('from_snippet') else ''} | {r['company']} | {r['title']} | "
                      f"{r['location'] or '–'} | "
                      f"{r['deadline'] or '–'} | {r['url']} | {r['job_id']} |" for i, r in enumerate(rows, start=1)]
            if any(r.get("from_snippet") for r in rows):
                lines += ["", "\\* The page could not be opened, so this was matched on the search summary only: "
                              "open the link to check the full posting."]
        else:
            lines.append("No open, relevant postings could be read and matched.")
        lines += ["", "Notes: " + "; ".join(notes)]
        if rejected:
            lines += ["", "## Skipped results", "", "| Result | Why skipped |", "| --- | --- |"]
            lines += [f"| [{r['title'][:60]}]({r['url']}) | {why} |" for r, why in rejected]
        text = "\n".join(lines)
        path = store.write_report(f"search-{date.today().isoformat()}", "jobs", text + "\n")
        head = text.split("\n## Skipped results")[0]
        return (f"{head}\n\nFull list including skipped results and reasons: {path}. "
                "Nothing was added to the tracker.")

    @tool
    def research_company(job_id: str) -> str:
        """Search the web about the employer and team of an analyzed job (products, focus, recent news)
        and write a short brief with numbered sources. Use when the user asks about the company or team,
        wants to prepare for an interview, or before writing a CV if the user wants it tailored to the team."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        try:
            text, path = run_research(job, llm, store)
        except web.WebError as e:
            return f"Research not possible: {e}"
        return text + (f"\n\nSaved to {path}" if path else "")

    @tool
    def match_job(job_id: str) -> str:
        """Check each responsibility and requirement of an analyzed job against the user's profile,
        with cited evidence. Returns a match score, verdicts and gaps, and saves a report."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        pct, path, results = run_match(job, index, store, llm)
        summary = [f"Score {pct}%. Report: {path}"]
        summary += [f"- [{r['type']}/{r['verdict']}] {r['text']} {', '.join(r['cited'])} — {r['reason']}"
                    for r in results]
        return "\n".join(summary)

    @tool
    def tailor_cv(job_id: str) -> str:
        """Draft CV bullets tailored to an analyzed job, each aimed at one responsibility or requirement,
        based only on evidence from the user's profile. Use when the user wants to improve CV bullets."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        text, path = run_tailor(job, index, store, llm)
        return f"{text}\n\nSaved to {path}"

    @tool
    def build_cv(job_id: str, language: str = "") -> str:
        """Write a CV tailored to an analyzed job as a one-page PDF (plus an editable Word file) in the style
        of the user's own CV, with their photo. Bullets are kept word for word from their CV; only order,
        selection and the objective change. Use when the user wants a CV / resume for this job.
        `language`: e.g. "Chinese" or "English" if the user asks for one; empty = the posting's language."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        cv_path = find_cv()
        if not cv_path:
            return ("Could not tell which file in profile/ is the CV. Name it with 'cv' or 'resume', "
                    "or set CV_FILE in .env.")
        result = run_build_cv(job, cv_text(cv_path), cv_llm, language)
        ats_lines = ats_report(job, result["cv"]) if config.CV_ATS_KEYWORDS else []
        images = find_images(cv_path, out_dir=store.root / "assets")
        # the CV as data too, for the CV quality evaluation (evaluate_cv.py)
        (store.reports_dir / f"{job['id']}-cv.json").write_text(
            result["cv"].model_dump_json(indent=2), encoding="utf-8")
        pdf, pages = render_pdf(result["cv"], store.reports_dir / f"{job['id']}-cv.pdf", **images)
        docx = render_docx(result["cv"], store.reports_dir / f"{job['id']}-cv.docx", **images)
        msg = [f"FILE: {pdf}", f"FILE: {docx}",
               f"Tailored CV from {cv_path.name}: PDF ({'one page' if pages == 1 else f'{pages} pages'}) "
               f"and an editable Word copy; photo {'included' if images['photo'] else 'not found'}.",
               f"Self-review found {len(result['issues'])} issue(s) and the draft was revised."
               if result["issues"] else "Self-review found no issues."]
        if result["translated"]:
            msg.append("This CV is a translation, so the word-for-word check was skipped: compare it with "
                       "the original yourself.")
        elif result["changes"]:
            msg.append("Word-for-word check fixed: " + "; ".join(result["changes"]))
        else:
            msg.append("Word-for-word check passed: every bullet uses only words from the original CV.")
        if result["objective"] == "same":
            msg.append("Objective: unchanged, the same words as in the original CV (do not say it was reworded).")
        elif result["objective"] == "reworded":
            msg.append("Objective: reworded for this job (the only part allowed to change).")
        if result["resorted"]:
            msg.append("Entries put back in newest-first order in: " + ", ".join(result["resorted"]))
        if result["numbers"]:
            msg.append("CHECK: these numbers are not in the original CV: " + ", ".join(result["numbers"]))
        if result["languages"]:
            msg.append("CHECK: these languages from the original CV are missing: " + ", ".join(result["languages"]))
        if pages > 1:
            msg.append("It did not fit on one page even at the smallest size: ask to leave out a project.")
        msg += ats_lines
        msg.append("The user should review the PDF before sending it.")
        return "\n".join(msg)

    def ats_report(job: dict, cv) -> list[str]:
        """ATS keyword check on the finished CV; adds only terms the user's own documents show."""
        # Compare the language, not the exact share: a Chinese CV full of English tool names (Python,
        # LangGraph) is ~65% Chinese characters while a Chinese posting is ~95%.
        if _is_chinese(cv_plain_text(cv)) != _is_chinese(job["text"]):
            return ["ATS keyword check skipped: the CV and the posting are in different languages."]
        try:
            terms = ats_terms(job, llm)
            store.put(job)
        except Exception:
            return ["ATS keyword check skipped: the model did not return the posting's key terms."]
        if not terms:
            return []
        index._load_bm25()
        chk = ats_check(terms, cv_plain_text(cv), index._chunks)
        lines = [f"ATS keywords: {len(chk['present'])} of {len(terms)} key terms of the posting are in the CV"
                 + (f" ({', '.join(chk['present'][:12])})." if chk["present"] else ".")]
        if chk["supported"]:
            label = add_keywords(cv, list(chk["supported"]))
            lines.append(f"ATS: added to the skills table as '{label}', because your documents show them: "
                         + "; ".join(f"{t} ({src})" for t, src in chk["supported"].items()))
        if chk["unsupported"]:
            lines.append("ATS: not added, not found in your documents (add them only if true): "
                         + ", ".join(chk["unsupported"]))
        return lines

    @tool
    def plan_gaps(job_id: str) -> str:
        """Turn each gap (partial or missing item) of a matched job into one concrete action:
        add evidence, reframe, build a small project, or learn. Use when the user asks how to close gaps."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        if "match" not in job:
            return "This job has not been matched yet. Call match_job first."
        gaps = [r for r in job["match"]["results"] if r["verdict"] != "strong"]
        if not gaps:
            return "No gaps: every item is strongly supported."
        gaps.sort(key=lambda r: (-WEIGHT[r["type"]], r["verdict"] != "missing"))
        a = job["analysis"]
        hits = index.search(" ".join(r["text"] for r in gaps[:6]), k=6)
        plan: GapPlan = structured(llm, GapPlan).invoke(GAP_PROMPT.format(
            title=a["title"], company=company_name(a), background=format_hits(hits),
            gaps="\n".join(f"{i}. ({r['type']}, {r['verdict']}) {r['text']} — {r['reason']}"
                           for i, r in enumerate(gaps, start=1))))
        label = {"add_evidence": "Add evidence", "reframe": "Reframe", "build": "Build", "learn": "Learn"}
        lines, used = [], set()
        for act in plan.actions:
            if 1 <= act.item <= len(gaps) and act.item not in used:
                used.add(act.item)
                g = gaps[act.item - 1]
                lines.append(f"- **{g['text']}** ({g['type']}, {g['verdict']})\n"
                             f"  {label[act.action]}: {act.step} — effort: {act.effort}; proof: {act.proof}")
        skipped = [g["text"] for i, g in enumerate(gaps, start=1) if i not in used]
        text = "\n".join(lines) or "No actions proposed."
        if skipped:
            text += "\n\nNo action proposed for: " + "; ".join(skipped)
        path = store.write_report(job["id"], "gaps", f"# Gap plan: {company_name(a)} – {a['title']}\n\n{text}\n")
        return f"{text}\n\nSaved to {path}"

    @tool
    def save_job(job_id: str, status: Literal["saved", "applied", "interview", "offer", "rejected"] = "saved",
                 company: str = "", url: str = "", cv_version: Literal["", "original", "tailored"] = "",
                 channel: str = "") -> str:
        """Add an analyzed job to the user's application tracker, or update its status (applied, interview,
        offer, rejected). The user is asked to approve first.
        Pass `company` if the posting did not name it and the user told you.
        Pass `url` if the user gave a link to the posting (in this message or earlier).
        Pass `cv_version` ("tailored" if they sent a CV made for this job, "original" if their usual CV) and
        `channel` (e.g. "campus site", "BOSS直聘", "referral", "HR WeChat", "email") when the user says so."""
        job = store.get(job_id)
        if not job:
            return f"Unknown job_id {job_id}. Call analyze_job first."
        if company.strip():
            job["analysis"]["company"] = company.strip()
        if url.strip():
            job["url"] = url.strip()
        store.put(job)
        a = job["analysis"]
        # Pause the graph. The UI shows this request and resumes with the user's answer.
        decision = interrupt({
            "action": "save_job", "job_id": job_id, "status": status, "cv_version": cv_version, "channel": channel,
            "company": company_name(a), "title": a["title"],
            "score": job.get("match", {}).get("score"), "url": job.get("url", ""),
        })
        if decision is True or (isinstance(decision, dict) and decision.get("approved")):
            row = store.track(job, status, cv_version=cv_version, channel=channel)
            return f"Saved to tracker: {row['company']} – {row['title']} ({status})."
        return "The user did not approve. Nothing was saved."

    return [search_profile, analyze_job, fetch_job, find_jobs, research_company, match_job,
            tailor_cv, build_cv, plan_gaps, save_job]
