"""Structured outputs. The LLM must fill these exactly; Pydantic validates the result."""
from typing import Literal

from pydantic import BaseModel, Field

ItemType = Literal["duty", "must", "nice"]


class Requirement(BaseModel):
    text: str = Field(description="One responsibility, skill or qualification, short and specific, in English")
    type: ItemType = Field(description="duty = a responsibility (what the person will do); "
                                        "must = required qualification; nice = preferred/bonus (优先, plus)")


class JobAnalysis(BaseModel):
    company: str = Field(description="Company name if the posting states it, otherwise an empty string")
    title: str = Field(description="Job title")
    location: str = Field(default="", description="City, or empty string")
    deadline: str = Field(default="", description="Application deadline as YYYY-MM-DD, or empty string")
    language: str = Field(description="Language the posting is written in, e.g. English or Chinese")
    requirements: list[Requirement] = Field(description="Responsibilities (duty) and requirements (must/nice)")


class Verdict(BaseModel):
    verdict: Literal["strong", "partial", "missing"]
    evidence: list[int] = Field(description="Numbers of the evidence passages that support the verdict")
    reason: str = Field(description="One sentence")
    tip: str = Field(description="One sentence on how to show or close the gap; empty if strong")


class SearchTerms(BaseModel):
    terms: list[str] = Field(description="4-8 short concrete terms a CV would literally contain, English and Chinese")


class CVBullet(BaseModel):
    target: int = Field(description="Number of the ONE job item this bullet addresses")
    bullet: str = Field(description="A CV bullet written for that item, using the posting's wording where honest")
    evidence: list[int] = Field(description="Numbers of the evidence passages this bullet is based on")


class CVEntry(BaseModel):
    title: str = Field(default="", description="Bold first line, e.g. 'Peking University, Beijing' or a project name")
    date: str = Field(default="", description="Right-aligned on the first line, e.g. '2024 – 2027 (expected)'")
    subtitle: str = Field(default="", description="Second line, e.g. degree, role, 'First author, under submission'")
    meta: str = Field(default="", description="Right-aligned on the second line, e.g. 'GPA: 3.66 / 4.0'")
    bullets: list[str] = Field(default_factory=list)


class CVSection(BaseModel):
    title: str = Field(description="Section title in the CV language, e.g. 教育经历 / Education")
    layout: str = Field(default="entries",
                        description='"table" = label + text rows, for skills; "text" = plain paragraphs without '
                                    'bullets (objective, languages); anything else = normal entries')
    entries: list[CVEntry]


class CVDocument(BaseModel):
    name: str = Field(description="Name in the CV language")
    name_alt: str = Field(default="", description="Name in the other script, if the CV has one")
    headline: str = Field(default="", description="One line under the name, e.g. degree and focus")
    contact: list[str] = Field(default_factory=list, description="Contact lines, exactly as in the CV")
    sections: list[CVSection]


class CVIssue(BaseModel):
    # Free text on purpose: a fixed list made the whole review fail when the model used another label.
    kind: str = Field(description="unsupported, stretched, irrelevant, ordering, too_long or wording")
    where: str = Field(description="Which section/bullet")
    problem: str
    fix: str


class CVCritique(BaseModel):
    issues: list[CVIssue]


class CVScore(BaseModel):
    relevance: int = Field(ge=1, le=5, description="1-5: how clearly the CV shows fit for THIS job's duties and must-haves")
    clarity: int = Field(ge=1, le=5, description="1-5: most relevant evidence first, easy to scan, no clutter")
    reason: str = Field(description="One sentence naming the strongest and the weakest point")


class SearchPlan(BaseModel):
    queries: list[str] = Field(description="2-3 web search queries, Chinese and/or English")


class JobCandidate(BaseModel):
    index: int = Field(description="Number of the search result")
    is_posting: bool = Field(description="True only if the page is ONE specific job posting (not a blog, "
                                         "news, list of many jobs, salary page or forum discussion)")
    is_open: bool = Field(description="False if the result says the posting is closed/expired (已截止, 已结束, closed)")
    relevant: bool = Field(description="True if the ROLE fits the request. City and graduation year are "
                                       "preferences, not requirements: do not reject for them")
    fits_city_year: bool = Field(default=False, description="True if city and graduation year visibly match")
    company: str = Field(default="")
    title: str = Field(default="")
    reason: str = Field(default="", description="A few words: why kept or rejected")


class JobCandidates(BaseModel):
    results: list[JobCandidate]


class BriefPoint(BaseModel):
    text: str
    sources: list[int] = Field(description="Numbers of the search results that support this point")


class CompanyBrief(BaseModel):
    overview: list[BriefPoint] = Field(description="What the company/team does, its products and focus")
    recent: list[BriefPoint] = Field(description="Recent news or developments relevant to the role")
    emphasize: list[str] = Field(description="Topics or keywords the candidate could emphasize (no claims)")
    questions: list[str] = Field(description="Good questions the candidate could ask in an interview")


class GapAction(BaseModel):
    item: int = Field(description="Number of the gap this action addresses")
    action: Literal["add_evidence", "build", "learn", "reframe"] = Field(
        description="add_evidence = the user may already have it but it is not written down; "
                    "build = a small project to create proof; learn = study for the interview; "
                    "reframe = existing experience, presented better")
    step: str = Field(description="One concrete step, specific enough to start today")
    effort: str = Field(description="Rough effort, e.g. '1 hour', 'a weekend'")
    proof: str = Field(description="What will show it is done, e.g. a CV line, a GitHub repo, a result")


class GapPlan(BaseModel):
    actions: list[GapAction]


class CVDraft(BaseModel):
    bullets: list[CVBullet]
    keywords: list[str] = Field(description="Words from the job posting worth using in the CV")
