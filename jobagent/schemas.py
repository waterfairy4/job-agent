"""Structured outputs. The LLM must fill these exactly; Pydantic validates the result."""
from typing import Literal

from pydantic import BaseModel, Field


class Requirement(BaseModel):
    text: str = Field(description="One skill or experience, short and specific, in English")
    type: Literal["must", "nice"] = Field(description="must = required; nice = preferred/bonus (优先, plus)")


class JobAnalysis(BaseModel):
    company: str = Field(description="Company name, or empty string if not stated")
    title: str = Field(description="Job title")
    location: str = Field(default="", description="City, or empty string")
    deadline: str = Field(default="", description="Application deadline as YYYY-MM-DD, or empty string")
    requirements: list[Requirement] = Field(description="At most 12 atomic requirements")


class Verdict(BaseModel):
    verdict: Literal["strong", "partial", "missing"]
    evidence: list[int] = Field(description="Numbers of the evidence passages that support the verdict")
    reason: str = Field(description="One sentence")
    tip: str = Field(description="One sentence on how to show or close the gap; empty if strong")


class CVBullet(BaseModel):
    bullet: str = Field(description="A CV bullet, rewritten to fit this job")
    evidence: list[int] = Field(description="Numbers of the passages this bullet is based on")
    targets: str = Field(description="Which job requirement it addresses")


class CVDraft(BaseModel):
    bullets: list[CVBullet]
    keywords: list[str] = Field(description="Words from the job description worth using in the CV")
