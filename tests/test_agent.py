"""End-to-end: job description -> analyze -> match -> save (paused for approval) -> resume."""
import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from jobagent.agent import build_agent, new_thread, step
from jobagent.rag import ProfileIndex, tokenize
from jobagent.schemas import CVBullet, CVDraft, JobAnalysis, Requirement, Verdict
from jobagent.store import JobStore
from jobagent.tools import build_tools, score
from tests.fakes import FakeLLM, tool_call

JD = "岗位要求：熟练掌握Python；有大模型评测经验；熟悉Kubernetes者优先。"


@pytest.fixture
def env(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "cv.md").write_text(
        "Skills: Python, PyTorch, LoRA fine-tuning.\n\n"
        "Built MLS-Eval, a multilingual LLM evaluation benchmark (大模型评测).", encoding="utf-8")
    index = ProfileIndex(DeterministicFakeEmbedding(size=32), persist_dir=str(tmp_path / "chroma"))
    assert index.build(str(profile)) > 0
    return index, JobStore(str(tmp_path / "data"))


def handlers():
    def judge(prompt):
        if "Python" in prompt.split("Evidence")[0]:
            return Verdict(verdict="strong", evidence=[1], reason="Listed in skills.", tip="")
        if "evaluation" in prompt.split("Evidence")[0]:
            # cites passage 99, which does not exist -> must be downgraded
            return Verdict(verdict="strong", evidence=[99], reason="MLS-Eval.", tip="Cite MLS-Eval.")
        return Verdict(verdict="missing", evidence=[], reason="Not mentioned.", tip="Try a small K8s demo.")

    return {
        JobAnalysis: lambda p: JobAnalysis(company="ByteDance", title="LLM Evaluation Engineer", location="Shanghai",
                                           requirements=[Requirement(text="Python", type="must"),
                                                         Requirement(text="LLM evaluation experience", type="must"),
                                                         Requirement(text="Kubernetes", type="nice"),
                                                         Requirement(text="  ", type="must")]),
        Verdict: judge,
        CVDraft: lambda p: CVDraft(bullets=[CVBullet(bullet="用 Python 构建多语言大模型评测基准", evidence=[1], targets="eval"),
                                            CVBullet(bullet="Led a team of 20 engineers", evidence=[], targets="lead")],
                                   keywords=["大模型评测"]),
    }


def test_full_flow_with_approval(env):
    index, store = env
    jid = store.job_id(JD)
    llm = FakeLLM([
        tool_call("analyze_job", {"job_text": JD}, "c1"),
        tool_call("match_job", {"job_id": jid}, "c2"),
        tool_call("tailor_cv", {"job_id": jid}, "c3"),
        AIMessage(content="Score 60%. Strong: Python. Gap: Kubernetes."),
        tool_call("save_job", {"job_id": jid}, "c4"),
        AIMessage(content="Saved."),
    ], handlers())
    graph = build_agent(build_tools(index, store, llm), llm)
    thread = new_thread()

    out = step(graph, thread, f"Match this job:\n{JD}")
    assert [t["tool"] for t in out["trace"]] == ["analyze_job", "match_job", "tailor_cv"]
    assert out["pending"] is None and "60%" in out["reply"]

    job = store.get(jid)
    verdicts = {r["text"]: r["verdict"] for r in job["match"]["results"]}
    assert verdicts == {"Python": "strong", "LLM evaluation experience": "partial", "Kubernetes": "missing"}
    assert job["match"]["score"] == 60  # (2*1 + 2*0.5 + 0) / 5
    assert len(job["analysis"]["requirements"]) == 3  # blank requirement dropped

    cv = (store.reports_dir / f"{jid}-cv.md").read_text(encoding="utf-8")
    assert "评测基准" in cv and "20 engineers" not in cv and "1 suggestion(s) removed" in cv

    # Saving pauses for approval; nothing is written yet
    out = step(graph, thread, "Save it please")
    assert out["pending"]["action"] == "save_job" and out["pending"]["score"] == 60
    assert store.tracker() == []

    out = step(graph, thread, resume=True)
    assert out["reply"] == "Saved."
    assert store.tracker()[0]["company"] == "ByteDance"


def test_rejected_save_writes_nothing(env):
    index, store = env
    jid = store.job_id(JD)
    llm = FakeLLM([
        tool_call("analyze_job", {"job_text": JD}, "c1"),
        tool_call("save_job", {"job_id": jid}, "c2"),
        AIMessage(content="Okay, not saved."),
    ], handlers())
    graph = build_agent(build_tools(index, store, llm), llm)
    thread = new_thread()
    assert step(graph, thread, JD)["pending"] is not None
    out = step(graph, thread, resume=False)
    assert "not saved" in out["reply"] and store.tracker() == []


def test_same_job_reuses_analysis(env):
    index, store = env
    llm = FakeLLM([], handlers())
    tools = {t.name: t for t in build_tools(index, store, llm)}
    a = json.loads(tools["analyze_job"].invoke({"job_text": JD}))
    b = json.loads(tools["analyze_job"].invoke({"job_text": "  " + JD + "\n\n"}))  # extra whitespace
    assert a["job_id"] == b["job_id"]
    assert sum("Job description" in p for p in llm.prompts) == 1  # second call hit the cache


def test_unknown_job_id(env):
    index, store = env
    tools = {t.name: t for t in build_tools(index, store, FakeLLM([], handlers()))}
    assert "Unknown job_id" in tools["match_job"].invoke({"job_id": "nope"})


def test_score_and_tokenizer():
    assert score([{"type": "must", "verdict": "strong"}, {"type": "nice", "verdict": "missing"}]) == 67
    assert "评测" in tokenize("大模型评测") and "python" in tokenize("Python")
