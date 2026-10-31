"""End-to-end: job posting -> analyze -> match -> tailor -> save (paused for approval) -> resume."""
import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from jobagent.agent import build_agent, new_thread, step
from jobagent.rag import ProfileIndex, tokenize
from jobagent.schemas import CVBullet, CVDraft, JobAnalysis, Requirement, SearchTerms, Verdict
from jobagent.store import JobStore
from jobagent.tools import ANALYZE_PROMPT, build_tools, run_match, score
from tests.fakes import FakeLLM, tool_call

JD = "岗位职责：设计大模型评测集与评分标准。岗位要求：熟练掌握Python；有大模型评测经验；熟悉Kubernetes者优先。"


@pytest.fixture
def env(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "cv.md").write_text(
        "Skills: Python, PyTorch, LoRA fine-tuning.\n\n"
        "Built MLS-Eval, a multilingual LLM evaluation benchmark (大模型评测) with scoring metrics.", encoding="utf-8")
    index = ProfileIndex(DeterministicFakeEmbedding(size=32), persist_dir=str(tmp_path / "chroma"))
    assert index.build(str(profile)) > 0
    return index, JobStore(str(tmp_path / "data"))


def handlers(company=""):
    def judge(prompt):
        item = prompt.split("Evidence")[0]
        if "Python" in item or "evaluation sets" in item:
            return Verdict(verdict="strong", evidence=[1], reason="Shown in CV.", tip="")
        if "LLM evaluation experience" in item:
            # cites passage 99, which does not exist -> must be downgraded
            return Verdict(verdict="strong", evidence=[99], reason="MLS-Eval.", tip="Cite MLS-Eval.")
        if "deep learning frameworks" in item:
            # claims "related" but cites nothing -> must become missing
            return Verdict(verdict="partial", evidence=[], reason="No framework named.", tip="Name PyTorch.")
        return Verdict(verdict="missing", evidence=[], reason="Not mentioned.", tip="Try a small K8s demo.")

    return {
        JobAnalysis: lambda p: JobAnalysis(
            company=company, title="LLM Evaluation Engineer", location="Shanghai", language="Chinese",
            requirements=[Requirement(text="Nice: Kubernetes", type="nice"),
                          Requirement(text="Design evaluation sets and scoring standards", type="duty"),
                          Requirement(text="Python", type="must"),
                          Requirement(text="LLM evaluation experience", type="must"),
                          Requirement(text="  ", type="must")]),
        Verdict: judge,
        SearchTerms: lambda p: SearchTerms(terms=["PyTorch", "TensorFlow", "深度学习框架"]),
        CVDraft: lambda p: CVDraft(bullets=[
            CVBullet(target=1, bullet="设计多语言大模型评测集与评分指标", evidence=[1]),
            CVBullet(target=2, bullet="Led a team of 20 engineers", evidence=[]),     # no evidence -> dropped
            CVBullet(target=42, bullet="用 Python 构建评测流水线", evidence=[1]),       # no such item -> dropped
        ], keywords=["大模型评测"]),
    }


def test_full_flow_with_approval(env):
    index, store = env
    jid = store.job_id(JD)
    llm = FakeLLM([
        tool_call("analyze_job", {"job_text": JD}, "c1"),
        tool_call("match_job", {"job_id": jid}, "c2"),
        tool_call("tailor_cv", {"job_id": jid}, "c3"),
        AIMessage(content="Score 75%."),
        tool_call("save_job", {"job_id": jid, "company": "ByteDance", "url": "https://jobs.example.com/1"}, "c4"),
        AIMessage(content="Saved."),
    ], handlers())
    graph = build_agent(build_tools(index, store, llm), llm)
    thread = new_thread()

    out = step(graph, thread, f"Match this job:\n{JD}")
    assert [t["tool"] for t in out["trace"]] == ["analyze_job", "match_job", "tailor_cv"]
    assert "does not name the company" in out["trace"][0]["result"]

    job = store.get(jid)
    types = [r["type"] for r in job["analysis"]["requirements"]]
    assert types == ["duty", "must", "must", "nice"]  # responsibilities first, blank item dropped
    verdicts = {r["text"]: r["verdict"] for r in job["match"]["results"]}
    assert verdicts["LLM evaluation experience"] == "partial"  # invented citation downgraded
    # duty 1.5*1 + must 2*1 + must 2*0.5 + nice 0  = 4.5 / 6.5
    assert job["match"]["score"] == 69

    report = (store.reports_dir / f"{jid}-match.md").read_text(encoding="utf-8")
    assert "Responsibilities: can you do this job?" in report and "Unknown company" in report

    cv = (store.reports_dir / f"{jid}-cv.md").read_text(encoding="utf-8")
    assert "评测集与评分指标" in cv and "for duty: Design evaluation sets" in cv
    assert "20 engineers" not in cv and "评测流水线" not in cv and "2 suggestion(s) removed" in cv
    tailor_prompt = [p for p in llm.prompts if "Write CV bullets" in p][0]
    assert "Write every bullet and keyword in Chinese" in tailor_prompt

    out = step(graph, thread, "Save it, it's ByteDance")
    p = out["pending"]
    assert p["action"] == "save_job" and p["company"] == "ByteDance" and p["score"] == 69
    assert store.tracker() == []

    assert step(graph, thread, resume=True)["reply"] == "Saved."
    row = store.tracker()[0]
    assert row["company"] == "ByteDance" and row["url"] == "https://jobs.example.com/1"
    assert p["url"] == "https://jobs.example.com/1"


def test_rejected_save_writes_nothing_and_unknown_company(env):
    index, store = env
    jid = store.job_id(JD)
    llm = FakeLLM([
        tool_call("analyze_job", {"job_text": JD}, "c1"),
        tool_call("save_job", {"job_id": jid}, "c2"),
        AIMessage(content="Okay, not saved."),
    ], handlers())
    graph = build_agent(build_tools(index, store, llm), llm)
    thread = new_thread()
    pending = step(graph, thread, JD)["pending"]
    assert pending["company"] == "Unknown company"
    out = step(graph, thread, resume=False)
    assert "not saved" in out["reply"] and store.tracker() == []


def test_cache_reused_and_old_versions_redone(env):
    index, store = env
    llm = FakeLLM([], handlers("Huawei"))
    tools = {t.name: t for t in build_tools(index, store, llm)}
    a = json.loads(tools["analyze_job"].invoke({"job_text": JD}))
    b = json.loads(tools["analyze_job"].invoke({"job_text": "  " + JD + "\n\n"}))  # extra whitespace
    assert a["job_id"] == b["job_id"]
    count = lambda: sum("Job posting:" in p for p in llm.prompts)
    assert count() == 1  # second call hit the cache

    old = store.get(a["job_id"])
    old.pop("version")  # simulate an analysis saved by the previous version
    store.put(old)
    tools["analyze_job"].invoke({"job_text": JD})
    assert count() == 2


def test_unknown_job_id(env):
    index, store = env
    tools = {t.name: t for t in build_tools(index, store, FakeLLM([], handlers()))}
    assert "Unknown job_id" in tools["match_job"].invoke({"job_id": "nope"})


def test_score_tokenizer_and_prompt():
    assert score([{"type": "must", "verdict": "strong"}, {"type": "nice", "verdict": "missing"}]) == 67
    assert score([{"type": "duty", "verdict": "partial"}]) == 50
    assert "评测" in tokenize("大模型评测") and "python" in tokenize("Python")
    assert "duty" in ANALYZE_PROMPT and "never guess" in ANALYZE_PROMPT


def test_strict_partial_and_query_expansion(env):
    index, store = env
    llm = FakeLLM([], handlers("Huawei"))
    tools = {t.name: t for t in build_tools(index, store, llm)}
    jid = json.loads(tools["analyze_job"].invoke({"job_text": JD}))["job_id"]
    job = store.get(jid)
    job["analysis"]["requirements"].append({"text": "Familiar with deep learning frameworks", "type": "must"})
    store.put(job)

    calls = []
    real_search = index.search
    index.search = lambda q, k: calls.append((q, k)) or real_search(q, k)

    run_match(store.get(jid), index, store, llm, expand=True, top_k=5)
    job = store.get(jid)
    verdicts = {r["text"]: r["verdict"] for r in job["match"]["results"]}
    assert verdicts["Familiar with deep learning frameworks"] == "missing"  # uncited partial
    assert any("PyTorch" in q and k == 5 for q, k in calls)                 # expanded query used
    assert job["match"]["settings"] == {"expand": True, "top_k": 5}
    n_expand = sum("search terms" in p for p in llm.prompts)
    assert n_expand == len(job["analysis"]["requirements"])

    run_match(store.get(jid), index, store, llm, expand=True, top_k=5)    # cached: no new expansion calls
    assert sum("search terms" in p for p in llm.prompts) == n_expand

    calls.clear()
    run_match(store.get(jid), index, store, llm, expand=False, top_k=3)    # v1-style retrieval
    assert all("PyTorch" not in q and k == 3 for q, k in calls)


def test_conversation_and_approval_survive_a_restart(env, tmp_path):
    from jobagent.agent import make_checkpointer, pending_approval, traces

    index, store = env
    jid = store.job_id(JD)
    db = str(tmp_path / "checkpoints.sqlite")
    script = [tool_call("analyze_job", {"job_text": JD}, "c1"), AIMessage(content="Analyzed."),
              tool_call("save_job", {"job_id": jid, "company": "ByteDance"}, "c2"), AIMessage(content="Saved.")]
    llm = FakeLLM(script, handlers())
    graph = build_agent(build_tools(index, store, llm), llm, checkpointer=make_checkpointer(db))
    thread = new_thread()
    step(graph, thread, f"Analyze:\n{JD}")
    assert step(graph, thread, "Save it")["pending"]["company"] == "ByteDance"

    # "restart": a new graph and a new connection on the same file, the same LLM script position
    graph2 = build_agent(build_tools(index, store, llm), llm, checkpointer=make_checkpointer(db))
    assert pending_approval(graph2, thread)["company"] == "ByteDance"   # the approval is still waiting
    messages = graph2.get_state({"configurable": {"thread_id": thread}}).values["messages"]
    first_reply = next(i for i, m in enumerate(messages) if isinstance(m, AIMessage) and m.content)
    assert [t["tool"] for t in traces(messages)[first_reply]] == ["analyze_job"]
    assert step(graph2, thread, resume=True)["reply"] == "Saved."
    assert store.tracker()[0]["company"] == "ByteDance"
    assert pending_approval(graph2, thread) is None


def test_last_thread_is_remembered(tmp_path):
    store = JobStore(str(tmp_path))
    assert store.last_thread() is None
    store.set_last_thread("abc")
    assert JobStore(str(tmp_path)).last_thread() == "abc"


def test_tool_failures_and_bad_arguments_do_not_crash_the_turn(env, monkeypatch):
    from langchain_core.messages import ToolMessage

    index, store = env
    llm = FakeLLM([
        tool_call("match_job", {"job_id": 123456}, "c1"),          # wrong type: must be a string
        tool_call("research_company", {"job_id": "x"}, "c2"),      # raises inside the tool
        lambda msgs: AIMessage(content=" | ".join(m.content for m in msgs if isinstance(m, ToolMessage))),
    ], handlers())
    tools = build_tools(index, store, llm)
    research = next(t for t in tools if t.name == "research_company")
    monkeypatch.setattr(store, "get", lambda job_id: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    graph = build_agent(tools, llm)
    out = step(graph, new_thread(), "Match and research")
    assert out["reply"].count("Tool error") == 1 and "disk on fire" in out["reply"]
    assert "fix the error" in out["reply"].split(" | ")[0]     # bad arguments: the model may correct them
    assert research.name == "research_company"
