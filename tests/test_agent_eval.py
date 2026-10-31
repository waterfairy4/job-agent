from pathlib import Path

from langchain_core.messages import AIMessage

from jobagent.agent import SYSTEM_PROMPT
from jobagent.agent_eval import (NONE, evaluate_retrieval, evaluate_routing, read_retrieval_cases,
                                 read_routing_cases, retrieval_markdown, routing_markdown)
from jobagent.store import JobStore
from jobagent.tools import build_tools

EVAL = Path(__file__).resolve().parent.parent / "eval"


def test_case_files_name_real_tools(tmp_path):
    names = {t.name for t in build_tools(None, JobStore(str(tmp_path)), llm=None)} | {NONE}
    cases = read_routing_cases(EVAL / "routing_cases.csv")
    assert len(cases) >= 30 and all(c["expected"] <= names for c in cases)
    assert {t for c in cases for t in c["expected"]} == names  # every tool (and "no tool") is tested
    assert len(read_retrieval_cases(EVAL / "retrieval_cases.csv")) >= 15


def test_routing_accuracy():
    class Router:  # a fake model: links -> fetch_job, "find" -> find_jobs, else no tool
        def invoke(self, messages):
            text = messages[-1].content
            assert messages[0].content == SYSTEM_PROMPT
            if "http" in text:
                return AIMessage(content="", tool_calls=[{"name": "fetch_job", "args": {}, "id": "1"}])
            if "find" in text.lower():
                return AIMessage(content="", tool_calls=[{"name": "find_jobs", "args": {}, "id": "2"}])
            return AIMessage(content="Hello")

    cases = [{"message": "https://x", "expected": {"fetch_job"}, "note": ""},
             {"message": "find jobs", "expected": {"find_jobs"}, "note": ""},
             {"message": "Hi", "expected": {NONE}, "note": ""},
             {"message": "Save it", "expected": {"save_job"}, "note": ""}]
    res = evaluate_routing(Router(), SYSTEM_PROMPT, cases)
    assert res["accuracy"] == 0.75 and res["per_tool"]["save_job"] == [0, 1]
    md = routing_markdown(res)
    assert "75.0%" in md and "| Save it | save_job | none |" in md


def test_retrieval_modes(tmp_path):
    from langchain_core.embeddings import DeterministicFakeEmbedding

    from jobagent.rag import ProfileIndex

    profile = tmp_path / "profile"
    profile.mkdir()
    notes = {"a": "Trained models with PyTorch and Python.", "b": "Built a 大模型评测 benchmark with metrics.",
             "c": "Full-stack web app with React and Spring Boot.", "d": "Survey statistics and weighting.",
             "e": "Congestion control with BBR in Linux containers.", "f": "Learning Korean and Turkish."}
    for name, text in notes.items():
        (profile / f"{name}.md").write_text(text, encoding="utf-8")
    index = ProfileIndex(DeterministicFakeEmbedding(size=32), persist_dir=str(tmp_path / "chroma"))
    assert index.build(str(profile)) == 6
    cases = [{"query": "Python PyTorch", "phrases": ["pytorch"], "note": ""},
             {"query": "大模型评测", "phrases": ["大模型评测"], "note": ""},
             {"query": "Kubernetes", "phrases": ["kubernetes"], "note": ""}]   # in no chunk -> skipped
    res = evaluate_retrieval(index, cases, k=5)
    assert res["n"] == 2 and res["skipped"] == ["Kubernetes"]
    assert set(res["modes"]) == {"bm25", "dense", "hybrid"}
    assert res["modes"]["bm25"]["hit"] == 1.0 and res["modes"]["hybrid"]["recall"] == 1.0
    assert "Warning" in retrieval_markdown(res)       # a tiny profile is flagged
