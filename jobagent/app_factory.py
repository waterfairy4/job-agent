"""Wire the real pieces together (used by the CLI and the web UI)."""
from jobagent.agent import build_agent
from jobagent.llm import get_llm
from jobagent.rag import ProfileIndex
from jobagent.store import JobStore
from jobagent.tools import build_tools


def create():
    index, store, llm = ProfileIndex(), JobStore(), get_llm()
    graph = build_agent(build_tools(index, store, llm), llm)
    return graph, index, store
