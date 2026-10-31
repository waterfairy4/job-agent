"""Wire the real pieces together (used by the CLI and the web UI)."""
from jobagent import config
from jobagent.agent import build_agent, make_checkpointer
from jobagent.llm import get_llm
from jobagent.rag import ProfileIndex
from jobagent.store import JobStore
from jobagent.tools import build_tools


def create():
    index, store, llm = ProfileIndex(), JobStore(), get_llm()
    cv_llm = get_llm(config.CV_MODEL) if config.CV_MODEL else llm
    graph = build_agent(build_tools(index, store, llm, cv_llm), llm, checkpointer=make_checkpointer())
    return graph, index, store
