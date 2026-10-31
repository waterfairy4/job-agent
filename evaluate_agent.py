"""Evaluate the agent's routing and the profile retrieval.

  python evaluate_agent.py routing v1   -> eval/routing-v1.md   (路由准确率: right first tool?)
  python evaluate_agent.py retrieval v1 -> eval/retrieval-v1.md (检索命中率: BM25 vs dense vs hybrid)
  python evaluate_agent.py compare v1 v2 -> is routing v2 really better? (paired McNemar test + bootstrap CI)

Test cases: eval/routing_cases.csv and eval/retrieval_cases.csv (edit or add rows freely).
Every run also adds a row per metric to eval/agent_history.csv, so versions can be compared.
Routing makes one LLM call per case and never runs a tool; retrieval makes no LLM calls.
"""
import csv
import sys
from datetime import date
from pathlib import Path

from jobagent import config
from jobagent.agent_eval import (compare_routing, evaluate_retrieval, evaluate_routing, load_routing_run,
                                 read_retrieval_cases, read_routing_cases, retrieval_markdown, routing_markdown,
                                 save_routing_run)

EVAL = Path("eval")
HISTORY = EVAL / "agent_history.csv"


def log(version: str, metric: str, value: float, settings: str):
    new = not HISTORY.exists()
    with open(HISTORY, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["version", "date", "metric", "value", "settings"])
        w.writerow([version, date.today().isoformat(), metric, f"{value:.4f}", settings])


def routing(version: str):
    from jobagent.agent import SYSTEM_PROMPT
    from jobagent.llm import get_llm
    from jobagent.store import JobStore
    from jobagent.tools import build_tools

    llm = get_llm()
    model = llm.bind_tools(build_tools(None, JobStore(), llm))  # tool schemas only: nothing is executed
    cases = read_routing_cases(EVAL / "routing_cases.csv")
    print(f"Routing {len(cases)} test messages with {config.LLM_MODEL}...")
    res = evaluate_routing(model, SYSTEM_PROMPT, cases)
    settings = f"model {config.LLM_MODEL}"
    text = routing_markdown(res, settings)
    (EVAL / f"routing-{version}.md").write_text(text, encoding="utf-8")
    save_routing_run(EVAL / "runs" / f"routing-{version}.csv", res)
    log(version, "routing_accuracy", res["accuracy"], settings)
    print(text)


def retrieval(version: str):
    from jobagent.rag import ProfileIndex

    index = ProfileIndex()
    cases = read_retrieval_cases(EVAL / "retrieval_cases.csv")
    res = evaluate_retrieval(index, cases, k=config.MATCH_TOP_K)
    text = retrieval_markdown(res)
    (EVAL / f"retrieval-{version}.md").write_text(text, encoding="utf-8")
    settings = f"{res['chunks']} chunks, chunk size {config.CHUNK_SIZE}, {config.EMBED_MODEL}"
    for mode, m in res["modes"].items():
        log(version, f"recall@{res['k']}_{mode}", m["recall"], settings)
    print(text)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    version = sys.argv[2] if len(sys.argv) > 2 else "latest"
    EVAL.mkdir(exist_ok=True)
    if cmd == "routing":
        routing(version)
    elif cmd == "retrieval":
        retrieval(version)
    elif cmd == "compare" and len(sys.argv) > 3:
        a, b = sys.argv[2], sys.argv[3]
        text = compare_routing(load_routing_run(EVAL, a), load_routing_run(EVAL, b), a, b)
        (EVAL / f"routing-compare-{a}-{b}.md").write_text(text, encoding="utf-8")
        print(text)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
