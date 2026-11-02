"""Two evaluations of the agent besides the judge (evaluation.py):

1. Routing accuracy (路由准确率): for each test message, does the agent's FIRST decision call the right
   tool (or no tool)? Only that first model call is made; tools are never executed, so a run costs one
   cheap LLM call per case and changes nothing.

2. Retrieval hit rate (检索命中率): for each test query, are the profile chunks that should be found in
   the top k? Compared for BM25 alone, dense (bge-m3) alone and the hybrid (RRF). A chunk counts as
   relevant when it contains one of the case's key phrases, so labels survive re-chunking.
"""
import csv
from collections import Counter
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

NONE = "none"  # the right answer is to reply or ask, without a tool


# ---------- routing ----------
def read_routing_cases(path: Path) -> list[dict]:
    """CSV columns: message, expected ("analyze_job", "none", or alternatives "a|b"), note."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("message") or "").strip()]
    return [{"message": r["message"].strip(), "expected": {x.strip() for x in r["expected"].split("|")},
             "note": (r.get("note") or "").strip()} for r in rows]


def first_tool(model, system_prompt: str, message: str) -> str:
    """The tool the agent would call first for this message, or "none"."""
    reply = model.invoke([SystemMessage(system_prompt), HumanMessage(message)])
    calls = getattr(reply, "tool_calls", None) or []
    return calls[0]["name"] if calls else NONE


def evaluate_routing(model, system_prompt: str, cases: list[dict]) -> dict:
    results = []
    for c in cases:
        try:
            got = first_tool(model, system_prompt, c["message"])
        except Exception as e:  # one failed call must not stop the run
            got = f"error: {type(e).__name__}"
        results.append({**c, "got": got, "ok": got in c["expected"]})
    n = len(results)
    per_tool = {}
    for r in results:
        for tool in r["expected"]:
            t = per_tool.setdefault(tool, [0, 0])
            t[0] += r["ok"]
            t[1] += 1
    return {"n": n, "accuracy": sum(r["ok"] for r in results) / n if n else 0.0,
            "per_tool": per_tool, "results": results,
            "confusions": Counter((" | ".join(sorted(r["expected"])), r["got"]) for r in results if not r["ok"])}


def routing_markdown(res: dict, settings: str = "") -> str:
    lines = ["# Routing accuracy (路由准确率)", "",
             f"{res['n']} test messages. First tool chosen correctly: **{res['accuracy']:.1%}**.",
             f"Settings: {settings}" if settings else "", "",
             "| Expected tool | Correct | Cases | Accuracy |", "| --- | --- | --- | --- |"]
    for tool, (ok, total) in sorted(res["per_tool"].items(), key=lambda kv: kv[1][0] / kv[1][1]):
        lines.append(f"| {tool} | {ok} | {total} | {ok / total:.0%} |")
    wrong = [r for r in res["results"] if not r["ok"]]
    if wrong:
        lines += ["", "## Wrong routes", "", "| Message | Expected | Got |", "| --- | --- | --- |"]
        lines += [f"| {r['message'][:80].replace('|', '/')} | {' or '.join(sorted(r['expected']))} | {r['got']} |"
                  for r in wrong]
    return "\n".join(line for line in lines if line is not None) + "\n"


# ---------- retrieval ----------
def read_retrieval_cases(path: Path) -> list[dict]:
    """CSV columns: query, relevant (key phrases separated by |), note."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("query") or "").strip()]
    return [{"query": r["query"].strip(),
             "phrases": [p.strip().lower() for p in r["relevant"].split("|") if p.strip()],
             "note": (r.get("note") or "").strip()} for r in rows]


def _squash(text: str) -> str:
    return "".join(text.lower().split())  # PDF text has line breaks and spaces in odd places


def relevant_ids(chunks: list, phrases: list[str]) -> set[str]:
    from jobagent.rag import chunk_body

    # the chunk's own text only: a section name added at its start must not make it "relevant"
    return {c.metadata["chunk_id"] for c in chunks
            if any(_squash(p) in _squash(chunk_body(c)) for p in phrases)}


def evaluate_retrieval(index, cases: list[dict], k: int = 5, modes=("bm25", "dense", "hybrid")) -> dict:
    index._load_bm25()
    chunks = index._chunks
    scored, skipped = [], []
    for c in cases:
        rel = relevant_ids(chunks, c["phrases"])
        if rel:
            scored.append({**c, "relevant": rel})
        else:
            skipped.append(c["query"])  # its phrases match no chunk: fix the labels or the profile
    out = {"n": len(scored), "k": k, "chunks": len(chunks), "skipped": skipped, "modes": {}, "per_query": []}
    for mode in modes:
        recalls, hits, rrs = [], [], []
        for c in scored:
            ids = [d.metadata["chunk_id"] for d in index.search(c["query"], k=k, mode=mode)]
            found = c["relevant"] & set(ids)
            recalls.append(len(found) / len(c["relevant"]))
            hits.append(1.0 if found else 0.0)
            first = next((i for i, cid in enumerate(ids, start=1) if cid in c["relevant"]), None)
            rrs.append(1 / first if first else 0.0)
            out["per_query"].append({"mode": mode, "query": c["query"], "recall": recalls[-1]})
        n = len(scored) or 1
        out["modes"][mode] = {"recall": sum(recalls) / n, "hit": sum(hits) / n, "mrr": sum(rrs) / n}
    return out


def retrieval_markdown(res: dict) -> str:
    k = res["k"]
    lines = ["# Retrieval hit rate (检索命中率)", "",
             f"{res['n']} queries over {res['chunks']} profile chunks, top {k}."]
    if res["chunks"] <= 3 * k:
        lines.append(f"**Warning:** with only {res['chunks']} chunks, the top {k} holds a large share of the "
                     f"whole profile, so these numbers say little. Add more material to profile/ first.")
    lines += ["", f"| Retriever | Recall@{k} | Hit@{k} | MRR |", "| --- | --- | --- | --- |"]
    for mode, m in res["modes"].items():
        lines.append(f"| {mode} | {m['recall']:.1%} | {m['hit']:.1%} | {m['mrr']:.3f} |")
    lines += ["", f"Recall@{k}: share of the relevant chunks found in the top {k}. Hit@{k}: share of queries with "
              f"at least one relevant chunk in the top {k}. MRR: 1 / rank of the first relevant chunk, averaged."]
    if res["skipped"]:
        lines += ["", "Skipped (their key phrases match no chunk): " + "; ".join(res["skipped"])]
    return "\n".join(lines) + "\n"


# ---------- comparing routing versions (paired: the same test messages) ----------
def save_routing_run(path: Path, res: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["message", "expected", "got", "ok"])
        for r in res["results"]:
            w.writerow([r["message"], "|".join(sorted(r["expected"])), r["got"], int(r["ok"])])


def load_routing_run(eval_dir: Path, version: str) -> dict[str, bool]:
    """{message: right first tool?}: from eval/runs/routing-<v>.csv, or rebuilt from an older report
    (every case not listed under "Wrong routes" was right)."""
    snap = eval_dir / "runs" / f"routing-{version}.csv"
    if snap.exists():
        with open(snap, encoding="utf-8-sig", newline="") as f:
            return {r["message"]: r["ok"] == "1" for r in csv.DictReader(f)}
    report = eval_dir / f"routing-{version}.md"
    if not report.exists():
        raise FileNotFoundError(f"No routing results for {version}: run `python evaluate_agent.py routing {version}`")
    text = report.read_text(encoding="utf-8")
    import re
    count = re.search(r"(\d+) test messages", text)
    cases = read_routing_cases(eval_dir / "routing_cases.csv")
    if count:  # cases added after that run are at the end of the file: leave them out of the comparison
        cases = cases[:int(count.group(1))]
    wrong = {line.strip("|").split(" | ")[0].strip() for line in text.split("## Wrong routes")[-1].splitlines()
             if "## Wrong routes" in text and line.startswith("| ") and not line.startswith(("| Message", "| ---"))}
    return {c["message"]: c["message"][:80].replace("|", "/") not in wrong for c in cases}


def compare_routing(a: dict[str, bool], b: dict[str, bool], name_a: str, name_b: str) -> str:
    from jobagent.stats import bootstrap_diff, mcnemar_exact, verdict_text

    keys = sorted(set(a) & set(b))
    items = [{"a": a[k], "b": b[k]} for k in keys]
    acc = lambda rows, v: sum(r[v] for r in rows) / len(rows)
    test = mcnemar_exact([r["a"] for r in items], [r["b"] for r in items])
    ci = bootstrap_diff(items, acc, "a", "b")
    return "\n".join([
        f"# Routing {name_a} vs {name_b} (paired, {len(items)} messages)", "",
        f"| | {name_a} | {name_b} | Difference | 95% CI |", "| --- | --- | --- | --- | --- |",
        f"| Accuracy | {acc(items, 'a'):.1%} | {acc(items, 'b'):.1%} | {100 * ci['diff']:+.1f} pts | "
        f"{100 * ci['low']:+.1f} to {100 * ci['high']:+.1f} pts |", "",
        f"McNemar exact test: fixed {test['b_only']}, broke {test['a_only']}; p = {test['p']:.3f}.", "",
        verdict_text(test["p"], ci["low"], ci["high"]), ""])
