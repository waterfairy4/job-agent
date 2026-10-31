from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from jobagent.agent import build_agent, make_checkpointer
from jobagent.api import create_app
from jobagent.tools import build_tools
from tests.fakes import FakeLLM, tool_call
from tests.test_agent import JD, env, handlers  # noqa: F401


def test_chat_approval_tracker_and_jobs(env, tmp_path):
    index, store = env
    jid = store.job_id(JD)
    llm = FakeLLM([tool_call("analyze_job", {"job_text": JD}, "c1"), AIMessage(content="Analyzed."),
                   tool_call("save_job", {"job_id": jid, "company": "ByteDance"}, "c2"), AIMessage(content="Saved.")],
                  handlers())
    graph = build_agent(build_tools(index, store, llm), llm, checkpointer=make_checkpointer(str(tmp_path / "c.sqlite")))
    client = TestClient(create_app(graph, store))

    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/chat", json={"message": f"Analyze:\n{JD}"}).json()
    thread = r["thread_id"]
    assert r["reply"] == "Analyzed." and r["tools"][0]["tool"] == "analyze_job" and r["pending"] is None
    assert client.get(f"/jobs/{jid}").json()["analysis"]["title"] == "LLM Evaluation Engineer"

    r = client.post("/chat", json={"message": "Save it", "thread_id": thread}).json()
    assert r["pending"]["company"] == "ByteDance" and client.get("/applications").json() == []
    assert client.post("/chat", json={"message": "hello", "thread_id": thread}).status_code == 409

    conv = client.get(f"/chat/{thread}").json()
    assert conv["pending"]["action"] == "save_job" and conv["messages"][0]["role"] == "user"

    r = client.post(f"/chat/{thread}/approval", json={"approve": True}).json()
    assert r["reply"] == "Saved." and client.get("/applications").json()[0]["company"] == "ByteDance"
    assert client.post(f"/chat/{thread}/approval", json={"approve": True}).status_code == 404

    assert "## Funnel" in client.get("/analytics").text
    assert client.delete(f"/applications/{jid}").json() == {"deleted": jid}
    assert client.delete(f"/applications/{jid}").status_code == 404
    assert client.get("/jobs/nope").status_code == 404 and client.get("/chat/nope").status_code == 404
    assert client.post("/chat", json={"message": ""}).status_code == 422      # validated by Pydantic


def test_openapi_export(tmp_path, monkeypatch):
    import json
    import sys

    import export_openapi

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["export_openapi.py", "https://tunnel.example.com"])
    export_openapi.main()
    spec = json.loads((tmp_path / "integrations" / "openapi.json").read_text(encoding="utf-8"))
    ops = {op["operationId"] for path in spec["paths"].values() for op in path.values()}
    assert {"chat", "answer_approval", "list_applications", "analytics"} <= ops
    assert spec["servers"] == [{"url": "https://tunnel.example.com"}]
