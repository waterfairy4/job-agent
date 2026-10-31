"""HTTP API in front of the agent (FastAPI):  uvicorn jobagent.api:app --port 8000
Interactive docs: http://localhost:8000/docs

  POST   /chat                       send a message (new conversation if no thread_id)
  POST   /chat/{thread_id}/approval  answer a waiting approval (e.g. "save this job?")
  GET    /chat/{thread_id}           the conversation and any waiting approval
  GET    /applications               the tracker (SQLite)
  DELETE /applications/{job_id}      remove a job from the tracker
  GET    /jobs/{job_id}              a job's analysis and match
  GET    /analytics                  funnel, CV A/B test and job segments (Markdown)
  GET    /health

The same agent, tools, checkpointer and tracker as the web UI; conversations are shared through the
SQLite checkpointer, so a thread started here can be continued in the UI and the other way round.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from jobagent.agent import new_thread, pending_approval, step


class ChatIn(BaseModel):
    message: str = Field(min_length=1, description="Your message, a pasted job posting or a link")
    thread_id: str | None = Field(default=None, description="Continue this conversation; empty = new one")


class ApprovalIn(BaseModel):
    approve: bool


class ToolCall(BaseModel):
    tool: str
    args: dict
    result: str | None = None


class ChatOut(BaseModel):
    thread_id: str
    reply: str
    pending: dict | None = Field(description="An action waiting for your approval, if any")
    tools: list[ToolCall]


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    thread_id: str
    messages: list[Message]
    pending: dict | None


def create_app(graph, store) -> FastAPI:
    api = FastAPI(title="Job Agent API", version="1.0",
                  description="Match jobs to your profile, build CVs and track applications.")

    def _out(thread_id: str, out: dict) -> ChatOut:
        return ChatOut(thread_id=thread_id, reply=out["reply"], pending=out["pending"],
                       tools=[ToolCall(tool=t["tool"], args=t["args"], result=t.get("result")) for t in out["trace"]])

    @api.get("/health", operation_id="health")
    def health():
        return {"status": "ok"}

    @api.post("/chat", response_model=ChatOut, operation_id="chat",
              summary="Send a message to Job Agent (job text, link, or request)")
    def chat(body: ChatIn):
        thread_id = body.thread_id or new_thread()
        if pending_approval(graph, thread_id):
            raise HTTPException(409, "This conversation is waiting for an approval: "
                                     f"POST /chat/{thread_id}/approval first.")
        return _out(thread_id, step(graph, thread_id, body.message))

    @api.post("/chat/{thread_id}/approval", response_model=ChatOut, operation_id="answer_approval",
              summary="Approve or reject the action the conversation is waiting on")
    def approval(thread_id: str, body: ApprovalIn):
        if not pending_approval(graph, thread_id):
            raise HTTPException(404, "Nothing is waiting for approval in this conversation.")
        return _out(thread_id, step(graph, thread_id, resume=body.approve))

    @api.get("/chat/{thread_id}", response_model=Conversation, operation_id="get_conversation",
             summary="Read a conversation and any waiting approval")
    def conversation(thread_id: str):
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        msgs = state.values.get("messages", []) if state.values else []
        if not msgs:
            raise HTTPException(404, "No conversation with this id.")
        shown = [Message(role="user" if isinstance(m, HumanMessage) else "assistant", content=str(m.content))
                 for m in msgs if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.content)]
        return Conversation(thread_id=thread_id, messages=shown, pending=pending_approval(graph, thread_id))

    @api.get("/applications", operation_id="list_applications", summary="The application tracker")
    def applications():
        return store.tracker()

    @api.delete("/applications/{job_id}", operation_id="delete_application",
                summary="Remove a job from the tracker")
    def delete_application(job_id: str):
        if not store.untrack(job_id):
            raise HTTPException(404, "This job is not in the tracker.")
        return {"deleted": job_id}

    @api.get("/analytics", response_class=PlainTextResponse, operation_id="analytics",
             summary="Funnel, CV A/B test and job segments (Markdown)")
    def analytics():
        """Funnel, response rate by channel, tailored-vs-original CV A/B test and job segments (Markdown)."""
        from jobagent.analytics import report

        return report(store.tracker(), store.all_jobs())

    @api.get("/jobs/{job_id}", operation_id="get_job", summary="A job's analysis and match")
    def job(job_id: str):
        found = store.get(job_id)
        if not found:
            raise HTTPException(404, "Unknown job_id. Analyze the job first.")
        return {k: found[k] for k in ("id", "url", "analysis", "match") if k in found}

    return api


def _create_default() -> FastAPI:
    from jobagent.app_factory import create

    graph, _, store = create()
    return create_app(graph, store)


def __getattr__(name):  # `uvicorn jobagent.api:app` builds the real app only when it is asked for
    if name == "app":
        globals()["app"] = _create_default()
        return globals()["app"]
    raise AttributeError(name)
