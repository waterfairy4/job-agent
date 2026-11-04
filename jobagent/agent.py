"""The LangGraph agent: a ReAct loop (think -> call tools -> observe -> repeat).

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                  \\--(no tool calls)--> END

save_job can pause the graph with interrupt(); `step()` reports that as `pending`,
and the next `step(..., resume=True/False)` continues from exactly that point.
"""
import sqlite3
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from jobagent import config

SYSTEM_PROMPT = """You are a job application assistant for one user.
Workflow:
- When the user pastes a job posting, call analyze_job, then match_job with the returned job_id.
  Pass the posting to analyze_job exactly as the user pasted it: do not reformat, shorten, translate or add
  lines (such as "公司：…"). The same posting must give the same job_id, so its analysis and score are reused
  instead of changing from run to run.
  This includes short or partial descriptions (a title plus a few duties or requirements) and requests
  like "analyze this job" or "帮我分析一下这个岗位": analyze first, never judge the fit from memory.
- When the user gives a LINK to a posting, call fetch_job, then match_job. If the page cannot be read,
  ask the user to paste the text.
- Call find_jobs when the user asks you to find or search for jobs. Pass their wishes as `request`
  (role, city, level, e.g. "2027届校招 大模型评测 上海"); ask first only if role or city is unclear.
  Show the shortlist as a table with match score, company, title and link. Never save jobs by yourself:
  the user picks; then use save_job with that job_id and its url.
- Call research_company when the user asks about the company or team, wants interview preparation,
  or wants the CV tailored to the team. Cite sources as [n] with their links.
- Use search_profile for questions about the user's own background.
- Call tailor_cv when the user wants better CV bullets for a job.
- Call build_cv when the user wants a CV / resume for a job ("update my CV", "make me a CV").
  It makes a one-page PDF and an editable Word copy. Pass language="Chinese" or "English" if the user
  asks for one. Describe changes only from the tool's lines (e.g. if it says the objective is unchanged,
  do not say it was reworded) and repeat every CHECK line.
- Call plan_gaps when the user asks how to close gaps or what to do to fit a job better.
- Call save_job only when the user wants to save or track a job, or reports progress on one (applied,
  interview, offer, rejected); the user will be asked to approve. Pass cv_version ("tailored" or
  "original") and channel (how they applied) when the user mentions them. News about a job id or a
  saved job ("X rejected me", "我拿到了面试", "got an offer from …") is such progress: call save_job with
  that status, then reply; never only comment on it.
- If the posting does not name the company, say so once and ask the user; pass their answer to save_job.
- If the user shared a link to the posting, pass it to save_job as `url`.
Rules:
- Never invent the user's experience. Base every claim about the user on tool results.
  When you summarize a match, CV or plan, name only roles, employers, degrees, projects, tools and numbers
  that appear word for word in the tool results of this conversation; never add a role or title from your
  own reading (e.g. "research assistant" when the results only show a paper). If unsure, leave it out.
- A verdict of "missing" means only that the retrieved evidence did not show the item. For basic facts the
  user is likely to have (degree, enrollment, a language), say "not found in your documents" and suggest
  checking the profile, not that the user lacks it.
- Be concise: give the score, strengths on responsibilities, the main gaps, and one next step.
- Language: reply in the language of the user's own words. If the message is only a pasted
  job posting, reply in the posting's language. Tool output in another language does not change this."""


def tool_error(e: Exception) -> str:
    """A tool that fails (bad arguments, network down, a bug) returns this message to the model instead
    of crashing the turn, so the agent can tell the user what went wrong. Invalid arguments are the
    model's own mistake: it gets the validation details and may correct the call."""
    if type(e).__name__ == "ToolInvocationError":
        return str(e)
    return (f"Tool error ({type(e).__name__}): {str(e)[:300]}. Tell the user briefly what failed and what "
            f"they can do; do not repeat the same call.")


def build_agent(tools: list, llm, checkpointer=None):
    model = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        reply = model.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=tool_error))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool calls -> "tools", else END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def make_checkpointer(path: str | None = None):
    """SQLite checkpointer: every step of every conversation is saved to one file, so a restart
    continues where it stopped (including an approval that was waiting). "" = in memory only."""
    path = config.CHECKPOINT_DB if path is None else path
    if not path:
        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Streamlit serves each rerun from a worker thread; SqliteSaver serializes access with its own lock.
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def new_thread() -> str:
    return uuid.uuid4().hex


def pending_approval(graph, thread_id: str):
    """The approval request a conversation is waiting on (e.g. after a restart), or None."""
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    waiting = [i for task in state.tasks for i in task.interrupts]
    return waiting[0].value if waiting else None


def traces(messages: list) -> dict[int, list[dict]]:
    """Tool calls grouped under the assistant reply that follows them: {message index: [calls]}.
    Rebuilt from the saved messages, so the tool details also survive a restart."""
    out, current, by_id = {}, [], {}
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            current = []
        elif isinstance(m, AIMessage):
            for c in m.tool_calls:
                t = {"id": c["id"], "tool": c["name"], "args": c["args"]}
                current.append(t)
                by_id[c["id"]] = t
            if m.content and not m.tool_calls:
                out[i], current = current, []
        elif isinstance(m, ToolMessage) and m.tool_call_id in by_id:
            by_id[m.tool_call_id]["result"] = str(m.content)
    return out


def step(graph, thread_id: str, message: str | None = None, resume=None) -> dict:
    """Run one user turn (or resume after an approval). Returns:
    reply   - the agent's last text
    pending - an approval request dict if the graph paused, else None
    trace   - tool calls made in this turn, for showing the user what happened
    """
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": config.MAX_STEPS}
    before = len(graph.get_state(cfg).values.get("messages", []))
    payload = Command(resume=resume) if resume is not None else {"messages": [HumanMessage(message)]}
    try:
        result = graph.invoke(payload, cfg)
    except GraphRecursionError:
        return {"reply": f"Stopped after {config.MAX_STEPS} steps to avoid a loop. Try a more specific request.",
                "pending": None, "trace": []}

    new = result["messages"][before:]
    trace = []
    for m in new:
        if isinstance(m, AIMessage):
            trace += [{"id": c["id"], "tool": c["name"], "args": c["args"]} for c in m.tool_calls]
        elif isinstance(m, ToolMessage):
            for t in trace:
                if t["id"] == m.tool_call_id:
                    t["result"] = str(m.content)[:300]
    interrupts = result.get("__interrupt__") or []
    last = result["messages"][-1]
    reply = last.content if isinstance(last, AIMessage) and not interrupts else ""
    return {"reply": reply, "pending": interrupts[0].value if interrupts else None, "trace": trace}
