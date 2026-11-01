"""The LangGraph agent: a ReAct loop (think -> call tools -> observe -> repeat).

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                  \\--(no tool calls)--> END

save_job can pause the graph with interrupt(); `step()` reports that as `pending`,
and the next `step(..., resume=True/False)` continues from exactly that point.
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from jobagent import config

SYSTEM_PROMPT = """You are a job application assistant for one user.
Workflow:
- When the user gives a job description, call analyze_job, then match_job with the returned job_id.
- Use search_profile for questions about the user's own background.
- Call tailor_cv when the user wants to prepare an application or improve their CV for a job.
- Call save_job only when the user wants to save or track a job; the user will be asked to approve.
Rules:
- Never invent the user's experience. Base every claim about the user on tool results.
- Be concise: give the score, the main strengths, the main gaps, and one next step.
- Reply in the language the user writes in."""


def build_agent(tools: list, llm, checkpointer=None):
    model = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        reply = model.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool calls -> "tools", else END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def new_thread() -> str:
    return uuid.uuid4().hex


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
