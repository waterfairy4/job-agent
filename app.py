"""Web UI:  streamlit run app.py"""
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from jobagent import config
from jobagent.agent import new_thread, step
from jobagent.app_factory import create

st.set_page_config(page_title="Job Agent", page_icon="🧭", layout="wide")


@st.cache_resource
def services():
    return create()


try:
    graph, index, store = services()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

ss = st.session_state
ss.setdefault("thread", new_thread())
ss.setdefault("pending", None)
ss.setdefault("traces", {})  # message index -> tool calls made before it
cfg = {"configurable": {"thread_id": ss.thread}}

# ---------- Sidebar: profile + tracker ----------
with st.sidebar:
    st.header("Profile")
    st.caption(f"{index.size()} chunks indexed from `{config.PROFILE_DIR}/`")
    if st.button("Re-index profile"):
        with st.spinner("Indexing..."):
            st.success(f"Indexed {index.build()} chunks.")
    st.header("Tracker")
    rows = store.tracker()
    if rows:
        df = pd.DataFrame(rows)[["company", "title", "score", "status", "deadline"]]
        st.dataframe(df.sort_values("score", ascending=False), hide_index=True, width="stretch")
    else:
        st.caption("No saved jobs yet.")
    if st.button("New conversation"):
        ss.thread, ss.pending, ss.traces = new_thread(), None, {}
        st.rerun()

# ---------- Chat history ----------
st.title("🧭 Job Agent")
st.caption("Paste a job description to match it, ask about your profile, or ask me to tailor your CV.")

messages = graph.get_state(cfg).values.get("messages", [])
for i, m in enumerate(messages):
    if isinstance(m, HumanMessage):
        st.chat_message("user").markdown(m.content)
    elif isinstance(m, AIMessage) and m.content:
        with st.chat_message("assistant"):
            for t in ss.traces.get(i, []):
                with st.expander(f"🔧 {t['tool']}", expanded=False):
                    st.json(t["args"])
                    if "result" in t:
                        st.code(t["result"])
            st.markdown(m.content)


def run(**kwargs):
    with st.spinner("Working..."):
        out = step(graph, ss.thread, **kwargs)
    n = len(graph.get_state(cfg).values.get("messages", [])) - 1
    ss.traces[n] = ss.traces.get(n, []) + out["trace"]
    ss.pending = out["pending"]
    st.rerun()


# ---------- Approval request ----------
if ss.pending:
    p = ss.pending
    with st.chat_message("assistant"):
        score = f" (match {p['score']}%)" if p.get("score") is not None else ""
        st.warning(f"Save **{p['company']} – {p['title']}**{score} to your tracker as **{p['status']}**?")
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve", width="stretch"):
            run(resume=True)
        if c2.button("❌ Reject", width="stretch"):
            run(resume=False)
elif prompt := st.chat_input("Paste a job description or ask something..."):
    run(message=prompt)
