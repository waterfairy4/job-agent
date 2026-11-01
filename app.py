"""Web UI:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from jobagent import config, theme
from jobagent.agent import new_thread, pending_approval, step, traces
from jobagent.app_factory import create
from jobagent.demo import EXAMPLE_JOBS

st.set_page_config(page_title="Job Agent", page_icon="💗" if config.APP_THEME != "plain" else "🧭",
                   layout="wide")
ui = theme.apply(st)


@st.cache_resource
def services():
    return create()


@st.cache_resource
def demo_shared():
    """Demo mode: one index of the fictional profile and one daily budget for the whole server."""
    import tempfile

    from jobagent import demo

    index, embeddings_name = demo.build_index(demo.new_workspace())
    limit = demo.DailyLimit(Path(tempfile.gettempdir()) / "jobagent-usage.json", config.DEMO_DAILY_LIMIT)
    return index, embeddings_name, limit


def demo_services():
    """Demo mode: each visitor gets a private temporary workspace, and uses their own key or the owner's."""
    import hashlib

    from jobagent import demo
    from jobagent.store import JobStore

    index, embeddings_name, limit = demo_shared()
    with st.sidebar:
        st.info("**Public demo** with a fictional sample candidate (林晓). Your data is temporary and private "
                f"to this browser session. Retrieval: {embeddings_name}.")
        key = st.text_input("Your DeepSeek API key (optional)", type="password",
                            help="Used only in this session, never stored. Without it, the demo key is used "
                                 f"({limit.remaining()} of {config.DEMO_DAILY_LIMIT} messages left today).")
    if not key and not config.LLM_API_KEY:
        st.warning("Enter your DeepSeek API key in the sidebar to try the demo.")
        st.stop()
    if not key and limit.remaining() == 0:  # the chat stays readable; only new messages are blocked
        st.sidebar.warning("Today's demo limit is used up: enter your own DeepSeek API key to continue.")
    s = st.session_state
    s.demo_owner_key = not key
    if "demo_store" not in s:
        s.demo_store = JobStore(str(demo.new_workspace()))
    signature = hashlib.sha256((key or "owner").encode()).hexdigest()
    if s.get("demo_signature") != signature:
        s.demo_graph = create(index=index, store=s.demo_store, api_key=key or None, checkpoint_db="")[0]
        s.demo_signature = signature
        s.pop("thread", None)
    return s.demo_graph, index, s.demo_store, limit


demo_limit = None
try:
    if config.DEMO_MODE:
        graph, index, store, demo_limit = demo_services()
    else:
        graph, index, store = services()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

ss = st.session_state
if "thread" not in ss:  # reopen the last conversation after a restart (it is saved in SQLite)
    ss.thread = store.last_thread() or new_thread()
    store.set_last_thread(ss.thread)
    ss.pending = pending_approval(graph, ss.thread)
cfg = {"configurable": {"thread_id": ss.thread}}

# ---------- Sidebar: profile + tracker ----------
with st.sidebar:
    st.header("🎀 Profile" if ui["user"] else "Profile")
    st.caption(f"{index.size()} chunks indexed from `{config.PROFILE_DIR}/`")
    if not config.DEMO_MODE and st.button("Re-index profile"):
        with st.spinner("Indexing..."):
            st.success(f"Indexed {index.build()} chunks.")
    st.header("💌 Tracker" if ui["user"] else "Tracker")
    rows = store.tracker()
    if rows:
        df = pd.DataFrame(rows).reindex(columns=["company", "title", "score", "status",
                                                 "saved_on", "applied_on", "deadline", "url"])
        df["saved_on"] = df["saved_on"].fillna(pd.DataFrame(rows).get("updated"))  # rows from older versions
        df["company"] = df["company"].replace("", "Unknown company")
        df["url"] = df["url"].fillna("").replace("", None)  # old rows have no link
        st.dataframe(df.sort_values("score", ascending=False), hide_index=True, width="stretch",
                     column_config={"url": st.column_config.LinkColumn("link", display_text="open")})
        with st.expander("Delete a job"):
            labels = {f"{r.get('company') or 'Unknown company'} – {r['title']} "
                      f"({r.get('saved_on') or r.get('updated', '')})": r["id"] for r in rows}
            choice = st.selectbox("Job", list(labels), index=None, placeholder="Choose a job")
            sure = st.checkbox("Yes, remove it from the tracker", disabled=choice is None)
            if st.button("🗑 Delete", disabled=not (choice and sure)):
                store.untrack(labels[choice])
                st.rerun()
    else:
        st.caption("No saved jobs yet.")
    with st.expander("📊 Analytics" if ui["user"] else "Analytics"):
        from jobagent.analytics import report as analytics_report

        st.markdown(analytics_report(store.tracker(), store.all_jobs()))
    if st.button("✨ New conversation" if ui["user"] else "New conversation"):
        ss.thread, ss.pending = new_thread(), None
        store.set_last_thread(ss.thread)
        st.rerun()

# ---------- Chat history ----------
st.title(ui["title"])
st.caption(ui["caption"])

messages = graph.get_state(cfg).values.get("messages", [])
calls = traces(messages)  # tool calls under each reply, rebuilt from the saved conversation
for i, m in enumerate(messages):
    if isinstance(m, HumanMessage):
        st.chat_message("user", avatar=ui["user"]).markdown(m.content)
    elif isinstance(m, AIMessage) and m.content:
        with st.chat_message("assistant", avatar=ui["assistant"]):
            for t in calls.get(i, []):
                with st.expander(f"🔧 {t['tool']}", expanded=False):
                    st.json(t["args"])
                    if "result" in t:
                        st.code(t["result"][:1500])
            st.markdown(m.content)
            for j, t in enumerate(calls.get(i, [])):
                if t["tool"] != "build_cv":
                    continue
                files = [Path(line[len("FILE: "):].strip()) for line in t.get("result", "").splitlines()
                         if line.startswith("FILE: ")]
                for k, path in enumerate(f for f in files if f.exists()):
                    pdf = path.suffix == ".pdf"
                    st.download_button("📄 Download CV (PDF)" if pdf else "📝 Editable Word copy",
                                       path.read_bytes(), file_name=path.name, key=f"dl-{i}-{j}-{k}",
                                       mime="application/pdf" if pdf else
                                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

def run(**kwargs):
    if demo_limit is not None and ss.get("demo_owner_key") and "message" in kwargs and not demo_limit.use():
        st.warning("The demo's daily limit is used up. Enter your own DeepSeek API key in the sidebar.")
        return
    with st.spinner(ui["thinking"]):
        out = step(graph, ss.thread, **kwargs)
    ss.pending = out["pending"]
    st.rerun()


# ---------- Approval request ----------
if ss.pending:
    p = ss.pending
    with st.chat_message("assistant", avatar=ui["assistant"]):
        score = f" (match {p['score']}%)" if p.get("score") is not None else ""
        link = f" with link {p['url']}" if p.get("url") else ""
        st.warning(f"Save **{p['company']} – {p['title']}**{score} to your tracker as **{p['status']}**{link}?")
        c1, c2 = st.columns(2)
        if c1.button(ui["yes"], width="stretch"):
            run(resume=True)
        if c2.button(ui["no"], width="stretch"):
            run(resume=False)
elif config.DEMO_MODE and not messages and (example := st.selectbox(
        "Try an example job posting", ["", *EXAMPLE_JOBS], format_func=lambda k: k or "Choose an example…")):
    run(message="Analyze this job and match it to my profile:\n\n" + EXAMPLE_JOBS[example])
elif prompt := st.chat_input("Paste a job, a link, or ask me anything… 💬"):
    run(message=prompt)
