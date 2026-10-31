"""Web UI:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from jobagent import config, theme
from jobagent.agent import new_thread, pending_approval, step, traces
from jobagent.app_factory import create

st.set_page_config(page_title="Job Agent", page_icon="💗" if config.APP_THEME != "plain" else "🧭",
                   layout="wide")
ui = theme.apply(st)


@st.cache_resource
def services():
    return create()


try:
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
    if st.button("Re-index profile"):
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
elif prompt := st.chat_input("Paste a job, a link, or ask me anything… 💬"):
    run(message=prompt)
