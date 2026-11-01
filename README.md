# Job Agent

An AI agent that matches job descriptions to your profile, tailors your CV with cited evidence,
and tracks your applications. Built with **LangChain** (RAG) and **LangGraph** (agent).
Works with Chinese and English job postings.

```
You: "Match this job and prepare my application"
            ┌────────────── LangGraph agent (ReAct loop) ───────────────┐
            │      think → call tools → observe → repeat → answer       │
            └───┬────────────┬────────────┬────────────┬─────────────┬──┘
                ▼            ▼            ▼            ▼             ▼
         search_profile  analyze_job   match_job    tailor_cv     save_job
         hybrid RAG      structured    cited        grounded      pauses for
         over your CV    output        verdicts     CV bullets    your approval
```

## What makes it more than a demo

| Problem | Solution |
| --- | --- |
| Dense search misses exact terms ("LoRA", "PyTorch", "评测") | **Hybrid retrieval**: Chroma vectors + BM25, merged with Reciprocal Rank Fusion |
| Chinese has no spaces, so word-based search and chunking fail | Chinese-aware splitter (breaks at 。！？) and character-bigram BM25 tokens |
| LLM output is free text and breaks parsers | **Structured outputs** with Pydantic schemas |
| LLMs flatter and invent experience | Code-enforced grounding: "strong" needs a real citation, ungrounded CV bullets are dropped |
| Agents can take wrong actions | **Human-in-the-loop**: `save_job` pauses the graph (`interrupt`) until you approve |
| Agents can loop | Step limit per message (`MAX_STEPS`) |
| Same job pasted twice costs twice | Job id = hash of the text; repeated analysis is served from cache |
| Hard to change safely | Offline tests with a scripted fake model + CI on every push |

## Project layout

| File | What it does |
| --- | --- |
| `jobagent/rag.py` | Load PDFs/notes, split, index (Chroma + BM25), hybrid search |
| `jobagent/schemas.py` | Pydantic schemas for structured outputs |
| `jobagent/tools.py` | The five tools and their grounding rules |
| `jobagent/agent.py` | LangGraph graph, checkpointer, `step()` with interrupt handling |
| `jobagent/store.py` | Job drafts, reports and the tracker (JSON files) |
| `app.py` | Web UI (Streamlit) |
| `cli.py` | Terminal chat |
| `ingest.py` | Indexes `profile/` |
| `tests/` | End-to-end tests with fake models (no API cost) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your DeepSeek or Qwen key in .env
mkdir profile                      # put your CV (PDF), papers, and a projects.md here
python ingest.py
```

## Use

```bash
streamlit run app.py     # web UI
python cli.py            # or the terminal (/job to paste a posting, /jobs for the tracker)
pytest                   # tests, offline and free
```

Try:
- *paste a job description* → analysis, match score, strengths and gaps
- "Prepare my CV for this job" → tailored bullets, each citing your own documents
- "Save it as applied" → approval card → tracker
- "Which of my projects show evaluation experience?" → answered from your profile with sources

Reports are saved in `data/reports/`. The matcher only finds what is written down,
so add skills and projects that are not on your CV to `profile/projects.md`.

## Cost

Embeddings and search run locally (free). An agent turn uses a few DeepSeek calls:
analysis is 1 call, matching is 1 call per requirement (≤ 12), CV tailoring is 1 call.

## Next: evaluation layer

- Label ~50 requirements yourself → measure how often the judge agrees with you
- Retrieval: recall@k for dense vs BM25 vs hybrid, on Chinese and English queries
- Agent: correct tool sequence and task success over repeated runs
- Trace latency, tokens and cost per turn
