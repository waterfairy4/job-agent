# Priya Nair (fictional candidate) — project notes

## Web browsing agent
- Built an agent that completes shopping and booking tasks in a real browser using Playwright and LLM tool calling.
- Planner–executor design: the planner writes a short plan, the executor calls tools (click, type, scroll, read page) and re-plans after failures.
- Long tasks use a memory module that stores visited pages and extracted facts in a vector store and retrieves them when needed.
- On a 200-task subset of WebArena the task success rate rose from 21% to 34%; most remaining failures were dropdown menus and pop-ups.

## Customer-support agent with human approval
- LangGraph agent for a telecom company that looks up orders, issues refunds and opens tickets.
- Refunds above 50 USD pause the graph and wait for a human supervisor to approve; all steps are logged for audit.
- Offline evaluation with 300 recorded conversations: tool-call accuracy 93%, correct escalation 88%.

## Agent evaluation harness
- Wrote a harness that replays recorded tasks against new agent versions and scores task success, number of steps, cost in tokens and unsafe actions.
- Each run is stored with its full trajectory, so failures can be inspected step by step and turned into new regression tests.
- An LLM judge grades open-ended answers; it was calibrated against 150 human-graded trajectories (agreement 86%).

## Document summarisation API
- FastAPI service that summarises long reports with a map-reduce prompt chain; added caching and rate limiting, p95 latency 4 seconds.

## Skills
- Python, TypeScript; LangGraph, Playwright, FastAPI; PostgreSQL.
