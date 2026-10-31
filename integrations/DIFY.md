# Use Job Agent as a tool in Dify (or Coze)

Job Agent is built in code (LangGraph). Low-code platforms such as Dify and Coze can still use it:
its FastAPI service describes itself in OpenAPI, and both platforms can turn an OpenAPI description
into a tool. A Dify app can then call Job Agent like any other tool, for example:
"paste a job → Job Agent analyzes and matches it → the Dify app formats the answer for a team".

## 1. Start the API

```bash
uvicorn jobagent.api:app --host 0.0.0.0 --port 8000
```

Check http://localhost:8000/docs in the browser. `--host 0.0.0.0` lets a Docker container reach it.

## 2. Export the description

```bash
python export_openapi.py                       # server: http://host.docker.internal:8000
```

This writes `integrations/openapi.json` with 8 operations: `chat`, `answer_approval`,
`get_conversation`, `list_applications`, `delete_application`, `get_job`, `analytics`, `health`.
`host.docker.internal` is how a program inside Docker (Dify) reaches your own computer.

## 3. Dify (self-hosted with Docker)

1. Install Dify with Docker Compose, following the official guide in the `langgenius/dify` repository.
2. In Dify: **Tools → Custom → Create custom tool**, give it a name (Job Agent), and paste the content
   of `integrations/openapi.json` into the schema box. Dify lists the operations it found.
3. Test the `chat` operation with `{"message": "帮我分析一下这个岗位：…"}`.
4. Create an **Agent** or **Workflow** app and add the Job Agent tool. Example workflow:
   Start (job text) → Job Agent `chat` → LLM node ("summarize the match for a hiring manager") → End.

Notes:
- Dify versions name menus slightly differently; look for "custom tool" and "OpenAPI schema".
- FastAPI writes OpenAPI 3.1. If your Dify version rejects the schema, report the error message:
  the export can be adjusted.
- `chat` may return `pending` (an approval, e.g. "save this job?"). Answer it with `answer_approval`
  and the same `thread_id`; nothing is saved without that call.

## 4. Coze (扣子)

Coze runs in the cloud, so it cannot reach `localhost` on your laptop. You need a public HTTPS address
for the API, e.g. a tunnel service (cpolar, natapp, or similar), then:

```bash
python export_openapi.py https://<your-tunnel-address>
```

and create a Coze plugin from the API description (Coze can import an OpenAPI file when creating a
plugin from an existing service; check its current docs). Anyone with the address can call your API,
so stop the tunnel when you are done.

## What this shows

The same agent is reachable from a web UI (Streamlit), a terminal, an HTTP API, and a low-code
platform. The agent logic, the approvals and the tracker stay in one place.
