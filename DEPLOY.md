# Deploy the public demo (free)

The demo runs with `DEMO_MODE=1`:

- a **fictional** sample candidate (`demo/profile/demo_cv.md`), never your real CV
- a private, temporary workspace for every visitor (jobs, tracker and CVs disappear on restart)
- the visitor's own DeepSeek key, or your key stored as a **secret** on the host, limited to
  `DEMO_DAILY_LIMIT` messages per day (default 40) for all visitors together
- light retrieval without PyTorch, so it starts fast on a free CPU machine (the full app uses bge-m3)
- example job postings to try with one click

Everything the host needs is in `Dockerfile` (Chinese fonts for the PDF CV included) and
`requirements-demo.txt`. Your `.env`, `profile/` and `data/` are never uploaded (`.dockerignore`).

## Option A: ModelScope 创空间 (opens in China without a VPN)

1. Log in at modelscope.cn → **创空间** → **创建创空间**.
   - Name: `job-agent`, visibility **公开** (public).
   - Type (SDK): **Docker** if it is offered. Otherwise **Streamlit** (see the note below).
2. In the new space: **设置 → 环境变量 / Secrets**, add:
   - `LLM_API_KEY` = your DeepSeek key (as a secret, so visitors never see it)
   - `DEMO_MODE` = `1`
   - optional: `DEMO_DAILY_LIMIT` = `40`, `LLM_MODEL` = `deepseek-flash`
3. Upload the code. In PowerShell, in `D:\Job Agent`:
   ```powershell
   git remote add modelscope https://www.modelscope.cn/studios/<your-name>/job-agent.git
   git push modelscope main
   ```
   Log in with your ModelScope user name and an **access token** (个人中心 → 访问令牌) as the password.
   If the space already has a starter commit, run `git pull modelscope main --allow-unrelated-histories`
   first.
4. The space builds (a few minutes the first time) and shows the app.

**Streamlit type instead of Docker:** the platform installs `requirements.txt`, which includes PyTorch
(slow, large). In the space, replace `requirements.txt` with the content of `requirements-demo.txt`, and
keep `DEMO_MODE=1`. Chinese text in the downloaded PDF CV may show as boxes, because the Docker image is
what installs the Chinese fonts.

## Option B: Hugging Face Spaces (for international viewers)

1. huggingface.co → **New Space** → SDK **Docker**, hardware **CPU basic (free)**.
2. **Settings → Variables and secrets**: secret `LLM_API_KEY`, variable `DEMO_MODE=1`.
3. Push the code to the Space's git address (`git remote add hf https://huggingface.co/spaces/<you>/job-agent`).
   The Space's `README.md` must start with this header (add it in the Space only, not on GitHub):
   ```yaml
   ---
   title: Job Agent
   sdk: docker
   app_port: 7860
   ---
   ```

## Check before sharing the link

- Open the app in a private browser window: the sidebar should say **Public demo** and show 林晓.
- Try an example job, then "make me a CV for this job in Chinese" and download the PDF.
- The daily limit protects your key; you can also leave `LLM_API_KEY` unset so visitors must use their own key.
