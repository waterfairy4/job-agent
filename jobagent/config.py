"""All settings in one place. Override any value in a .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

# Paths
# Public demo (hosting): see the DEMO_* settings below and jobagent/demo.py
DEMO_MODE = os.getenv("DEMO_MODE", "0") in ("1", "true", "yes")
PROFILE_DIR = os.getenv("PROFILE_DIR", "demo/profile" if DEMO_MODE else "profile")  # your CV, papers, notes
DATA_DIR = os.getenv("DATA_DIR", "data")            # vector DB, jobs, drafts, reports
# Which file in profile/ is your CV (for build_cv). Empty: a file named like cv/resume/简历, or the only PDF.
CV_FILE = os.getenv("CV_FILE", "")

# Chunking (characters; works for Chinese and English)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
# Optional: cut files at their section titles first and start each chunk with its section name.
# Off by default: on 28 retrieval queries it did not help (hybrid MRR 0.964 both ways, recall@5 87.8% -> 82.3%).
SECTION_CHUNKS = os.getenv("SECTION_CHUNKS", "0") in ("1", "true", "yes")

# Embeddings (runs locally, free). bge-m3 is multilingual (~2 GB).
# Quick English-only test: EMBED_MODEL=BAAI/bge-small-en-v1.5
# Slow downloads on campus: HF_ENDPOINT=https://hf-mirror.com
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))              # search_profile results
MATCH_TOP_K = int(os.getenv("MATCH_TOP_K", "5"))  # evidence chunks per job item (v1 used 3)
# Hybrid fusion: weight of the BM25 list in Reciprocal Rank Fusion (dense = 1.0). 1.0 = equal weights.
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "1.0"))
# Optional reranker: a cross-encoder re-scores the top RERANK_POOL hybrid results (needs sentence-transformers,
# one-time model download; slow downloads: HF_ENDPOINT=https://hf-mirror.com). Off by default until measured.
RERANK = os.getenv("RERANK", "0") in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_POOL = int(os.getenv("RERANK_POOL", "20"))
# Expand each job item into concrete search terms before retrieval ("DL frameworks" -> PyTorch, ...)
QUERY_EXPANSION = os.getenv("QUERY_EXPANSION", "1") not in ("0", "false", "no")

# Conversation memory (LangGraph checkpointer): a SQLite file, so chats and pending approvals survive a
# restart. Empty = in memory only (lost on restart).
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", os.path.join(DATA_DIR, "checkpoints.sqlite"))

# Agent safety: max graph steps per user message (stops runaway loops)
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))

# LLM (any OpenAI-compatible API with tool calling)
# DeepSeek: LLM_BASE_URL=https://api.deepseek.com  LLM_MODEL=deepseek-flash (or deepseek-v4-pro)
# Qwen:     LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL=qwen-plus
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")
# DeepSeek only: keep "disabled" (thinking mode + tools needs reasoning passed back; see llm.py)
LLM_THINKING = os.getenv("LLM_THINKING", "disabled")
# Optional stronger model just for writing the CV (few calls per CV), e.g. deepseek-v4-pro
CV_MODEL = os.getenv("CV_MODEL", "")
# CV fonts: Latin text and Chinese text (both installed on Windows by default)
CV_FONT_LATIN = os.getenv("CV_FONT_LATIN", "Times New Roman")
CV_FONT_CJK = os.getenv("CV_FONT_CJK", "SimSun")
# Extra folder with fonts for the PDF (system fonts are found automatically)
CV_FONT_DIR = os.getenv("CV_FONT_DIR", "")
# Your standing CV rules (e.g. "keep Korean", "never add words to my bullets"), read by build_cv
CV_PREFERENCES_FILE = os.getenv("CV_PREFERENCES_FILE", "cv_preferences.md")
# Applicant tracking systems (ATS): build_cv lists the posting's key terms, and adds to the skills table the
# missing ones that your own documents show (never others). 0 = report only, never add.
CV_ATS_KEYWORDS = os.getenv("CV_ATS_KEYWORDS", "1") not in ("0", "false", "no")
# ATS-friendly PDF layout: dates and GPA on the same line as the school or project (not right-aligned),
# skills as "label: values" lines, so parsers keep them together. Off = the classic layout.
CV_ATS_LAYOUT = os.getenv("CV_ATS_LAYOUT", "0") in ("1", "true", "yes")

# App look: "cute" (pink, background picture) or "plain"
APP_THEME = os.getenv("APP_THEME", "cute").lower()

# Public demo (hosting): a fictional sample profile, a private temporary workspace per visitor, and either
# the visitor's own API key or the owner's key (LLM_API_KEY, kept secret on the host) with a daily limit.
DEMO_PROFILE_DIR = PROFILE_DIR if DEMO_MODE else "demo/profile"
DEMO_DAILY_LIMIT = int(os.getenv("DEMO_DAILY_LIMIT", "40"))  # messages per day on the owner's key
APP_BACKGROUND = os.getenv("APP_BACKGROUND", "assets/background.jpg")

# Web search for research_company and find_jobs: "baidu" (百度, via Qianfan 千帆), "bocha" (博查) or "tavily"
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "bocha").lower()
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
# Network calls (search, job pages): attempts per call and the first wait in seconds (doubles each time)
HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
HTTP_BACKOFF = float(os.getenv("HTTP_BACKOFF", "1.0"))
# find_jobs searches only these sites (job boards with one posting per page), so news and
# forum threads don't fill the results. Comma separated; empty = the whole web.
JOB_SITES = [s.strip() for s in os.getenv(
    "JOB_SITES",
    "www.nowcoder.com,www.yingjiesheng.com,www.shixiseng.com,www.zhipin.com,www.liepin.com,"
    "jobs.bytedance.com,careers.tencent.com,talent.alibaba.com,career.huawei.com,app.mokahr.com",
).split(",") if s.strip()]
# Saved browser login (made by `python login.py`): job pages are opened with your own session.
# Only cookies are stored here, never your password. Delete the folder to log out.
BROWSER_PROFILE = os.getenv("BROWSER_PROFILE", os.path.join(DATA_DIR, "browser"))
# Only results from roughly this period: "week", "month", "semiyear" or "year"
JOB_RECENCY = os.getenv("JOB_RECENCY", "semiyear")
