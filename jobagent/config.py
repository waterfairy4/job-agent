"""All settings in one place. Override any value in a .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

# Paths
PROFILE_DIR = os.getenv("PROFILE_DIR", "profile")   # your CV, papers, notes (.pdf / .md / .txt)
DATA_DIR = os.getenv("DATA_DIR", "data")            # vector DB, jobs, drafts, reports

# Chunking (characters; works for Chinese and English)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Embeddings (runs locally, free). bge-m3 is multilingual (~2 GB).
# Quick English-only test: EMBED_MODEL=BAAI/bge-small-en-v1.5
# Slow downloads on campus: HF_ENDPOINT=https://hf-mirror.com
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))              # search_profile results
MATCH_TOP_K = int(os.getenv("MATCH_TOP_K", "3"))  # evidence per job requirement

# Agent safety: max graph steps per user message (stops runaway loops)
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))

# LLM (any OpenAI-compatible API with tool calling)
# DeepSeek: LLM_BASE_URL=https://api.deepseek.com  LLM_MODEL=deepseek-chat
# Qwen:     LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  LLM_MODEL=qwen-plus
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
