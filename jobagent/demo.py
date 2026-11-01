"""Public demo mode (DEMO_MODE=1), for free hosting such as ModelScope 创空间 or Hugging Face Spaces.

- Profile: a FICTIONAL sample candidate (demo/profile/), never the owner's real CV.
- Each visitor gets a private temporary workspace (jobs, tracker, CVs) that disappears on restart.
- API key: the visitor's own DeepSeek key, or the owner's key (LLM_API_KEY, stored as a host secret)
  limited to DEMO_DAILY_LIMIT messages per day for all visitors together.
- Embeddings: if sentence-transformers is not installed (the light demo image), a small hashing
  embedding is used, so the demo starts in seconds without downloading a 2 GB model. Hybrid search
  then relies mostly on BM25; the full app uses bge-m3.
"""
import hashlib
import json
import math
import tempfile
from datetime import date
from pathlib import Path

from langchain_core.embeddings import Embeddings

from jobagent import config
from jobagent.rag import tokenize


class HashEmbeddings(Embeddings):
    """Bag of tokens (words, Chinese characters and bigrams) hashed into a fixed-size unit vector."""

    def __init__(self, size: int = 512):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.size
        for t in tokenize(text):
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
            v[h % self.size] += 1.0 if (h >> 64) % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def demo_embeddings():
    """The real model when it is installed, else the light hashing embedding."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return HashEmbeddings(), "light keyword vectors (demo)"
    from jobagent.rag import default_embeddings

    return default_embeddings(), config.EMBED_MODEL


def build_index(workdir: Path):
    """The shared index of the fictional profile (built once per server start)."""
    from jobagent.rag import ProfileIndex

    embeddings, name = demo_embeddings()
    index = ProfileIndex(embeddings, persist_dir=str(workdir / "chroma"))
    index.build(config.DEMO_PROFILE_DIR)
    return index, name


def new_workspace() -> Path:
    return Path(tempfile.mkdtemp(prefix="jobagent-visitor-"))


class DailyLimit:
    """Counts messages sent with the owner's key, per calendar day, in a small JSON file."""

    def __init__(self, path: Path, limit: int):
        self.path, self.limit = Path(path), limit

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return data if data.get("day") == date.today().isoformat() else {"day": date.today().isoformat(), "used": 0}

    def remaining(self) -> int:
        return max(0, self.limit - self._read()["used"])

    def use(self) -> bool:
        """Take one message from today's budget. False when it is used up."""
        data = self._read()
        if data["used"] >= self.limit:
            return False
        data["used"] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return True


EXAMPLE_JOBS = {
    "LLM evaluation engineer (示例)": """示例科技 – 大模型评测工程师（2027届校招，上海）
岗位职责：
1. 设计大模型评测集、评测指标与打分标准；
2. 搭建自动化评测流水线，分析评测结果并定位模型问题；
3. 探索 LLM-as-a-judge 方法，提升自动评测与人工评测的一致性。
任职要求：
1. 计算机、人工智能等相关专业硕士；
2. 熟练掌握 Python，熟悉 PyTorch；
3. 有大模型评测或 NLP 研究经验；有多语言经验者优先。""",
    "Agent application engineer (sample)": """Example AI Co. – LLM Agent Application Engineer (2027 graduates, Beijing)
Responsibilities:
- Build LLM agents with tool calling for internal business workflows
- Build RAG retrieval over company documents and evaluate retrieval quality
- Deploy services with FastAPI and monitor them in production
Requirements:
- Master's degree in CS or related field
- Strong Python; experience with LangChain or LangGraph
- Bonus: Docker, SQL, experience evaluating LLM applications""",
}
