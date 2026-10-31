"""RAG over your profile: load -> split -> index -> hybrid search.

Hybrid search = dense vectors (meaning) + BM25 (exact terms like "PyTorch", "LoRA", "评测"),
merged with Reciprocal Rank Fusion. Technical job requirements are full of exact terms,
so dense search alone misses things.
"""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from jobagent import config

CJK = re.compile(r"[㐀-鿿豈-﫿]")
TEXT_TYPES = {".md", ".txt"}

# Sentence-ending punctuation for Chinese and English, so chunks break at natural points.
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ", ", " ", ""]


# ---------- Loading ----------
def load_documents(folder: str = config.PROFILE_DIR) -> list[Document]:
    """One Document per PDF page (keeps page numbers for citations) or per text file."""
    from pypdf import PdfReader

    docs = []
    for path in sorted(Path(folder).glob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for n, page in enumerate(PdfReader(path).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": path.name, "page": n}))
        elif suffix in TEXT_TYPES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": path.name, "page": 1}))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator="end",  # "。" stays with its own sentence
    )
    chunks = splitter.split_documents(docs)
    counters: dict[tuple, int] = {}
    for c in chunks:
        key = (c.metadata["source"], c.metadata["page"])
        counters[key] = counters.get(key, 0) + 1
        c.metadata["chunk_id"] = f"{key[0]}#p{key[1]}#c{counters[key]}"
    return chunks


# ---------- BM25 tokenizer ----------
def tokenize(text: str) -> list[str]:
    """English/Turkish: lowercase words. Chinese: single characters + character bigrams.

    Chinese has no spaces, so whitespace tokenization would make a whole sentence one token.
    Bigrams approximate words ("评测" , "模型") without needing a segmenter.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9+#._-]*", text)
    for run in re.findall(r"[㐀-鿿豈-﫿]+", text):
        tokens.extend(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


# ---------- Index ----------
def default_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


class ProfileIndex:
    def __init__(self, embeddings=None, persist_dir: str | None = None):
        from langchain_chroma import Chroma

        self.vs = Chroma(
            collection_name="profile",
            embedding_function=embeddings or default_embeddings(),
            persist_directory=persist_dir or str(Path(config.DATA_DIR) / "chroma"),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._chunks: list[Document] = []

    def build(self, folder: str = config.PROFILE_DIR) -> int:
        """Rebuild from scratch, so deleted or edited files never leave stale chunks."""
        chunks = split_documents(load_documents(folder))
        self.vs.reset_collection()
        if chunks:
            self.vs.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        self._bm25 = None
        return len(chunks)

    def _load_bm25(self):
        if self._bm25 is None:
            data = self.vs.get(include=["documents", "metadatas"])
            self._chunks = [Document(page_content=t, metadata=m)
                            for t, m in zip(data["documents"], data["metadatas"])]
            if self._chunks:
                self._bm25 = BM25Okapi([tokenize(c.page_content) for c in self._chunks])
        return self._bm25

    def size(self) -> int:
        return len(self.vs.get(include=[])["ids"])

    MODES = ("hybrid", "dense", "bm25")

    def search(self, query: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Document]:
        """Hybrid search with Reciprocal Rank Fusion: score = sum of 1 / (60 + rank).
        mode="dense" or "bm25" uses one retriever alone (for comparing them in evaluation)."""
        bm25 = self._load_bm25()
        if bm25 is None:
            raise RuntimeError("Profile index is empty. Add files to profile/ and run: python ingest.py")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        pool = k * 3 if mode == "hybrid" else k
        rankings = []

        if mode in ("hybrid", "dense"):
            dense = self.vs.similarity_search(query, k=min(pool, len(self._chunks)))
            rankings.append([d.metadata["chunk_id"] for d in dense])
        if mode in ("hybrid", "bm25"):
            scores = bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            rankings.append([self._chunks[i].metadata["chunk_id"] for i in order[:pool] if scores[i] > 0])

        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)

        by_id = {c.metadata["chunk_id"]: c for c in self._chunks}
        best = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [by_id[cid] for cid in best if cid in by_id]


def format_hits(hits: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.metadata['source']}, p.{h.metadata['page']})\n{h.page_content}"
        for i, h in enumerate(hits, start=1)
    )
