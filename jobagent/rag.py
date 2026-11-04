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


# Section titles of a CV or paper (PDF text has no markdown "#"): a short line that is only one of these.
SECTION_WORDS = (
    "求职意向", "个人简介", "教育经历", "教育背景", "项目经历", "科研经历", "研究经历", "实习经历", "工作经历",
    "专业技能", "技术能力", "技能", "论文", "发表论文", "出版物", "语言", "语言与荣誉", "荣誉", "获奖", "奖项",
    "objective", "summary", "education", "experience", "work experience", "research experience", "projects",
    "publications", "skills", "technical skills", "languages", "honors", "awards",
    "abstract", "introduction", "related work", "method", "methods", "experiments", "results", "discussion",
    "conclusion", "conclusions", "limitations", "references", "appendix")
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# a numbered English paper heading, "3.2 Hybrid Retrieval" (not "2023 年 7 月" or "3 个发送端")
NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z:-]*){0,8})$")


def _heading(line: str) -> tuple[int, str] | None:
    """(level, title) if the line is a section title, else None."""
    s = line.strip()
    m = MD_HEADING.match(s)
    if m:
        return len(m.group(1)), m.group(2).strip()
    bare = s.rstrip("：:").strip()
    if 0 < len(bare) <= 24 and bare.lower() in SECTION_WORDS:
        return 2, bare
    m = NUMBERED.match(s)
    if m and not s.endswith(("。", ".", "，", ",", "；", ";")):
        return 2 + m.group(1).count("."), s
    return None


def split_sections(text: str) -> list[tuple[str, str]]:
    """Cut a text at its section titles. Returns (section name, text) pairs; the text keeps its title line.
    The name is the nearest title below the document's own title (a "# " heading over the whole file)."""
    sections, path, buf = [], {}, []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            levels = sorted(path)
            # skip a lone level-1 file title when a deeper heading exists ("# My profile" > "## Job Agent")
            names = [path[lv] for lv in levels if not (lv == 1 and len(levels) > 1)]
            sections.append((" › ".join(names[-2:]), body))

    for line in text.splitlines():
        h = _heading(line)
        if h:
            flush()
            buf = []
            level, title = h
            path = {lv: name for lv, name in path.items() if lv < level}
            path[level] = title
        buf.append(line)
    flush()
    return sections


def split_documents(docs: list[Document], by_section: bool | None = None) -> list[Document]:
    """Chunks of about CHUNK_SIZE characters. With SECTION_CHUNKS (optional, off by default), a document is first cut at its
    section titles, so a chunk never mixes two projects, and each chunk starts with its section name
    ("【求职智能体（Job Agent）】"), which helps both BM25 and the embeddings and names the project in citations."""
    by_section = config.SECTION_CHUNKS if by_section is None else by_section
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator="end",  # "。" stays with its own sentence
    )
    chunks = []
    for doc in docs:
        if not by_section:
            chunks += splitter.split_documents([doc])
            continue
        for name, text in split_sections(doc.page_content):
            prefix = f"【{name}】\n" if name else ""
            for piece in splitter.split_text(text):
                meta = {**doc.metadata, "section": name, "prefix_len": len(prefix)}
                chunks.append(Document(page_content=prefix + piece, metadata=meta))
    counters: dict[tuple, int] = {}
    for c in chunks:
        key = (c.metadata["source"], c.metadata["page"])
        counters[key] = counters.get(key, 0) + 1
        c.metadata["chunk_id"] = f"{key[0]}#p{key[1]}#c{counters[key]}"
    return chunks


def chunk_body(chunk: Document) -> str:
    """The chunk's own text, without the section name added at its start."""
    return chunk.page_content[int(chunk.metadata.get("prefix_len", 0) or 0):]


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
    def __init__(self, embeddings=None, persist_dir: str | None = None, reranker=None):
        """`reranker`: a function (query, list of texts) -> list of scores; None = the cross-encoder in
        config.RERANK_MODEL, loaded only when a reranked search is first made."""
        from langchain_chroma import Chroma

        self.vs = Chroma(
            collection_name="profile",
            embedding_function=embeddings or default_embeddings(),
            persist_directory=persist_dir or str(Path(config.DATA_DIR) / "chroma"),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._chunks: list[Document] = []
        self._reranker = reranker

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

    MODES = ("hybrid", "dense", "bm25", "rerank")

    def search(self, query: str, k: int = config.TOP_K, mode: str | None = None,
               bm25_weight: float | None = None) -> list[Document]:
        """Hybrid search with weighted Reciprocal Rank Fusion: score = sum of weight / (60 + rank), dense weight 1,
        BM25 weight `bm25_weight` (default config.BM25_WEIGHT).
        mode "dense" or "bm25": one retriever alone (for comparing them in evaluation).
        mode "rerank": the top RERANK_POOL hybrid results re-scored by a cross-encoder.
        Default mode: "rerank" if config.RERANK is on, else "hybrid"."""
        mode = mode or ("rerank" if config.RERANK else "hybrid")
        bm25 = self._load_bm25()
        if bm25 is None:
            raise RuntimeError("Profile index is empty. Add files to profile/ and run: python ingest.py")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if mode == "rerank":
            pool = self.search(query, k=max(k, config.RERANK_POOL), mode="hybrid", bm25_weight=bm25_weight)
            scores = self._rerank_scores(query, [d.page_content for d in pool])
            order = sorted(range(len(pool)), key=lambda i: -scores[i])
            return [pool[i] for i in order[:k]]
        weight = config.BM25_WEIGHT if bm25_weight is None else bm25_weight
        pool = k * 3 if mode == "hybrid" else k
        rankings = []   # (weight, ranked chunk ids)

        if mode in ("hybrid", "dense"):
            dense = self.vs.similarity_search(query, k=min(pool, len(self._chunks)))
            rankings.append((1.0, [d.metadata["chunk_id"] for d in dense]))
        if mode in ("hybrid", "bm25"):
            scores = bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            rankings.append((weight if mode == "hybrid" else 1.0,
                             [self._chunks[i].metadata["chunk_id"] for i in order[:pool] if scores[i] > 0]))

        fused: dict[str, float] = {}
        for w, ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + w / (60 + rank)

        by_id = {c.metadata["chunk_id"]: c for c in self._chunks}
        best = sorted(fused, key=lambda cid: -fused[cid])[:k]
        return [by_id[cid] for cid in best if cid in by_id]

    def _rerank_scores(self, query: str, texts: list[str]) -> list[float]:
        if self._reranker is None:
            self._reranker = load_reranker()
        return list(self._reranker(query, texts))


_RERANKER = None


def load_reranker(model: str | None = None):
    """The cross-encoder as a function (query, texts) -> scores. Loaded once per process."""
    global _RERANKER
    if _RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError("The reranker needs sentence-transformers (pip install sentence-transformers).") from e
        encoder = CrossEncoder(model or config.RERANK_MODEL, max_length=512)
        _RERANKER = lambda query, texts: encoder.predict([(query, t) for t in texts]) if texts else []
    return _RERANKER


def format_hits(hits: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.metadata['source']}, p.{h.metadata['page']})\n{h.page_content}"
        for i, h in enumerate(hits, start=1)
    )
