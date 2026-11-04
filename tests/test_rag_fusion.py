"""Weighted Reciprocal Rank Fusion and the optional reranker (with a fake cross-encoder: no download)."""
import pytest

from jobagent import config
from jobagent.demo import HashEmbeddings
from jobagent.rag import ProfileIndex

DOCS = {
    "a.md": "LoRA 微调：用 PEFT 对 Qwen 做参数高效微调。",
    "b.md": "Hybrid retrieval with BM25 and dense vectors, fused by reciprocal rank fusion.",
    "c.md": "评测集构建与 LLM-as-a-judge 一致性分析。",
    "d.md": "Kubernetes autoscaling and Prometheus monitoring for model serving.",
    "e.md": "数据合成：从种子指令扩展出指令数据。",
    "f.md": "Turkish speech recognition with Whisper.",
}


@pytest.fixture
def index(tmp_path):
    folder = tmp_path / "profile"
    folder.mkdir()
    for name, text in DOCS.items():
        (folder / name).write_text(text, encoding="utf-8")
    calls = []

    def fake_reranker(query, texts):          # prefers the passage about monitoring, whatever the query
        calls.append(len(texts))
        return [1.0 if "Prometheus" in t else 0.0 for t in texts]

    idx = ProfileIndex(HashEmbeddings(), persist_dir=str(tmp_path / "chroma"), reranker=fake_reranker)
    idx.build(str(folder))
    idx.calls = calls
    return idx


def ids(docs):
    return [d.metadata["source"] for d in docs]


def test_bm25_weight_zero_gives_the_dense_ranking(index):
    q = "LoRA 微调 retrieval 评测"
    assert ids(index.search(q, k=3, mode="hybrid", bm25_weight=0.0)) == ids(index.search(q, k=3, mode="dense"))


def test_bm25_weight_changes_the_fusion(index):
    q = "BM25 dense fusion"
    equal = index.search(q, k=6, mode="hybrid", bm25_weight=1.0)
    heavy = index.search(q, k=6, mode="hybrid", bm25_weight=5.0)
    assert ids(heavy)[0] == ids(index.search(q, k=1, mode="bm25"))[0]   # a heavy BM25 weight puts its top first
    assert set(ids(equal)) == set(ids(heavy))                          # same candidates, different order allowed


def test_rerank_reorders_the_hybrid_pool(index, monkeypatch):
    monkeypatch.setattr(config, "RERANK_POOL", 6)
    top = index.search("LoRA 微调", k=2, mode="rerank")
    assert ids(top)[0] == "d.md" and index.calls == [6]                # the reranker saw the pool and decided


def test_default_mode_follows_the_setting(index, monkeypatch):
    monkeypatch.setattr(config, "RERANK", False)
    index.search("LoRA", k=2)
    assert index.calls == []                                           # off by default: no reranker call
    monkeypatch.setattr(config, "RERANK", True)
    index.search("LoRA", k=2)
    assert len(index.calls) == 1
