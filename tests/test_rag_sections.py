"""Section-aware chunking: a chunk never mixes two projects and starts with its section name."""
from langchain_core.documents import Document

from jobagent.agent_eval import relevant_ids
from jobagent.rag import chunk_body, split_documents, split_sections

PROFILE = """# 我的项目资料

说明：供检索使用。

## 求职智能体（Job Agent）

- 基于 LangGraph 构建 ReAct 智能体。
- 混合检索：bge-m3 与 BM25，用 RRF 融合。

## 面向特定人群的 LoRA 微调

- 用 PEFT 对 Qwen2.5-7B 做 LoRA 微调。
"""

CV_PAGE = """程桂花
项目经历
多语言评测框架 2025
用 Python 实现评估流程。
实习经历
Netaş 软件开发实习生 2023 年 7 月 – 9 月
构建全栈 Web 应用。"""


def test_markdown_is_cut_at_headings():
    names = [name for name, _ in split_sections(PROFILE)]
    assert names == ["我的项目资料", "求职智能体（Job Agent）", "面向特定人群的 LoRA 微调"]


def test_each_chunk_has_one_section_and_its_name():
    chunks = split_documents([Document(page_content=PROFILE, metadata={"source": "projects.md", "page": 1})],
                             by_section=True)
    agent = [c for c in chunks if c.metadata["section"] == "求职智能体（Job Agent）"]
    assert agent and all(c.page_content.startswith("【求职智能体（Job Agent）】\n") for c in agent)
    assert all("LoRA" not in chunk_body(c) for c in agent)            # never mixed with the next project
    assert [c.metadata["chunk_id"] for c in chunks] == [f"projects.md#p1#c{i}" for i in range(1, len(chunks) + 1)]


def test_cv_pdf_text_is_cut_at_cv_section_titles_but_not_at_dates():
    names = [name for name, _ in split_sections(CV_PAGE)]
    assert names == ["", "项目经历", "实习经历"]                          # "2023 年 7 月" is not a title


def test_plain_chunks_when_turned_off():
    chunks = split_documents([Document(page_content=PROFILE, metadata={"source": "p.md", "page": 1})],
                             by_section=False)
    assert all(not c.page_content.startswith("【") and "section" not in c.metadata for c in chunks)


def test_section_name_does_not_make_a_chunk_relevant():
    chunks = split_documents([Document(page_content=PROFILE, metadata={"source": "p.md", "page": 1})],
                             by_section=True)
    # "Job Agent" is in the title line, which belongs to the first chunk's own text, and in every prefix
    rel = relevant_ids(chunks, ["Job Agent"])
    assert rel == {c.metadata["chunk_id"] for c in chunks if "Job Agent" in chunk_body(c)}


def test_index_with_sections_builds_and_finds_the_right_project(tmp_path, monkeypatch):
    from jobagent import config
    from jobagent.demo import HashEmbeddings
    from jobagent.rag import ProfileIndex

    folder = tmp_path / "profile"
    folder.mkdir()
    (folder / "projects.md").write_text(PROFILE, encoding="utf-8")
    (folder / "cv.txt").write_text(CV_PAGE, encoding="utf-8")
    monkeypatch.setattr(config, "SECTION_CHUNKS", True)  # optional setting, off by default
    index = ProfileIndex(HashEmbeddings(), persist_dir=str(tmp_path / "chroma"))
    assert index.build(str(folder)) >= 5
    top = index.search("LoRA 微调 PEFT", k=1)[0]
    assert top.metadata["section"] == "面向特定人群的 LoRA 微调"


def test_synthetic_benchmark_is_consistent():
    """Every benchmark query has at least one relevant chunk, and no fictional document mentions the owner."""
    from pathlib import Path

    from jobagent.agent_eval import read_retrieval_cases
    from jobagent.rag import load_documents

    bench = Path(__file__).parent.parent / "eval" / "retrieval_bench"
    chunks = split_documents(load_documents(str(bench / "docs")), by_section=False)
    cases = read_retrieval_cases(bench / "cases.csv")
    assert len(cases) >= 40 and len(chunks) >= 30
    assert all(relevant_ids(chunks, c["phrases"]) for c in cases)
    # each key phrase must point to exactly one document: matching ignores spaces, so a short phrase can hit an
    # unrelated document by accident ("DPO" inside "and pop-ups")
    from jobagent.agent_eval import _squash

    for c in cases:
        for p in c["phrases"]:
            docs = {ch.metadata["source"] for ch in chunks if _squash(p) in _squash(ch.page_content)}
            assert len(docs) == 1, f"{c['query']!r}: key phrase {p!r} matches {sorted(docs)}"
    assert all("fictional" in d.page_content.lower() or "虚构" in d.page_content
               for d in load_documents(str(bench / "docs")))
