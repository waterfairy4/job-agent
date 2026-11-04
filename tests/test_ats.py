"""ATS: key-term check (only terms the user's documents show are added) and the ATS-friendly PDF layout."""
from langchain_core.documents import Document
from pypdf import PdfReader

from jobagent.cvpdf import render_pdf
from jobagent.schemas import ATSTerms, CVDocument, CVEntry, CVSection
from jobagent.tools import add_keywords, ats_check, ats_contains, ats_terms
from tests.fakes import FakeLLM
from tests.test_agent import env  # noqa: F401  (pytest fixture)

POSTING = "岗位职责：参与大模型评测体系建设，设计评测指标。岗位要求：熟练使用 Python 进行数据分析；了解 AI 与 NLP。"


def test_ats_contains_matches_whole_english_words_and_ignores_chinese_spaces():
    assert ats_contains("Built an AI agent", "AI") and not ats_contains("Evaluated OpenAI models", "AI")
    assert ats_contains("熟练使用 Python 进行", "python")
    assert ats_contains("大模型 评测体系", "大模型评测")


def test_ats_terms_keep_only_terms_that_are_really_in_the_posting():
    llm = FakeLLM([], {ATSTerms: lambda p: ATSTerms(terms=["大模型评测", "Python", "Kubernetes", "评测指标",
                                                          "python", "数据分析", "x"])})
    job = {"text": POSTING}
    assert ats_terms(job, llm) == ["大模型评测", "Python", "评测指标", "数据分析"]   # invented, duplicate, too short dropped
    assert ats_terms(job, None) == job["ats_terms"]                                  # cached: no second model call


def test_ats_check_adds_only_what_the_documents_show():
    chunks = [Document(page_content="技能：Python、PyTorch；做过大模型评测与数据分析。", metadata={"source": "projects.md"})]
    cv = CVDocument(name="程桂花", sections=[
        CVSection(title="项目经历", entries=[CVEntry(title="MLS-Eval", bullets=["多语言大模型评测流程"])]),
        CVSection(title="技术能力", layout="table", entries=[CVEntry(title="编程语言", bullets=["Python"])])])
    chk = ats_check(["Python", "大模型评测", "数据分析", "评测指标"], "程桂花 多语言大模型评测流程 Python", chunks)
    assert chk["present"] == ["Python", "大模型评测"]
    assert chk["supported"] == {"数据分析": "projects.md"} and chk["unsupported"] == ["评测指标"]
    assert add_keywords(cv, list(chk["supported"])) == "关键词"
    assert cv.sections[1].entries[-1].bullets == ["数据分析"]                     # joined into the skills table


def test_ats_layout_keeps_dates_and_labels_on_the_same_line(tmp_path):
    cv = CVDocument(name="程桂花", sections=[
        CVSection(title="教育经历", entries=[CVEntry(title="北京大学", date="2024 – 2027（预计）",
                                                   subtitle="计算机科学与技术，硕士", meta="GPA：3.66 / 4.0")]),
        CVSection(title="技术能力", layout="table", entries=[CVEntry(title="编程语言", bullets=["Python"])])])
    pdf, pages = render_pdf(cv, tmp_path / "ats.pdf", ats=True)
    lines = [" ".join(l.split()) for l in PdfReader(pdf).pages[0].extract_text().splitlines()]
    assert pages == 1
    assert any("北京大学" in l and "2024" in l for l in lines)
    assert any("硕士" in l and "GPA" in l for l in lines)
    assert any("编程语言" in l and "Python" in l for l in lines)


def test_build_cv_adds_supported_keywords_and_reports_the_rest(env, monkeypatch, tmp_path):
    import json

    import jobagent.cvdoc as cvdoc
    from jobagent.schemas import CVCritique
    from jobagent.tools import build_tools
    from tests.test_agent import JD, handlers

    index, store = env
    profile = tmp_path / "profile"
    (profile / "cv.md").write_text("技能：PyTorch。\n\n做过大模型评测（MLS-Eval）。\n\n语言：土耳其语、英语", encoding="utf-8")
    (profile / "projects.md").write_text("熟练使用 Python；设计过评分标准。", encoding="utf-8")
    index.build(str(profile))
    monkeypatch.setattr(cvdoc.config, "PROFILE_DIR", str(profile))
    monkeypatch.setattr(cvdoc.config, "CV_FILE", "cv.md")
    cv = CVDocument(name="程桂花", sections=[
        CVSection(title="项目经历", entries=[CVEntry(title="MLS-Eval", bullets=["做过大模型评测（MLS-Eval）。"])]),
        CVSection(title="技术能力", layout="table", entries=[CVEntry(title="框架", bullets=["PyTorch"])]),
        CVSection(title="语言", layout="text", entries=[CVEntry(title="语言：", bullets=["土耳其语、英语"])])])
    h = handlers("ByteDance")
    h.update({CVDocument: lambda p: cv.model_copy(deep=True), CVCritique: lambda p: CVCritique(issues=[]),
              ATSTerms: lambda p: ATSTerms(terms=["大模型评测", "Python", "Kubernetes", "评分标准", "Docker"])})
    llm = FakeLLM([], h)
    tools = {t.name: t for t in build_tools(index, store, llm)}
    jid = json.loads(tools["analyze_job"].invoke({"job_text": JD}))["job_id"]

    out = tools["build_cv"].invoke({"job_id": jid})
    assert "ATS keywords: 1 of 4 key terms of the posting are in the CV (大模型评测)." in out   # Docker: not in posting
    assert "added to the skills table as '关键词'" in out and "Python (projects.md)" in out and "评分标准" in out
    assert "not found in your documents (add them only if true): Kubernetes" in out
    saved = CVDocument.model_validate_json((store.reports_dir / f"{jid}-cv.json").read_text(encoding="utf-8"))
    assert saved.sections[1].entries[-1].title == "关键词"
    assert saved.sections[1].entries[-1].bullets == ["Python、评分标准"]
    assert "Kubernetes" not in saved.model_dump_json()                                     # never added


def test_chinese_cv_with_english_terms_counts_as_chinese():
    from jobagent.tools import _is_chinese
    cv = ("求职智能体（Job Agent）：基于 LangGraph 构建 ReAct 智能体，BM25 + bge-m3 混合检索，"
          "Python、PyTorch、Hugging Face Transformers、PEFT/TRL、pandas/NumPy/SciPy、Docker、Linux、Git")
    assert _is_chinese(cv)
    assert _is_chinese("【实习】大模型评测实习生 参与大模型评测体系的设计与优化，熟练使用Python进行数据处理。")
    assert not _is_chinese("LLM Evaluation Intern at Meituan (美团), Beijing. Python, data analysis, LLM-as-a-judge.")
