"""build_cv (draft -> critique -> revise, number check, photo, Word file) and plan_gaps."""
import json

import pytest
from docx import Document
from PIL import Image

import jobagent.cvdoc as cvdoc
from jobagent.schemas import (CVCritique, CVDocument, CVEntry, CVIssue, CVSection, GapAction, GapPlan)
from jobagent.tools import build_tools
from tests.fakes import FakeLLM
from tests.test_agent import JD, env, handlers  # noqa: F401  (env is a pytest fixture)


def _doc(bullets, langs="Languages: Turkish, English, Korean"):
    return CVDocument(
        name="Mechi Test", name_alt="泽伊内普", headline="MSc CS", contact=["mechi@example.com"],
        sections=[CVSection(title="Research", entries=[CVEntry(
                      title="MLS-Eval", date="", subtitle="Team lead", meta="", bullets=bullets)]),
                  CVSection(title="Skills", layout="table",
                            entries=[CVEntry(title="ML", bullets=["PyTorch", "LoRA"])]),
                  CVSection(title="Languages", entries=[CVEntry(bullets=[langs])])])


def cv_handler(prompt):
    if prompt.startswith("Revise"):   # fixes the invented number, but slips in a keyword and an invented line
        return _doc(["Built MLS-Eval, a multilingual LLM evaluation benchmark (大模型评测) (高并发)",
                     "Led a team of 20 engineers on Kubernetes infrastructure"])
    return _doc(["Built MLS-Eval, a multilingual LLM evaluation benchmark (大模型评测)", "Evaluated 37 models"],
                langs="Languages: Turkish, English")          # dropped Korean


def critique_handler(prompt):
    return CVCritique(issues=[CVIssue(kind="stretched", where="Research", problem="overclaims",
                                      fix="say what the project did")])


def _plan(prompt):
    return GapPlan(actions=[
        GapAction(item=1, action="build", step="Deploy a small K8s demo of the eval pipeline",
                  effort="a weekend", proof="GitHub repo"),
        GapAction(item=9, action="learn", step="ghost item", effort="1h", proof="-"),   # no such gap
    ])


def _tools(env, extra, cv_llm=False):
    index, store = env
    h = handlers("ByteDance")
    h.update(extra)
    llm = FakeLLM([], h)
    tools = {t.name: t for t in build_tools(index, store, llm, llm if cv_llm else None)}
    jid = json.loads(tools["analyze_job"].invoke({"job_text": JD}))["job_id"]
    return tools, store, jid, llm


def _profile(tmp_path):
    return next(p for p in tmp_path.iterdir() if p.name == "profile")


def test_build_cv_draft_critique_revise(env, monkeypatch, tmp_path):
    from pypdf import PdfReader

    profile = _profile(tmp_path)
    (profile / "cv.md").write_text(
        "Skills: Python, PyTorch, LoRA fine-tuning.\n\nBuilt MLS-Eval, a multilingual LLM evaluation benchmark "
        "(大模型评测) with scoring metrics.\n\nLanguages: Turkish, English, Korean", encoding="utf-8")
    Image.new("RGB", (60, 80), "blue").save(profile / "photo.jpg")
    monkeypatch.setattr(cvdoc.config, "PROFILE_DIR", str(profile))
    monkeypatch.setattr(cvdoc.config, "CV_FILE", "")
    prefs = tmp_path / "prefs.md"
    prefs.write_text("- Keep Korean.", encoding="utf-8")
    monkeypatch.setattr(cvdoc.config, "CV_PREFERENCES_FILE", str(prefs))
    tools, store, jid, llm = _tools(env, {CVDocument: cv_handler, CVCritique: critique_handler})

    out = tools["build_cv"].invoke({"job_id": jid})
    pdf, docx = store.reports_dir / f"{jid}-cv.pdf", store.reports_dir / f"{jid}-cv.docx"
    assert CVDocument.model_validate_json((store.reports_dir / f"{jid}-cv.json").read_text(encoding="utf-8")).name
    assert out.startswith(f"FILE: {pdf}\nFILE: {docx}") and pdf.exists() and docx.exists()
    assert "one page" in out and "photo included" in out
    # the draft's problems were fed to the revision
    revise = next(p for p in llm.prompts if p.startswith("Revise"))
    assert "numbers are not in the original CV: 37" in revise and "Korean" in revise and "Keep Korean." in revise
    # code removed what the revision slipped in
    assert "（高并发）" not in out and "(高并发)" in out and "Led a team of 20 engineers" in out
    assert "Team lead" in out                                   # invented subtitle removed
    text = "".join(page.extract_text() for page in PdfReader(pdf).pages)
    assert "大模型评测" in text and "高并发" not in text and "20 engineers" not in text and "Team lead" not in text
    assert "Korean" in text and len(PdfReader(pdf).pages) == 1

    d = Document(docx)
    assert len(d.inline_shapes) == 1                             # the photo
    assert "PyTorch; LoRA" in "\n".join(c.text for t in d.tables for r in t.rows for c in r.cells)


def test_build_cv_without_identifiable_cv(env, monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(cvdoc.config, "PROFILE_DIR", str(empty))
    monkeypatch.setattr(cvdoc.config, "CV_FILE", "")
    tools, _, jid, _ = _tools(env, {CVDocument: cv_handler, CVCritique: critique_handler})
    assert "Could not tell which file" in tools["build_cv"].invoke({"job_id": jid})


def test_find_cv_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(cvdoc.config, "CV_FILE", "")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF")
    assert cvdoc.find_cv(str(tmp_path)) == tmp_path / "paper.pdf"          # only PDF
    (tmp_path / "other.pdf").write_bytes(b"%PDF")
    assert cvdoc.find_cv(str(tmp_path)) is None                              # ambiguous
    (tmp_path / "Zeynep_CV_Huawei.pdf").write_bytes(b"%PDF")
    assert cvdoc.find_cv(str(tmp_path)).name == "Zeynep_CV_Huawei.pdf"       # named like a CV
    monkeypatch.setattr(cvdoc.config, "CV_FILE", "other.pdf")
    assert cvdoc.find_cv(str(tmp_path)).name == "other.pdf"                  # explicit setting


def test_images_from_pdf(tmp_path):
    """A portrait image in the CV PDF becomes the photo, a square one the logo."""
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")

    Image.new("RGB", (300, 400), "red").save(tmp_path / "p.png")
    Image.new("RGB", (200, 200), "green").save(tmp_path / "l.png")
    c = canvas.Canvas(str(tmp_path / "cv.pdf"))
    c.drawImage(str(tmp_path / "p.png"), 400, 650, 90, 120)
    c.drawImage(str(tmp_path / "l.png"), 50, 680, 80, 80)
    c.save()
    empty = tmp_path / "profile"
    empty.mkdir()
    found = cvdoc.find_images(tmp_path / "cv.pdf", folder=str(empty), out_dir=tmp_path / "assets")
    assert Image.open(found["photo"]).size == (300, 400)
    assert Image.open(found["logo"]).size == (200, 200)
    Image.new("RGB", (10, 10)).save(empty / "photo.jpg")     # an explicit photo wins
    assert cvdoc.find_images(tmp_path / "cv.pdf", folder=str(empty),
                             out_dir=tmp_path / "assets")["photo"] == empty / "photo.jpg"


def test_strip_added_and_languages():
    src = "实现 REST API、异步请求处理与数据库集成。语言：土耳其语（母语）· 英语 · 韩语（会话水平）。荣誉：奖学金。研究：英语和汉语评估"
    assert cvdoc.strip_added("实现 REST API、异步请求处理（并发）与数据库集成。", src) == \
        ("实现 REST API、异步请求处理与数据库集成。", ["（并发）"])
    assert cvdoc.strip_added("实现 REST API，异步请求处理与数据库集成", src)[1] == []   # punctuation only
    assert cvdoc.strip_added("Led a team of 20 engineers", src)[0] is None          # invented line
    assert cvdoc.missing_languages("语言：土耳其语 · 英语", src) == ["韩语"]          # 汉语 is not a language line item
    assert cvdoc.missing_languages("Languages: Turkish, English, Korean", src) == []  # English names count


def test_unknown_numbers_and_length():
    src = "12,120 queries, 101 questions, +0.012 to +0.061, 2024 – 2027"
    assert cvdoc.unknown_numbers("12120 queries and 101 items in 2024", src) == []
    assert cvdoc.unknown_numbers("99% accuracy on 101 items", src) == ["99"]
    assert cvdoc.visual_length("ab 评测") == 7    # 2 latin + space + 2 CJK counted double


def test_plan_gaps(env):
    tools, store, jid, _ = _tools(env, {GapPlan: _plan})
    assert "not been matched" in tools["plan_gaps"].invoke({"job_id": jid})
    tools["match_job"].invoke({"job_id": jid})
    out = tools["plan_gaps"].invoke({"job_id": jid})
    assert "Build: Deploy a small K8s demo" in out and "ghost item" not in out
    assert "No action proposed for:" in out                    # remaining gaps are listed, not hidden
    assert (store.reports_dir / f"{jid}-gaps.md").exists()


def test_build_cv_survives_a_failing_review(env, monkeypatch, tmp_path):
    """The review step must never block the CV: an unexpected label or a crash just skips it."""
    profile = _profile(tmp_path)
    monkeypatch.setattr(cvdoc.config, "PROFILE_DIR", str(profile))
    monkeypatch.setattr(cvdoc.config, "CV_FILE", "")

    assert CVIssue(kind="irrelevant ordering", where="x", problem="y", fix="z").kind == "irrelevant ordering"

    def broken_review(prompt):
        raise ValueError("validation error")

    tools, store, jid, _ = _tools(env, {CVDocument: cv_handler, CVCritique: broken_review})
    out = tools["build_cv"].invoke({"job_id": jid})
    assert out.startswith("FILE: ") and (store.reports_dir / f"{jid}-cv.pdf").exists()


def test_newest_first_and_objective_status():
    from jobagent.schemas import CVDocument, CVEntry, CVSection
    from jobagent.tools import objective_status, sort_by_date

    source = "求职意向\n大模型方向的全职岗位（2027 届校园招聘）。\n项目经历\nLoRA 2026 年\nRL 2025 年春"
    cv = CVDocument(name="X", sections=[
        CVSection(title="求职意向", entries=[CVEntry(title="大模型方向的全职岗位（2027 届校园招聘）")]),
        CVSection(title="项目经历", entries=[CVEntry(title="RL", date="2025 年春"),
                                            CVEntry(title="LoRA", date="2026 年"),
                                            CVEntry(title="Thesis", date="2026 年春 – 至今")]),
        CVSection(title="其他", entries=[CVEntry(title="b", date="2020"), CVEntry(title="a")])])
    assert sort_by_date(cv) == ["项目经历"]
    assert [e.title for e in cv.sections[1].entries] == ["Thesis", "LoRA", "RL"]
    assert [e.title for e in cv.sections[2].entries] == ["b", "a"]          # no date: left alone
    assert objective_status(cv, source) == "same"
    cv.sections[0].entries[0].bullets = ["热爱消费电子与全球化产品"]
    assert objective_status(cv, source) == "reworded"


def test_objective_may_be_reworded_and_paragraph_sections(tmp_path):
    from jobagent.cvdoc import enforce_faithful, normalize_layout
    from jobagent.cvpdf import render_pdf
    from jobagent.schemas import CVDocument, CVEntry, CVSection

    source = ("求职意向\n大语言模型方向的研究与工程类全职岗位。\n项目经历\n用 Python 设计评估流程。\n"
              "语言与荣誉\n语言：土耳其语（母语）· 韩语（会话水平）。荣誉：北京市政府奖学金。")
    cv = CVDocument(name="X", sections=[
        CVSection(title="求职意向", entries=[CVEntry(bullets=["大语言模型方向的研究与工程类实习岗位。"])]),
        CVSection(title="项目经历", entries=[CVEntry(title="P", bullets=["用 Python 设计并发评估流程。"])]),
        CVSection(title="语言与荣誉", entries=[CVEntry(title="语言", bullets=["土耳其语（母语）", "韩语（会话水平）"]),
                                              CVEntry(title="荣誉", bullets=["北京市政府奖学金。"])])])
    changes = enforce_faithful(cv, source)
    assert cv.sections[0].entries[0].bullets == ["大语言模型方向的研究与工程类实习岗位。"]   # objective kept
    assert cv.sections[1].entries[0].bullets == ["用 Python 设计评估流程。"] and changes      # bullet still checked
    normalize_layout(cv)
    assert cv.sections[0].layout == "text" and cv.sections[2].layout == "text"
    assert [e.title for e in cv.sections[2].entries] == ["语言：", "荣誉："]
    _, pages = render_pdf(cv, tmp_path / "cv.pdf")
    assert pages == 1


def test_skills_and_languages_in_second_line_fields_are_printed(tmp_path):
    """Seen in the demo: "编程语言：" printed with nothing after it, because the model put the skills in
    `subtitle`, which the table/text layouts did not print."""
    from pypdf import PdfReader

    from jobagent.cvdoc import normalize_layout
    from jobagent.cvpdf import render_pdf
    from jobagent.schemas import CVDocument, CVEntry, CVSection

    cv = CVDocument(name="林晓", sections=[
        CVSection(title="技术能力", layout="text", entries=[
            CVEntry(title="编程语言：", subtitle="Python（主要）、SQL、JavaScript"),
            CVEntry(title="工程：", meta="FastAPI、Docker")]),
        CVSection(title="技能", layout="table", entries=[CVEntry(title="机器学习", subtitle="PyTorch")]),
        CVSection(title="语言", entries=[CVEntry(title="语言", subtitle="中文（母语）· 英语（专业工作水平）")])])
    normalize_layout(cv)
    assert cv.sections[0].entries[0].bullets == ["Python（主要）、SQL、JavaScript"]
    assert cv.sections[2].entries[0].bullets == ["中文（母语）· 英语（专业工作水平）"]
    assert all(not e.subtitle and not e.meta for s in cv.sections for e in s.entries)
    pdf, _ = render_pdf(cv, tmp_path / "cv.pdf")
    text = "".join(PdfReader(pdf).pages[0].extract_text().split())
    for word in ("SQL", "FastAPI", "PyTorch", "英语"):
        assert word in text


def test_structured_output_retries_when_the_model_returns_nothing():
    import pytest

    from jobagent.llm import structured
    from jobagent.schemas import CVScore

    answers = [None, None, CVScore(relevance=4, clarity=3, reason="ok")]

    class Flaky:
        def with_structured_output(self, schema, **kw):
            return type("R", (), {"invoke": lambda self, p: answers.pop(0)})()

    assert structured(Flaky(), CVScore).invoke("x").relevance == 4
    answers[:] = [None, None, None]
    with pytest.raises(RuntimeError, match="did not return a valid CVScore after 3 tries"):
        structured(Flaky(), CVScore).invoke("x")


def test_dropped_language_is_put_back_word_for_word():
    from jobagent.cvdoc import cv_plain_text, restore_languages

    source = "语言与荣誉\n语言：土耳其语（母语）· 英语（专业工作水平）· 韩语（会话水平）。荣誉：北京市政府奖学金。"
    cv = CVDocument(name="X", sections=[CVSection(title="语言与荣誉", layout="text", entries=[
        CVEntry(title="语言：", bullets=["土耳其语（母语）· 英语（专业工作水平）。"]),
        CVEntry(title="荣誉：", bullets=["北京市政府奖学金。"])])])
    assert restore_languages(cv, source) == ["韩语"]
    assert cv.sections[0].entries[0].bullets == ["土耳其语（母语）· 英语（专业工作水平） · 韩语（会话水平）。"]
    assert restore_languages(cv, source) == [] and "韩语（会话水平）" in cv_plain_text(cv)


def test_gap_plan_prompt_forbids_upgrading_experience():
    """Seen in use: plan_gaps suggested presenting a portfolio demo as an "internal pilot" with user numbers."""
    from jobagent.tools import GAP_PROMPT

    for phrase in ("Never upgrade it", "pilot", "users", "measure X first", "A portfolio demo stays a demo"):
        assert phrase in GAP_PROMPT
