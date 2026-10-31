"""Full tailored CV: find the CV and its photo/logo, run checks, render a Word file in the CV's own style.

Layout (modelled on a typical Chinese academic CV): logo left, centered name and contact block,
photo right; section titles with a rule underneath; entries with the title left and dates right;
skills as a two-column table.
"""
import re
from pathlib import Path

from jobagent import config

CV_NAME = re.compile(r"(cv|resume|résumé|简历|履历|özgeçmiş)", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
CJK = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]")
IMAGE_TYPES = (".jpg", ".jpeg", ".png")


# ---------- finding the inputs ----------
def find_cv(folder: str | None = None) -> Path | None:
    """The CV file: CV_FILE from .env, else a file named like a CV, else the only PDF in the folder."""
    folder = folder or config.PROFILE_DIR
    if config.CV_FILE:
        path = Path(folder) / config.CV_FILE
        return path if path.exists() else None
    files = [p for p in sorted(Path(folder).glob("*")) if p.suffix.lower() in (".pdf", ".md", ".txt")]
    named = [p for p in files if CV_NAME.search(p.stem)]
    if named:
        return named[0]
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    return pdfs[0] if len(pdfs) == 1 else None


def cv_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def find_images(cv_path: Path | None, folder: str | None = None, out_dir: Path | None = None) -> dict:
    """Photo and logo for the CV header.

    1. profile/photo.(jpg|png) and profile/logo.(jpg|png) if present;
    2. otherwise taken from the first page of the CV PDF: a portrait image is the photo,
       a roughly square one is the logo.
    Returns {"photo": Path | None, "logo": Path | None}.
    """
    folder = Path(folder or config.PROFILE_DIR)
    found = {}
    for kind in ("photo", "logo"):
        for ext in IMAGE_TYPES:
            if (folder / f"{kind}{ext}").exists():
                found[kind] = folder / f"{kind}{ext}"
                break
    if cv_path and cv_path.suffix.lower() == ".pdf" and len(found) < 2:
        out_dir = Path(out_dir or Path(config.DATA_DIR) / "assets")
        try:
            from pypdf import PdfReader

            images = [img.image for img in PdfReader(cv_path).pages[0].images]
        except Exception:  # unusual PDFs or a missing Pillow: just go without pictures
            images = []
        portraits = [im for im in images if im.height > 1.15 * im.width and im.width >= 100]
        squares = [im for im in images if 0.85 <= im.height / max(im.width, 1) <= 1.15 and im.width >= 80]
        for kind, candidates in (("photo", portraits), ("logo", squares)):
            if kind not in found and candidates:
                best = max(candidates, key=lambda im: im.width * im.height)
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"{kind}.png"
                best.convert("RGB").save(path)
                found[kind] = path
    return {"photo": found.get("photo"), "logo": found.get("logo")}


# ---------- checks ----------
def _norm(n: str) -> str:
    return n.replace(",", "")


def unknown_numbers(text: str, source: str) -> list[str]:
    """Numbers in `text` that never appear in `source`: a tailored CV must not invent figures."""
    known = {_norm(n) for n in NUMBER.findall(source)}
    return sorted({n for n in NUMBER.findall(text) if _norm(n) not in known})


def visual_length(text: str) -> int:
    """Approximate width on the page: a Chinese character is about twice as wide as a Latin one."""
    compact = re.sub(r"\s+", " ", text).strip()
    return len(compact) + len(CJK.findall(compact))


def cv_plain_text(cv) -> str:
    parts = [cv.name, cv.name_alt, cv.headline, *cv.contact]
    for sec in cv.sections:
        parts.append(sec.title)
        for e in sec.entries:
            parts += [e.title, e.date, e.subtitle, e.meta, *e.bullets]
    return "\n".join(p for p in parts if p)


# ---------- rendering ----------
def _set_fonts(run, size=None, bold=None, italic=None):
    from docx.oxml.ns import qn
    from docx.shared import Pt

    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    run.font.name = config.CV_FONT_LATIN
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(fonts)
    fonts.set(qn("w:eastAsia"), config.CV_FONT_CJK)


def _rule_below(paragraph):
    """A thin line under a section title. Inserted at its schema position (before spacing etc.),
    because Word rejects pPr children in the wrong order."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    ppr = paragraph._p.get_or_add_pPr()
    border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "000000")):
        bottom.set(qn(k), v)
    border.append(bottom)
    ppr.insert_element_before(border, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku",
                              "w:wordWrap", "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE",
                              "w:autoSpaceDN", "w:bidi", "w:adjustRightInd", "w:snapToGrid",
                              "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
                              "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
                              "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle",
                              "w:rPr", "w:sectPr", "w:pPrChange")


def _plain_table(table, widths):
    """No borders, fixed column widths. Widths go on the table, the grid and every cell,
    because Word and LibreOffice each read a different one. Elements are inserted in schema order."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Emu

    widths = [Emu(int(w)) for w in widths]  # arithmetic on lengths gives plain ints
    tblpr = table._tbl.tblPr
    total = OxmlElement("w:tblW")
    total.set(qn("w:w"), str(int(sum(w.twips for w in widths))))
    total.set(qn("w:type"), "dxa")
    for old in tblpr.findall(qn("w:tblW")):
        tblpr.remove(old)
    tblpr.insert_element_before(total, "w:jc", "w:tblCellSpacing", "w:tblInd", "w:tblBorders", "w:shd",
                                "w:tblLayout", "w:tblCellMar", "w:tblLook")
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tblpr.insert_element_before(borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblpr.insert_element_before(layout, "w:tblCellMar", "w:tblLook")
    for col, w in zip(table._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        col.set(qn("w:w"), str(int(w.twips)))
    table.autofit = False
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = w


def _para(container, text="", size=10.5, bold=False, align=None, before=0, after=0):
    from docx.shared import Pt

    p = container.add_paragraph()
    fmt = p.paragraph_format
    fmt.space_before, fmt.space_after, fmt.line_spacing = Pt(before), Pt(after), 1.08
    if align is not None:
        p.alignment = align
    if text:
        _set_fonts(p.add_run(text), size=size, bold=bold)
    return p


def _two_sided(doc, left, right, width, size=10.5, bold_left=False):
    """`left` at the margin, `right` aligned to the right edge (dates, GPA)."""
    from docx.enum.text import WD_TAB_ALIGNMENT

    p = _para(doc, size=size)
    p.paragraph_format.tab_stops.add_tab_stop(width, WD_TAB_ALIGNMENT.RIGHT)
    if left:
        _set_fonts(p.add_run(left), size=size, bold=bold_left)
    if right:
        _set_fonts(p.add_run("\t" + right), size=size)
    return p


def render_docx(cv, path: Path, photo: Path | None = None, logo: Path | None = None) -> Path:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    doc = Document()
    from docx.oxml.ns import qn as _qn

    zoom = doc.settings.element.find(_qn("w:zoom"))  # python-docx's template omits a required attribute
    if zoom is not None and zoom.get(_qn("w:percent")) is None:
        zoom.set(_qn("w:percent"), "100")
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)  # A4
    sec.left_margin = sec.right_margin = Cm(1.6)
    sec.top_margin, sec.bottom_margin = Cm(1.2), Cm(1.1)
    width = sec.page_width - sec.left_margin - sec.right_margin
    doc.styles["Normal"].font.size = Pt(10.5)

    # Header: logo | name + contact | photo
    side = Cm(2.6)
    header = doc.add_table(rows=1, cols=3)
    header.alignment = WD_TABLE_ALIGNMENT.CENTER
    _plain_table(header, (side, width - 2 * side, side))
    left, mid, right = header.rows[0].cells
    if logo:
        left.paragraphs[0].add_run().add_picture(str(logo), width=Cm(2.2))
    if photo:
        right.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
        right.paragraphs[0].add_run().add_picture(str(photo), width=Cm(2.4))
    first = mid.paragraphs[0]
    first.alignment = WD_ALIGN_PARAGRAPH.CENTER
    first.paragraph_format.space_after = Pt(2)
    _set_fonts(first.add_run(cv.name), size=18, bold=True)
    center = WD_ALIGN_PARAGRAPH.CENTER
    if cv.name_alt:
        _para(mid, cv.name_alt, size=9.5, align=center)
    if cv.headline:
        _para(mid, cv.headline, size=10, align=center)
    for line in cv.contact:
        _para(mid, line, size=9.5, align=center)

    for section in cv.sections:
        title = _para(doc, section.title, size=11.5, bold=True, before=7, after=3)
        _rule_below(title)
        if section.layout == "table":
            table = doc.add_table(rows=len(section.entries), cols=2)
            _plain_table(table, (Cm(2.6), width - Cm(2.6)))
            for e, r in zip(section.entries, table.rows):
                row = r.cells
                for c in row:
                    f = c.paragraphs[0].paragraph_format
                    f.space_before, f.space_after, f.line_spacing = Pt(1), Pt(1), 1.08
                _set_fonts(row[0].paragraphs[0].add_run(e.title), size=10.5, bold=True)
                _set_fonts(row[1].paragraphs[0].add_run("；".join(e.bullets) if _is_cjk(e.bullets)
                                                        else "; ".join(e.bullets)), size=10.5)
            continue
        for e in section.entries:
            if e.title or e.date:
                _two_sided(doc, e.title, e.date, width, bold_left=True).paragraph_format.space_before = Pt(2)
            if e.subtitle or e.meta:
                _two_sided(doc, e.subtitle, e.meta, width)
            for b in e.bullets:
                p = _para(doc, size=10.5)
                p.paragraph_format.left_indent = Cm(0.45)
                p.paragraph_format.first_line_indent = Cm(-0.3)
                _set_fonts(p.add_run("·  " + b), size=10.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    return path


def _is_cjk(texts: list[str]) -> bool:
    return any(CJK.search(t) for t in texts)


# ---------- faithfulness: no added words ----------
_SKIP = re.compile(r"[\s·‧•●▪]")
_DASHES = str.maketrans({"‑": "-", "‐": "-", "–": "-", "—": "-", "⸺": "-", "－": "-"})
_PUNCT = set("，。、；：！？,.;:!?()（）[]【】“”\"'‘’-/·|～~ ")


def _norm_map(text: str) -> tuple[str, list[int]]:
    """Normalized text (no spaces/bullet dots, unified dashes, lower case) + index of each char in `text`."""
    chars, index = [], []
    for i, ch in enumerate(text):
        if _SKIP.match(ch):
            continue
        chars.append(ch.translate(_DASHES).lower())
        index.append(i)
    return "".join(chars), index


def added_runs(text: str, source: str, k: int = 3) -> list[tuple[int, int]]:
    """Spans of `text` not covered by any k-character window that also occurs in `source`.
    Rewording that reuses the original's words is covered; inserted words like "（并发）" are not."""
    t, idx = _norm_map(text)
    s, _ = _norm_map(source)
    covered = [False] * len(t)
    for i in range(len(t) - k + 1):
        if t[i:i + k] in s:
            for j in range(i, i + k):
                covered[j] = True
    if len(t) < k and t in s:
        covered = [True] * len(t)
    runs, i = [], 0
    while i < len(t):
        if covered[i]:
            i += 1
            continue
        j = i
        while j < len(t) and not covered[j]:
            j += 1
        piece = t[i:j]
        if sum(ch not in _PUNCT for ch in piece) >= 2:  # ignore pure punctuation changes
            runs.append((idx[i], idx[j - 1] + 1))
        i = j
    return runs


def strip_added(text: str, source: str, max_share: float = 0.25) -> tuple[str | None, list[str]]:
    """Remove text that is not in the source. Returns (clean text, removed pieces).
    If more than `max_share` of the text is new, the whole line is invented: returns (None, [text])."""
    runs = added_runs(text, source)
    if not runs:
        return text, []
    removed = [text[a:b] for a, b in runs]
    if sum(len(_norm_map(r)[0]) for r in removed) > max_share * max(len(_norm_map(text)[0]), 1):
        return None, [text]
    for a, b in reversed(runs):
        text = text[:a] + text[b:]
    return re.sub(r"\s{2,}", " ", text).strip(), removed


LANGUAGES = {"英语": "english", "汉语": "chinese", "中文": "chinese", "普通话": "mandarin",
             "土耳其语": "turkish", "韩语": "korean", "日语": "japanese", "法语": "french",
             "德语": "german", "西班牙语": "spanish", "俄语": "russian", "阿拉伯语": "arabic"}


def missing_languages(cv_text_out: str, source: str) -> list[str]:
    """Languages listed in the original CV (in Chinese or English) that the new CV dropped."""
    m = re.search(r"(语言\s*[:：]|languages?\s*[:：])(.{0,400})", source, re.IGNORECASE | re.DOTALL)
    if m:  # only the languages line, not e.g. "English and Chinese evaluation" in a research summary
        source = re.split(r"荣誉|honou?rs|awards?", m.group(2), flags=re.IGNORECASE)[0]
    out, src = cv_text_out.lower(), source.lower()
    missing = []
    for zh, en in LANGUAGES.items():
        if (zh in source or en in src) and zh not in cv_text_out and en not in out:
            missing.append(zh if zh in source else en)
    return missing


def restore_languages(cv, source: str) -> list[str]:
    """Put back languages the model dropped, copied word for word from the original CV
    (e.g. "韩语（会话水平）"). A prompt rule alone did not stop the model from dropping them."""
    lost = missing_languages(cv_plain_text(cv), source)
    if not lost:
        return []
    from jobagent.schemas import CVEntry, CVSection

    phrases = []
    for lang in lost:
        m = re.search(re.escape(lang) + r"\s*[（(][^）)]{0,30}[）)]", source, re.IGNORECASE)
        phrases.append(m.group(0) if m else lang)
    names = [*LANGUAGES, *LANGUAGES.values()]
    target = None
    for sec in cv.sections:
        for e in sec.entries:
            text = " ".join([e.title, *e.bullets]).lower()
            if "语言" in e.title or "language" in e.title.lower() or (
                    any(n in text for n in names) and any(k in sec.title.lower() for k in ("语言", "language"))):
                target = e
    if target is None:
        sec = next((s for s in cv.sections if "语言" in s.title or "language" in s.title.lower()), None)
        if sec is None:
            sec = CVSection(title="语言与荣誉" if CJK.search(source) else "Languages", layout="text", entries=[])
            cv.sections.append(sec)
        target = CVEntry(title="语言：" if CJK.search(source) else "Languages: ", bullets=[])
        sec.entries.insert(0, target)
    if target.bullets:
        last = target.bullets[-1].rstrip().rstrip("。.")
        target.bullets[-1] = last + "".join(f" · {p}" for p in phrases) + ("。" if CJK.search(last) else "")
    else:
        target.bullets.append(" · ".join(phrases))
    return lost


OBJECTIVE = ("求职意向", "求职目标", "个人简介", "objective", "summary", "profile")
PARAGRAPHS = OBJECTIVE + ("语言", "荣誉", "language", "honor", "award")


def is_objective(title: str) -> bool:
    return any(k in title.lower() for k in OBJECTIVE)


def normalize_layout(cv) -> None:
    """Objective and Languages & Honors are plain paragraphs (no bullet dots), entry titles are bold
    labels ending in '：', like the original CV."""
    for sec in cv.sections:
        if sec.layout != "table" and any(k in sec.title.lower() for k in PARAGRAPHS):
            sec.layout = "text"
            for e in sec.entries:
                if e.title and e.bullets and not e.title.rstrip().endswith(("：", ":")):
                    e.title = e.title.rstrip() + ("：" if CJK.search(e.title) else ": ")


def enforce_faithful(cv, source: str) -> list[str]:
    """Edit `cv` in place: strip words added to lines that come from the CV, drop invented lines.
    Section titles, the name block and the objective (the one part that may be reworded) are left
    alone. Returns a list of human-readable changes."""
    changes = []

    def check_field(obj, field):
        value = getattr(obj, field)
        if not value:
            return
        clean, removed = strip_added(value, source)
        if clean is None:
            setattr(obj, field, "")
            changes.append(f"removed \"{value[:40]}\" (not in the original CV)")
        elif removed:
            setattr(obj, field, clean)
            changes.append(f"removed {'、'.join(removed)} from \"{clean[:30]}\"")

    check_field(cv, "headline")
    for sec in cv.sections:
        if is_objective(sec.title):
            continue
        for e in sec.entries:
            for field in ("title", "date", "subtitle", "meta"):
                check_field(e, field)
            kept = []
            for b in e.bullets:
                clean, removed = strip_added(b, source)
                if clean is None:
                    changes.append(f"dropped a line not in the original CV: \"{b[:60]}\"")
                    continue
                if removed:
                    changes.append(f"removed {'、'.join(removed)} from \"{clean[:30]}…\"")
                kept.append(clean)
            e.bullets = kept
    return changes
