"""One-page PDF CV with Typst (pip install typst): no Word, WPS or LibreOffice needed.

The CV data is passed to the template as JSON, so no text ever needs escaping. If the result is
longer than one page, the layout is scaled down step by step (font size, spacing) until it fits.
"""
import json
import shutil
import tempfile
from pathlib import Path

from jobagent import config

# The first font that has a character is used: Latin fonts first, then Chinese ones.
# Windows: Times New Roman + SimSun (bold Chinese: Microsoft YaHei). Linux/Mac fallbacks follow.
SERIF = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "SimSun", "Songti SC",
         "Noto Serif CJK SC", "Source Han Serif SC", "Microsoft YaHei", "Noto Sans CJK SC"]
# SimSun has no bold weight, so bold Chinese uses fonts that do.
BOLD = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "Noto Serif CJK SC",
        "Source Han Serif SC", "Microsoft YaHei", "SimHei", "Songti SC", "Noto Sans CJK SC"]
SCALES = [1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.82]

TEMPLATE = r"""
#let cv = json(bytes(sys.inputs.at("cv")))
#let s = float(sys.inputs.at("scale"))
#let serif = SERIF_FONTS
#let boldf = BOLD_FONTS
#set page(paper: "a4", margin: (x: 1.55cm, top: 1.1cm * s, bottom: 1.0cm * s))
#set text(font: serif, size: 10.5pt * s, lang: "zh", region: "cn")
#set par(leading: 0.62em, spacing: 0.62em * s, justify: false)
#show strong: set text(font: boldf, weight: "bold")

#let photo = sys.inputs.at("photo", default: "")
#let logo = sys.inputs.at("logo", default: "")
#grid(
  columns: (2.6cm, 1fr, 2.6cm),
  align: (left + horizon, center + horizon, right + horizon),
  if logo != "" { image(logo, width: 2.2cm * s) },
  [
    #text(size: 18pt * s)[*#cv.name*]
    #v(1pt)
    #if cv.at("name_alt", default: "") != "" { linebreak(); text(size: 9.5pt * s, cv.name_alt) }
    #if cv.at("headline", default: "") != "" { linebreak(); text(size: 10pt * s, cv.headline) }
    #for line in cv.at("contact", default: ()) { linebreak(); text(size: 9.5pt * s, line) }
  ],
  if photo != "" { image(photo, width: 2.3cm * s) },
)

#let section(title) = {
  v(7pt * s)
  block(width: 100%, stroke: (bottom: 0.6pt), inset: (bottom: 2.5pt), below: 5pt * s,
        text(size: 11.5pt * s)[*#title*])
}

#let two(left, right, bold: false) = grid(
  columns: (1fr, auto),
  if bold { strong(left) } else { left },
  right,
)

#for sec in cv.sections {
  section(sec.title)
  if sec.at("layout", default: "entries") == "table" {
    grid(
      columns: (2.5cm, 1fr), row-gutter: 6pt * s, column-gutter: 4pt,
      ..sec.entries.map(e => (strong(e.title), e.bullets.join("；"))).flatten(),
    )
  } else if sec.at("layout", default: "entries") == "text" {
    // plain paragraphs (objective, languages): no bullet dots; an entry title is a bold label
    for e in sec.entries {
      let parts = e.at("bullets", default: ())
      let body = parts.enumerate().map(((i, b)) => if i == 0 or b.starts-with("·") { b } else { " · " + b }).join()
      par(if e.at("title", default: "") != "" { strong(e.title) } + body)
    }
  } else {
    for e in sec.entries {
      if e.at("title", default: "") != "" or e.at("date", default: "") != "" {
        v(3pt * s)
        two(e.at("title", default: ""), e.at("date", default: ""), bold: true)
      }
      if e.at("subtitle", default: "") != "" or e.at("meta", default: "") != "" {
        two(e.at("subtitle", default: ""), e.at("meta", default: ""))
      }
      for b in e.at("bullets", default: ()) {
        pad(left: 0.75em, top: 1pt * s, grid(columns: (0.75em, 1fr), [·], b))
      }
    }
  }
}
"""


def _template() -> str:
    return (TEMPLATE.replace("SERIF_FONTS", "(" + ", ".join(json.dumps(f) for f in SERIF) + ",)")
            .replace("BOLD_FONTS", "(" + ", ".join(json.dumps(f) for f in BOLD) + ",)"))


def _pages(pdf: bytes) -> int:
    import io

    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(pdf)).pages)


def render_pdf(cv, path: Path, photo: Path | None = None, logo: Path | None = None) -> tuple[Path, int]:
    """Write the CV as a PDF, scaled to fit one page if possible. Returns (path, number of pages)."""
    import typst

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "cv.typ").write_text(_template(), encoding="utf-8")
        inputs = {"cv": cv.model_dump_json()}
        for name, src in (("photo", photo), ("logo", logo)):
            if src:
                dst = tmp / f"{name}{Path(src).suffix.lower()}"
                shutil.copy(src, dst)
                inputs[name] = dst.name
        font_paths = [config.CV_FONT_DIR] if config.CV_FONT_DIR else []
        pdf, pages = b"", 0
        for scale in SCALES:
            pdf = typst.compile(str(tmp / "cv.typ"), root=str(tmp), font_paths=font_paths,
                                sys_inputs={**inputs, "scale": str(scale)})
            pages = _pages(pdf)
            if pages <= 1:
                break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf)
    return path, pages
