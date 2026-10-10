"""Render the in-application manual as a standalone PDF.

    py -3.11 tools/build_manual.py

The text is not written twice. It is read from ``facet.ui.help``, which is the
same source the Help menu shows, so the printed manual cannot describe a
version of the application the manual inside it does not. A reader away from
their machine gets the identical words.

Chrome does the layout; its own print header stamps the ``file://`` URL onto
every page, so that is suppressed and the footer drawn afterwards with PyMuPDF.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
OUT_DIR = ROOT / "docs"

CSS = """
@page { size: A4; margin: 20mm 18mm 22mm 18mm; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif;
       font-size: 10pt; line-height: 1.52; color: #15181c; margin: 0; }
h1 { font-size: 15pt; font-weight: 600; margin: 0 0 10px 0;
     padding-bottom: 5px; border-bottom: 1.4px solid #2a78d6;
     page-break-before: always; page-break-after: avoid; }
h2 { font-size: 11pt; font-weight: 600; margin: 15px 0 5px 0;
     page-break-after: avoid; }
p { margin: 0 0 8px 0; }
ol, ul { margin: 0 0 9px 0; padding-left: 18px; }
li { margin-bottom: 6px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 11px 0;
        font-size: 9pt; page-break-inside: avoid; }
td { border-bottom: 0.5px solid #e1e4e8; padding: 4px 6px;
     vertical-align: top; }
code { font-family: "Consolas", monospace; font-size: 9pt; background: #f2f4f6;
       padding: 0 3px; }
i { color: #1d4e89; font-style: normal; font-weight: 600; }
b { font-weight: 600; }

.cover { page-break-after: always; padding-top: 52mm; text-align: center; }
.cover h1 { border: none; page-break-before: avoid; font-size: 30pt;
            margin: 0 0 4px 0; padding: 0; letter-spacing: -0.5px; }
.cover .tag { font-size: 12pt; color: #3c4147; margin-bottom: 30px; }
.cover .what { font-size: 11pt; margin: 0 auto 34px auto; max-width: 118mm;
               color: #15181c; text-align: left; line-height: 1.6; }
.cover .meta { font-size: 9pt; color: #6b7177; }
.cover img { width: 86px; margin-bottom: 16px; }

.toc { page-break-after: always; }
.toc h1 { page-break-before: avoid; }
.toc div { margin-bottom: 4px; font-size: 10pt; }
.toc .g { margin: 13px 0 5px 0; font-size: 8.4pt; font-weight: 600;
          letter-spacing: 0.8px; text-transform: uppercase; color: #2a78d6; }
"""

GROUPS = [
    ("Start here", ("start",)),
    ("Tutorials", ("tut-quartz", "tut-eulytite", "tut-polymorphs",
                   "tut-cryolite", "tut-hosts", "tut-figure")),
    ("Reference", ("examples", "why", "window", "site", "lonepair", "utilities",
                   "diffraction", "exafs", "planes", "disorder", "appearance",
                   "shortcuts")),
    ("Limits and licences", ("limits", "licences")),
]


def _logo_data_uri() -> str:
    """The application mark, drawn by the same code that draws the icon."""
    try:
        from PySide6.QtCore import QBuffer, QByteArray

        from facet.ui.branding import logo_pixmap

        pixmap = logo_pixmap(256, ground=False)
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QBuffer.WriteOnly)
        pixmap.save(buffer, "PNG")
        buffer.close()
        import base64

        return ("data:image/png;base64,"
                + base64.b64encode(bytes(data)).decode("ascii"))
    except Exception:
        return ""


def build_html() -> str:
    from facet.ui import help as helpmod
    from facet.version import NAME, TAGLINE, __version__

    sections = {key: (title, body) for key, title, body in helpmod._sections()}
    today = _dt.date.today().strftime("%d %B %Y")
    logo = _logo_data_uri()

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<title>{NAME} user manual</title><style>{CSS}</style></head><body>"]

    parts.append(
        "<div class='cover'>"
        + (f"<img src='{logo}'>" if logo else "")
        + f"<h1>{NAME}</h1>"
        f"<div class='tag'>{TAGLINE}</div>"
        "<div class='what'>A coordination number is not measured. It is "
        "produced by choosing a cutoff, and where a structure gives no gap for "
        "that cutoff to fall in, the number reported says as much about the "
        "person reporting it as about the crystal.<br><br>"
        "FACET cuts by bond valence rather than by distance, and reports not "
        "one coordination number but the range of threshold each one survives "
        "&mdash; so that a number which is a property of the structure can be "
        "told apart from one that is a property of the choice.</div>"
        f"<div class='meta'>User manual &middot; version {__version__}"
        f"<br>{today}</div></div>")

    # contents
    parts.append("<div class='toc'><h1>Contents</h1>")
    n = 0
    for group, keys in GROUPS:
        parts.append(f"<div class='g'>{group}</div>")
        for key in keys:
            if key not in sections:
                continue
            n += 1
            parts.append(f"<div>{n}. &nbsp;{sections[key][0]}</div>")
    parts.append("</div>")

    for _group, keys in GROUPS:
        for key in keys:
            if key in sections:
                parts.append(sections[key][1])

    parts.append("</body></html>")
    return "".join(parts)


def stamp(pdf: Path, label: str) -> int:
    import fitz

    doc = fitz.open(pdf)
    total = doc.page_count
    grey = (0.42, 0.45, 0.48)
    rule = (0.84, 0.86, 0.88)
    for i, page in enumerate(doc, start=1):
        if i == 1:
            continue                      # the cover carries no furniture
        w, h = page.rect.width, page.rect.height
        y = h - 34
        page.draw_line(fitz.Point(51, y), fitz.Point(w - 51, y),
                       color=rule, width=0.6)
        page.insert_text(fitz.Point(51, y + 11), label, fontname="helv",
                         fontsize=7.4, color=grey)
        text = "%d of %d" % (i, total)
        width = fitz.get_text_length(text, fontname="helv", fontsize=7.4)
        page.insert_text(fitz.Point(w - 51 - width, y + 11), text,
                         fontname="helv", fontsize=7.4, color=grey)
    doc.saveIncr()
    doc.close()
    return total


def main() -> int:
    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])

    from facet.version import NAME, __version__

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / f"{NAME}-manual.html"
    pdf_path = OUT_DIR / f"{NAME}-manual.pdf"

    html_path.write_text(build_html(), encoding="utf-8")
    print(f"  html  -> {html_path.relative_to(ROOT)}")

    if not CHROME.exists():
        print("  Chrome not found; the HTML is written but no PDF was made.")
        return 1

    result = subprocess.run(
        [str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
        capture_output=True)
    if not pdf_path.exists():
        print("  PDF was not produced:", result.stderr.decode()[:300])
        return 1

    pages = stamp(pdf_path, f"{NAME} {__version__} user manual")
    size = pdf_path.stat().st_size / 1e6
    print(f"  pdf   -> {pdf_path.relative_to(ROOT)}  ({pages} pages, {size:.1f} MB)")
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
