"""Build the library manual: run its examples, write the outputs under them.

    py -3.11 tools/build_library_manual.py            # write docs/library_manual.md, .html, .pdf
    py -3.11 tools/build_library_manual.py --check    # exit 1 if docs/library_manual.md is out of date
    py -3.11 tools/build_library_manual.py --no-pdf   # the Markdown and the figures only

The manual is written in ``docs/library_manual.src.md``. Every fenced block
opened with three backticks and ``python`` is run, in order, in one
namespace, from an empty working directory; what it prints is written under
it as a ``text`` block, and the figures it saves are copied to
``docs/library_manual/``. A block opened with ``python no-run`` is shown and
not run (an installation line, a loop over files the reader has and the
repository has not). The outputs are never typed by hand, so the manual
states what this version of the library does; ``--check`` is what the test
suite runs to say so.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "library_manual.src.md"
OUT_MD = ROOT / "docs" / "library_manual.md"
OUT_HTML = ROOT / "docs" / "FACET-library-manual.html"
OUT_PDF = ROOT / "docs" / "FACET-library-manual.pdf"
FIGURES = ROOT / "docs" / "library_manual"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

sys.path.insert(0, str(ROOT))

FENCE = re.compile(r"^```(.*)$")
WIDTH = 100            # an output line longer than this is cut, with a mark


def _blocks(text: str):
    """(kind, info, lines): 'text' runs of prose and 'code' fenced blocks."""
    kind, info, lines = "text", "", []
    for line in text.splitlines():
        found = FENCE.match(line)
        if found and kind == "text":
            if lines:
                yield "text", "", lines
            kind, info, lines = "code", found.group(1).strip(), []
        elif found and kind == "code":
            yield "code", info, lines
            kind, info, lines = "text", "", []
        else:
            lines.append(line)
    if lines:
        yield kind, info, lines


def _clip(output: str) -> str:
    lines = []
    for line in output.rstrip().splitlines():
        line = line.rstrip()
        lines.append(line if len(line) <= WIDTH else line[:WIDTH - 2] + " …")
    return "\n".join(lines)


def run(source: str, figures: Path | None = None) -> str:
    """The manual with every runnable block's output under it."""
    from facet.version import __version__

    namespace: dict = {"__name__": "__manual__"}
    out: list[str] = []
    work = Path(tempfile.mkdtemp(prefix="facet_manual_"))
    before = Path.cwd()
    os.chdir(work)
    try:
        for kind, info, lines in _blocks(source):
            if kind == "text":
                out += lines
                continue
            words = info.split()
            language = words[0] if words else ""
            shown = "python" if language == "python" else info
            out.append(f"```{shown}")
            out += lines
            out.append("```")
            if language != "python" or "no-run" in words:
                continue
            code = "\n".join(lines)
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    exec(compile(code, "<manual>", "exec"), namespace)
            except Exception:
                raise SystemExit("a block of the manual did not run:\n\n" + code
                                 + "\n\n" + traceback.format_exc())
            printed = _clip(buffer.getvalue())
            if printed:
                out += ["", "```text", printed, "```"]
        text = "\n".join(out).rstrip() + "\n"
        text = text.replace("{{version}}", __version__)
        if figures is not None:
            figures.mkdir(exist_ok=True)
            wanted = set(re.findall(r"\]\(library_manual/([^)]+)\)", text))
            for name in sorted(wanted):
                made = work / name
                if not made.is_file():
                    raise SystemExit(f"the manual shows {name}, which no block saved")
                shutil.copyfile(made, figures / name)
            for old in figures.iterdir():
                if old.name not in wanted:
                    old.unlink()
        return text
    finally:
        os.chdir(before)
        shutil.rmtree(work, ignore_errors=True)


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10pt;
       color: #16181c; line-height: 1.45; }
body { margin: 0; }
h1 { font-size: 21pt; margin: 0 0 4pt 0; color: #10233f; }
h2 { font-size: 14.5pt; margin: 20pt 0 6pt 0; padding-top: 6pt; color: #10233f;
     border-top: 1.2pt solid #10233f; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 13pt 0 4pt 0; color: #1d3a63; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 5pt 0; }
ul, ol { margin: 4pt 0 4pt 0; padding-left: 16pt; }
a { color: #1d4f91; text-decoration: none; }
code { font-family: Consolas, "Cascadia Mono", "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: #eef1f5; padding: 0.5pt 2.5pt; border-radius: 2pt; overflow-wrap: anywhere; }
pre { font-family: Consolas, "Cascadia Mono", "DejaVu Sans Mono", monospace; font-size: 7.6pt;
      line-height: 1.32; margin: 5pt 0; padding: 6pt 8pt; border-radius: 3pt;
      white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid-page; }
pre code { background: none; padding: 0; font-size: inherit; }
pre.python { background: #f3f5f8; border-left: 2.5pt solid #1d4f91; }
pre.output { background: #fbfaf6; border-left: 2.5pt solid #b9a878; color: #2c2a25; }
pre.other { background: #f3f5f8; border-left: 2.5pt solid #7a8594; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.8pt; width: 100%; }
th { text-align: left; background: #10233f; color: #fff; padding: 3pt 6pt; font-weight: 600; }
td { padding: 3pt 6pt; border-bottom: 0.5pt solid #d5dae2; vertical-align: top; }
td code, th code { white-space: normal; overflow-wrap: anywhere; }
img { max-width: 62%; display: block; margin: 6pt auto; }
blockquote { margin: 7pt 0; padding: 5pt 10pt; background: #eef4fb; border-left: 2.5pt solid #1d4f91; }
blockquote p { margin: 2pt 0; }
.cover { background: #10233f; color: #fff; padding: 16pt 18pt; border-radius: 4pt; margin-bottom: 12pt; }
.cover h1 { color: #fff; }
.cover p { margin: 2pt 0; color: #d7e0ee; }
"""


def to_html(markdown_text: str) -> str:
    import markdown

    # the title and the lines under it, up to the first section, are the cover
    head, _, rest = markdown_text.partition("\n## ")
    cover = markdown.markdown(head, extensions=["tables"])
    body = markdown.markdown("## " + rest, extensions=["tables", "fenced_code", "sane_lists"])
    body = body.replace('<pre><code class="language-python">', '<pre class="python"><code>')
    body = body.replace('<pre><code class="language-text">', '<pre class="output"><code>')
    body = re.sub(r'<pre><code( class="language-[a-z-]+")?>', '<pre class="other"><code>', body)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<title>FACET library manual</title>"
            f"<style>{CSS}</style></head><body>"
            f"<div class='cover'>{cover}</div>{body}</body></html>")


def to_pdf(html_path: Path, pdf_path: Path) -> None:
    if not CHROME.is_file():
        raise SystemExit(f"the PDF needs Chrome at {CHROME}")
    subprocess.run([str(CHROME), "--headless", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
                    html_path.resolve().as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the library manual")
    parser.add_argument("--check", action="store_true",
                        help="compare with docs/library_manual.md and write nothing")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    if args.check:
        built = run(source)
        current = OUT_MD.read_text(encoding="utf-8") if OUT_MD.is_file() else ""
        if built != current:
            import difflib

            diff = list(difflib.unified_diff(current.splitlines(), built.splitlines(),
                                             "docs/library_manual.md", "rebuilt",
                                             lineterm="", n=1))
            print("\n".join(diff[:60]))
            print("docs/library_manual.md is out of date: "
                  "py -3.11 tools/build_library_manual.py")
            return 1
        print("docs/library_manual.md states what the library does")
        return 0
    built = run(source, FIGURES)
    OUT_MD.write_text(built, encoding="utf-8", newline="\n")
    print(f"markdown -> {OUT_MD.relative_to(ROOT)}  ({len(built.splitlines())} lines)")
    html = to_html(built)
    OUT_HTML.write_text(html, encoding="utf-8", newline="\n")
    print(f"html     -> {OUT_HTML.relative_to(ROOT)}")
    if not args.no_pdf:
        to_pdf(OUT_HTML, OUT_PDF)
        print(f"pdf      -> {OUT_PDF.relative_to(ROOT)}  ({OUT_PDF.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
