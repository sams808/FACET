"""Write facet/data/CHANGELOG.md from the commit history.

    py -3.11 tools/make_changelog.py

Every commit's title becomes a line; a commit whose message names a version
("Version 0.9.0.", "version 0.6.0") starts a new section for it, so the file
reads as what each version did. The application shows the file under
Help > What changed, and the build bundles it, so a user on one version can
read what the next one changed without the repository.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "facet" / "data" / "CHANGELOG.md"
VERSION_RE = re.compile(r"\bversion (\d+\.\d+\.\d+)\b", re.IGNORECASE)


def commits() -> list[tuple[str, str, str, str]]:
    """(hash, date, title, body) newest first."""
    raw = subprocess.run(
        ["git", "log", "--date=short", "--format=%h%x1f%ad%x1f%s%x1f%b%x1e"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=True).stdout
    out = []
    for record in raw.split("\x1e"):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split("\x1f")
        if len(parts) < 4:
            continue
        out.append((parts[0], parts[1], parts[2].strip(), parts[3]))
    return out


def build(entries, current_version: str) -> str:
    sections: list[tuple[str, str, list[str]]] = []
    name, date, lines = f"{current_version} (in progress)", "", []
    for h, d, title, body in entries:
        found = VERSION_RE.search(body) or VERSION_RE.search(title)
        if found:
            # this commit begins the version it names: close the one above
            if lines or sections:
                sections.append((name, date, lines))
            name, date, lines = found.group(1), d, []
            if name == current_version:
                name = current_version
        lines.append(f"- {title}  `{h}`  {d}")
    sections.append((name, date, lines))
    text = ["# What changed", "",
            "One line per commit, newest first; a version heading marks the "
            "commit that began it.", ""]
    seen = set()
    for name, date, lines in sections:
        if not lines:
            continue
        heading = name if name not in seen else f"{name} (continued)"
        seen.add(name)
        text.append(f"## {heading}" + (f" — {date}" if date else ""))
        text.append("")
        text.extend(lines)
        text.append("")
    return "\n".join(text)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from facet.version import __version__
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(commits(), __version__), encoding="utf-8")
    print(f"changelog -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
