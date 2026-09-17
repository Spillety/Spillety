#!/usr/bin/env python3
from pathlib import Path

SKILL = Path(".agents/skills/spillety/SKILL.md")
START = "<!-- AUTO-GEN:START -->"
END = "<!-- AUTO-GEN:END -->"
PATTERNS = ["docs/*.md", "docs/notebooks/*.ipynb", "docs/theory/*.md", "spillety/**/*.py", ""]
MUST = ["README.md", "pyproject.toml"]
EXCLUDE = {"node_modules", "__pycache__", ".git", ".venv", "venv", "dist", "build", "package-lock.json"}


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE for part in path.parts)


def collect():
    files = []
    for pat in PATTERNS:
        for p in Path(".").glob(pat):
            if p.is_file() and not _is_excluded(p):
                files.append(p.as_posix())
    for m in MUST:
        if m not in files and Path(m).exists():
            files.append(m)
    return sorted(set(files))


def build_block(paths):
    if not paths:
        return f"{START}\n<!-- no files -->\n{END}"
    return "\n".join([START] + [f"- `https://github.com/Spillety/Spillety/{p}`" for p in paths] + [END])


def main():
    if not SKILL.exists():
        print(f"SKILL not found: {SKILL}")
        return 1
    text = SKILL.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("markers not found")
        return 1
    paths = collect()
    block = build_block(paths)
    before, _, after = text.partition(START)
    _, _, after2 = after.partition(END)
    new_text = before + block + after2
    if new_text != text:
        SKILL.write_text(new_text, encoding="utf-8")
        print(f"updated {SKILL}: {len(paths)} links")
    else:
        print(f"no changes {SKILL}: {len(paths)} links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
