#!/usr/bin/env python3
"""Check K-BRAIN docstring style: summary first line starts with ``## ``."""

import ast
import sys
from pathlib import Path

DEFAULT_ROOTS = ("spillety", "tests", "scripts")


def _first_line(doc):
    stripped = doc.strip()
    return stripped.splitlines()[0] if stripped else "<empty>"


def check_file(path):
    """Collect style violations in one file."""
    violations = []
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return [f"{path}:1 unparseable :: {exc}"]
    mod = ast.get_docstring(tree, clean=False)
    if mod is not None and not _first_line(mod).startswith("## "):
        violations.append(f"{path}:1 module docstring :: {_first_line(mod)[:80]}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and not _first_line(doc).startswith("## "):
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                violations.append(
                    f"{path}:{node.lineno} {kind} {node.name} :: {_first_line(doc)[:80]}"
                )
    return violations


def collect(targets):
    """Expand CLI targets to .py files (default: repo roots)."""
    if not targets:
        targets = DEFAULT_ROOTS
    files = []
    for target in targets:
        p = Path(target)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        elif p.is_file() and p.suffix == ".py":
            files.append(p)
    return [f for f in files if "__pycache__" not in f.parts]


def main(argv):
    """Entry point: --check reports violations, exit 1 when found."""
    targets = [a for a in argv[1:] if a != "--check"]
    violations = []
    for path in collect(targets):
        violations.extend(check_file(path))
    if violations:
        print("\n".join(violations))
        print(f"docstring style: {len(violations)} violation(s)")
        return 1
    print("docstring style: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
