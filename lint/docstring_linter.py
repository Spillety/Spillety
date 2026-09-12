#!/usr/bin/env python3
"""
Docstring formatter linter for K-BRAIN project.

Fixes triple-quoted strings that don't follow the project convention:
- All docstrings must start with \"\"\"\\n
- All docstrings must end with \\n\"\"\"
- Indentation preserved on content lines.

Usage:
    python lint/docstring_linter.py [--fix] [--check] [PATHS...]
    python lint/docstring_linter.py --version

Pre-commit integration: .pre-commit-config.yaml

"""

import argparse
import re
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional


# Match triple-quoted strings (both """ and ''')
DOCSTRING_PATTERN = re.compile(
    r'(\"\"\".*?\"\"\")',
    re.DOTALL
)

SINGLE_LINE_DOCSTRING = re.compile(
    r'(\"\"\")(.+?)(\"\"\")',
    re.DOTALL
)

MULTI_LINE_DOCSTRING = re.compile(
    r'(\"\"\")\n(.+?)(\n\s*\"\"\")',
    re.DOTALL
)


def fix_docstring(content: str, file_path: str = "") -> Tuple[str, List[str]]:
    """Fix all docstrings in the given content.

    Returns (fixed_content, list_of_changes).
    Each change is a string describing what was fixed.
    """
    changes: List[str] = []
    fixed = content

    # Find all triple-quoted strings
    for match in DOCSTRING_PATTERN.finditer(content):
        full_match = match.group(1)
        start_pos = match.start()

        # Skip if already has correct format (starts with \"\"\"\n)
        if full_match.startswith('"""\n'):
            continue

        # Determine indentation from the line where the docstring starts
        line_start = fixed.rfind('\n', 0, start_pos) + 1
        indent = fixed[line_start:start_pos]

        # Check if it's a single-line docstring
        if '\n' not in full_match or full_match.count('\n') == 0:
            # Single-line: """text""" -> """\ntext\n"""
            inner_match = SINGLE_LINE_DOCSTRING.match(full_match)
            if inner_match:
                opening = inner_match.group(1)
                doc_text = inner_match.group(2)
                closing = inner_match.group(3)

                # Preserve the original indentation from the content
                # Check if the closing quotes are on the same line with indentation
                new_docstring = f'{opening}\n{indent}{doc_text}\n{indent}{closing}'

                fixed = fixed[:start_pos] + new_docstring + fixed[match.end():]
                changes.append(f"Fixed single-line docstring at line {content[:start_pos].count(chr(10)) + 1}")

        else:
            # Multi-line docstring without leading newline
            # Pattern: """text\n    more text\n    """
            multi_match = re.match(
                r'(\"\"\")(.+?)(\n)(\s*)(\"\"\")',
                full_match,
                re.DOTALL
            )

            if multi_match:
                opening = multi_match.group(1)
                first_line = multi_match.group(2)
                newline_before_close = multi_match.group(3)
                close_indent = multi_match.group(4)
                closing = multi_match.group(5)

                # Reconstruct with leading newline
                new_docstring = f'{opening}\n{close_indent}{first_line}{newline_before_close}{close_indent}{closing}'

                fixed = fixed[:start_pos] + new_docstring + fixed[match.end():]
                changes.append(f"Fixed multi-line docstring at line {content[:start_pos].count(chr(10)) + 1}")
            else:
                # Try another pattern: """text\n    text\n    """
                # Where first line doesn't have \n after opening """
                multi_match2 = re.match(
                    r'(\"\"\")(.+?)(\n\s*\"\"\")',
                    full_match,
                    re.DOTALL
                )
                if multi_match2:
                    opening = multi_match2.group(1)
                    content_between = multi_match2.group(2)
                    closing_part = multi_match2.group(3)

                    # Get indentation from closing part
                    close_indent_match = re.match(r'(\s*)', closing_part)
                    close_indent = close_indent_match.group(1) if close_indent_match else ""

                    # Determine proper indentation from context
                    line_start = fixed.rfind('\n', 0, start_pos) + 1
                    context_indent = fixed[line_start:start_pos]

                    new_docstring = f'{opening}\n{context_indent}{content_between}{closing_part}'

                    fixed = fixed[:start_pos] + new_docstring + fixed[match.end():]
                    changes.append(f"Fixed docstring at line {content[:start_pos].count(chr(10)) + 1}")

    return fixed, changes


def format_docstring_line(line: str) -> str:
    """Format a single line of a docstring to ensure proper leading newline.

    Handles:
    - Single-line docstring without leading newline -> fixed
    - Multi-line docstring without leading newline -> fixed
    """
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]

    if stripped.startswith('"""') and not stripped.startswith('"""\n'):
        # Single-line or start of multi-line without newline
        if stripped.endswith('"""') and stripped.count('"""') == 2:
            # Single-line docstring
            inner = stripped[3:-3]
            if inner.strip():
                return f'{indent}"""\n{indent}{inner.strip()}\n{indent}"""'
            else:
                return line  # Empty docstring, leave as-is
        elif stripped.endswith('"""'):
            # First line of multi-line docstring
            inner = stripped[3:]
            return f'{indent}"""\n{indent}{inner}'
    return line


def process_file(file_path: str, fix: bool = False) -> Tuple[int, List[str]]:
    """Process a single Python file.

    Returns (number_of_changes, list_of_change_descriptions).
    """
    path = Path(file_path)
    if not path.exists():
        return 0, []

    content = path.read_text(encoding='utf-8')
    original = content
    all_changes: List[str] = []

    # Process line by line for single-line fixes
    new_lines = []
    in_docstring = False
    docstring_lines: List[str] = []
    docstring_start_line = 0
    docstring_indent = ""

    i = 0
    while i < len(content):
        # Check for triple-quoted string start
        if content[i:i+3] == '"""' and not in_docstring:
            # Check if it's followed by a newline
            if content[i+3:i+4] == '\n':
                # Already has leading newline - check closing
                in_docstring = True
                docstring_start_line = content[:i].count('\n') + 1
                docstring_indent = content[max(0, content.rfind('\n', 0, i)+1):i]
                new_lines.append('"""')
                i += 3
                continue
            else:
                # No leading newline - need to fix
                # Find the end of this docstring
                end_match = re.search(r'"""', content[i+3:])
                if end_match:
                    full_docstring = content[i:end_match.end()+i+3]
                    docstring_start_line = content[:i].count('\n') + 1
                    docstring_indent = content[max(0, content.rfind('\n', 0, i)+1):i]

                    inner = full_docstring[3:-3]
                    if '\n' in inner:
                        # Multi-line without leading newline
                        lines = inner.split('\n')
                        first_line = lines[0].rstrip()
                        remaining = '\n'.join(lines[1:])

                        if remaining.strip():
                            new_docstring = f'"""\n{docstring_indent}{first_line}\n{remaining}\n{docstring_indent}"""'
                        else:
                            new_docstring = f'"""\n{docstring_indent}{first_line}\n{docstring_indent}"""'
                        all_changes.append(f"Fixed docstring at line {docstring_start_line}")
                        new_lines.append(new_docstring)
                    else:
                        # Single-line
                        new_docstring = f'"""\n{docstring_indent}{inner.strip()}\n{docstring_indent}"""'
                        all_changes.append(f"Fixed single-line docstring at line {docstring_start_line}")
                        new_lines.append(new_docstring)

                    i = end_match.end() + i + 3
                    continue

        new_lines.append(content[i])
        i += 1

    new_content = ''.join(new_lines)

    if fix and new_content != original:
        path.write_text(new_content, encoding='utf-8')
    elif not fix and new_content != original:
        # Just report changes
        pass

    return len(all_changes), all_changes


def find_python_files(paths: List[str]) -> List[str]:
    """Find all Python files in the given paths."""
    py_files: List[str] = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.py':
            py_files.append(str(path))
        elif path.is_dir():
            for f in path.rglob('*.py'):
                py_files.append(str(f))
    return sorted(py_files)


def main():
    parser = argparse.ArgumentParser(
        description='K-BRAIN Docstring Linter — fixes triple-quoted string formatting'
    )
    parser.add_argument('paths', nargs='*', default=['.'],
                        help='Files or directories to lint (default: current directory)')
    parser.add_argument('--fix', action='store_true',
                        help='Apply fixes automatically')
    parser.add_argument('--check', action='store_true',
                        help='Check only, report changes without applying')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')

    args = parser.parse_args()

    paths = args.paths if args.paths else ['.']
    py_files = find_python_files(paths)

    if not py_files:
        print("No Python files found.")
        sys.exit(0)

    total_changes = 0
    all_issues: List[str] = []

    for file_path in py_files:
        changes, descriptions = process_file(file_path, fix=args.fix)
        if changes > 0:
            total_changes += changes
            for desc in descriptions:
                issue = f"{file_path}: {desc}"
                all_issues.append(issue)
                if args.check or args.verbose:
                    print(f"ISSUE: {issue}")
            if args.fix:
                print(f"FIXED: {file_path} ({changes} docstrings)")

    if args.check:
        if total_changes > 0:
            print(f"\n{total_changes} docstring(s) need fixing across {len(py_files)} files.")
            sys.exit(1)
        else:
            print("All docstrings properly formatted.")
            sys.exit(0)
    elif args.fix:
        print(f"\nTotal: {total_changes} docstring(s) fixed in {len(py_files)} file(s).")
        sys.exit(0)
    else:
        # Report mode
        for issue in all_issues:
            print(f"ISSUE: {issue}")
        if total_changes > 0:
            print(f"\n{total_changes} docstring(s) need fixing. Use --fix to apply.")
            sys.exit(1)
        else:
            print("All docstrings properly formatted.")
            sys.exit(0)


if __name__ == '__main__':
    main()
