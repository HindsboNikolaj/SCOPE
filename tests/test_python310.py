"""Every module runs on the oldest Python the package supports (requires-python >= 3.10)."""
import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")))


def _quote(token: str) -> str:
    body = token.lstrip("rRbBuUfF")
    return body[:3] if body[:3] in ('"""', "'''") else body[:1]


def reused_fstring_quotes(source: str):
    """Lines where an f-string reuses its own quote inside a replacement field (Python 3.12+ only)."""
    if sys.version_info < (3, 12):   # older tokenizers reject it: compile() reports it below
        return []
    bad, stack = [], []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        kind = tokenize.tok_name[tok.type]
        if kind == "FSTRING_START":
            if stack and _quote(tok.string) == stack[-1]:
                bad.append(tok.start[0])
            stack.append(_quote(tok.string))
        elif kind == "FSTRING_END":
            stack.pop()
        elif kind == "STRING" and stack and _quote(tok.string) == stack[-1]:
            bad.append(tok.start[0])
    return bad


def test_sources_run_on_python_310():
    # A PEP 701 f-string such as f"{row.get("x")}" is fine on 3.12 and a SyntaxError on 3.10
    # and 3.11; one in judge.py stopped the judge on those versions.
    problems = []
    for path in SOURCES:
        source = path.read_text()
        try:
            compile(source, str(path), "exec")
        except SyntaxError as e:
            problems.append(f"{path.relative_to(ROOT)}:{e.lineno}: {e.msg}")
        problems += [f"{path.relative_to(ROOT)}:{n}: f-string reuses its quote"
                     for n in reused_fstring_quotes(source)]
    assert not problems, problems
