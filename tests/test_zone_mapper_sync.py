"""test_zone_mapper_sync.py — the mapper must not lie about your labels.

WHY THIS EXISTS
    A zone's NAME decides what it can answer. tools/zone_mapper_v2.html shows
    you the role your typed name will get, so you can tell BEFORE a 30-minute
    run whether "plant_mask" does anything. That preview is a hand-copied
    duplicate of kevacv/helpers.py ZONE_ROLE_KEYWORDS, and it drifted:

        the mapper was missing  gate, doorway, passageway, holding,
                                podium, desk, seating
        and the roles           mask  and  walkway  ENTIRELY

    So "plant_mask" and "corridor_west" both previewed as
    "other -- drives NO metric, rename it", advising the operator to rename
    labels the engine handles perfectly. A tool that misreports the contract is
    worse than no tool, because it is trusted.

    Two copies of one table cannot be prevented -- the mapper is a standalone
    HTML file that must open with no server and no Python. So they are pinned
    together HERE instead, and drift becomes a failing test rather than a
    rename that quietly removes a zone from the analysis.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kevacv.helpers import ZONE_ROLE_KEYWORDS, classify_zones  # noqa: E402

MAPPER = ROOT / "tools" / "zone_mapper_v2.html"
_fail = False


def check(ok, what, detail=""):
    global _fail
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        _fail = True


def mapper_keywords(html):
    """Parse the JS object literal into the same shape as the Python dict."""
    m = re.search(r"const ZONE_ROLE_KEYWORDS = \{(.*?)\n\};", html, re.S)
    if not m:
        return None
    out = {}
    for role, body in re.findall(r"(\w+)\s*:\s*\[(.*?)\]", m.group(1), re.S):
        out[role] = [w for w in re.findall(r'"([^"]+)"', body)]
    return out


print("=" * 74)
print("  the mapper's role table mirrors the engine's")
print("=" * 74)

check(MAPPER.exists(), "the mapper exists where the docs say it does", MAPPER.name)
html = MAPPER.read_text(encoding="utf-8")
js = mapper_keywords(html)
check(js is not None, "its ZONE_ROLE_KEYWORDS is parseable")

if js is not None:
    check(set(js) == set(ZONE_ROLE_KEYWORDS),
          "every ROLE the engine knows is offered by the mapper",
          f"only in python: {sorted(set(ZONE_ROLE_KEYWORDS) - set(js))} · "
          f"only in html: {sorted(set(js) - set(ZONE_ROLE_KEYWORDS))}")
    for role in sorted(set(js) & set(ZONE_ROLE_KEYWORDS)):
        check(js[role] == list(ZONE_ROLE_KEYWORDS[role]),
              f"role '{role}' has the same keywords, in the same order",
              f"python={list(ZONE_ROLE_KEYWORDS[role])} html={js[role]}")

print()
print("=" * 74)
print("  the labels this venue actually uses all mean something")
print("=" * 74)

# The real regression: these are the names in zones/CAM.112_zone.json. Every
# one of them must drive a role, in BOTH implementations.
LIVE = ["reception", "main_entrance", "waiting_area", "corridor_west",
        "plant_mask", "staff_door", "dining"]
py_roles = classify_zones(LIVE)
for n in LIVE:
    got = py_roles.get(n) or []
    check(got and got != ["other"], f"engine: '{n}' -> {got or 'other'}")
    if js is not None:
        low = n.lower()
        jr = [r for r in js if any(k in low for k in js[r])]
        check(bool(jr), f"mapper: '{n}' -> {jr or 'other (WOULD TELL YOU TO RENAME IT)'}")

print()
print("=" * 74)
print("  the page can actually DRAW")
print("=" * 74)

# THE BUG THIS SECTION EXISTS FOR, 2026-08-13: an over-greedy edit that removed
# the hardcoded VENUE block also removed the declarations that lived under it --
# canvas, context, polygon state, mode(). The page loaded with no error and drew
# NOTHING. It got shipped because the checks were the wrong ones: `node --check`
# validates SYNTAX, and a missing global is a RUNTIME error; the handler check
# only compared onclick="x()" names, and mode()/setStatus() are not handlers.
#
# So assert the things a click actually needs, by name.
REQUIRED = [
    ("const cv ", "the canvas -- without it nothing can be drawn"),
    ("ctx = cv.getContext", "the 2d context"),
    ("const video ", "the video element"),
    ("const COLORS", "the zone palette"),
    ("let polygons", "polygon state"),
    ("entryLines", "door-line state"),
    ("function mode(", "polygon-vs-line switch, read on every click"),
    ("function setStatus(", "the status line every handler writes to"),
    ("function onNameType(", "the live role preview, called from oninput"),
    ("function closeShape(", "finishing a shape"),
    ("function draw(", "the render loop"),
    ("cv.onclick", "the click handler itself"),
]
for frag, why in REQUIRED:
    check(frag in html, f"page defines `{frag.strip()}`", why)

# Every identifier an inline handler calls must exist as a function.
handlers = set(re.findall(r'on\w+="(\w+)\(', html))
defined = set(re.findall(r"function (\w+)\(", html))
missing = sorted(handlers - defined - {"document"})
check(not missing, "every inline on*= handler resolves to a real function",
      missing or "none dangling")

# And every global those functions reference must be declared somewhere.
declared = set(defined)
for stmt in re.findall(r"(?:const|let|var)\s+([^;\n]+)", html):
    # `let polygons = {}, entryLines = {}, current = [], scale = 1;` declares
    # FOUR globals. Taking only the first name is how this check would have
    # reported a false failure on code that is perfectly fine.
    for part in stmt.split(","):
        m = re.match(r"\s*(\w+)", part)
        if m:
            declared.add(m.group(1))
for name in ("cv", "ctx", "video", "polygons", "entryLines", "current",
             "scale", "COLORS", "classifyRoles"):
    check(name in declared, f"global `{name}` is declared, not assumed")

print()
print("=" * 74)
print("  the mapper writes the schema the engine reads")
print("=" * 74)

check("entry_lines:" in html or "entry_lines" in html,
      "it exports entry_lines (many doors), not entry_line (one)")
check(re.search(r"payload = \{[^}]*entry_line\s*:", html) is None,
      "and the single-line key is gone from the payload",
      "a one-line export collapses a 3-door venue into one number")
check("roles:" in html or '"roles"' in html or "roles: roles" in html,
      "it writes roles explicitly, so a later rename cannot silently "
      "change what a zone measures")
check("j.entry_line ?" in html or "j.entry_line" in html,
      "but it still OPENS old single-line files")

print()
print("=" * 74)
print("  ALL PASS" if not _fail else "  FAILURES ABOVE")
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (1 if _fail else 0), "module-level checks in this file failed"


if __name__ == "__main__":
    sys.exit(1 if _fail else 0)
