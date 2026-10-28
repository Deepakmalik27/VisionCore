"""test_build_id.py — the output must say which code produced it.

WHY THIS EXISTS
    Five days of runs produced identical output while the source changed daily.
    The GPU box was running a copy of kevacv/ made once by setup_pod.sh, and
    bootstrap.sh skips setup whenever the venv already exists — so it re-ran
    frozen code forever and nothing in the output said so.

    engine.py has burned _BUILD_ID onto the annotated video's HUD the whole
    time, with a comment reading "we could not tell which build a reviewed
    video came from, so fixes looked like no-ops". Nothing on the codebase
    path ever set it, so every video said "build ?".

    These tests pin the properties that make the id worth trusting: it is
    derived from content, it is stable, and it MOVES when the code moves.

Run: python tests/test_build_id.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv import build_id

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def _pkg(tmp, files):
    d = Path(tmp)
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
    return d


def test_stable_for_identical_content():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        files = {"x.py": "A = 1\n", "y.py": "B = 2\n"}
        check(build_id.compute(_pkg(a, files)) == build_id.compute(_pkg(b, files)),
              "two checkouts with identical content share an id")


def test_changes_when_content_changes():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        one = build_id.compute(_pkg(a, {"x.py": "A = 1\n"}))
        two = build_id.compute(_pkg(b, {"x.py": "A = 2\n"}))
        check(one != two, "a one-character change moves the id", f"{one} vs {two}")


def test_changes_when_a_file_is_added():
    """The stale-pod failure was a MISSING module, not a changed one."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        one = build_id.compute(_pkg(a, {"x.py": "A = 1\n"}))
        two = build_id.compute(_pkg(b, {"x.py": "A = 1\n", "new.py": "C = 3\n"}))
        check(one != two, "adding a module moves the id")


def test_filename_is_part_of_the_id():
    """Same bytes under a different name is different code to every importer."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        one = build_id.compute(_pkg(a, {"x.py": "A = 1\n"}))
        two = build_id.compute(_pkg(b, {"z.py": "A = 1\n"}))
        check(one != two, "a rename moves the id")


def test_diff_localises_the_mismatch():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        da = _pkg(a, {"same.py": "S = 1\n", "moved.py": "M = 1\n",
                      "only_here.py": "H = 1\n"})
        db = _pkg(b, {"same.py": "S = 1\n", "moved.py": "M = 2\n",
                      "only_there.py": "T = 1\n"})
        d = build_id.diff(build_id.manifest(db), da)
        check(d["changed"] == ["moved.py"], "the changed file is named",
              str(d["changed"]))
        check(d["only_here"] == ["only_here.py"], "a file missing there is named",
              str(d["only_here"]))
        check(d["only_there"] == ["only_there.py"], "an orphan there is named",
              str(d["only_there"]))


def test_real_package_has_an_id():
    bid = build_id.compute()
    check(len(bid) == 12 and bid.isalnum(),
          "the live package produces a 12-char id", bid)
    check(bid in build_id.describe(), "and describe() shows it")


def test_hud_truncation_is_safe():
    """engine.py renders str(_BUILD_ID)[:12] onto the HUD band. If the id were
    shorter the check would still work, but it must never be empty."""
    check(len(build_id.compute()[:12]) == 12, "12 chars survive HUD truncation")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_stable_for_identical_content,
               test_changes_when_content_changes,
               test_changes_when_a_file_is_added,
               test_filename_is_part_of_the_id,
               test_diff_localises_the_mismatch,
               test_real_package_has_an_id,
               test_hud_truncation_is_safe):
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
