"""test_render_path.py — the overlay code that 390 passing tests never touch.

WHY THIS FILE EXISTS
    On 2026-08-26 two bugs shipped into render_annotated() while the suite was
    fully green and tools/verify_fresh.py reported FRESH:

      1. a trail fix referenced `frame_w`, which is not in scope in the
         renderer -> NameError, and the run died an hour in.
      2. the overlay blend was inserted AFTER the h264 pipe write. The live
         path writes through that pipe, so the setting would have reported as
         applied while the video looked identical.

    Neither is subtle. Both survived because nothing in tests/ imports the
    render path at all -- 390 tests, zero coverage of the code that produces
    the only artefact anybody actually watches.

    These are static checks, not a render. A real render needs a video, a GPU
    and three minutes; that is what the 20-second smoke run is for. What a
    static check CAN do is fail in 20 milliseconds on the exact two mistakes
    above, which is the difference between finding them here and finding them
    after an hour of GPU time.

Run: python tests/test_render_path.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


SRC = (ROOT / "kevacv" / "engine.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)
FN = next((n for n in ast.walk(TREE)
           if isinstance(n, ast.FunctionDef) and n.name == "render_annotated"), None)

print()
print("=" * 76)
print("  the renderer only uses names it actually has")
print("=" * 76)
check(FN is not None, "render_annotated() found in engine.py")
if FN is not None:
    # Every name bound anywhere inside the function, plus its arguments, plus
    # module-level names. Anything referenced but not in that set is the
    # `frame_w` class of bug: a name borrowed from a DIFFERENT function.
    bound = {a.arg for a in FN.args.args}
    for n in ast.walk(FN):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                bound.add((al.asname or al.name).split(".")[0])
    module_level = {n.targets[0].id for n in TREE.body
                    if isinstance(n, ast.Assign) and n.targets
                    and isinstance(n.targets[0], ast.Name)}
    for n in TREE.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            module_level.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                module_level.add((al.asname or al.name).split(".")[0])
    # engine.py does `from .config import *`, so every config constant is a
    # legitimate global that no AST scan of engine.py alone can see. Include
    # them -- but ONLY them. `frame_w` is not a config constant, so the check
    # that caught it stays sharp; see the self-test below.
    import kevacv.config as _C
    cfg_names = {n for n in dir(_C) if not n.startswith("__")}
    # `__builtins__` is a MODULE when a file is run directly and a DICT when
    # it is imported -- so vars(__builtins__) works under __main__ and raises
    # TypeError under pytest. Importing the module explicitly behaves the same
    # either way, which is the only reason this test collects at all.
    import builtins as _B
    known = bound | module_level | cfg_names | set(dir(_B))
    # names from THIS function's enclosing scope that the renderer legitimately
    # closes over would show here too; engine.py's renderer is module-level, so
    # anything unresolved is genuinely unbound.
    unbound = sorted({n.id for n in ast.walk(FN)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                     - known)
    check(not unbound, "no name is used that the renderer does not define",
          f"unbound: {unbound[:6]}" if unbound else "")

print()
print("=" * 76)
print("  ...and that check would still catch the bug it was written for")
print("=" * 76)
# A check that passes because it was loosened until it passed is worthless.
# This proves the unbound-name scan still fires on the REAL 2026-08-26 bug:
# `frame_w`, used in the renderer, defined only in process_video.
_probe = ast.parse("def render_probe(frame):\n"
                   "    lim = max(120.0, frame_w * 0.06)\n"
                   "    return lim\n").body[0]
_pb = {a.arg for a in _probe.args.args}
for _n in ast.walk(_probe):
    if isinstance(_n, ast.Name) and isinstance(_n.ctx, ast.Store):
        _pb.add(_n.id)
_probe_unbound = {n.id for n in ast.walk(_probe)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - _pb - known
check("frame_w" in _probe_unbound,
      "the scan still flags `frame_w` used in a renderer",
      "if this ever passes silently the check has been loosened into uselessness")

print()
print("=" * 76)
print("  the overlay blend reaches the LIVE output path")
print("=" * 76)
# RENDER_DIRECT_H264 pipes frames to ffmpeg, and that branch runs FIRST.
# A blend placed after it is dead code on the only path that ships.
i_blend = SRC.find("_clean_frame is not None")
i_pipe = SRC.find("_ff_pipe.stdin.write")
i_writer = SRC.find("writer.write(frame)")
check(i_blend != -1, "the blend exists at all")
check(i_blend != -1 and i_pipe != -1 and i_blend < i_pipe,
      "blend happens BEFORE the h264 pipe write",
      "the pipe is the live path; blending after it changes nothing")
check(i_blend != -1 and i_writer != -1 and i_blend < i_writer,
      "blend happens BEFORE the cv2 writer too")
check("frame[88:]" in SRC and "_clean_frame[88:]" in SRC,
      "only the video area is blended, not the HUD letterbox",
      "the HUD sits on black; fading it just looks broken")

print()
print("=" * 76)
print("  no dead branches left behind in the write path")
print("=" * 76)
check("and False" not in SRC,
      "no `if ... and False:` stub survived an edit",
      "left one behind on 2026-08-26 while restructuring the write branch")

print()
print("=" * 76)
print("  playback targeting cannot silently drop frames")
print("=" * 76)
check("PLAYBACK_TARGET_S" in SRC, "duration targeting is wired into the renderer")
check("CLAMPED" in SRC,
      "hitting the fps clamp is REPORTED, not silent",
      "an 8 h chunk cannot honestly become 2 minutes")
check("PLAYBACK_FPS_MIN" in SRC and "PLAYBACK_FPS_MAX" in SRC,
      "the clamp is bounded at both ends")

print()
print("=" * 76)
print("  trails are not drawn through jumps")
print("=" * 76)
check("frame.shape[1]" in SRC and "_lim" in SRC,
      "the trail split threshold reads the width from the frame in scope",
      "frame_w belongs to process_video, not to the renderer")

print()
print("=" * 76)


def test_script_level_checks_passed():
    assert not FAILED, f"render-path checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 76)
