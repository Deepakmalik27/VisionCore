"""test_run_config.py — config/*.yaml must actually reach the run.

WHY THIS EXISTS
    config/cam112.yaml declared `analysis.fps: 8`. kevacv/config.py declared
    FPS_TARGET = 15. The notebook declared 7. Nothing read the yaml — run.sh
    only `sed`s it into the log — so 15 ran, and the log displayed a file
    saying 8 directly above it.

    That is the worst failure mode a config file has: printed, trusted, and
    inert. These tests pin the two halves of the fix — the yaml wins, and
    anything it says that we do NOT understand is reported rather than dropped.

Run: python tests/test_run_config.py
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.config import apply_run_config, describe_run_config

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def _stub():
    """Stand-in for kevacv.engine — the real module needs torch, and none of
    this logic does."""
    m = types.ModuleType("stub_engine")
    m.FPS_TARGET = 15
    m.YOLO_IMGSZ = 1280
    m.TRACKER_MODE = "botsort-reid"
    m.ANALYSIS_MAX_W = 1280
    return m


def _write(tmp_path, text):
    p = tmp_path / "run.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_shipped_config_sets_fps():
    """The real file, not a fixture — this is the regression that shipped.

    UPDATED 2026-08-26. This asserted `FPS_TARGET == 8` and had been failing
    for some time: cam112.yaml was changed from 8 to 15 and the test was not.
    A permanently-red test is worse than no test, because the next real
    regression hides inside "yeah, that one always fails".

    WHY 15 IS THE VALUE THAT MATTERS
        MOT ground truth is frame-INDEXED, so it is only valid for a run whose
        frame sampling matches the clip it was labelled on. Both GT clips were
        cut at 15 fps. Sample at any other rate and every frame renumbers, so
        the ground truth silently aligns to the wrong moments and the HOTA it
        reports is confident nonsense.

        So this no longer asserts a literal. It asserts that the shipped config
        and the labelled clips AGREE -- which is the property that would
        actually break scoring if it drifted again.
    """
    stub = _stub()
    r = apply_run_config(ROOT / "config" / "cam112.yaml", target=stub)
    GT_CLIP_FPS = 15                      # both eval clips, verified by ffprobe
    check(stub.FPS_TARGET == GT_CLIP_FPS,
          f"cam112.yaml drives FPS_TARGET to {GT_CLIP_FPS}, matching the "
          f"fps the ground-truth clips were cut at",
          f"got {stub.FPS_TARGET}; a mismatch renumbers every frame and "
          f"misaligns the ground truth with no error")
    check("FPS_TARGET" in r["applied"], "the change is reported to the caller")
    # `from` is the TARGET's value before the call -- here the stub's 15, NOT
    # the package default of 8. Worth stating: I asserted 8 and it failed,
    # because apply_run_config reports what it actually overwrote, which is
    # exactly what a provenance log should say.
    check(r["applied"]["FPS_TARGET"]["from"] == _stub().FPS_TARGET,
          "the value it replaced is reported too, so the log shows the move")


def test_caller_owned_keys_are_not_unknown():
    """detector/reid_weights are real settings consumed by the caller. If they
    were reported as unrecognised, every run would print a false warning and
    the genuine ones would stop being read."""
    stub = _stub()
    r = apply_run_config(ROOT / "config" / "cam112.yaml", target=stub)
    check(r["unknown"] == [], "no false 'unrecognised' on the shipped config",
          str(r["unknown"]))


def test_unknown_key_is_reported_not_dropped(tmp_path):
    stub = _stub()
    p = _write(tmp_path, "analysis:\n  fps: 5\n  bogus_setting: 99\n")
    r = apply_run_config(p, target=stub)
    check("analysis.bogus_setting" in r["unknown"],
          "an unrecognised key is reported", str(r["unknown"]))
    check(stub.FPS_TARGET == 5, "and the keys we DO understand still apply")
    check("UNRECOGNISED" in describe_run_config(r),
          "and it is visible in the printed block")


def test_missing_file_is_reported_not_silent(tmp_path):
    stub = _stub()
    r = apply_run_config(tmp_path / "nope.yaml", target=stub)
    check(stub.FPS_TARGET == 15, "defaults survive a missing config")
    check(r["unknown"] and "not found" in r["unknown"][0],
          "and the absence is reported", str(r["unknown"]))


def test_only_analysis_is_applied(tmp_path):
    """measured_baseline records a PAST run. Applying it would silently
    reconfigure this one from a historical observation."""
    stub = _stub()
    p = _write(tmp_path, "analysis:\n  fps: 6\n"
                    "measured_baseline:\n  fps: 999\n"
                    "targets:\n  hota_floor: 0.4\n")
    r = apply_run_config(p, target=stub)
    check(stub.FPS_TARGET == 6, "analysis applied", f"got {stub.FPS_TARGET}")
    check(r["unknown"] == [],
          "non-analysis sections are ignored without being called unknown",
          str(r["unknown"]))


def test_empty_config_changes_nothing(tmp_path):
    stub = _stub()
    p = _write(tmp_path, "camera:\n  id: CAM.112\n")
    r = apply_run_config(p, target=stub)
    check(r["applied"] == {}, "nothing applied from a config with no analysis block")
    check(stub.FPS_TARGET == 15, "module defaults untouched")
    check("nothing applied" in describe_run_config(r),
          "and the log says so plainly rather than printing an empty block")


def main():
    import tempfile
    print(__doc__.strip().splitlines()[0])
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        for fn in (test_shipped_config_sets_fps,
                   test_caller_owned_keys_are_not_unknown,
                   test_unknown_key_is_reported_not_dropped,
                   test_missing_file_is_reported_not_silent,
                   test_only_analysis_is_applied,
                   test_empty_config_changes_nothing):
            print(f"\n{fn.__name__}")
            fn(tmp_path) if fn.__code__.co_argcount else fn()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
