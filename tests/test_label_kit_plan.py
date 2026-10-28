"""entry_label_kit is the only path to the ground truth this project has never
had. Two faults made it unusable:

  * `plan` could emit the SAME window twice -- on a short clip the "busy" tail
    slice still contained the zero-activity windows the "quiet" slice had
    already taken. A duplicate is counted twice by `check` when it decides
    whether the reference is usable, and twice again by `score`.
  * `sheets` was documented in the module docstring and did not exist, so the
    workflow stopped after choosing windows and no frames ever reached a human.
"""
import gzip
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "entry_label_kit", ROOT / "tools" / "entry_label_kit.py")
KIT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(KIT)

DOOR_X, DOOR_Y = 1150, 380


def _run_dir(tmp_path, busy_windows=(( 20, 33, 4),)):
    frames = []
    for fi in range(15 * 60):
        t = fi / 15.0
        boxes = []
        for t0, t1, n in busy_windows:
            if t0 <= t < t1:
                boxes = [[100 + k, DOOR_X - 20, DOOR_Y - 40,
                          DOOR_X + 40, DOOR_Y + 60] for k in range(n)]
        frames.append([fi, t, boxes])
    d = tmp_path / "run"
    d.mkdir()
    with gzip.open(d / "CAM.112_frames.json.gz", "wt") as fh:
        json.dump(frames, fh)
    return d


def _plan(tmp_path, n=6):
    out = tmp_path / "plan.json"
    KIT.plan(str(_run_dir(tmp_path)), n=n, out=str(out), camera="CAM.112")
    return json.loads(out.read_text())


def test_plan_never_emits_the_same_window_twice(tmp_path):
    doc = _plan(tmp_path)
    keys = [(w["t0"], w["t1"]) for w in doc["windows"]]
    assert len(keys) == len(set(keys)), keys


def test_plan_still_holds_out_the_busiest_window(tmp_path):
    doc = _plan(tmp_path)
    loud = [w for w in doc["windows"] if w["door_track_ids"] > 0]
    assert loud, "synthetic clip should produce at least one busy window"
    busiest = max(loud, key=lambda w: w["door_track_ids"])
    assert busiest["held_out"] and busiest["sealed"]


def test_plan_keeps_a_quiet_window(tmp_path):
    doc = _plan(tmp_path)
    assert any(w["door_track_ids"] == 0 for w in doc["windows"])


def test_sheets_command_exists():
    # the docstring advertised it for as long as the file has existed
    assert hasattr(KIT, "sheets")
    assert "sheets" in KIT.__doc__
