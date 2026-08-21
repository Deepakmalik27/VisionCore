"""The wired slit estimator must pick the VENUE door in SOURCE coordinates.

Two landmines this pins:
  - CAM.112's entry_lines dict lists 'dining entry' FIRST; picking by
    iteration order counts the interior threshold.
  - the zones file is in 3840x2160 while the analysed frame is 1920x1080;
    scaling to the analysed size halves the line onto the reception desk.
"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from kevacv.analytics import venue_entry_lines

Z = "zones/CAM.112_zone.json"


def test_picks_the_venue_door_not_the_first_key():
    lines = json.load(open(Z))["entry_lines"]
    assert list(lines)[0] == "dining entry", "fixture changed"
    door = venue_entry_lines(set(lines))
    picked = next(k for k in lines if k in door)
    assert picked == "entry line", f"picked {picked!r}, an interior threshold"


def test_line_is_in_source_coordinates():
    cfg = json.load(open(Z))
    ref = cfg["frame_size"]
    assert tuple(ref) == (3840, 2160), ref
    ln = cfg["entry_lines"]["entry line"]
    # unscaled, the line must sit in the RIGHT half of a 3840-wide frame
    assert min(p[0] for p in ln) > 1920, "line is not in source coords"
    # and halving it (the analysed-size mistake) lands it left of centre
    assert max(p[0] / 2 for p in ln) < 1920, "the halving mistake is detectable"


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
