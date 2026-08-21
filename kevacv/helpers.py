"""helpers.py — the small shared functions engine.py needs.

EXTRACTED FROM notebook Cells 2, 4 and 6. These are not a coherent subsystem;
they are the utilities the engine happened to reach for out of the notebook's
shared namespace — a clock formatter, a zone loader, a colour map, an id
coercer.

WHY THEY MATTER MORE THAN THEY LOOK
    engine.py CALLS every one of them. In the notebook that worked because all
    cells share one namespace. As a module it would have raised NameError at
    runtime — after the video decoded, after the models loaded, deep inside
    process_video. Python resolves a global at call time, so the import
    succeeded and the failure waited.

    That is exactly the class of bug this project keeps producing: something
    that looks fine until the expensive part has already run.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .log import get_logger

_log = get_logger("helpers")

VIDEO_START_CLOCK = None    # set per run; wall() falls back to video time

# matplotlib is only needed by the two plotting helpers. Importing it at module
# level would make the whole package require a display stack on a headless
# server, so it is imported lazily inside those functions.
plt = None


def _require_plt():
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")      # no display on a server
        import matplotlib.pyplot as _plt
        plt = _plt
    return plt


# ── palette ─────────────────────────────────────────────────────────────────
# Chosen to stay distinguishable under the three common colour-vision
# deficiencies, because these colours are the ONLY thing separating one zone
# from another in the annotated video. A palette that collapses for a
# red-green viewer makes the review artefact useless to them.
ZONE_HEXES = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#008300",
              "#e87ba4", "#eb6834", "#e34948"]
ROLE_HEXES = {"customer": "#2a78d6", "staff": "#eb6834", "unknown": "#898781"}
INK = "#1a1a1a"        # primary text on light backgrounds
INK2 = "#6b6b6b"       # secondary text / axis labels

# ── zone roles ──────────────────────────────────────────────────────────────
# A zone's NAME decides what questions it can answer, so these keywords are
# behaviour, not decoration. "entry" must win over "seating" in a compound
# name like dining_entrance, which is why classify_zones checks entry
# indicators first rather than taking the first keyword that matches.
ZONE_ROLE_KEYWORDS = {
    "entry":   ["entry", "door", "entrance", "gate", "doorway", "passageway"],
    "wait":    ["wait", "queue", "lobby", "line", "holding"],
    "staff":   ["staff", "reception", "host", "counter", "register",
                "checkout", "till", "cashier", "podium", "desk"],
    "seating": ["table", "seat", "dining", "booth", "seating"],
    "service": ["service", "bar", "kitchen", "prep"],
    "mask":    ["mask", "ignore", "mirror", "reflection", "phantom"],
    "walkway": ["walkway", "corridor", "path", "aisle"],
}

# Per-venue manual overrides, consulted before the keywords. Empty by default:
# a guess that cannot be corrected is worse than no guess.
ZONE_AI_OVERRIDES = {}

# Written only when a venue has no zone file at all, so a first run produces
# something drawable rather than crashing. These coordinates are a placeholder
# and will be wrong for every real camera — the file exists to be edited, or
# replaced by learn_entry_zones() fitting the door from the data.
GENERIC_TEMPLATE = {
    "entry_line": [[100, 600], [1000, 600]],
    "polygons": {
        "wait_zone":      [[60, 300], [500, 300], [500, 850], [60, 850]],
        "staff_zone":     [[950, 250], [1300, 250], [1300, 650], [950, 650]],
        "seating_zone_1": [[520, 100], [900, 100], [900, 420], [520, 420]],
    },
}

def _safe_id(tid):
    """Safe ID converter for track IDs: handles both integer IDs (41) and named string IDs ('receptionist_sarah')."""
    if isinstance(tid, str) and not tid.isdigit():
        return tid
    try:
        return int(tid)
    except Exception:
        return str(tid)

def load_zone_config(path, frame_size=None):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(GENERIC_TEMPLATE, indent=2))
        print(f"wrote GENERIC TEMPLATE {path} — edit the coordinates to match this camera view!")
    cfg = json.loads(path.read_text())
    polygons = {name: np.array(pts, dtype=float) for name, pts in cfg.get("polygons", {}).items()}
    # U3: a venue can have more than one door. "entry_lines" is a name -> 2-point
    # map; the old singular "entry_line" still works and becomes {"entry": ...}.
    # Returning ONE line meant a two-door venue could not be counted at all.
    _lines = dict(cfg.get("entry_lines") or {})
    if not _lines and cfg.get("entry_line"):
        _lines["entry"] = cfg["entry_line"]
    entry_lines = {n: [list(map(float, p)) for p in pts]
                   for n, pts in _lines.items() if pts and len(pts) == 2}

    if frame_size:
        fw, fh = frame_size
        ref = cfg.get("frame_size")
        if not ref:
            all_pts = np.vstack(list(polygons.values()) +
                                [np.array(p, dtype=float) for p in entry_lines.values()])
            max_x, max_y = all_pts[:, 0].max(), all_pts[:, 1].max()
            if max_x > fw or max_y > fh:
                ref = (max_x, max_y)
        if ref:
            rw, rh = ref
            sx, sy = fw / rw, fh / rh
            polygons = {name: pts * [sx, sy] for name, pts in polygons.items()}
            entry_lines = {n: [[x * sx, y * sy] for x, y in pts]
                           for n, pts in entry_lines.items()}
            if ref != (fw, fh):
                print(f"↕️  {path.name}: scaled zones from reference "
                      f"{rw:.0f}x{rh:.0f} -> actual {fw}x{fh} "
                      f"(factor {sx:.3f}x, {sy:.3f}y)")

    polygons = {name: pts.astype(int) for name, pts in polygons.items()}
    entry_lines = {n: [[int(x), int(y)] for x, y in pts]
                   for n, pts in entry_lines.items()}
    # U7: an explicit roles map in the zones file wins over keyword guessing, so
    # a zone named in any language still gets its role instead of silently
    # becoming "other" and having its metrics vanish from the report.
    # C8 (2026-08-19): CLEAR FIRST. This is a process-global that was written
    # to and never reset, so camera A's role map persisted into classify_zones
    # for camera B and for the next chunk in the same process. Zone names
    # collide across venues by design ("reception", "dining", "queue"), and
    # roles decide which zone counts as entry, interior and staff -- so the
    # wrong role map silently changes who is a guest and where an arrival is.
    # run_night.py and any multi-camera loop hit this.
    ZONE_AI_OVERRIDES.clear()
    for zname, rs in (cfg.get("roles") or {}).items():
        ZONE_AI_OVERRIDES[zname] = list(rs) if isinstance(rs, (list, tuple)) else [rs]
    return polygons, entry_lines

def uses_centre_anchor(zone_name, staff_zones):
    """Which point on a body decides whether it is inside this zone?

    Feet everywhere EXCEPT staff zones. At a desk the body is clipped by the
    counter, so the bottom of the box is the counter edge and the real feet
    land OUTSIDE the polygon — anchoring on feet there loses the receptionist
    from their own zone (run-2 QA).

    WHY THIS IS A FUNCTION
        The rule was written out three times — the zone triggers in
        process_video, the per-frame counts in render_annotated, and the
        per-zone static policy. All three agreed, but only by coincidence:
        each derived `staff_zones` separately and each re-implemented the
        centre-vs-feet choice inline. Three copies of a geometry rule is how
        boxes and polygons end up disagreeing about where a person is
        standing, which is invisible in the code and obvious in the output.
    """
    return zone_name in (staff_zones or ())


def anchor_point(box, centre=False):
    """The (x, y) that uses_centre_anchor() selected. box is xyxy.

    THE FOOT IS NOT AT THE BOTTOM OF THE BOX ON THIS CAMERA.

    Measured against 600 hand-labelled boxes on CAM.112 (Delilah Dallas,
    2026-08-17): the detector's boxes are 1.6x TALLER than the person inside
    them. CrowdHuman is street-level footage where a standing body is h/w ~2.5;
    a ceiling camera sees head-and-shoulders at h/w ~1.14, so the model
    stretches its learned shape onto a foreshortened person.

        pipeline box   194 x 542 px   h/w 2.80
        hand-labelled  290 x 330 px   h/w 1.14

    So `y2` sits a median 260 px BELOW the real feet, and every consumer of
    this point has been asking about a spot roughly half a body underneath the
    person: zone membership, entry-line crossings, the ground-plane fit, and
    the "expected height at this footline" model that D0 and D1 both trust.
    Several defects chased separately this week were all this one bug.

    Calibrated position of the true foot inside the predicted box (n=442):
        p25 0.501    MEDIAN 0.590    p75 0.951
    Horizontally the centre is already right (median 0.474 of the width), so
    only y moves.

    FOOT_ANCHOR_FRAC is that fraction. 1.0 restores the old behaviour exactly
    and is the shipping default until an A/B against ground truth says
    otherwise — the spread above is wide, so 0.59 is a better estimate, not a
    perfect one.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    if centre:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    try:
        from . import config as _cfg
        frac = float(getattr(_cfg, "FOOT_ANCHOR_FRAC", 1.0) or 1.0)
    except Exception:
        frac = 1.0
    if frac >= 1.0:
        return ((x1 + x2) / 2.0, y2)
    return ((x1 + x2) / 2.0, y1 + (y2 - y1) * frac)


def classify_zones(zone_names):
    roles = {}
    for name in zone_names:
        if name in ZONE_AI_OVERRIDES:
            roles[name] = ZONE_AI_OVERRIDES[name]
            continue
        low = str(name).lower()
        # keep in sync with Cell 2's classify_zones — this later definition
        # overwrites it, and it was missing archway/portal
        is_entry_indicator = any(kw in low for kw in ["gate", "door", "entry", "entrance", "passageway", "archway", "portal"])
        matched = []
        for role, kws in ZONE_ROLE_KEYWORDS.items():
            if role == "seating" and is_entry_indicator:
                if not any(explicit in low for explicit in ["table", "booth", "chair", "seat"]):
                    continue
            if any(kw in low for kw in kws):
                matched.append(role)
        roles[name] = matched or ["other"]
    return roles

def mmss(s):
    s = max(0, int(s))
    return f"{s // 60:02d}:{s % 60:02d}"

def wall(t):
    """Video-time -> wall-clock string using the burned-in start time."""
    if not VIDEO_START_CLOCK:
        return ""
    base = datetime.strptime(VIDEO_START_CLOCK, "%H:%M:%S")
    return (base + timedelta(seconds=float(t))).strftime("%H:%M:%S")

def zone_color_map(zone_names):
    """Fixed sorted order -> fixed colors, shared by charts AND video overlay."""
    return {z: ZONE_HEXES[i % len(ZONE_HEXES)] for i, z in enumerate(sorted(zone_names))}

def show_gallery(snaps, title, ncols=3, max_items=9):
    items = snaps[:max_items]
    if not items:
        print(f"({title}: no frames captured)")
        return
    rows = math.ceil(len(items) / ncols)
    fig, axes = plt.subplots(rows, ncols, figsize=(5.2 * ncols, 3.1 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (t, img) in zip(axes, items):
        ax.imshow(img)
        ax.set_title(f"t = {mmss(t)}", fontsize=10, color=INK2)
    fig.suptitle(title, fontsize=13, color=INK)
    plt.tight_layout()
    plt.show()

def plot_reid_pair_audit(pairs, track_crops, title, max_pairs=6):
    """(v36) The actual crops behind a calibration number, side by side --
    turns "why don't the numbers separate" into something you can just
    look at, instead of another round of threshold guessing.

    pairs: [(a, b, sim), ...] -- e.g. calibration_report["same_pairs_worst"]
           or ["diff_pairs_worst"], already worst-first.
    track_crops: {track_id: [(_, crop_bgr), ...]} -- same structure used for
                 face embedding; picks the LARGEST banked crop per track as
                 the representative image (most informative for a human to
                 judge), and prints its pixel size so you can see directly
                 if crops are simply too small/blurry for even a human to
                 tell two people apart -- if you can't tell either, the
                 model failing isn't a threshold problem, it's a resolution/
                 occlusion/uniform-clothing problem no threshold fixes.
    """
    _require_plt()
    _require_plt()
    items = pairs[:max_pairs]
    if not items:
        print(f"({title}: no pairs to show)")
        return
    fig, axes = plt.subplots(len(items), 2, figsize=(6.4, 2.6 * len(items)))
    axes = np.atleast_2d(axes)
    for row, (a, b, sim) in enumerate(items):
        for col, tid in enumerate((a, b)):
            ax = axes[row][col]
            ax.axis("off")
            crops = track_crops.get(tid, [])
            if crops:
                _, crop = max(crops, key=lambda c: c[1].shape[0] * c[1].shape[1])
                ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                ax.set_title(f"ID {tid}  ({crop.shape[1]}x{crop.shape[0]}px)",
                            fontsize=9)
            else:
                ax.set_title(f"ID {tid} (no banked crop)", fontsize=9)
        axes[row][0].set_ylabel(f"sim={sim:.3f}", fontsize=10, rotation=0,
                                labelpad=32, ha="right", va="center")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
