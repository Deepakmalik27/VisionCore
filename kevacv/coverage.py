"""coverage.py — what is the model ACTUALLY seeing, across the whole frame?

WHY THIS EXISTS
    Every diagnostic this pipeline has answers "how many". The funnel counts
    detections per stage. The scorecard counts events. None of them can answer
    the question a reviewer actually asks first:

        "Is there a part of this frame where we never see anybody,
         and is that because nobody goes there -- or because we are blind?"

    Those two have identical signatures in every number we currently print:
    zero. validity.py already states the principle -- "I did not observe" is
    not "I observed nothing" -- and applies it to TIME. This applies it to
    SPACE, which is the half nobody built.

THE THREE-WAY DIAGNOSIS
    A cell with zero detections means one of three completely different things,
    and only the zone map and the dead-area masks can tell them apart:

        zero + inside a zone we count      -> WE ARE MISSING PEOPLE   (bug)
        zero + under a dead-area mask      -> correct, by design       (fine)
        zero + neither                     -> wall, ceiling, sky       (ignore)

    Reporting "12% of the frame is blind" without that split is a rumour. So
    this module refuses to emit a blind-spot finding unless it was given the
    polygons to classify it with.

WHY A GRID AND NOT A HEATMAP IMAGE
    The grid IS the data; the image is one rendering of it. Keeping the counts
    as numbers means the scorecard can diff two runs mechanically -- "the far
    corner went from 400 detections to 3" is a regression you can catch in CI,
    where a picture is something a human has to remember to look at.

KARPATHY'S STEP 1, MADE AUTOMATIC
    "Become one with the data... write simple filtering/sorting code to
    visualize distributions and outliers -- this almost always uncovers bugs in
    data quality or preprocessing." That habit is manual and nobody sustains
    it. This runs it every time, unprompted.
"""
from __future__ import annotations

import math

# A person at the far end of CAM.112 is ~60 px tall in a 1920-wide analysed
# frame. 32x18 cells over 1920x1080 makes each cell 60x60 -- about one person.
# Finer than that and every cell is noise; coarser and a whole doorway hides
# inside one bucket.
DEFAULT_COLS, DEFAULT_ROWS = 32, 18


class CoverageGrid:
    """Per-cell accounting of what the detector saw, and what was taken away.

    Pure: no cv2, no file, no model. It takes numbers and returns numbers, so
    it is testable without a video -- the same reason observe.py is shaped this
    way.
    """

    def __init__(self, frame_w, frame_h, cols=DEFAULT_COLS, rows=DEFAULT_ROWS):
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError(f"frame size must be positive, got {frame_w}x{frame_h}")
        self.w, self.h = int(frame_w), int(frame_h)
        self.cols, self.rows = int(cols), int(rows)
        n = self.cols * self.rows
        self.seen = [0] * n          # detections that SURVIVED the filter chain
        self.conf_sum = [0.0] * n    # to report a mean confidence per cell
        self.killed = [0] * n        # detections the chain removed here
        self.height_sum = [0.0] * n  # observed person height, for the scale field
        self.height_n = [0] * n
        self.frames = 0

    # ── geometry ────────────────────────────────────────────────────────────
    def cell_of(self, x, y):
        """Pixel -> cell index, or None if the point is outside the frame.

        Out-of-frame is NOT clamped to the edge cell. A box whose anchor lands
        at x = -30 is a bug upstream, and folding it into column 0 would hide
        that bug inside a legitimate-looking count.
        """
        if not (0 <= x < self.w and 0 <= y < self.h):
            return None
        c = min(self.cols - 1, int(x * self.cols / self.w))
        r = min(self.rows - 1, int(y * self.rows / self.h))
        return r * self.cols + c

    def cell_box(self, idx):
        """Cell index -> (x1, y1, x2, y2) in frame pixels."""
        r, c = divmod(idx, self.cols)
        return (int(c * self.w / self.cols), int(r * self.h / self.rows),
                int((c + 1) * self.w / self.cols), int((r + 1) * self.h / self.rows))

    # ── accumulation ────────────────────────────────────────────────────────
    def add_frame(self):
        self.frames += 1

    def add_detection(self, anchor_xy, conf=None, height_px=None):
        """One SURVIVING detection, at the anchor that decides its zone.

        Deliberately keyed on the ANCHOR, not the box centre: the anchor is
        what the counter uses, so the coverage map answers "where do we count
        people", not "where do boxes float".
        """
        i = self.cell_of(*anchor_xy)
        if i is None:
            return None
        self.seen[i] += 1
        if conf is not None:
            self.conf_sum[i] += float(conf)
        if height_px:
            self.height_sum[i] += float(height_px)
            self.height_n[i] += 1
        return i

    def add_kill(self, anchor_xy):
        """One detection the filter chain removed, at the same anchor."""
        i = self.cell_of(*anchor_xy)
        if i is None:
            return None
        self.killed[i] += 1
        return i

    # ── read-out ────────────────────────────────────────────────────────────
    def mean_conf(self, idx):
        return (self.conf_sum[idx] / self.seen[idx]) if self.seen[idx] else None

    def mean_height(self, idx):
        return (self.height_sum[idx] / self.height_n[idx]) if self.height_n[idx] else None

    def kill_ratio(self, idx):
        """Killed / (killed + seen). None where nothing at all was proposed."""
        tot = self.killed[idx] + self.seen[idx]
        return (self.killed[idx] / tot) if tot else None

    def blind_cells(self, zone_cells=None, mask_cells=None):
        """Cells with ZERO surviving detections, split three ways.

        -> {"in_zone": [...], "masked": [...], "elsewhere": [...]}
        `zone_cells` / `mask_cells` are sets of cell indices. Pass None for
        zone_cells and the in_zone bucket stays empty -- this refuses to guess
        which blind spots matter without being told where we count.
        """
        zone_cells = set(zone_cells or ())
        mask_cells = set(mask_cells or ())
        out = {"in_zone": [], "masked": [], "elsewhere": []}
        for i in range(self.cols * self.rows):
            if self.seen[i]:
                continue
            if i in mask_cells:
                out["masked"].append(i)
            elif i in zone_cells:
                out["in_zone"].append(i)
            else:
                out["elsewhere"].append(i)
        return out

    def cells_for_polygon(self, poly):
        """Which cells a polygon touches. Cheap: tests each cell's centre.

        A cell whose CENTRE is outside but whose corner clips the polygon is
        not counted. That is the conservative direction -- it under-reports the
        zone's footprint, so a blind-spot warning is never raised for a cell the
        zone barely grazes.
        """
        out = set()
        for i in range(self.cols * self.rows):
            x1, y1, x2, y2 = self.cell_box(i)
            if _point_in_poly(poly, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)):
                out.add(i)
        return out

    def findings(self, zone_cells=None, mask_cells=None, min_frames=100):
        """-> [(level, message)], in the shape preflight/capabilities use."""
        out = []
        if self.frames < min_frames:
            return [("INFO", f"coverage map: only {self.frames} frame(s) "
                             f"accumulated — too few to judge blind spots")]
        total = self.cols * self.rows
        blind = self.blind_cells(zone_cells, mask_cells)
        n_seen_cells = sum(1 for i in range(total) if self.seen[i])
        out.append(("INFO",
                    f"coverage: the detector produced a surviving box in "
                    f"{n_seen_cells}/{total} cells "
                    f"({100.0 * n_seen_cells / total:.0f}% of the frame) "
                    f"over {self.frames} frames"))
        if blind["in_zone"]:
            out.append(("ERROR",
                        f"{len(blind['in_zone'])} cell(s) INSIDE A COUNTED ZONE "
                        f"never produced a single detection. Either nobody ever "
                        f"stood there, or we are blind there — and those two "
                        f"are worth different money. Check the zone overlay "
                        f"against the coverage map before trusting the count."))
        # A cell where almost everything proposed is destroyed is either a
        # correct dead area or a filter eating real people. Worth a look either
        # way, and invisible in the aggregate funnel percentage.
        hot = [i for i in range(total)
               if (self.kill_ratio(i) or 0) > 0.9 and self.killed[i] >= 20]
        if hot:
            out.append(("WARN",
                        f"{len(hot)} cell(s) had >90% of their detections "
                        f"removed by the filter chain (>=20 kills each). If "
                        f"they are not under a dead-area mask, a filter is "
                        f"eating real people there."))
        return out

    def as_dict(self):
        """The grid as plain data, so two runs can be diffed mechanically."""
        return {
            "frame_size": [self.w, self.h],
            "grid": [self.cols, self.rows],
            "frames": self.frames,
            "seen": list(self.seen),
            "killed": list(self.killed),
            "conf_sum": [round(v, 3) for v in self.conf_sum],
            "height_sum": [round(v, 1) for v in self.height_sum],
            "height_n": list(self.height_n),
        }


def _point_in_poly(poly, pt):
    """Ray casting. Same convention as engine._in_poly, kept local so this
    module stays importable without cv2."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def render(grid, frame_bgr, alpha=0.45):
    """Paint the coverage map over a reference frame. Needs cv2; import-local.

    Sequential single hue, light -> dark with density, because this encodes
    MAGNITUDE. Never a rainbow: a rainbow implies categories, and there are
    none here.
    """
    import cv2
    import numpy as np

    out = frame_bgr.copy()
    overlay = out.copy()
    peak = max(grid.seen) or 1
    for i in range(grid.cols * grid.rows):
        x1, y1, x2, y2 = grid.cell_box(i)
        n = grid.seen[i]
        if n:
            # one hue (blue), light -> dark with log density so a single busy
            # doorway does not flatten the whole rest of the frame to nothing
            f = math.log1p(n) / math.log1p(peak)
            col = (int(245 - 145 * f), int(190 - 120 * f), int(90 - 40 * f))
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), col, -1)
        else:
            # blind: hatched, never a solid colour. A solid fill would read as
            # a measurement; this is an absence.
            for d in range(0, (x2 - x1) + (y2 - y1), 10):
                cv2.line(overlay, (x1 + d, y1), (x1, y1 + d), (70, 70, 75), 1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    for i in range(grid.cols * grid.rows):
        x1, y1, x2, y2 = grid.cell_box(i)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (60, 60, 60), 1)
    return out


def demo():
    """One runnable check: the three-way blind diagnosis actually separates."""
    g = CoverageGrid(1920, 1080, cols=8, rows=4)   # 240x270 cells
    assert g.cell_of(0, 0) == 0
    assert g.cell_of(1919, 1079) == 8 * 4 - 1
    # out of frame is None, NOT clamped — clamping would hide an upstream bug
    assert g.cell_of(-5, 100) is None and g.cell_of(100, 5000) is None
    assert g.cell_of(1920, 100) is None

    for _ in range(500):
        g.add_frame()
    # a busy doorway in one cell
    door = g.cell_of(300, 900)
    for _ in range(120):
        g.add_detection((300, 900), conf=0.8, height_px=180)
    assert g.seen[door] == 120
    assert abs(g.mean_conf(door) - 0.8) < 1e-9
    assert abs(g.mean_height(door) - 180) < 1e-9
    assert g.mean_conf(0) is None, "a cell with no detections has no mean"

    # a cell where the filter chain destroys everything
    hot = g.cell_of(1700, 200)
    for _ in range(50):
        g.add_kill((1700, 200))
    assert g.kill_ratio(hot) == 1.0
    assert g.kill_ratio(door) == 0.0
    assert g.kill_ratio(g.cell_of(900, 500)) is None, "nothing proposed -> None"

    # THE THREE-WAY SPLIT. A zone covering the left half, a mask on the right.
    zone = g.cells_for_polygon([[0, 0], [900, 0], [900, 1079], [0, 1079]])
    mask = g.cells_for_polygon([[1500, 0], [1919, 0], [1919, 1079], [1500, 1079]])
    assert door in zone, "the doorway must fall inside the zone we drew"
    assert hot in mask, "the kill-heavy cell must fall under the mask"
    b = g.blind_cells(zone_cells=zone, mask_cells=mask)
    assert door not in b["in_zone"], "a cell WITH detections is not blind"
    assert hot in b["masked"], "a blind cell under a mask is 'masked', not a bug"
    assert all(i not in zone for i in b["elsewhere"])
    # every blind cell lands in exactly one bucket, and buckets never overlap
    allb = b["in_zone"] + b["masked"] + b["elsewhere"]
    assert len(allb) == len(set(allb)), "a cell was reported in two buckets"
    assert len(allb) == sum(1 for i in range(32) if g.seen[i] == 0)

    f = g.findings(zone_cells=zone, mask_cells=mask)
    lv = {l for l, _ in f}
    assert "ERROR" in lv, "blind cells inside a counted zone must be an ERROR"
    assert any("removed by the filter chain" in m for _, m in f), \
        "a >90% kill cell must be reported"

    # WITHOUT the zone map it must refuse to raise the blind-spot error --
    # "12% of the frame is blind" with no idea which 12% is a rumour.
    f2 = g.findings()
    assert not any(l == "ERROR" for l, _ in f2), \
        "no zone map -> no blind-spot ERROR, because it cannot be classified"

    # too few frames -> says so instead of judging
    small = CoverageGrid(100, 100, 2, 2)
    small.add_frame()
    assert "too few" in small.findings()[0][1]

    d = g.as_dict()
    assert d["grid"] == [8, 4] and len(d["seen"]) == 32 and d["frames"] == 500

    try:
        CoverageGrid(0, 100)
        raise AssertionError("a zero-width frame must raise, not divide by zero")
    except ValueError:
        pass
    print(f"coverage: OK  ({g.cols}x{g.rows} grid, "
          f"{len(b['in_zone'])} blind-in-zone, {len(b['masked'])} masked, "
          f"{len(b['elsewhere'])} elsewhere)")


if __name__ == "__main__":
    demo()
