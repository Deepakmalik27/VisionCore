"""funnel.py — where detections are born, and where each one dies.

WHY THIS EXISTS
    The pipeline drops detections in eight places between YOLO and a number on
    a report: the head/person split, carried-object suppression, the
    implausible-size filter, dead-area masks, dedup NMS, the tracker's own
    birth rules, the static/phantom sweep, and the minimum-event threshold.

    Every one of them is a filter somebody added for a good reason, and every
    one of them can silently eat real people. When the output is "we counted 12
    guests" and the truth is 40, the only useful question is WHICH STAGE lost
    them — and nothing recorded that. The run log printed a couple of totals
    from two of the eight stages and nothing from the rest.

    This is the missing ledger. One line per stage: how many detections went
    in, how many came out, and what share of the original that stage removed.

HOW TO READ IT
    The field's diagnostic split is HOTA = sqrt(DetA x AssA) — DetA asks "did
    we FIND the people", AssA asks "did we KEEP them the same person"
    (kevacv/eval_harness.py implements both). That split needs labelled ground
    truth. This funnel needs none, and answers the DetA half's first question
    on any run: of everything the detector saw, what survived to be counted.

    A stage removing ~0% is doing nothing and should be questioned. A stage
    removing 30%+ is either the most valuable filter here or the reason the
    counts are wrong, and the only way to tell is to look at what it dropped.

WHAT IT DELIBERATELY IS NOT
    Not a profiler (kevacv/log.py stages own timing) and not accuracy
    (eval_harness owns that, against ground truth). It is a conservation
    check: detections in, detections out, per stage, in order.
"""
from __future__ import annotations

from collections import OrderedDict


class DetectionFunnel:
    """Per-stage detection accounting for one video chunk.

    Stages are keyed by name and reported in FIRST-SEEN order, which is the
    order they run in — so the table reads top to bottom as the frame does.
    """

    def __init__(self, label=""):
        self.label = label
        self.stages = OrderedDict()
        self._raw_total = 0

    def record(self, stage, n_in, n_out):
        """One stage, one frame. Returns n_out so call sites can chain."""
        s = self.stages.get(stage)
        if s is None:
            s = self.stages[stage] = {"in": 0, "out": 0, "frames": 0,
                                      "frames_emptied": 0}
        s["in"] += int(n_in)
        s["out"] += int(n_out)
        s["frames"] += 1
        # A stage that takes a frame from "someone is here" to "nobody is
        # here" is how a person blinks out of the annotated video and comes
        # back with a new id. Counted separately because the aggregate share
        # hides it: one detection lost from a frame of one is 100% of that
        # frame and a rounding error in the total.
        if n_in > 0 and n_out == 0:
            s["frames_emptied"] += 1
        return n_out

    def record_first(self, stage, n):
        """The detector's raw output — the denominator everything else is a
        share of."""
        self._raw_total += int(n)
        return self.record(stage, n, n)

    @property
    def raw_total(self):
        return self._raw_total

    def dropped(self, stage):
        s = self.stages.get(stage)
        return 0 if s is None else s["in"] - s["out"]

    def as_dict(self):
        out = {"label": self.label, "raw_total": self._raw_total, "stages": []}
        for name, s in self.stages.items():
            drop = s["in"] - s["out"]
            out["stages"].append({
                "stage": name,
                "in": s["in"],
                "out": s["out"],
                "dropped": drop,
                "share_of_raw": (drop / self._raw_total) if self._raw_total else 0.0,
                "frames": s["frames"],
                "frames_emptied": s["frames_emptied"],
            })
        return out

    def describe(self, width=78):
        """The table, as a human reads it in the run log."""
        d = self.as_dict()
        title = f"  DETECTION FUNNEL{' · ' + self.label if self.label else ''}"
        L = ["=" * width, title, "=" * width]
        if not d["stages"]:
            L.append("  (no detections recorded — the frame loop never ran)")
            L.append("=" * width)
            return "\n".join(L)
        L.append(f"  {'stage':<26}{'in':>9}{'out':>9}{'dropped':>9}"
                 f"{'%raw':>8}{'emptied':>9}")
        L.append("  " + "-" * (width - 4))
        for s in d["stages"]:
            L.append(f"  {s['stage']:<26}{s['in']:>9}{s['out']:>9}"
                     f"{s['dropped']:>9}{100 * s['share_of_raw']:>7.1f}%"
                     f"{s['frames_emptied']:>9}")
        L.append("  " + "-" * (width - 4))
        first, last = d["stages"][0], d["stages"][-1]
        kept = (last["out"] / first["in"]) if first["in"] else 0.0
        L.append(f"  {'SURVIVED':<26}{first['in']:>9}{last['out']:>9}"
                 f"{first['in'] - last['out']:>9}{100 * (1 - kept):>7.1f}%")
        L.append("")
        L.append("  'emptied' = frames this stage took from >=1 detection to 0.")
        L.append("  Those are the blink-outs: a person vanishes for a frame and")
        L.append("  the tracker gives them a new id when they come back.")
        L.append("=" * width)
        return "\n".join(L)

    def findings(self, dead_stage_share=0.001, greedy_stage_share=0.25):
        """The lines worth raising to ERROR/WARN, as (level, message).

        Thresholds are arguments, not literals, because the right value is a
        property of the venue: a busy doorway and an empty corridor at 03:00
        do not share a definition of 'greedy'.
        """
        out = []
        d = self.as_dict()
        for s in d["stages"][1:]:      # [0] is the raw detector, drops nothing
            share = s["share_of_raw"]
            if share >= greedy_stage_share:
                out.append(("WARN",
                            f"{s['stage']} removed {100 * share:.0f}% of all "
                            f"detections ({s['dropped']}). Confirm it is "
                            f"dropping phantoms and not people before trusting "
                            f"any count downstream of it."))
            elif s["dropped"] == 0:
                out.append(("INFO",
                            f"{s['stage']} removed nothing all chunk — it is "
                            f"either unnecessary here or misconfigured."))
            elif share < dead_stage_share:
                out.append(("INFO",
                            f"{s['stage']} removed {s['dropped']} detection(s), "
                            f"under {100 * dead_stage_share:.1f}% of the total."))
            if s["frames_emptied"]:
                out.append(("INFO",
                            f"{s['stage']} emptied {s['frames_emptied']} frame(s) "
                            f"that had someone in them — each is a candidate "
                            f"id break."))
        return out
