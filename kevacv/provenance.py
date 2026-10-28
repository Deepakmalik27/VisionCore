"""What produced this number. Stamped onto the events, not just the log.

WHY
---
follow_up.txt puts it plainly: without version stamping,

    "Why did yesterday produce 2,104 entries and today's pipeline produce
     2,083?"

is unanswerable. This project has lived that question all day. Three examples
from a single session, every one of which would have been a five-second lookup
with a stamp on the events:

  * A funnel came back byte-identical after a detector change, because 3 of 5
    inference call sites had been patched. The run "looked the same" and there
    was nothing on the output saying which code produced it.
  * A GMC A/B reported "identical to baseline" because a duplicated yaml key
    meant the flag never applied. The events carried no record of the value
    that actually ran -- only the file said `false`, and the file was wrong.
  * A held-out score of 1/3 turned out to be measured against ground truth
    from a DIFFERENT HOUR of footage.

`tools/stamp_build.py` already computes a build id. Nothing attached it to an
event. So the id was in the log, the log was on a box, and the JSON someone
actually reads had no idea what made it.

DESIGN
------
Small and flat. A stamp is written ONCE per run and referenced by the events,
rather than copied onto every row -- a crossing is a handful of floats and a
per-row copy of the whole build would dwarf it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def build_stamp(*, build_id: Optional[str] = None,
                config_path: Optional[str] = None,
                zones_path: Optional[str] = None,
                video: Optional[str] = None,
                detector: Optional[str] = None,
                tracker: Optional[str] = None,
                reid_weights: Optional[str] = None,
                changed: Optional[Dict[str, Any]] = None,
                ground_plane: Optional[Dict[str, Any]] = None,
                fps: Optional[float] = None,
                analysis_w: Optional[float] = None,
                imgsz: Optional[float] = None) -> Dict[str, Any]:
    """One record identifying everything that could change a count."""
    stamp = {
        "build_id": build_id or "?",
        "config": config_path or "?",
        "zones": zones_path or "?",
        "video": video or "?",
        "detector": detector or "?",
        "tracker": tracker or "?",
        "reid_weights": reid_weights or "?",
        "fps": fps,
        "analysis_w": analysis_w,
        "imgsz": imgsz,
        # The plane is a VERSIONED INPUT, not a detail: it is the ruler for
        # every metre-based gate, and it is refitted per run. Two runs of
        # identical code on identical video can disagree because the plane
        # landed differently, and that must be visible on the output.
        "calibration": {
            "camera_h_m": (ground_plane or {}).get("camera_h_m"),
            "horizon_row": (ground_plane or {}).get("horizon_row"),
            "mode": (ground_plane or {}).get("mode"),
        },
        # Every knob that differs from the shipped default. This is the field
        # that would have caught the duplicated-key GMC failure: the file said
        # false, the RUN said true, and only this records what the run did.
        "changed": dict(changed or {}),
    }
    stamp["fingerprint"] = fingerprint(stamp)
    return stamp


def fingerprint(stamp: Dict[str, Any]) -> str:
    """Short, stable hash of everything that could change a count.

    Two runs with the same fingerprint should produce the same numbers. If they
    do not, the difference is nondeterminism and that is itself a finding --
    which is exactly how the determinism check was framed.
    """
    keys = ("detector", "tracker", "reid_weights", "fps", "analysis_w",
            "imgsz", "zones", "video", "changed")
    blob = json.dumps({k: stamp.get(k) for k in keys},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """What differs between two stamps — the answer to 'why did it change?'"""
    out: Dict[str, Any] = {}
    for k in ("build_id", "config", "zones", "video", "detector", "tracker",
              "reid_weights", "fps", "analysis_w", "imgsz"):
        if a.get(k) != b.get(k):
            out[k] = [a.get(k), b.get(k)]
    ca, cb = (a.get("calibration") or {}), (b.get("calibration") or {})
    for k in set(ca) | set(cb):
        if ca.get(k) != cb.get(k):
            out[f"calibration.{k}"] = [ca.get(k), cb.get(k)]
    ka, kb = (a.get("changed") or {}), (b.get("changed") or {})
    for k in sorted(set(ka) | set(kb)):
        if ka.get(k) != kb.get(k):
            out[f"changed.{k}"] = [ka.get(k), kb.get(k)]
    return out
