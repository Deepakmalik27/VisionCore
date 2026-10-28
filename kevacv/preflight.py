"""preflight.py — P1 Pre-Flight Validation Engine (Fail-Loud Architecture).

WHY THIS EXISTS
    In actual surveillance deployments (e.g. CAM.112), silent geometry and configuration
    failures occur (e.g. short entry line, missing seating polygon, invalid ground plane).
    When these occur, downstream analytics yield zero crossings, missing dwell times, and 
    silently absent metrics without raising errors.

THE PRINCIPLE
    Fail Loud & Pre-Flight Gatekeeper.
    Before spending compute on video processing:
      1. Validate zone schema completeness and polygon geometry.
      2. Test entry line geometry using synthetic trajectory crossing tests across the doorway.
      3. Verify ground plane calibration parameters for realistic camera height (1.5m - 7.0m) and horizon location.
      4. Raise a structured PreflightValidationError with exact remediation steps if any critical check fails.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


class PreflightValidationError(Exception):
    """Raised when pre-flight validation fails on critical schema, line, or geometry rules."""
    def __init__(self, message: str, errors: List[str], warnings: List[str]):
        super().__init__(message)
        self.errors = errors
        self.warnings = warnings

    def __str__(self) -> str:
        lines = [f"\n{'='*70}", "PRE-FLIGHT VALIDATION FAILED — CRITICAL CONFIGURATION ERRORS", f"{'='*70}"]
        for err in self.errors:
            lines.append(f"  [ERROR] {err}")
        if self.warnings:
            lines.append(f"\n  Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  [WARN]  {warn}")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)


def _line_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                    q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
    """Check if line segment p1-p2 intersects line segment q1-q2."""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


def validate_polygons(zones_cfg: Dict[str, Any], frame_size: Tuple[int, int]) -> Tuple[List[str], List[str]]:
    """Validate polygon definitions in zones configuration."""
    errors, warnings = [], []
    fw, fh = frame_size

    polygons = zones_cfg.get("polygons", {})
    if not polygons:
        errors.append("No polygons found in zones configuration ('polygons' dictionary is missing or empty).")
        return errors, warnings

    for name, poly in polygons.items():
        if not isinstance(poly, list) or len(poly) < 3:
            errors.append(f"Zone '{name}' must be a polygon with at least 3 vertices (got {len(poly) if isinstance(poly, list) else 0}).")
            continue
        for idx, pt in enumerate(poly):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                errors.append(f"Zone '{name}' vertex #{idx} is invalid: {pt!r}. Expected [x, y].")
                continue
            try:
                x, y = pt
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    errors.append(f"Zone '{name}' vertex #{idx} has non-numeric coordinates: ({x!r}, {y!r}).")
                    continue
                if not (0 <= x <= fw * 1.2 and 0 <= y <= fh * 1.2):
                    warnings.append(f"Zone '{name}' vertex #{idx} ({x}, {y}) is outside frame bounds ({fw}x{fh}).")
            except (TypeError, ValueError) as e:
                errors.append(f"Zone '{name}' vertex #{idx} could not be unpacked: {pt!r} ({e}).")

    # Check essential semantic roles
    roles = zones_cfg.get("roles", {})
    has_entry = False
    for zname, rlist in roles.items():
        if "entry" in rlist or "entrance" in zname.lower() or "main_entrance" in zname.lower():
            has_entry = True
            break
    
    if not has_entry and "main_entrance" not in polygons and "entrance" not in polygons:
        warnings.append("No entrance/entry zone found in polygons or roles. Region-based arrival tracking will be disabled.")

    return errors, warnings


def _point_in_poly(pt, poly):
    """Ray casting; poly is [(x, y), ...]."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1
            if x < xint:
                inside = not inside
    return inside


def _entry_lines_of(zones_cfg):
    """Every entry line in a zone config, whatever shape it is written in.

    #22 (2026-08-20). This validator read a TOP-LEVEL "entry_line" key:

        entry_line = zones_cfg.get("entry_line")
        if not entry_line:
            warnings.append("No 'entry_line' specified ...")
            return errors, warnings        # <- every real run took this path

    Real zone files -- including zones/CAM.112_zone.json, the only one in
    production -- store a PLURAL DICT instead:

        "entry_lines": {"dining entry": [...], "staff entry": [...],
                        "entry line": [...]}

    Note the venue door is spelled "entry line" WITH A SPACE, so even a naive
    singular lookup would have missed it. The consequence: validate_entry_line
    has never executed a single check on this camera. The short-line check, the
    walk-around-the-ends check and the mask-over-line check all returned early
    behind a warning that read like "you have not configured a line yet"
    rather than "this validator cannot read your file".

    That is why the run that produced 'entry line IN 0 | OUT 0' for 300s
    reported no preflight error: not because errors are non-fatal, but because
    the check never ran. Accept both shapes.
    """
    out = []
    single = zones_cfg.get("entry_line")
    if single:
        out.append(("entry_line", single))
    for name, line in (zones_cfg.get("entry_lines") or {}).items():
        out.append((str(name), line))
    return out


def validate_entry_line(zones_cfg: Dict[str, Any], frame_size: Tuple[int, int]) -> Tuple[List[str], List[str]]:
    """Validate every entry line: shape, length, open ends, mask collision."""
    errors, warnings = [], []
    lines = _entry_lines_of(zones_cfg)
    if not lines:
        warnings.append("No entry line specified in zones config "
                        "(neither 'entry_line' nor 'entry_lines'). "
                        "Line-crossing count will be unavailable.")
        return errors, warnings
    for name, line in lines:
        _e, _w = _validate_one_entry_line(name, line, zones_cfg, frame_size)
        errors.extend(_e)
        warnings.extend(_w)
    return errors, warnings


def _validate_one_entry_line(name, entry_line, zones_cfg, frame_size):
    errors, warnings = [], []
    fw, fh = frame_size
    tag = f"Entry line {name!r}"

    if not isinstance(entry_line, list) or len(entry_line) != 2:
        errors.append(f"Invalid {tag} format: {entry_line!r}. "
                      f"Expected [[x1, y1], [x2, y2]].")
        return errors, warnings

    p1, p2 = entry_line[0], entry_line[1]
    if not isinstance(p1, (list, tuple)) or len(p1) != 2:
        errors.append(f"{tag}[0] must be [x, y], got: {p1!r}")
        return errors, warnings
    if not isinstance(p2, (list, tuple)) or len(p2) != 2:
        errors.append(f"{tag}[1] must be [x, y], got: {p2!r}")
        return errors, warnings
    try:
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        line_len = math.hypot(dx, dy)
    except (TypeError, ValueError) as e:
        errors.append(f"{tag} vertices are malformed: {entry_line!r} ({e}).")
        return errors, warnings

    # #20: 20 ABSOLUTE pixels is nothing at 4K -- 0.5% of the width. Express
    # the bar as a FRACTION of the frame, this package's convention elsewhere.
    _min_frac = 0.08
    if line_len < max(20.0, _min_frac * math.hypot(fw, fh)):
        errors.append(
            f"{tag} is too short ({line_len:.0f} px = "
            f"{line_len / math.hypot(fw, fh):.1%} of the frame diagonal). "
            f"A short line is the documented CAM.112 failure: people walk "
            f"around its ends and the counter reads 0 while the doorway is "
            f"busy. It must span the physical threshold, wall to wall.")
        return errors, warnings

    # #20: the old test built a perpendicular through the line's OWN midpoint
    # and asked whether it intersects the line. It always does -- true by
    # construction. A test that cannot fail is worse than no test. What broke
    # on CAM.112 was people walking AROUND the ends: only 2 of 67 tracks
    # crossed the drawn segment while 9 changed side of the infinite line.
    _ends_free = []
    for _px, _py, _which in ((float(p1[0]), float(p1[1]), "start"),
                             (float(p2[0]), float(p2[1]), "end")):
        _margin = min(_px, _py, fw - _px, fh - _py)
        _ends_free.append((_which, _margin))
    _gap = max(m for _w, m in _ends_free)
    _open_frac = _gap / max(1.0, min(fw, fh))
    if _open_frac > 0.12:
        _w = max(_ends_free, key=lambda t: t[1])[0]
        warnings.append(
            f"{tag}'s {_w} lies {_gap:.0f}px ({_open_frac:.0%} of the "
            f"frame) from the nearest edge, so there is open floor to walk "
            f"around it. Anchor both ends on a wall, a frame edge, or a "
            f"physical obstruction -- this is the exact failure that produced "
            f"'entry line IN 0 | OUT 0' while people streamed through the door.")

    # A mask over the line is fatal ONLY while the mask DELETES.
    #
    # MASK_REQUIRE_MOTION (engine._drop_masked) changes what a mask zone means:
    #   0.0  -> "delete every detection whose feet land here"  -> FATAL, nobody
    #           can ever be seen crossing. Measured on CAM.112: 93% of
    #           'entry line' lies inside 'plant area mask', and it read
    #           IN 0 | OUT 0 for a whole run.
    #   >0   -> "a known static distractor lives here, require motion" -> a
    #           walking guest survives, so the overlap is undesirable but NOT
    #           fatal. Downgrade to a warning, otherwise --strict refuses to
    #           run the very fix that makes the overlap survivable.
    try:
        from . import config as _C
        _mask_motion = float(getattr(_C, "MASK_REQUIRE_MOTION", 0.0) or 0.0)
    except Exception:
        _mask_motion = 0.0

    for _zname, _poly in (zones_cfg.get("polygons") or {}).items():
        if "mask" not in str(_zname).lower():
            continue
        try:
            _pts = [(float(a), float(b)) for a, b in _poly]
        except (TypeError, ValueError):
            continue
        _mid = ((float(p1[0]) + float(p2[0])) / 2.0,
                (float(p1[1]) + float(p2[1])) / 2.0)
        _hits = [_point_in_poly(_q, _pts)
                 for _q in ((float(p1[0]), float(p1[1])), _mid,
                            (float(p2[0]), float(p2[1])))]
        if not any(_hits):
            continue
        if _mask_motion > 0:
            warnings.append(
                f"{tag} passes through mask zone {_zname!r} "
                f"(start/mid/end inside: {_hits}). Survivable only because "
                f"MASK_REQUIRE_MOTION={_mask_motion:g} makes that zone require "
                f"motion rather than delete. A guest standing still at the "
                f"door can still be suppressed -- move the line or shrink the "
                f"mask when you can.")
        else:
            errors.append(
                f"{tag} passes through mask zone {_zname!r} "
                f"(start/mid/end inside: {_hits}). Masked detections are "
                f"DROPPED (MASK_REQUIRE_MOTION=0), so nobody can be seen "
                f"crossing there and the line will read 0 however busy the "
                f"door is. Set analysis.mask_require_motion > 0 or redraw.")

    return errors, warnings


def validate_ground_plane(zones_cfg: Dict[str, Any], ground_plane_obj: Any,
                          strict: bool = False) -> Tuple[List[str], List[str]]:
    """Validate camera ground plane calibration quality."""
    errors, warnings = [], []
    
    # #21 (2026-08-19): this returned warnings ONLY, so strict=True could
    # never trip on a bad calibration -- the strict flag was decorative for
    # this check. An uncalibrated plane means every metre-based gate
    # (REID_HANDOFF_M, REID_STATIONARY_M, MAX_WALK_SPEED_MPS) is running on a
    # fallback fit that this repo's own notes call "implausible camera
    # heights". Under strict that is an error, not a note.
    if ground_plane_obj is None or not getattr(ground_plane_obj, "ok", False):
        _msg = ("Ground plane is inactive/uncalibrated. Every metre-based gate "
                "falls back to raw pixels, and the auto perspective fit has "
                "produced implausible camera heights on this camera. Run "
                "tools/calibrate_plane.py and fill ground_points in the zones "
                "file.")
        (errors if strict else warnings).append(_msg)
        return errors, warnings

    if hasattr(ground_plane_obj, "sanity"):
        frame_h = zones_cfg.get("frame_size", [1920, 1080])[1]
        sanity_issues = ground_plane_obj.sanity(frame_h)
        for issue in sanity_issues:
            warnings.append(f"Ground plane calibration warning: {issue}")

    return errors, warnings


def run_preflight_checks(zones_cfg: Dict[str, Any],
                         ground_plane_obj: Optional[Any] = None,
                         strict: bool = True) -> Dict[str, Any]:
    """Execute complete fail-loud pre-flight validation suite.
    
    Raises PreflightValidationError if critical errors are present and strict=True.
    Returns status report summary dict.
    """
    frame_size = tuple(zones_cfg.get("frame_size", [3840, 2160]))
    
    all_errors: List[str] = []
    all_warnings: List[str] = []

    # 1. Polygon validation
    p_err, p_warn = validate_polygons(zones_cfg, frame_size)
    all_errors.extend(p_err)
    all_warnings.extend(p_warn)

    # 2. Entry line validation
    l_err, l_warn = validate_entry_line(zones_cfg, frame_size)
    all_errors.extend(l_err)
    all_warnings.extend(l_warn)

    # 3. Ground plane validation
    g_err, g_warn = validate_ground_plane(zones_cfg, ground_plane_obj, strict=strict)
    all_errors.extend(g_err)
    all_warnings.extend(g_warn)

    report = {
        "status": "PASSED" if not all_errors else "FAILED",
        "errors": all_errors,
        "warnings": all_warnings,
        "frame_size": frame_size
    }

    if all_errors and strict:
        raise PreflightValidationError(
            f"Pre-flight validation failed with {len(all_errors)} error(s).",
            errors=all_errors,
            warnings=all_warnings
        )

    return report
