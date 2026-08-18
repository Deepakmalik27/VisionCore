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


def validate_entry_line(zones_cfg: Dict[str, Any], frame_size: Tuple[int, int]) -> Tuple[List[str], List[str]]:
    """Validate entry line presence and conduct synthetic crossing simulations."""
    errors, warnings = [], []
    fw, fh = frame_size

    entry_line = zones_cfg.get("entry_line")
    if not entry_line:
        warnings.append("No 'entry_line' specified in zones config. Line-crossing count will be unavailable.")
        return errors, warnings

    if not isinstance(entry_line, list) or len(entry_line) != 2:
        errors.append(f"Invalid 'entry_line' format: {entry_line!r}. Expected [[x1, y1], [x2, y2]].")
        return errors, warnings

    try:
        p1, p2 = entry_line[0], entry_line[1]
        # Validate that vertices are subscriptable coordinate pairs
        if not isinstance(p1, (list, tuple)) or len(p1) != 2:
            errors.append(f"entry_line[0] must be [x, y], got: {p1!r}")
            return errors, warnings
        if not isinstance(p2, (list, tuple)) or len(p2) != 2:
            errors.append(f"entry_line[1] must be [x, y], got: {p2!r}")
            return errors, warnings
        dx = float(p2[0]) - float(p1[0])
        dy = float(p2[1]) - float(p1[1])
        line_len = math.hypot(dx, dy)
    except (TypeError, ValueError, IndexError) as e:
        errors.append(f"entry_line vertices are malformed: {entry_line!r} ({e}).")
        return errors, warnings

    if line_len < 20:
        errors.append(f"Entry line is too short ({line_len:.1f} px). Must span the physical entrance threshold.")
        return errors, warnings

    # Synthetic crossing test: simulate a person walking through the line perpendicularly
    mid_x = (p1[0] + p2[0]) / 2.0
    mid_y = (p1[1] + p2[1]) / 2.0
    
    # Perpendicular unit vector
    perp_x = -dy / line_len
    perp_y = dx / line_len

    traj_start = (mid_x - perp_x * 100, mid_y - perp_y * 100)
    traj_end = (mid_x + perp_x * 100, mid_y + perp_y * 100)

    if not _line_intersect(tuple(p1), tuple(p2), traj_start, traj_end):
        errors.append("Synthetic crossing simulation failed: test trajectory did not intersect the configured entry line.")

    return errors, warnings


def validate_ground_plane(zones_cfg: Dict[str, Any], ground_plane_obj: Any) -> Tuple[List[str], List[str]]:
    """Validate camera ground plane calibration quality."""
    errors, warnings = [], []
    
    if ground_plane_obj is None or not getattr(ground_plane_obj, "ok", False):
        warnings.append("Ground plane is inactive/uncalibrated. Spatial metrics will default to raw pixel distances.")
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
    g_err, g_warn = validate_ground_plane(zones_cfg, ground_plane_obj)
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
