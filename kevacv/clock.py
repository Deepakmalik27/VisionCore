"""clock.py — what time is it, really, and is that clock trustworthy?

WHY THIS EXISTS
    The entire product is timestamps. "The desk was empty from 16:47" is the
    deliverable; the tracking is only how we get there. Yet every time-related
    guard lived in notebook Cell 2e, so kevacv.pipeline — the path meant to be
    universal — had no clock verification at all.

    Three separate ways the clock lies, all of which produce a confident,
    plausible, wrong report:

    1. VARIABLE FRAME RATE. frame_index/fps is not the clock if the container
       lies about fps. Durations drift by a growing amount, so an early gap
       reads correctly and a late one is minutes out. Detected by comparing
       assumed time against CAP_PROP_POS_MSEC at several points.

    2. DST. The clock is parsed from a filename as naive local time. An
       overnight run across a spring-forward or fall-back boundary shifts every
       timestamp by an hour — and fall-back makes one local hour happen twice,
       so two different real moments print identically.

    3. NON-MONOTONIC TIMESTAMPS. A decoder hiccup can repeat or reverse a
       timestamp. merge_intervals, covered_windows and every dwell computation
       assume time only moves forward; a reversal produces negative durations
       that quietly cancel real ones out.

    A wrong-video/wrong-clock pairing already shipped once here: CHUNK_FILTER
    selected the 7:30pm file, the 4:30pm file was on disk, and the run stamped
    19:30 onto 16:30 footage. verify_provenance() is that guard, in code that
    travels with the pipeline instead of with one notebook.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from .log import get_logger

_log = get_logger("clock")

# "CAM.112 (PP.09_12) 7-28-2026, 4.30.00pm CDT - 7-28-2026, 5.30.00pm CDT.mp4"
_STAMP = re.compile(
    r"(?P<m>\d{1,2})-(?P<d>\d{1,2})-(?P<y>\d{4})\s*,\s*"
    r"(?P<hh>\d{1,2})\.(?P<mm>\d{2})\.(?P<ss>\d{2})\s*(?P<ap>[ap]m)",
    re.I)

VFR_DRIFT_PCT = 1.0        # above this, frame_index/fps is not the clock


def parse_start(name):
    """-> naive datetime of the chunk's START, from its filename.

    Reads the FIRST stamp only. Filenames carry a range ("4.30pm - 5.30pm") and
    a bare substring search matched the previous hour's END time, which is how
    a chunk filter once selected the wrong file.
    """
    m = _STAMP.search(str(name))
    if not m:
        return None
    g = m.groupdict()
    hh = int(g["hh"]) % 12
    if g["ap"].lower() == "pm":
        hh += 12
    try:
        return datetime(int(g["y"]), int(g["m"]), int(g["d"]),
                        hh, int(g["mm"]), int(g["ss"]))
    except ValueError:
        return None


def localize(dt, tz_name):
    """Attach a real timezone. Returns (aware_dt, findings).

    Naive local time is not a time. Two hours a year it is ambiguous, and one
    hour a year it does not exist.
    """
    findings = []
    if dt is None:
        return None, [("ERROR", "no timestamp could be parsed from the filename "
                                "— wall-clock times would be video time only")]
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        return dt, [("WARN", "zoneinfo unavailable; times remain naive local")]
    try:
        zone = ZoneInfo(tz_name)
    except Exception:
        return dt, [("WARN", f"unknown timezone {tz_name!r}; times remain naive")]

    aware = dt.replace(tzinfo=zone)
    # fold=1 names the SECOND pass through a repeated local hour. If the two
    # folds have different UTC offsets, this local time happens twice today.
    if aware.utcoffset() != dt.replace(tzinfo=zone, fold=1).utcoffset():
        findings.append(("ERROR",
                         f"{dt} is AMBIGUOUS in {tz_name} — the clock went back "
                         f"and this local hour occurs twice. Two different real "
                         f"moments would print the same time."))
    # A time that does not exist round-trips to something else.
    if aware.astimezone(ZoneInfo("UTC")).astimezone(zone).replace(tzinfo=None) != dt:
        findings.append(("ERROR",
                         f"{dt} does not exist in {tz_name} — the clock went "
                         f"forward through it."))
    return aware, findings


def check_dst_span(start, hours, tz_name):
    """Does this run cross a DST transition? -> findings.

    Checked on the SPAN, not just the start: a 10-hour overnight chunk can
    begin in one offset and end in another, and every timestamp after the
    boundary is then an hour out.
    """
    if start is None:
        return []
    aware, f = localize(start, tz_name)
    if aware is None or aware.tzinfo is None:
        return f
    end = aware + timedelta(hours=float(hours or 0))
    if aware.utcoffset() != end.utcoffset():
        f.append(("ERROR",
                  f"this run crosses a DST change ({aware.utcoffset()} -> "
                  f"{end.utcoffset()}). Timestamps after the boundary are off "
                  f"by the difference unless every conversion is done in UTC."))
    return f


def check_frame_clock(probe_times, fps):
    """Is frame_index/fps the real clock? -> (source, worst_drift_pct, findings).

    probe_times: [(frame_index, actual_seconds), ...] read from the decoder.
    Returns "frame_index" or "pos_msec" — the caller should use the latter as
    its time source when drift is high.
    """
    if not probe_times or not fps:
        return "frame_index", 0.0, [("WARN", "clock could not be verified; "
                                             "assuming constant frame rate")]
    worst = 0.0
    for idx, actual in probe_times:
        assumed = idx / float(fps)
        if actual and actual > 0:
            worst = max(worst, abs(assumed - actual) / actual * 100.0)
    if worst > VFR_DRIFT_PCT:
        return "pos_msec", worst, [
            ("ERROR", f"VARIABLE FRAME RATE — worst drift {worst:.2f}%. "
                      f"frame_index/fps is NOT the clock here; every duration "
                      f"would be wrong by a growing amount. Use "
                      f"CAP_PROP_POS_MSEC as the time source.")]
    return "frame_index", worst, []


def verify_provenance(selected_name, decoded_name, clock_source_name):
    """The file we picked, the file we decoded, and the file the clock came
    from must be the SAME file. -> findings.

    This exact mismatch shipped: a chunk filter selected the 7:30pm file, the
    4:30pm file was already on disk, and the run stamped 19:30 onto 16:30
    footage. Every timestamp in that report was three hours wrong and nothing
    complained, because each step was individually correct.
    """
    names = {"selected": str(selected_name or ""),
             "decoded": str(decoded_name or ""),
             "clock": str(clock_source_name or "")}
    stamps = {k: parse_start(v) for k, v in names.items() if v}
    distinct = {v for v in stamps.values() if v is not None}
    if len(distinct) > 1:
        detail = "  ".join(f"{k}={v}" for k, v in stamps.items())
        return [("ERROR",
                 f"PROVENANCE MISMATCH — the selected file, the decoded file "
                 f"and the clock source do not agree: {detail}. Every "
                 f"timestamp in this report would be wrong.")]
    base = {k: v.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for k, v in names.items() if v}
    if len(set(base.values())) > 1:
        return [("WARN",
                 f"selected/decoded/clock filenames differ but parse to the "
                 f"same start time: {base}")]
    return []


def describe(source, drift_pct, findings):
    L = [f"clock source   {source}   worst drift {drift_pct:.2f}%"]
    for lvl, msg in findings:
        L.append(f"  [{lvl}] {msg}")
    if not findings:
        L.append("  no clock problems found")
    return "\n".join(L)
