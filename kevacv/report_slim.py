"""report_slim.py — one file a manager reads, one file a analyst reads.

WHY THIS EXISTS
    The run produces an 8-sheet workbook, events.csv, minute_summaries.json,
    coverage_by_minute.csv, an id_timeline.csv, an id_audit.txt, a
    night_summary.json and a REPORT.md. Nobody opens nine files. The numbers
    that matter get lost among the numbers that were easy to compute.

    This module writes exactly three things:
        SUMMARY.txt    the whole night, readable in 30 seconds
        people.csv     one row per person, with a confidence and a crop
        snaps/P*.jpg   so "P4" is a face you can look at, not a token

    Everything else belongs in debug/ and is for whoever is fixing the
    pipeline, not for whoever is running the venue.

THE ONE RULE
    A number is printed with the tier it earned. EXACT, PROXY, ESTIMATE and
    WEAK are not decoration — a reader must be able to tell, without asking,
    which numbers they may act on. A number whose tier is unknown prints as
    UNKNOWN, never as bare text.

CONTRACT
    Pure data in, strings out. No notebook globals, no cv2, no file reads —
    so it is testable on a laptop with dicts. write_slim_outputs() is the only
    function that touches disk.
"""
from __future__ import annotations

TIERS = ("EXACT", "PROXY", "ESTIMATE", "WEAK", "UNKNOWN")

TIER_LEGEND = [
    ("EXACT",    "measured directly", "coverage, gaps, timings"),
    ("PROXY",    "staff stood near guest >=3s", "NOT proof of conversation"),
    ("ESTIMATE", "identity had to hold for minutes", "a range, not one number"),
    ("WEAK",     "may over-count", "never act on this alone"),
]

_BAR = "=" * 78
_SUB = "-" * 78


def _hm(seconds):
    """Seconds of video time -> H:MM, or pass a 'HH:MM' string straight back."""
    if seconds is None:
        return "--:--"
    if isinstance(seconds, str):
        return seconds
    s = int(seconds)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}"


def _fmt(value, unit=""):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}{unit}"
    return f"{value}{unit}"


def _tier(tag):
    tag = (tag or "UNKNOWN").upper()
    return tag if tag in TIERS else "UNKNOWN"


def coverage_strip(covered_windows, t_end, width=48):
    """An ASCII picture of when the desk was covered.

    '#' covered, '.' not covered, '?' no footage. A percentage tells you how
    much; this tells you WHEN, which is the part a manager acts on.
    """
    if not t_end or t_end <= 0:
        return "?" * width
    cells = []
    for i in range(width):
        a = t_end * i / width
        b = t_end * (i + 1) / width
        overlap = 0.0
        for (s, e) in covered_windows or []:
            overlap += max(0.0, min(b, e) - max(a, s))
        cells.append("#" if overlap >= (b - a) * 0.5 else ".")
    return "".join(cells)


def _target_line(label, actual, target, higher_is_better=True, unit="%"):
    if actual is None or target is None:
        return f"  TARGET {label}: not measured"
    ok = actual >= target if higher_is_better else actual <= target
    delta = actual - target
    verdict = "PASS" if ok else "FAIL"
    dots = "." * max(1, 34 - len(label))
    return (f"  TARGET {label} >= {target}{unit}   ACTUAL {actual}{unit} "
            f"{dots} {verdict} {delta:+.1f}")


def summary_txt(meta, answers, staff, anomalies, notes=None,
                covered_windows=None):
    """Build SUMMARY.txt.

    meta      dict: camera, date, start, end, footage_h, missing_h, source,
                    provenance_ok, hota (dict or None), t_end_s, video (dict
                    or None: annotated, codec, size_mb, speedup, clips, why)
    answers   list of dicts: label, value, tier, extra(list of (label,value,tier))
    staff     list of dicts: name, minutes, pct, source, confidence
    anomalies list of dicts: time, what, clip, severity
    notes     list of strings — what weakened this run
    """
    L = [_BAR]
    L.append(f"  RECEPTION  ·  {meta.get('camera', '?')}"
             f"{' ' * 6}{meta.get('date', '')}")
    L.append(f"  {meta.get('start', '?')} -> {meta.get('end', '?')}   ·   "
             f"{_fmt(meta.get('footage_h'), ' h')} footage   ·   "
             f"{_fmt(meta.get('missing_h'), ' h')} missing")
    prov = "PROVENANCE OK  file == clock" if meta.get("provenance_ok") \
        else "!! PROVENANCE UNVERIFIED — do not trust the times below"
    L.append(f"  source  {meta.get('source', '?')}")
    L.append(f"          [{prov}]")
    hota = meta.get("hota")
    if hota:
        L.append(f"  quality HOTA {hota.get('day', '--')} (day) / "
                 f"{hota.get('ir', '--')} (IR)   [{hota.get('verdict', '?')}]")
    else:
        L.append("  quality HOTA NOT MEASURED — every accuracy claim below "
                 "is an estimate")
    L.append(_BAR)
    L.append("")

    if covered_windows is not None:
        L.append("  DESK COVERAGE")
        _w = 48
        L.append("  " + coverage_strip(covered_windows, meta.get("t_end_s"), _w))
        _a, _b = str(meta.get("start", "")), str(meta.get("end", ""))
        L.append("  " + "^" + " " * (_w - 2) + "^")
        L.append("  " + _a.ljust(_w - len(_b)) + _b)
        L.append("  # covered   . not covered")
        L.append("")

    L.append(_SUB)
    L.append("  THE ANSWERS")
    L.append(_SUB)
    for a in answers:
        L.append(f"  {a['label']:<38} {str(a['value']):<18} [{_tier(a.get('tier'))}]")
        for (lbl, val, tier) in a.get("extra", []):
            L.append(f"      {lbl:<34} {str(val):<18} [{_tier(tier)}]")
        L.append("")

    L.append(_SUB)
    L.append("  WHO WORKED THE DESK")
    L.append(_SUB)
    if not staff:
        L.append("  nobody was identified at the desk this run")
    for s in staff:
        pct = s.get("pct") or 0
        bar = "#" * int(pct / 4) + "." * (25 - int(pct / 4))
        L.append(f"  {s.get('name') or '?':<22} {_fmt(s.get('minutes'), ' min'):>10} "
                 f"{pct:>3.0f}%  {bar}  [{s.get('source') or '?'}] "
                 f"conf {s.get('confidence') or '--'}")
    L.append("")

    L.append(_SUB)
    L.append(f"  WHAT WENT WRONG   ({len(anomalies)} for a human to watch)")
    L.append(_SUB)
    if not anomalies:
        L.append("  nothing flagged")
    for an in anomalies:
        # .get(k, default) returns None when the key EXISTS and is None, which
        # is exactly how these rows encode "not measured" — and None blows up
        # an f-string width spec. `or default` covers both. A report that
        # crashes on a missing field is worse than one that prints a blank.
        L.append(f"  {an.get('time') or '--:--':<8} {an.get('what') or '':<46} "
                 f"{an.get('clip') or '':<14} {an.get('severity') or ''}")
    L.append("")

    vid = meta.get("video") or {}
    if vid:
        L.append(_SUB)
        L.append("  WATCH")
        L.append(_SUB)
        ann = vid.get("annotated")
        if ann:
            size = f"  {vid['size_mb']:.0f} MB" if vid.get("size_mb") else ""
            L.append(f"  full run    {ann}{size}")
            if vid.get("speedup"):
                L.append(f"              plays {vid['speedup']:.1f}x faster than "
                         f"real time — {_fmt(meta.get('footage_h'), ' h')} of "
                         f"footage in about "
                         f"{(meta.get('footage_h') or 0) * 60 / vid['speedup']:.0f} min")
            L.append(f"              codec {vid.get('codec', 'unknown')}"
                     + ("  — plays in Windows Media Player, QuickTime, Chrome"
                        if vid.get("codec", "").startswith("h264") else ""))
        else:
            # The 68b97311f9 run printed "nobody appeared in any analysed frame"
            # while a valid 58 MB file sat on disk under the _h264 name. Never
            # let a missing path imply an empty venue again.
            L.append("  full run    NOT FOUND")
            L.append(f"              reason: {vid.get('why', 'not recorded')}")
            L.append("              a missing file is NOT evidence the venue "
                     "was empty — check for *_annotated_h264.mp4")
        clips = vid.get("clips") or []
        if clips:
            L.append(f"  moments     {len(clips)} clip(s), one per flagged event above")
            for c in clips[:6]:
                L.append(f"              {c}")
        L.append("")

    L.append(_SUB)
    L.append("  HOW MUCH TO TRUST THIS")
    L.append(_SUB)
    for tag, means, caveat in TIER_LEGEND:
        L.append(f"  [{tag:<8}]  {means:<32} {caveat}")
    if notes:
        L.append("")
        L.append("  WEAKENING THIS RUN")
        for n in notes:
            L.append(f"    {n}")
    L.append("")
    L.append("  NOT MEASURED / NOT CLAIMED")
    L.append("    what was actually said · customers across days · "
             "areas off-camera")
    L.append(_BAR)
    return "\n".join(L)


PEOPLE_COLUMNS = ["person", "snap", "role", "role_from", "first_seen",
                  "last_seen", "minutes", "waited_s", "greeted", "greet_s",
                  "confidence", "flags"]


def people_rows(people):
    """Normalise person dicts into the fixed column order, filling gaps.

    A missing value is written as "" and never as 0 — a zero that means
    "we did not measure this" is the same lie the report exists to stop.
    """
    rows = []
    for p in people:
        rows.append({c: ("" if p.get(c) is None else p.get(c))
                     for c in PEOPLE_COLUMNS})
    return rows


def people_csv(people):
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=PEOPLE_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in people_rows(people):
        w.writerow(r)
    return buf.getvalue()


def write_slim_outputs(out_dir, meta, answers, staff, anomalies, people,
                       notes=None, covered_windows=None, snaps=None):
    """Write SUMMARY.txt, people.csv and snaps/. Returns the paths written.

    `snaps` is {person_id: BGR ndarray}. Written only if cv2 imports — the
    rest of the module stays importable on a machine without it.
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    p = out / "SUMMARY.txt"
    p.write_text(summary_txt(meta, answers, staff, anomalies, notes,
                             covered_windows), encoding="utf-8")
    written.append(p)

    p = out / "people.csv"
    p.write_text(people_csv(people), encoding="utf-8")
    written.append(p)

    if snaps:
        try:
            import cv2
            sd = out / "snaps"
            sd.mkdir(exist_ok=True)
            for pid, img in snaps.items():
                if img is None:
                    continue
                # append, never with_suffix: ids and camera names carry dots
                f = sd / f"{pid}.jpg"
                cv2.imwrite(str(f), img)
                written.append(f)
        except ImportError:
            pass
    return written


def describe_video(annotated_path, clips=None, analysed_fps=None,
                   playback_fps=None):
    """Build the `video` block for summary_txt() from what is on disk.

    Resolves the direct-h264 filename the renderer actually writes. The
    pipeline once looked for `*_annotated.mp4`, found nothing because the file
    was `*_annotated_h264.mp4`, and reported "nobody appeared in any analysed
    frame" for a chunk containing 24,426 rendered frames. Checking both names
    here means the report cannot repeat that.
    """
    from pathlib import Path
    if not annotated_path:
        return {"annotated": None, "why": "renderer was not run"}
    p = Path(annotated_path)
    candidates = [p, p.with_name(p.stem + "_h264.mp4")]
    if p.stem.endswith("_h264"):
        candidates.append(p.with_name(p.stem[:-5] + ".mp4"))
    found = next((c for c in candidates if c.exists()), None)
    if found is None:
        return {"annotated": None,
                "why": (f"none of {[c.name for c in candidates]} exist on disk"),
                "clips": list(clips or [])}
    out = {"annotated": str(found),
           "size_mb": found.stat().st_size / 1_048_576,
           "codec": "h264" if "_h264" in found.stem else "unknown (mp4v?)",
           "clips": list(clips or [])}
    if analysed_fps and playback_fps and analysed_fps > 0:
        out["speedup"] = playback_fps / analysed_fps
    return out
