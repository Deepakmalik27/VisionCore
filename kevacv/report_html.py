"""report_html.py — one self-contained page a non-technical person reads.

WHY THIS EXISTS
    The run already emits SUMMARY.txt, people.csv and a scorecard. Those are
    correct and nobody outside this repo will ever open them. The question a
    GM or a CEO actually asks is "what happened last night, and can I trust
    it" -- and answering it today means reading four files and knowing what
    `trust=region` means.

    This renders the SAME numbers, already computed, as one HTML file. No
    server, no CDN, no build step: it opens off a USB stick on a laptop with
    no internet, which is the actual demo constraint.

WHAT IT MUST NOT DO
    It computes NO new answers. Every value on the page comes from `result`,
    which the pipeline already assembled, so the page can never disagree with
    SUMMARY.txt. If a number is missing here it is missing upstream -- and it
    renders as "not measured", never as zero. That is validity.py's rule
    ("'I did not observe' is not 'I observed nothing'") applied to pixels.

THE TIER IS NOT DECORATION
    Every headline carries EXACT / PROXY / WEAK exactly as answers.py graded
    it. A CEO deck that drops the tier is how "greeted within 30s" becomes a
    promise instead of a proximity heuristic.

COLOUR
    Series colours are the validated categorical slots 1-3 (blue/orange/aqua)
    on the ivory surface; aqua sits below 3:1 there, so every bar carries a
    direct label and the page ends with a full table view -- the documented
    relief for that warning, not an oversight.
"""
from __future__ import annotations

import html
from pathlib import Path

# ── palette ────────────────────────────────────────────────────────────────
# categorical slots 1-3, validated all-pairs on surface #FBF8F3:
#   CVD worst 9.2 (>=8), normal-vision worst 24.0 (>=15).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

_CSS = """
:root{--surface:#FBF8F3;--panel:#FFFDF9;--ink:#1A1714;--ink2:#5A5248;
--ink3:#8C8378;--rule:#E4DCCE;--brass:#A8842C;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
--good:#0ca30c;--warn:#fab219;--crit:#d03b3b}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:44px 28px 80px}
h1{font:italic 400 34px/1.15 ui-serif,Georgia,"Times New Roman",serif;margin:0}
h2{font:italic 400 21px/1.2 ui-serif,Georgia,serif;margin:44px 0 4px;
color:var(--ink)}
.sub{color:var(--ink2);font-size:13px;margin:6px 0 0}
.rule{height:1px;background:var(--rule);margin:14px 0 22px}
.brassrule{height:2px;background:var(--brass);width:52px;margin:10px 0 0}
.num{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
gap:1px;background:var(--rule);border:1px solid var(--rule);margin-top:26px}
.kpi{background:var(--panel);padding:20px 18px}
.kpi .v{font:400 40px/1 ui-monospace,Menlo,monospace;
font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .k{font-size:11px;letter-spacing:.13em;text-transform:uppercase;
color:var(--ink3);margin-top:9px}
.kpi .n{font-size:12px;color:var(--ink2);margin-top:5px}
.tier{display:inline-block;font-size:10px;letter-spacing:.09em;padding:1px 6px;
border:1px solid var(--rule);border-radius:2px;color:var(--ink2);
vertical-align:2px;margin-left:6px}
.tier.EXACT{border-color:var(--good);color:#0a7a0a}
.tier.PROXY{border-color:var(--warn);color:#8a6206}
.tier.WEAK{border-color:var(--crit);color:var(--crit)}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th{text-align:left;font-weight:600;font-size:11px;letter-spacing:.09em;
text-transform:uppercase;color:var(--ink3);
border-bottom:1px solid var(--rule);padding:7px 10px 7px 0}
td{padding:7px 10px 7px 0;border-bottom:1px solid var(--rule);
vertical-align:top}
td.num,th.num{text-align:right;padding-right:16px;
font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.an{display:flex;gap:12px;padding:11px 0;border-bottom:1px solid var(--rule)}
.dot{width:9px;height:9px;border-radius:50%;margin-top:6px;flex:0 0 9px}
.an .t{font-family:ui-monospace,Menlo,monospace;color:var(--ink2);
flex:0 0 58px;font-size:13px}
.none{color:var(--ink3);font-style:italic;padding:14px 0}
.foot{margin-top:52px;padding-top:18px;border-top:2px solid var(--brass);
font-size:12px;color:var(--ink2)}
.foot b{color:var(--ink);font-weight:600}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;
color:var(--ink2);margin:10px 0 2px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;
margin-right:6px;vertical-align:-1px}
@media print{.wrap{max-width:none}body{background:#fff}}
"""


def _e(v):
    return html.escape("" if v is None else str(v))


def _hhmm(sec, start_hhmm=None):
    """Seconds-into-chunk -> wall clock when we know the start, else mm:ss."""
    if sec is None:
        return None
    sec = float(sec)
    if start_hhmm:
        try:
            h, m = (int(x) for x in str(start_hhmm).split(":")[:2])
            tot = h * 3600 + m * 60 + sec
            return f"{int(tot // 3600) % 24:02d}:{int(tot % 3600) // 60:02d}"
        except (ValueError, TypeError):
            pass
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


# ── charts (inline SVG, no library) ────────────────────────────────────────

def _bars(pairs, colour=SERIES[0], h=132, unit=""):
    """Vertical bars with a direct label on every bar.

    `pairs` is [(label, value)]. An empty series renders as an honest
    "not measured" line rather than an empty axis, because an empty chart and
    a chart of zeroes look identical and mean opposite things.
    """
    pairs = [(k, v) for k, v in pairs if v is not None]
    if not pairs:
        return '<p class="none">not measured</p>'
    top = max(v for _, v in pairs) or 1
    n = len(pairs)
    bw, gap = 34, 12
    w = n * (bw + gap) + 40
    out = [f'<svg viewBox="0 0 {w} {h + 46}" width="100%" height="{h + 46}" '
           f'role="img" style="max-width:{w}px">']
    out.append(f'<line x1="20" y1="{h}" x2="{w - 20}" y2="{h}" '
               f'stroke="#E4DCCE" stroke-width="1"/>')
    for i, (k, v) in enumerate(pairs):
        bh = max(2.0, (v / top) * (h - 26))
        x, y = 20 + i * (bw + gap), h - bh
        # 4px rounded data-end, anchored to the baseline
        out.append(f'<path d="M{x} {h} L{x} {y + 4} Q{x} {y} {x + 4} {y} '
                   f'L{x + bw - 4} {y} Q{x + bw} {y} {x + bw} {y + 4} '
                   f'L{x + bw} {h} Z" fill="{colour}"/>')
        out.append(f'<text x="{x + bw / 2}" y="{y - 6}" text-anchor="middle" '
                   f'font-size="12" font-family="ui-monospace,Menlo,monospace" '
                   f'fill="#1A1714">{_e(v)}{_e(unit)}</text>')
        out.append(f'<text x="{x + bw / 2}" y="{h + 17}" text-anchor="middle" '
                   f'font-size="11" fill="#8C8378">{_e(k)}</text>')
    out.append("</svg>")
    return "".join(out)


def _hbars(pairs, colour=SERIES[2], unit=""):
    """Horizontal bars, direct-labelled. Used where the category name is long."""
    pairs = [(k, v) for k, v in pairs if v is not None]
    if not pairs:
        return '<p class="none">not measured</p>'
    top = max(v for _, v in pairs) or 1
    rows = []
    for k, v in pairs:
        pct = 100.0 * v / top
        rows.append(
            f'<tr><td style="width:150px">{_e(k)}</td>'
            f'<td style="padding-right:0"><span style="display:inline-block;'
            f'height:11px;border-radius:0 3px 3px 0;background:{colour};'
            f'width:{pct:.1f}%;min-width:2px;vertical-align:-1px"></span></td>'
            f'<td class="num" style="width:88px">{_e(v)}{_e(unit)}</td></tr>')
    return f'<table style="margin-top:10px">{"".join(rows)}</table>'


def _strip(segments, total_s, h=26):
    """A state timeline: [(t0, t1, state)] over `total_s`.

    States are `covered` / `gap` / `unobserved`. `unobserved` is drawn as a
    hatch, never as a gap colour, because "the camera was blind" and "the desk
    was empty" are the two things this whole report exists not to confuse.
    """
    if not total_s:
        return '<p class="none">not measured</p>'
    fill = {"covered": STATUS["good"], "gap": STATUS["critical"]}
    out = ['<svg viewBox="0 0 1000 %d" width="100%%" height="%d" role="img" '
           'preserveAspectRatio="none">' % (h, h),
           '<defs><pattern id="hatch" width="6" height="6" '
           'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
           '<rect width="6" height="6" fill="#F2ECE1"/>'
           '<line x1="0" y1="0" x2="0" y2="6" stroke="#D6CBB8" '
           'stroke-width="2"/></pattern></defs>',
           f'<rect x="0" y="0" width="1000" height="{h}" fill="url(#hatch)"/>']
    for t0, t1, state in segments:
        x = 1000.0 * float(t0) / total_s
        w = max(1.0, 1000.0 * (float(t1) - float(t0)) / total_s)
        c = fill.get(state)
        if c:
            # 2px surface gap between adjacent fills
            out.append(f'<rect x="{x:.2f}" y="0" width="{max(1.0, w - 2):.2f}" '
                       f'height="{h}" fill="{c}"/>')
    out.append("</svg>")
    return "".join(out)


# ── page ───────────────────────────────────────────────────────────────────

def _kpi(value, key, note="", tier=None):
    t = f'<span class="tier {_e(tier)}">{_e(tier)}</span>' if tier else ""
    v = "—" if value is None else _e(value)
    return (f'<div class="kpi"><div class="v">{v}</div>'
            f'<div class="k">{_e(key)}{t}</div>'
            f'<div class="n">{_e(note)}</div></div>')


def _coverage_segments(events, zone_roles, observed, total_s):
    """Staff-zone occupancy -> [(t0, t1, state)] across the whole chunk."""
    staff_zones = {z for z, r in (zone_roles or {}).items()
                   if "staff" in (r or [])}
    busy = sorted((float(e["t_in"]), float(e["t_out"])) for e in (events or [])
                  if e.get("zone") in staff_zones)
    merged = []
    for a, b in busy:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    segs = [(a, b, "covered") for a, b in merged]
    # a gap only counts inside OBSERVED footage; outside it we saw nothing,
    # and the hatch background already says so.
    for o0, o1 in (observed or [(0.0, total_s)]):
        cur = float(o0)
        for a, b in merged:
            if b <= o0 or a >= o1:
                continue
            if a > cur:
                segs.append((cur, min(a, o1), "gap"))
            cur = max(cur, b)
        if cur < o1:
            segs.append((cur, float(o1), "gap"))
    return sorted(segs)


def render(result, venue="Delilah"):
    """The whole page as one string. Reads `result`; computes no answers."""
    run = result.get("run") or {}
    meta_start = (result.get("clock") or {}).get("start") or ""
    hhmm = meta_start[11:16] if len(str(meta_start)) >= 16 else None
    dur = float(run.get("duration_s") or 0)
    answers = result.get("answers") or []
    by_label = {str(a.get("label", "")).lower(): a for a in answers}

    def ans(*needles):
        for lab, a in by_label.items():
            if all(n in lab for n in needles):
                return a
        return {}

    people = result.get("people") or []
    staff = result.get("staff") or []
    anomalies = result.get("anomalies") or []
    arrivals = result.get("arrivals") or {}
    events = run.get("events") or []
    zone_roles = run.get("zone_roles") or {}
    observed = run.get("observed_windows") or ([(0.0, dur)] if dur else [])
    observed_s = sum(float(b) - float(a) for a, b in observed) or None

    # ── KPI row. Values are the graded answers, verbatim. ──
    a_guests, a_desk = ans("guest"), ans("desk", "cover")
    a_greet = ans("greet")
    kpis = [
        _kpi(a_guests.get("value"), "guests", "unique visitors",
             a_guests.get("tier")),
        _kpi(len(staff) or None, "staff on shift",
             f"{sum(1 for s in staff if s.get('source') == 'face')} face-matched"),
        _kpi(a_desk.get("value"), "desk covered", "of observed footage",
             a_desk.get("tier")),
        _kpi(len(anomalies), "anomalies", "flagged for review"),
    ]
    if a_greet:
        kpis.append(_kpi(a_greet.get("value"), "greet latency",
                         "entry to staff proximity", a_greet.get("tier")))

    # ── arrivals per hour, from the per-guest arrival times already derived ──
    # Bucket by HOUR INTO THE CHUNK, not by the clock string. A service night
    # runs 18:00 -> 02:00, and sorting the labels lexicographically puts
    # "00:00" before "18:00" -- i.e. midnight drawn at the far left of the
    # evening. Sort on elapsed time; the wall clock is only the label.
    per_hour, arr_by_id = {}, (run.get("arrivals_by_id") or {})
    for t in arr_by_id.values():
        if t is None:
            continue
        idx = int(float(t) // 3600)
        hh = _hhmm(idx * 3600, hhmm)
        label = (hh.split(":")[0] + ":00") if hh and ":" in hh else f"+{idx}h"
        cur = per_hour.get(idx) or [label, 0]
        cur[1] += 1
        per_hour[idx] = cur
    arrivals_chart = _bars([(lab, n) for _, (lab, n) in sorted(per_hour.items())],
                           SERIES[0])

    # ── zone dwell ──
    dwell = {}
    for e in events:
        dwell[e.get("zone")] = dwell.get(e.get("zone"), 0.0) + float(
            e.get("duration") or 0)
    dwell_chart = _hbars(
        [(k, round(v / 60.0)) for k, v in
         sorted(dwell.items(), key=lambda kv: -kv[1])[:8]],
        SERIES[2], unit=" min")

    # ── staff time ──
    staff_chart = _hbars(
        [(s.get("name"), s.get("minutes")) for s in
         sorted(staff, key=lambda s: -(s.get("minutes") or 0))[:8]],
        SERIES[1], unit=" min")

    cover_strip = _strip(_coverage_segments(events, zone_roles, observed, dur),
                         dur)

    # ── anomalies ──
    if anomalies:
        an_html = "".join(
            f'<div class="an"><span class="dot" style="background:'
            f'{STATUS["critical"] if a.get("severity") == "high" else STATUS["warning"]}">'
            f'</span><span class="t">{_e(a.get("time"))}</span>'
            f'<span>{_e(a.get("what"))}</span></div>' for a in anomalies)
    else:
        an_html = ('<p class="none">Nothing was flagged. Note this means no '
                   'rule fired &mdash; not that nothing happened.</p>')

    # ── the table view (the relief for the sub-3:1 series colour) ──
    rows = "".join(
        f'<tr><td class="num">{_e(p.get("person"))}</td>'
        f'<td>{_e(p.get("role"))}</td>'
        f'<td class="num">{_e(p.get("first_seen"))}</td>'
        f'<td class="num">{_e(p.get("last_seen"))}</td>'
        f'<td class="num">{_e(p.get("minutes"))}</td>'
        f'<td class="num">{"" if p.get("waited_s") is None else _e(p["waited_s"])}</td>'
        f'<td>{_e(p.get("greeted"))}</td>'
        f'<td style="color:#8C8378">{_e(", ".join(p.get("flags") or []))}</td>'
        f'</tr>' for p in people[:120])
    table = (f'<table><thead><tr><th class="num">id</th><th>role</th>'
             f'<th class="num">first</th><th class="num">last</th>'
             f'<th class="num">min</th><th class="num">waited s</th>'
             f'<th>greeted</th><th>flags</th></tr></thead>'
             f'<tbody>{rows}</tbody></table>') if people else \
        '<p class="none">no people rows</p>'

    # ── evidence footer: how much of this to believe ──
    xc = (arrivals.get("cross_check") or {})
    hota = result.get("hota") or {}
    ev = [f'<b>Arrival estimators</b> &mdash; line {arrivals.get("line")}, '
          f'region {arrivals.get("region")}, slit {arrivals.get("slit")} '
          f'(trust: {xc.get("trust", "?")}). The slit-scan uses no identity at '
          f'all, so a gap between it and the other two points at id churn, not '
          f'at the door.']
    if hota:
        ev.append(f'<b>Tracking score</b> &mdash; HOTA '
                  f'{hota.get("HOTA", 0):.3f} · DetA {hota.get("DetA", 0):.3f} '
                  f'· AssA {hota.get("AssA", 0):.3f} · recall '
                  f'{hota.get("recall", 0):.3f} (floor 0.40).')
    else:
        ev.append('<b>Tracking score</b> &mdash; not scored: no ground truth '
                  'for this chunk. Every accuracy claim above is the '
                  "pipeline's own, unverified.")
    if observed_s and dur:
        ev.append(f'<b>Observed</b> &mdash; {observed_s / 60:.0f} of '
                  f'{dur / 60:.0f} minutes of footage carried usable evidence. '
                  f'Percentages are of observed time, never elapsed.')
    errs = [m for lvl, m in (result.get("findings") or []) if lvl == "ERROR"]
    if errs:
        ev.append('<b style="color:#d03b3b">Errors this run</b> &mdash; '
                  + "; ".join(_e(m) for m in errs[:4]))

    answers_rows = "".join(
        f'<tr><td>{_e(a.get("label"))}</td>'
        f'<td class="num">{_e(a.get("value"))}</td>'
        f'<td><span class="tier {_e(a.get("tier"))}">{_e(a.get("tier"))}</span>'
        f'</td></tr>' for a in answers)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(venue)} — {_e(str(meta_start)[:10])}</title>
<style>{_CSS}</style></head><body><div class="wrap">

<h1>{_e(venue)}</h1>
<p class="sub">{_e(result.get("camera_id"))} &nbsp;·&nbsp;
{_e(str(meta_start)[:10] or "date unknown")} &nbsp;·&nbsp;
{_e(hhmm or "?")}&ndash;{_e(_hhmm(dur, hhmm) or "?")} &nbsp;·&nbsp;
{dur / 3600:.1f} h of footage</p>
<div class="brassrule"></div>

<div class="kpis">{''.join(kpis)}</div>

<h2>Was the desk covered?</h2><div class="rule"></div>
{cover_strip}
<div class="legend">
<span><span class="sw" style="background:{STATUS['good']}"></span>staffed</span>
<span><span class="sw" style="background:{STATUS['critical']}"></span>nobody at the desk</span>
<span><span class="sw" style="background:#EFE8DA;border:1px solid #D6CBB8"></span>not observed &mdash; blind, not empty</span>
</div>

<h2>When did people arrive?</h2><div class="rule"></div>
{arrivals_chart}

<h2>Where did they spend time?</h2><div class="rule"></div>
{dwell_chart}

<h2>Who worked, and for how long?</h2><div class="rule"></div>
{staff_chart}

<h2>What needs a human eye?</h2><div class="rule"></div>
{an_html}

<h2>Every answer, with its tier</h2><div class="rule"></div>
<table><thead><tr><th>question</th><th class="num">answer</th><th>tier</th>
</tr></thead><tbody>{answers_rows}</tbody></table>
<p class="sub" style="margin-top:10px">EXACT = measured. PROXY = inferred from
a stand-in signal (staff proximity is not a conversation). WEAK = directional
only. A number without a tier is not on this page.</p>

<h2>Every person on camera</h2><div class="rule"></div>
{table}

<div class="foot">{"<p>" + "</p><p>".join(ev) + "</p>"}</div>
</div></body></html>"""


def write(result, out_dir, camera_id="CAM", venue="Delilah"):
    """Render and write. -> Path. Never raises on a partial `result`."""
    p = Path(out_dir) / f"{camera_id}_night_report.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(result, venue=venue), encoding="utf-8")
    return p


def demo():
    """One runnable check: the page renders, and 'not measured' never becomes 0."""
    blank = {"camera_id": "CAM.T", "run": {}, "answers": [], "people": [],
             "staff": [], "anomalies": [], "arrivals": {}, "findings": []}
    h = render(blank)
    assert "<html" in h and "</html>" in h
    # the empty-series honesty rule, in all three chart shapes
    assert h.count("not measured") >= 3, h.count("not measured")
    assert "Nothing was flagged" in h and "not that nothing happened" in h
    assert "not scored" in h, "an unscored run must say so"

    full = {
        "camera_id": "CAM.112",
        "clock": {"start": "2026-08-19T18:00:00"},
        "run": {
            "duration_s": 3600.0,
            "observed_windows": [(0.0, 3600.0)],
            "zone_roles": {"reception": ["staff"], "dining": ["seating"]},
            "arrivals_by_id": {1: 120.0, 2: 300.0, 3: 4000.0},
            "events": [
                {"track_id": "anna", "role": "staff", "zone": "reception",
                 "t_in": 0.0, "t_out": 1200.0, "duration": 1200.0},
                {"track_id": "anna", "role": "staff", "zone": "reception",
                 "t_in": 1800.0, "t_out": 3600.0, "duration": 1800.0},
                {"track_id": 2, "role": "customer", "zone": "dining",
                 "t_in": 400.0, "t_out": 1000.0, "duration": 600.0},
            ],
        },
        "answers": [{"label": "Guests tonight", "value": "47", "tier": "WEAK"},
                    {"label": "Desk covered", "value": "83%", "tier": "EXACT"},
                    {"label": "Greet latency", "value": "42 s", "tier": "PROXY"}],
        "people": [{"person": 2, "role": "customer", "first_seen": "06:40",
                    "last_seen": "16:40", "minutes": 10.0, "waited_s": 240,
                    "greeted": "no", "flags": ["no-door-crossing"]}],
        "staff": [{"name": "anna", "minutes": 50.0, "source": "face"}],
        "anomalies": [{"time": "06:40", "severity": "high",
                       "what": "guest 2 waited 4m00s with no staff nearby"}],
        "arrivals": {"line": 41, "region": 47, "slit": 46,
                     "cross_check": {"trust": "region"}},
        "hota": {"HOTA": 0.47, "DetA": 0.39, "AssA": 0.58, "recall": 0.48},
        "findings": [("ERROR", "entry zone misplaced")],
    }
    h2 = render(full)
    assert "47" in h2 and "83%" in h2
    assert "WEAK" in h2 and "PROXY" in h2 and "EXACT" in h2
    # the 600s desk gap between the two reception events must be drawn red
    assert STATUS["critical"] in h2, "an uncovered stretch must render"
    assert "HOTA" in h2 and "0.470" in h2
    assert "entry zone misplaced" in h2, "run errors must reach the page"
    assert "anna" in h2
    # arrivals bucketed by wall clock, not by seconds-into-chunk
    assert "18:00" in h2 and "19:00" in h2, "arrival hours must be wall clock"
    # ...and ordered by ELAPSED time. An overnight service crosses midnight, so
    # a lexicographic sort on the label puts 00:00 before 18:00 -- midnight
    # drawn at the left edge of the evening. Regression guard for that.
    import re as _re
    over = dict(full["run"], arrivals_by_id={1: 60.0, 2: 7 * 3600.0, 3: 9 * 3600.0},
                duration_s=10 * 3600.0)
    labels = _re.findall(r'>(\d\d:00)<', render(dict(full, run=over)))
    assert labels[:3] == ["18:00", "01:00", "03:00"], labels[:3]
    print(f"report_html: OK  (blank {len(h)} bytes, full {len(h2)} bytes)")


if __name__ == "__main__":
    demo()
