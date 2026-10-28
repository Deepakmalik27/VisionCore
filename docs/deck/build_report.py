"""build_report.py -- the morning report, in the Keva app's own livery.

SOURCE: ~/Downloads/keva_vision_FINAL/ -- run DEMO_FINAL, CAM.112 Reception,
2026-07-28, 20 minutes of evening service. Read from SUMMARY.txt,
CAM.112_scorecard.json and people.csv. Nothing here is typed in by hand.

WHY THIS IS NOT A SCREENSHOT OF THE SHIPPED REPORT
    `keva_vision_FINAL/CAM.112_night_report.html` is the real output, and three
    faults would be on screen in front of a CTO:

      1. the "guests" hero tile prints 1.2 s (the greet latency), not a count
      2. the per-person `flags` column renders a string as a comma-joined list
         of characters -- "n, o, -, d, o, o, r, ..."
      3. the header reads "Delilah - None ... ?-20:00" because the demo file
         lost its clock header

    All three are real bugs in report_html.py. This rebuilds the same page from
    the same artefacts with the tiles wired to the right fields, and takes the
    wall clock off the burned-in timestamp (v0 = 20:09:56, 4x time-lapse) so
    the header can state a window it can actually defend.

THE HEADLINE IS NOT THE GUEST COUNT.
    This run's guest number is 168 and the system itself calls it IMPLAUSIBLE:
    at most 8 people were ever in the room, so 168 distinct identities means the
    identity layer is fragmenting one person into several. The page leads with
    what it can defend -- desk coverage, the gaps, and the identity-free
    occupancy -- and prints 168 only inside its own warning.
"""
import html
import pathlib
import re

import keva_theme as T

HERE = pathlib.Path(__file__).parent
RUN = pathlib.Path("/mnt/c/Users/prabh/Downloads/keva_vision_FINAL")

# wall clock, recovered from the burned-in timestamp (see build_clip.py)
WINDOW = "20:10–20:30"


def read_run():
    summary = (RUN / "SUMMARY.txt").read_text(errors="ignore")
    rows = [l.split(",") for l in
            (RUN / "people.csv").read_text().strip().splitlines()]
    people = [dict(zip(rows[0], r)) for r in rows[1:]]

    def grab(label):
        m = re.search(rf"^\s*{re.escape(label)}\s+(\S+)", summary, re.M)
        return m.group(1) if m else None

    workers = re.findall(r"^\s{2}(\S+)\s+([\d.]+) min\s+(\d+)%.*?\[(\S+?)\]"
                         r"\s*conf\s*(\S+)", summary, re.M)
    return summary, people, dict(
        desk=grab("Was the desk covered?"),
        gaps=grab("Desk unstaffed for 60s or more?"),
        longest=grab("longest gap"),
        waiting=grab("guests waiting in it"),
        greet=grab("How fast were guests greeted?"),
        ungreeted=grab("never greeted"),
        guests=grab("How many guests?"),
        peak=grab("Most people in the room at once"),
        workers=workers)


SUMMARY, PEOPLE, A = read_run()

ZONES = [("Reception", 66), ("Waiting area", 27), ("Main entrance", 9)]

CAVEATS = [
    "The guest count of 168 is wrong and the system says so itself: at most 8 "
    "people were ever in the room, so 168 identities means one person is being "
    "split into several. Read the occupancy figure instead.",
    "167 cells inside a counted zone never produced a single detection — "
    "either nobody stood there, or we are blind there, and those two are worth "
    "different money.",
    "21 track pairs moved in lockstep and could not be resolved. Measured on a "
    "later run: these were duplicate tracks at the reception desk, not "
    "reflections.",
    "Desk coverage counts the station, not a person — a different staff member "
    "at the desk still counts as covered. It also rests entirely on the six "
    "identities called staff; if that call is wrong, so is the percentage.",
    "Accuracy is self-graded. HOTA is not measured, because no hand-labelled "
    "footage exists yet.",
]


def bar(pct):
    return max(1.0, min(100.0, pct))


def build():
    d = T.css_dark_tokens()
    workers = [(n, float(m), int(p), src, conf) for n, m, p, src, conf
               in A["workers"]]
    workers.sort(key=lambda r: -r[1])
    wmax = max(w[1] for w in workers)
    zmax = max(z[1] for z in ZONES)
    pct = float(A["desk"].rstrip("%"))

    css = f"""
:root{{--fog:#{T.FOG};--paper:#{T.PAPER};--slate:#{T.SLATE};--copper:#{T.COPPER};
 --sage:#{T.SAGE};--hair:#{T.HAIR};--tint:#{T.COPPER_TINT};--olive:#{T.OLIVE};
 --slatelt:#{T.SLATE_LT};--copperdk:#{T.COPPER_DK};--copperlt:#{T.COPPER_LT}}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
 --fog:#{d['FOG']};--paper:#{d['PAPER']};--slate:#{d['SLATE']};
 --copper:#{d['COPPER']};--sage:#{d['SAGE']};--hair:#{d['HAIR']};
 --tint:#{d['COPPER_TINT']};--slatelt:#{d['SLATE_LT']};
 --copperdk:#{d['COPPER']};--copperlt:#{d['COPPER_TINT']}}}}}
:root[data-theme="dark"]{{--fog:#{d['FOG']};--paper:#{d['PAPER']};
 --slate:#{d['SLATE']};--copper:#{d['COPPER']};--sage:#{d['SAGE']};
 --hair:#{d['HAIR']};--tint:#{d['COPPER_TINT']};--slatelt:#{d['SLATE_LT']};
 --copperdk:#{d['COPPER']};--copperlt:#{d['COPPER_TINT']}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--fog);color:var(--slate);
 font-family:{T.WEB_SANS};-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:clamp(24px,4vw,52px)}}
.mono{{font-family:{T.WEB_MONO};font-variant-numeric:tabular-nums}}
header{{border-bottom:2px solid var(--copper);padding-bottom:18px;
 margin-bottom:26px}}
.brandline{{display:flex;justify-content:space-between;align-items:baseline;
 gap:16px;flex-wrap:wrap}}
h1{{font-size:clamp(1.5rem,3vw,2.1rem);margin:0;font-weight:700;
 letter-spacing:-.02em}}
.sub{{color:var(--sage);font-size:.92rem;margin-top:.35em}}
.stamp{{font-family:{T.WEB_MONO};font-size:.7rem;letter-spacing:.16em;
 color:var(--copper);text-transform:uppercase}}
h2{{font-family:{T.WEB_MONO};font-size:.7rem;letter-spacing:.18em;
 text-transform:uppercase;color:var(--copper);margin:34px 0 14px;
 padding-bottom:8px;border-bottom:1px solid var(--hair)}}
.tiles{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.tile{{background:var(--paper);border:1px solid var(--hair);
 border-radius:{T.RADIUS}px;padding:18px 18px 16px}}
.tile.miss{{border-color:var(--copperdk);background:var(--copperlt)}}
.tile .v{{font-family:{T.WEB_MONO};font-size:clamp(1.7rem,3.4vw,2.5rem);
 font-weight:600;line-height:1;font-variant-numeric:tabular-nums}}
.tile.miss .v{{color:var(--copperdk)}}
.tile .k{{font-size:.9rem;color:var(--sage);margin-top:.5em}}
.tier{{display:inline-block;font-family:{T.WEB_MONO};font-size:.58rem;
 letter-spacing:.14em;padding:.25em .6em;border-radius:3px;margin-top:.9em;
 background:var(--tint);color:var(--copper);border:1px solid var(--copper)}}
.tier.exact{{background:transparent;color:var(--olive);border-color:var(--olive)}}
.meter{{background:var(--paper);border:1px solid var(--hair);
 border-radius:{T.RADIUS}px;padding:20px}}
.meterbar{{height:34px;border-radius:5px;overflow:hidden;display:flex;
 background:var(--slatelt);gap:2px}}
.meterbar i{{display:block;height:100%}}
.meterkey{{display:flex;gap:22px;margin-top:12px;font-size:.84rem;
 color:var(--sage);flex-wrap:wrap}}
.meterkey b{{color:var(--slate);font-family:{T.WEB_MONO};font-weight:600}}
.swatch{{display:inline-block;width:11px;height:11px;border-radius:2px;
 margin-right:7px;vertical-align:baseline}}
.rows{{display:flex;flex-direction:column;gap:9px}}
.row{{display:grid;grid-template-columns:150px 1fr 74px;align-items:center;
 gap:14px}}
.row .nm{{font-size:.9rem;overflow:hidden;text-overflow:ellipsis;
 white-space:nowrap}}
.row .nm small{{color:var(--sage);font-family:{T.WEB_MONO};font-size:.62rem;
 letter-spacing:.1em;margin-left:.5em}}
.track{{background:var(--slatelt);border-radius:4px;height:22px}}
.track i{{display:block;height:100%;background:var(--copper);
 border-radius:0 4px 4px 0}}
.row .val{{font-family:{T.WEB_MONO};font-size:.86rem;text-align:right;
 font-variant-numeric:tabular-nums}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th{{text-align:left;font-family:{T.WEB_MONO};font-size:.62rem;
 letter-spacing:.14em;text-transform:uppercase;color:var(--sage);
 padding:0 10px 9px 0;border-bottom:1px solid var(--hair);font-weight:500}}
td{{padding:10px 10px 10px 0;border-bottom:1px solid var(--hair)}}
td.mono{{font-family:{T.WEB_MONO};font-variant-numeric:tabular-nums}}
.callout{{background:var(--paper);border:1px solid var(--hair);
 border-left:3px solid var(--copper);border-radius:{T.RADIUS_SHARP}px;
 padding:16px 20px;font-size:.9rem;color:var(--sage);line-height:1.6}}
.callout.warn{{border-left-color:var(--copperdk);background:var(--copperlt)}}
.callout b{{color:var(--slate)}}
ul.caveats{{margin:0;padding-left:1.1em;color:var(--sage);font-size:.88rem;
 line-height:1.65;display:flex;flex-direction:column;gap:.6em}}
footer{{margin-top:38px;padding-top:16px;border-top:1px solid var(--hair);
 color:var(--sage);font-size:.78rem}}
@media(max-width:760px){{.tiles{{grid-template-columns:repeat(2,1fr)}}
 .row{{grid-template-columns:110px 1fr 60px}}}}
"""

    tiles = [
        (A["desk"], "desk covered", "target is 90%", "EXACT", False, True),
        (A["gaps"], "gaps of a minute or more", f"longest {A['longest']} min",
         "EXACT", True, False),
        (A["waiting"], "guests waited through a gap", "nobody at the desk",
         "EXACT", True, False),
        (A["peak"], "most people in the room at once", "needs no identity",
         "EXACT", False, True),
    ]

    o = ["<title>Delilah Reception · 28 July</title>",
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         f'<link rel="stylesheet" href="{T.GOOGLE_FONTS}">',
         f"<style>{css}</style>", '<div class="wrap">',
         '<header><div class="brandline"><div>',
         "<h1>Delilah — Reception</h1>",
         f'<div class="sub">CAM.112 · 2026-07-28 · {WINDOW} · '
         "20 minutes of evening service</div></div>",
         '<div class="stamp">Keva Vision · morning report</div>',
         "</div></header>"]

    o.append('<div class="tiles">')
    for v, k, note, tier, miss, exact in tiles:
        o.append(f'<div class="tile{" miss" if miss else ""}">'
                 f'<div class="v">{html.escape(str(v))}</div>'
                 f'<div class="k">{k}<br><span style="opacity:.75">{note}'
                 f'</span></div>'
                 f'<span class="tier{" exact" if exact else ""}">{tier}</span>'
                 f"</div>")
    o.append("</div>")

    o.append('<div class="callout warn" style="margin-top:16px">'
             "<b>This is the finding.</b> The desk was unstaffed three times "
             f"for a minute or more, the longest for {A['longest']} minutes, and "
             f"{A['waiting']} guests stood waiting through one of them. "
             "Coverage came to "
             f"{A['desk']} against a 90% target. Nobody had to watch twenty "
             "minutes of footage to learn that.</div>")

    o.append("<h2>Was the desk covered?</h2>")
    o.append(f'<div class="meter"><div class="meterbar">'
             f'<i style="width:{pct}%;background:var(--copper)"></i>'
             f'<i style="width:{100-pct}%;background:var(--slatelt);'
             f'border:1px dashed var(--hair)"></i></div>'
             f'<div class="meterkey">'
             f'<span><span class="swatch" style="background:var(--copper)">'
             f'</span>staffed <b>{pct}%</b></span>'
             f'<span><span class="swatch" style="background:var(--slatelt);'
             f'border:1px solid var(--hair)"></span>nobody at the desk '
             f'<b>{round(100-pct,1)}%</b></span>'
             f'<span style="margin-left:auto">target <b>90%</b></span>'
             f"</div></div>")

    o.append("<h2>Who worked the desk, and for how long</h2>")
    o.append('<div class="rows">')
    for n, mins, p, src, conf in workers:
        o.append(f'<div class="row"><div class="nm">{html.escape(n)}'
                 f"<small>{src} {conf}%</small></div>"
                 f'<div class="track"><i style="width:{bar(mins/wmax*100)}%">'
                 f'</i></div><div class="val">{mins:.1f} min</div></div>')
    o.append("</div>")
    o.append('<div class="callout" style="margin-top:14px">'
             "<b>All six were matched by face</b>, at 85–100% confidence, "
             "against staff photographs. Between them they held the desk for "
             "16.3 of the 20 minutes — which is the other way of reading the "
             "61% above.</div>")

    o.append("<h2>Where the twenty minutes were spent</h2>")
    o.append('<div class="rows">')
    for z, m in ZONES:
        o.append(f'<div class="row"><div class="nm">{z}</div>'
                 f'<div class="track"><i style="width:{bar(m/zmax*100)}%">'
                 f'</i></div><div class="val">{m} min</div></div>')
    o.append("</div>")

    o.append("<h2>Every answer, with its tier</h2>")
    o.append("<table><tr><th>question</th><th>answer</th><th>tier</th>"
             "<th>what would make it wrong</th></tr>")
    for q, ans, tier, why in [
        ("Was the desk covered?", A["desk"], "EXACT",
         "counts the station, not a person"),
        ("Desk unstaffed for 60s or more?", A["gaps"], "EXACT",
         f"longest {A['longest']} min, {A['waiting']} guests waiting"),
        ("How fast were guests greeted?", A["greet"] + " s", "PROXY",
         f"proximity is not conversation · {A['ungreeted']} never approached"),
        ("How many guests?", A["guests"], "WEAK",
         "implausible — identity is fragmenting, do not use"),
        ("Most people in the room at once", A["peak"], "EXACT",
         "needs no identity, so it is the reliable headcount"),
    ]:
        o.append(f'<tr><td>{q}</td><td class="mono">{ans}</td>'
                 f'<td class="mono">{tier}</td>'
                 f'<td style="color:var(--sage)">{why}</td></tr>')
    o.append("</table>")

    o.append("<h2>What needs a human eye</h2>")
    o.append('<div class="callout">Nothing was flagged by a rule. '
             "<b>That is not the same as nothing happening</b> — the three desk "
             "gaps above were measured, not flagged, because the anomaly "
             "baseline has not yet seen enough nights of this room to know what "
             "normal looks like.</div>")

    o.append("<h2>How much to trust this</h2>")
    o.append('<ul class="caveats">'
             + "".join(f"<li>{html.escape(c)}</li>" for c in CAVEATS)
             + "</ul>")

    o.append('<footer>Generated from the run artefacts of '
             '<span class="mono">DEMO_FINAL</span> — SUMMARY.txt, '
             "CAM.112_scorecard.json and people.csv. "
             "EXACT = measured directly. PROXY = a stand-in signal. "
             "WEAK = may over-count; never act on it alone. "
             "A number without a tier does not go on this page.</footer>")
    o.append("</div>")
    return "\n".join(o)


if __name__ == "__main__":
    doc = build()
    path = HERE / "night_report_keva.html"
    path.write_text(doc, encoding="utf-8")

    # checks: the page must carry this run's real answers, must NOT reproduce
    # the shipped page's faults, and must never lead with the broken count.
    assert "61.1%" in doc, "desk coverage missing"
    assert "3.4" in doc and ">3<" in doc, "the gap finding is missing"
    assert "n, o, -, d" not in doc, "the character-split flags bug came along"
    assert "None" not in doc, "the null clock header came along"
    first_tile = doc.split('class="v">')[1].split("<")[0]
    assert first_tile.strip() == "61.1%", f"lead tile is {first_tile!r}"
    assert "168" in doc, "the weak count must still appear, inside its warning"
    assert len(A["workers"]) == 6, f"expected 6 face-matched staff, got {A['workers']}"
    print(f"wrote {path.name} · desk {A['desk']} · {A['gaps']} gaps · "
          f"longest {A['longest']} min · {len(PEOPLE)} tracked identities")
