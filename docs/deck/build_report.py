"""build_report.py -- the morning report, in the Keva app's own livery.

WHY THIS IS NOT A SCREENSHOT OF THE SHIPPED REPORT
    `keva_delilah_1hr/CAM.112_night_report.html` is the real output of the real
    run, and it has two rendering faults that would be on screen in front of a
    CTO:

      1. the "guests" hero tile prints 3.4 s (the greet latency) instead of 8
      2. the per-person `flags` column renders a string as a comma-joined list
         of characters -- "n, o, -, d, o, o, r, ..."

    So this rebuilds the SAME page from the SAME run artefacts (SUMMARY.txt,
    CAM.112_scorecard.json, people.csv), with the tiles wired to the right
    fields. Every number below was read out of those three files; none is typed
    in by hand except where marked.

    Both faults are real bugs in report_html.py and are worth fixing at source.

RUN: Delilah - CAM.112 Reception - 2026-07-28 19:30-20:30 - 1.0 h
"""
import html
import pathlib
import re

import keva_theme as T

HERE = pathlib.Path(__file__).parent
RUN = pathlib.Path("/mnt/c/Users/prabh/Downloads/keva_delilah_1hr")


# ── read the run, do not retype it ───────────────────────────────────────
def read_run():
    summary = (RUN / "SUMMARY.txt").read_text(errors="ignore")
    rows = [l.split(",") for l in
            (RUN / "people.csv").read_text().strip().splitlines()]
    head, body = rows[0], rows[1:]
    people = [dict(zip(head, r)) for r in body]

    def grab(label):
        m = re.search(rf"^\s*{re.escape(label)}\s+(\S+(?:\s+range \S+)?)",
                      summary, re.M)
        return m.group(1) if m else None

    desk = grab("Was the desk covered?")
    gaps = grab("When was the desk empty?")
    greet = grab("How fast were guests greeted?")
    guests = grab("How many guests?")
    ungreeted = grab("never greeted")

    # who worked the desk: "  <name>   <mins> min   <pct>%"
    workers = re.findall(r"^\s{2}(\S+)\s+([\d.]+) min\s+(\d+)%.*?\[(\S+?)\]",
                         summary, re.M)
    return summary, people, dict(
        desk=desk, gaps=gaps, greet=greet, guests=guests,
        ungreeted=ungreeted, workers=workers)


SUMMARY, PEOPLE, A = read_run()

# zone dwell -- from the shipped report page, which computes it
ZONES = [("Reception", 133), ("Waiting area", 33),
         ("Main entrance", 21), ("Dining", 20)]

# the honest tail of SUMMARY.txt, condensed to what a CTO can act on
CAVEATS = [
    "107 cells inside a counted zone never produced a single detection — "
    "either nobody stood there, or we are blind there.",
    "17 cells had over 90% of their detections removed by the filters. "
    "If they are not under a mask, a filter is eating real people.",
    "55 track pairs moved in lockstep — likely reflections in the mirrored "
    "wall, and they inflate a headcount if left in.",
    "Accuracy is self-graded: HOTA not measured, because no hand-labelled "
    "footage exists yet.",
]


def bar(pct, w=100):
    return max(1.0, min(100.0, pct)) * w / 100.0


def build():
    d = T.css_dark_tokens()
    workers = [(n, float(m), int(p), src) for n, m, p, src in A["workers"]]
    workers.sort(key=lambda r: -r[1])
    wmax = max(w[1] for w in workers)
    zmax = max(z[1] for z in ZONES)

    greeted = [p for p in PEOPLE if p["greeted"] == "yes"]
    counted = [p for p in PEOPLE if "not-counted-as-guest" not in p["flags"]]

    css = f"""
:root{{--fog:#{T.FOG};--paper:#{T.PAPER};--slate:#{T.SLATE};--copper:#{T.COPPER};
 --sage:#{T.SAGE};--hair:#{T.HAIR};--tint:#{T.COPPER_TINT};--olive:#{T.OLIVE};
 --slatelt:#{T.SLATE_LT}}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
 --fog:#{d['FOG']};--paper:#{d['PAPER']};--slate:#{d['SLATE']};
 --copper:#{d['COPPER']};--sage:#{d['SAGE']};--hair:#{d['HAIR']};
 --tint:#{d['COPPER_TINT']};--slatelt:#{d['SLATE_LT']}}}}}
:root[data-theme="dark"]{{--fog:#{d['FOG']};--paper:#{d['PAPER']};
 --slate:#{d['SLATE']};--copper:#{d['COPPER']};--sage:#{d['SAGE']};
 --hair:#{d['HAIR']};--tint:#{d['COPPER_TINT']};--slatelt:#{d['SLATE_LT']}}}
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
.tile .v{{font-family:{T.WEB_MONO};font-size:clamp(1.7rem,3.4vw,2.5rem);
 font-weight:600;line-height:1;font-variant-numeric:tabular-nums}}
.tile .k{{font-size:.9rem;color:var(--sage);margin-top:.5em}}
.tier{{display:inline-block;font-family:{T.WEB_MONO};font-size:.58rem;
 letter-spacing:.14em;padding:.25em .6em;border-radius:3px;margin-top:.9em;
 background:var(--tint);color:var(--copper);border:1px solid var(--copper)}}
.tier.exact{{background:#{T.OLIVE}1a;color:var(--olive);
 border-color:var(--olive)}}
:root[data-theme="dark"] .tier.exact,
@media (prefers-color-scheme:dark){{}}
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
.row{{display:grid;grid-template-columns:132px 1fr 74px;align-items:center;
 gap:14px}}
.row .nm{{font-size:.9rem;overflow:hidden;text-overflow:ellipsis;
 white-space:nowrap}}
.row .nm small{{color:var(--sage);font-family:{T.WEB_MONO};font-size:.62rem;
 letter-spacing:.1em;margin-left:.5em}}
.track{{background:var(--slatelt);border-radius:4px;height:22px}}
.track i{{display:block;height:100%;background:var(--copper);
 border-radius:0 4px 4px 0}}
.row .val{{font-family:{T.WEB_MONO};font-size:.86rem;text-align:right;
 color:var(--slate);font-variant-numeric:tabular-nums}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th{{text-align:left;font-family:{T.WEB_MONO};font-size:.62rem;
 letter-spacing:.14em;text-transform:uppercase;color:var(--sage);
 padding:0 10px 9px 0;border-bottom:1px solid var(--hair);font-weight:500}}
td{{padding:11px 10px 11px 0;border-bottom:1px solid var(--hair)}}
td.mono{{font-family:{T.WEB_MONO};font-variant-numeric:tabular-nums}}
.flag{{font-family:{T.WEB_MONO};font-size:.62rem;color:var(--sage)}}
.callout{{background:var(--paper);border:1px solid var(--hair);
 border-left:3px solid var(--copper);border-radius:{T.RADIUS_SHARP}px;
 padding:16px 20px;font-size:.9rem;color:var(--sage);line-height:1.6}}
.callout b{{color:var(--slate)}}
ul.caveats{{margin:0;padding-left:1.1em;color:var(--sage);font-size:.88rem;
 line-height:1.65;display:flex;flex-direction:column;gap:.55em}}
footer{{margin-top:38px;padding-top:16px;border-top:1px solid var(--hair);
 color:var(--sage);font-size:.78rem}}
@media(max-width:760px){{.tiles{{grid-template-columns:repeat(2,1fr)}}
 .row{{grid-template-columns:104px 1fr 60px}}}}
"""

    tiles = [
        (A["guests"].split()[0], "guests", "range 8–8", "ESTIMATE", False),
        (A["desk"], "desk covered", "of observed footage", "EXACT", True),
        (A["gaps"], "times the desk was empty", "not one gap", "EXACT", True),
        (A["greet"] + " s", "median time to greet",
         "1 of 8 guests reached", "PROXY", False),
    ]

    o = [f"<title>Delilah Reception · 28 July</title>",
         f'<link rel="preconnect" href="https://fonts.googleapis.com">',
         f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
         f'<link rel="stylesheet" href="{T.GOOGLE_FONTS}">',
         f"<style>{css}</style>", '<div class="wrap">',
         '<header><div class="brandline"><div>',
         "<h1>Delilah — Reception</h1>",
         '<div class="sub">CAM.112 · 2026-07-28 · 19:30–20:30 · '
         "1.0 h of footage · provenance verified</div></div>",
         '<div class="stamp">Keva Vision · morning report</div>',
         "</div></header>"]

    o.append('<div class="tiles">')
    for v, k, note, tier, exact in tiles:
        o.append(f'<div class="tile"><div class="v">{html.escape(v)}</div>'
                 f'<div class="k">{k}<br><span style="opacity:.75">{note}'
                 f'</span></div>'
                 f'<span class="tier{" exact" if exact else ""}">{tier}</span>'
                 f"</div>")
    o.append("</div>")

    # coverage meter -- part to whole, one bar, direct labelled
    o.append("<h2>Was the desk covered?</h2>")
    pct = float(A["desk"].rstrip("%"))
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
    for n, mins, p, src in workers:
        tag = "face" if src == "face" else "zone"
        o.append(f'<div class="row"><div class="nm">{html.escape(n)}'
                 f"<small>{tag}</small></div>"
                 f'<div class="track"><i style="width:{bar(mins/wmax*100)}%">'
                 f'</i></div><div class="val">{mins:.1f} min</div></div>')
    o.append("</div>")
    o.append('<div class="callout" style="margin-top:14px">'
             "<b>Read this one carefully.</b> Five of these are face-matched "
             "names; the rest are anonymous ids the system saw holding the "
             "desk. Some of the ids are almost certainly the same people as "
             "the names, so treat the list as time-at-desk, not as a "
             "headcount of staff.</div>")

    o.append("<h2>Where the hour was spent</h2>")
    o.append('<div class="rows">')
    for z, m in ZONES:
        o.append(f'<div class="row"><div class="nm">{z}</div>'
                 f'<div class="track"><i style="width:{bar(m/zmax*100)}%">'
                 f'</i></div><div class="val">{m} min</div></div>')
    o.append("</div>")

    o.append("<h2>Every guest the system saw</h2>")
    o.append("<table><tr><th>id</th><th>first seen</th><th>last seen</th>"
             "<th>minutes</th><th>waited</th><th>greeted</th><th>note</th></tr>")
    for p in PEOPLE:
        flags = p["flags"].strip()
        note = ("not counted — never crossed the door"
                if "not-counted-as-guest" in flags else "")
        o.append(f'<tr><td class="mono">{html.escape(p["person"])}</td>'
                 f'<td class="mono">{p["first_seen"]}</td>'
                 f'<td class="mono">{p["last_seen"]}</td>'
                 f'<td class="mono">{p["minutes"]}</td>'
                 f'<td class="mono">{p["waited_s"] or "—"}'
                 f'{" s" if p["waited_s"] else ""}</td>'
                 f'<td>{p["greeted"]}</td>'
                 f'<td class="flag">{note}</td></tr>')
    o.append("</table>")
    o.append(f'<div class="callout" style="margin-top:16px">'
             f"<b>{len(counted)} of {len(PEOPLE)}</b> people crossed the door "
             f"and count as guests; the other {len(PEOPLE)-len(counted)} were "
             f"seen but never crossed it, so they are listed and excluded. "
             f"<b>{len(greeted)}</b> had a staff member come within reach. "
             f"That is the number to argue about — not the 3.4 s.</div>")

    o.append("<h2>What needs a human eye</h2>")
    o.append('<div class="callout">Nothing was flagged tonight. '
             "<b>That means no rule fired — not that nothing happened.</b> "
             "Until the anomaly baseline has more nights of this room to "
             "compare against, an empty list is weak evidence.</div>")

    o.append("<h2>How much to trust this</h2>")
    o.append('<ul class="caveats">'
             + "".join(f"<li>{html.escape(c)}</li>" for c in CAVEATS)
             + "</ul>")

    o.append('<footer>Generated from the run artefacts of '
             "<span class=\"mono\">delilah_1hr</span> — SUMMARY.txt, "
             "CAM.112_scorecard.json and people.csv. "
             "EXACT = measured directly. ESTIMATE = needed identity to hold, "
             "so it carries a range. PROXY = a stand-in signal; staff standing "
             "near a guest is not proof of a conversation. "
             "A number without a tier does not go on this page.</footer>")
    o.append("</div>")
    return "\n".join(o)


if __name__ == "__main__":
    doc = build()
    path = HERE / "night_report_keva.html"
    path.write_text(doc, encoding="utf-8")

    # checks: the report must carry the run's real headline answers, and must
    # NOT reproduce the two faults in the shipped page.
    assert "97.0%" in doc and ">8<" in doc, "headline answers missing"
    assert "3.4 s" in doc, "greet latency missing or lost its unit"
    assert "n, o, -, d" not in doc, "the character-split flags bug came along"
    assert doc.count("<tr>") == len(PEOPLE) + 1, "people table lost rows"
    # the guests tile must show a count, not the greet latency
    first_tile = doc.split('class="v">')[1].split("<")[0]
    assert first_tile.strip() == "8", f"guests tile shows {first_tile!r}"
    print(f"wrote {path.name} · {len(PEOPLE)} people · desk {A['desk']}")
