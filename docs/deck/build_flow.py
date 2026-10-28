"""build_flow.py -- one node list -> keva_vision_flow.svg + keva_vision_flow.drawio

Both artefacts come from the SAME NODES/EDGES below, so the editable draw.io
file and the SVG embedded in the decks can never drift apart. Edit here, rerun.

Every label is traceable to the repo:
  stage names          kevacv/pipeline.py  `with stage(...)`
  publish gate         docs/SUCCESS_CRITERIA.md Q4 "never publishes if camera moved"
  device counts        docs/delilah_la_devices.json
"""
import html
import pathlib

W, H = 1640, 1140

import keva_theme as T

INK = "#" + T.SLATE
IVORY = "#" + T.FOG
BRASS = "#" + T.COPPER
MUTED = "#" + T.SAGE
LINE = "#" + T.HAIR
STOP = "#" + T.COPPER_DK   # warm, authoritative; ERROR red fights the palette
GO = "#" + T.OLIVE
LANE_BG = "#" + T.SLATE_LT

SERIF = "'Space Grotesk','Segoe UI',sans-serif"
MONO = "'JetBrains Mono',Consolas,monospace"

# (id, x, y, w, h, title, subtitle, kind)
NODES = [
    # -- lane A : the venue
    ("a1", 70, 128, 300, 96, "39 cameras + 3 door readers",
     "UniFi Protect · 44 devices · all online", "box"),
    ("a2", 440, 128, 250, 96, "UNVR Pro recorder",
     "10.0.14.10 · already installed", "box"),
    ("a3", 760, 128, 320, 96, "≈312 camera-hours a night",
     "nobody watches them", "alert"),

    # -- lane B : the pass, serpentine
    ("b1", 70, 336, 290, 104, "1 · PREFLIGHT",
     "Is this run trustworthy at all?", "stage"),
    ("b2", 420, 336, 290, 104, "2 · ANALYSE",
     "Find every person, follow them", "stage"),
    ("b3", 770, 336, 290, 104, "3 · PHANTOMS",
     "Drop reflections, statues, posters", "stage"),
    ("b4", 1120, 336, 290, 104, "4 · IDENTITY",
     "One person = one id", "stage"),
    ("b5", 1120, 530, 290, 104, "5 · ANSWERS",
     "4 questions, each with a tier", "stage"),
    ("d1", 610, 524, 340, 116, "Camera moved, or data invalid?",
     "", "gate"),
    ("b6", 70, 530, 290, 104, "6 · REPORT",
     "One page, published at 06:00", "stage"),
    ("stop", 1080, 664, 340, 96, "NOTHING IS PUBLISHED",
     "a wrong number is worse than none", "stop"),

    # -- lane C : what a human sees
    ("c1", 70, 852, 320, 108, "The morning report",
     "read in 2 minutes, not 8 hours", "out"),
    ("c2", 450, 852, 340, 108, "Anomalies to review",
     "only the deviations, with a timestamp", "out"),
    ("c3", 850, 852, 320, 108, "Annotated video",
     "jump to the moment worth watching", "out"),
    ("c4", 450, 1000, 720, 92, "Learned baseline · this venue\u2019s normal per zone, hour and weekday",
     "built from every night that came before", "learn"),
]

# (src, dst, label, style)  style: solid | dashed
EDGES = [
    ("a1", "a2", "", "solid"),
    ("a2", "a3", "", "solid"),
    ("a3", "b1", "every night", "solid"),
    ("b1", "b2", "", "solid"),
    ("b2", "b3", "", "solid"),
    ("b3", "b4", "", "solid"),
    ("b4", "b5", "", "solid"),
    ("b5", "d1", "", "solid"),
    ("d1", "stop", "YES", "solid"),
    ("d1", "b6", "NO", "solid"),
    ("b6", "c1", "", "solid"),
    ("b6", "c2", "", "solid"),
    ("b6", "c3", "", "solid"),
    ("c4", "c2", "scores tonight against every night before it", "dashed"),
]

LANES = [
    (96, 260, "THE VENUE", "hardware that is already in the ceiling"),
    (296, 776, "KEVA VISION", "the automated pass · kevacv/pipeline.py"),
    (812, 1112, "WHAT A HUMAN SEES", "the whole point"),
]

N = {n[0]: n for n in NODES}


def anchor(a, b):
    """Edge endpoints on the two boxes' facing sides."""
    _, ax, ay, aw, ah = a[:5]
    _, bx, by, bw, bh = b[:5]
    acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2, by + bh / 2
    if abs(bcy - acy) > max(ah, bh) * 0.6:          # mostly vertical
        return (acx, ay + ah if bcy > acy else ay,
                bcx, by if bcy > acy else by + bh)
    return (ax + aw if bcx > acx else ax, acy,      # mostly horizontal
            bx if bcx > acx else bx + bw, bcy)


# ---------------------------------------------------------------- SVG
def svg():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="{SERIF}">',
         '<defs>',
         f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         f'markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker>',
         f'<marker id="arD" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
         f'markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="{BRASS}"/></marker>',
         '</defs>',
         f'<rect width="{W}" height="{H}" fill="{IVORY}"/>']

    for y0, y1, name, sub in LANES:
        o.append(f'<rect x="24" y="{y0}" width="{W-48}" height="{y1-y0}" rx="10" '
                 f'fill="{LANE_BG}" stroke="{LINE}"/>')
        o.append(f'<text x="44" y="{y0+26}" font-size="15" letter-spacing="2.5" '
                 f'fill="{BRASS}" font-family="{MONO}">{html.escape(name)}</text>')
        o.append(f'<text x="{44+len(name)*13.5+26}" y="{y0+26}" font-size="14" '
                 f'fill="{MUTED}" font-style="italic">{html.escape(sub)}</text>')

    for src, dst, label, style in EDGES:
        x1, y1, x2, y2 = anchor(N[src], N[dst])
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
            d = f'M{x1} {y1} L{x1} {my} L{x2} {my} L{x2} {y2}'
        else:
            d = f'M{x1} {y1} L{x2} {y2}'
        col = BRASS if style == "dashed" else MUTED
        dash = ' stroke-dasharray="7 6"' if style == "dashed" else ''
        mk = 'arD' if style == "dashed" else 'ar'
        o.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"{dash} '
                 f'marker-end="url(#{mk})"/>')
        if label:
            o.append(f'<rect x="{mx-len(label)*3.6-8}" y="{my-13}" '
                     f'width="{len(label)*7.2+16}" height="22" rx="4" fill="{IVORY}"/>')
            o.append(f'<text x="{mx}" y="{my+3}" font-size="13" fill="{col}" '
                     f'text-anchor="middle" font-style="italic">{html.escape(label)}</text>')

    for _id, x, y, w, h, title, sub, kind in NODES:
        fill, stroke, tcol, sw = "#FFFFFF", LINE, INK, 1.5
        if kind == "stage":
            fill, stroke, sw = "#FFFFFF", BRASS, 2
        elif kind == "alert":
            fill, stroke, tcol = "#" + T.COPPER_LT, STOP, STOP
        elif kind == "stop":
            fill, stroke, tcol, sw = STOP, STOP, "#FFF7F5", 2
        elif kind == "out":
            fill, stroke, sw = "#FFFFFF", INK, 2
        elif kind == "learn":
            fill, stroke = "#" + T.COPPER_TINT, BRASS

        if kind == "gate":
            cx, cy = x + w / 2, y + h / 2
            o.append(f'<path d="M{cx} {y} L{x+w} {cy} L{cx} {y+h} L{x} {cy} z" '
                     f'fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
            o.append(f'<text x="{cx}" y="{cy+5}" font-size="15" fill="{INK}" '
                     f'text-anchor="middle">{html.escape(title)}</text>')
            continue

        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        fam = MONO if kind == "stage" else SERIF
        size = 16 if kind == "stage" else 19
        o.append(f'<text x="{x+18}" y="{y+ (40 if sub else h/2+6)}" font-size="{size}" '
                 f'fill="{tcol}" font-family="{fam}" letter-spacing="{0.5 if kind=="stage" else 0}">'
                 f'{html.escape(title)}</text>')
        if sub:
            o.append(f'<text x="{x+18}" y="{y+68}" font-size="14" fill="'
                     f'{"#FCEADB" if kind=="stop" else MUTED}" font-style="italic">'
                     f'{html.escape(sub)}</text>')

    o.append('</svg>')
    return "\n".join(o)


# ------------------------------------------------------------- draw.io
DRAWIO_STYLE = {
    "box": f"rounded=1;arcSize=8;fillColor=#FFFFFF;strokeColor={LINE};"
           f"fontColor={INK};fontFamily=Space Grotesk;fontSize=15;whiteSpace=wrap;html=1;",
    "alert": f"rounded=1;arcSize=8;fillColor=#FCEADB;strokeColor={STOP};"
             f"fontColor={STOP};fontFamily=Space Grotesk;fontSize=15;whiteSpace=wrap;html=1;",
    "stage": f"rounded=1;arcSize=8;fillColor=#FFFFFF;strokeColor={BRASS};strokeWidth=2;"
             f"fontColor={INK};fontFamily=JetBrains Mono;fontSize=14;whiteSpace=wrap;html=1;",
    "stop": f"rounded=1;arcSize=8;fillColor={STOP};strokeColor={STOP};"
            f"fontColor=#FFF7F5;fontFamily=Space Grotesk;fontSize=15;whiteSpace=wrap;html=1;",
    "out": f"rounded=1;arcSize=8;fillColor=#FFFFFF;strokeColor={INK};strokeWidth=2;"
           f"fontColor={INK};fontFamily=Space Grotesk;fontSize=15;whiteSpace=wrap;html=1;",
    "learn": f"rounded=1;arcSize=8;fillColor=#FDF5EF;strokeColor={BRASS};"
             f"fontColor={INK};fontFamily=Space Grotesk;fontSize=15;whiteSpace=wrap;html=1;",
    "gate": f"rhombus;fillColor=#FFFFFF;strokeColor={INK};strokeWidth=2;"
            f"fontColor={INK};fontFamily=Space Grotesk;fontSize=14;whiteSpace=wrap;html=1;",
}


def drawio():
    c = ['<mxfile host="app.diagrams.net">',
         '<diagram name="Keva Vision — how a night becomes a page">',
         '<mxGraphModel dx="1400" dy="900" grid="0" page="1" pageWidth="1640" '
         'pageHeight="940" math="0" shadow="0"><root>',
         '<mxCell id="0"/><mxCell id="1" parent="0"/>']

    for i, (y0, y1, name, sub) in enumerate(LANES):
        c.append(f'<mxCell id="lane{i}" value="{html.escape(name)} &#8212; '
                 f'{html.escape(sub)}" style="rounded=1;fillColor={LANE_BG};'
                 f'strokeColor={LINE};verticalAlign=top;align=left;spacingLeft=14;'
                 f'spacingTop=4;fontColor={BRASS};fontFamily=JetBrains Mono;fontSize=13;'
                 f'html=1;" vertex="1" parent="1">'
                 f'<mxGeometry x="24" y="{y0}" width="{W-48}" height="{y1-y0}" '
                 f'as="geometry"/></mxCell>')

    for _id, x, y, w, h, title, sub, kind in NODES:
        label = html.escape(title)
        if sub:
            label += f'<br><i style="color:{MUTED};font-size:12px">{html.escape(sub)}</i>'
        c.append(f'<mxCell id="{_id}" value="{label}" style="{DRAWIO_STYLE[kind]}" '
                 f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" '
                 f'height="{h}" as="geometry"/></mxCell>')

    for i, (src, dst, label, style) in enumerate(EDGES):
        col = BRASS if style == "dashed" else MUTED
        dash = "dashed=1;" if style == "dashed" else ""
        c.append(f'<mxCell id="e{i}" value="{html.escape(label)}" '
                 f'style="edgeStyle=orthogonalEdgeStyle;rounded=1;{dash}'
                 f'strokeColor={col};fontColor={col};fontSize=12;fontStyle=2;html=1;" '
                 f'edge="1" parent="1" source="{src}" target="{dst}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    c.append('</root></mxGraphModel></diagram></mxfile>')
    return "\n".join(c)


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    (here / "keva_vision_flow.svg").write_text(svg(), encoding="utf-8")
    (here / "keva_vision_flow.drawio").write_text(drawio(), encoding="utf-8")
    import cairosvg   # the decks embed the raster, so it is not a manual step
    cairosvg.svg2png(url=str(here / "keva_vision_flow.svg"),
                     write_to=str(here / "keva_vision_flow.png"), scale=2)

    # one runnable check: every edge endpoint must name a real node, and no two
    # boxes may overlap -- an overlap is invisible in the XML and obvious on stage.
    ids = {n[0] for n in NODES}
    for s, d, _, _ in EDGES:
        assert s in ids and d in ids, f"edge {s}->{d} names a missing node"
    for i, a in enumerate(NODES):
        for b in NODES[i + 1:]:
            ax, ay, aw, ah = a[1:5]
            bx, by, bw, bh = b[1:5]
            assert not (ax < bx + bw and bx < ax + aw
                        and ay < by + bh and by < ay + ah), \
                f"{a[0]} overlaps {b[0]}"
    print("wrote keva_vision_flow.svg + .drawio  · %d nodes, %d edges, checks pass"
          % (len(NODES), len(EDGES)))
