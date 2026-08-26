"""build_deck.py -- one SLIDES list -> Keva_Vision_CTO_COO.pptx + index.html

Audience: a CTO and a COO who do not read code.

EVERY NUMBER ON THESE SLIDES IS MEASURED. The source is the 1-hour Delilah run
in `keva_delilah_1hr/` (SUMMARY.txt, CAM.112_scorecard.json, people.csv) —
CAM.112 Reception, 2026-07-28, 19:30–20:30. Nothing is a mock-up any more; the
earlier illustrative report slide has been replaced by that run's real output.

The ONE blank left on purpose is the infrastructure figure on the last slide.
Fill it in yourself rather than letting me invent a number.

Livery is the Keva restaurant app's own — see keva_theme.py.
"""
import pathlib

import keva_theme as T

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent.parent

INK = T.SLATE
IVORY = T.FOG
BRASS = T.COPPER
MUTED = T.SAGE
LINE = T.HAIR
STOP = T.COPPER_DK
PAPER = T.PAPER
TINT = T.COPPER_TINT
OLIVE = T.OLIVE

SERIF = T.PPT_SANS          # the app is a grotesk, not a serif
MONO = T.PPT_MONO

VIDEO = "docs/deck/keva_vision_demo_clip.mp4"

SLIDES = [
    dict(kind="title",
         eyebrow="KEVA VISION · DELILAH LA",
         title="Nobody watches the cameras.",
         sub="What we are building, what it already measures, "
             "and what it will take to finish."),

    dict(kind="statement",
         eyebrow="THE PROBLEM",
         title="The problem is not the cameras.\nIt is the hours.",
         stats=[("39", "cameras at Delilah LA"),
                ("~8 h", "of service, every night"),
                ("≈312", "camera-hours a night")],
         body=[
             "Nobody can watch 312 hours. So in practice, nobody watches any of it.",
             "Footage gets opened only after something has already gone wrong. "
             "It works as evidence. It has never worked as management.",
             "The result: leadership learns what happened in the room from a "
             "complaint the next morning, not from the room itself.",
         ]),

    dict(kind="image",
         eyebrow="WHAT IS ALREADY IN THE CEILING",
         title="No new hardware for the pilot.",
         image="camera_wall.png",
         caption="Every device here is live today on the venue's own UniFi "
                 "Protect recorder. Keva Vision reads what is already "
                 "recording. CAM.112 — the reception camera — is the pilot."),

    dict(kind="table",
         eyebrow="WHAT IT ANSWERS",
         title="Four questions, and how sure we can honestly be.",
         head=["The question", "Why the floor cares",
               "What one hour of real footage gave us"],
         rows=[
             ["Was the desk covered?",
              "An unstaffed host stand is a guest who walks back out.",
              "97.0% covered, zero gaps. MEASURED — it only asks 'is anyone "
              "standing here', so it never depends on telling people apart."],
             ["How many guests?",
              "A number to reconcile against the POS.",
              "8 guests. Two independent methods — a door line and a floor "
              "region — both returned 8. An ESTIMATE that agreed with itself."],
             ["How fast were guests greeted?",
              "The first sixty seconds decide the review.",
              "3.4 s — but only 1 of the 8 was ever approached. "
              "This is the number we do not yet trust. See the honest slide."],
             ["What happened last night?",
              "One page, on the desk at 06:00.",
              "Published, with every figure carrying its own confidence "
              "label. A number without one does not go on the page."],
         ],
         foot="Targets are written down in docs/SUCCESS_CRITERIA.md, so "
              "'good enough' is a decision rather than a moving line."),

    dict(kind="full",
         eyebrow="HOW IT WORKS",
         title="One night in. One page out.",
         image="keva_vision_flow.png"),

    dict(kind="image",
         eyebrow="WHAT THE MACHINE IS LOOKING AT",
         title="The room, divided once.",
         image="frame_zones.png",
         caption="Reception · waiting area · dining · the entry line · the "
                 "main entrance. Drawn once per camera, then reused every "
                 "night. The plant is masked out — it moves, and moving "
                 "leaves look like people."),

    dict(kind="video",
         eyebrow="THE RESULT, PLAYED",
         title="Sixty seconds of it working.",
         file=VIDEO,
         meta="CAM.112 · 2026-07-28, 19:43–19:47 · 62 s · 4× speed "
              "(one hour of service renders to fifteen minutes)",
         look=[
             "Orange box = staff. Blue box = guest. The system works the role "
             "out from behaviour — it is not reading uniforms.",
             "Watch the names: 'staff1' and 'sarah' are face-matched against "
             "the staff photos. The rest stay anonymous ids on purpose.",
             "Bottom right, the entry-line counter ticks over as people cross. "
             "Crossing the line is what counts as an arrival; standing near "
             "the door does not.",
             "Watch an id stay with a person as they cross the room. That is "
             "the hard part, and it is what August was spent fixing.",
         ]),

    dict(kind="split",
         eyebrow="FROM PIXELS TO A SENTENCE",
         title="This is the raw material. Nobody will ever read it.",
         image="frame_routes.png",
         stats=[("193", "paths walked"), ("44", "came in"), ("23", "went out")],
         body=[
             "That is a single twenty-minute window with every path drawn on "
             "top of it. It is honest, complete, and completely useless to a "
             "manager.",
             "The whole job of Keva Vision is turning that picture into one "
             "sentence a GM can act on:",
             "“The reception desk held all hour, eight guests came through "
             "the door, and only one of them was ever approached.”",
         ]),

    dict(kind="imagelist",
         eyebrow="ONE HOUR, MEASURED",
         title="What the pilot actually produced.",
         image="report_answers.png",
         eye=[
             "Two independent counting methods — a line across the door and a "
             "region on the floor — both returned 8. In July they disagreed, "
             "2 against 6.",
             "Desk coverage is the number we are willing to promise on: it "
             "only asks whether anyone is standing there, so it never depends "
             "on telling people apart.",
             "Nothing was flagged for review — which means no rule fired, not "
             "that nothing happened. The page says so in those words.",
         ]),

    dict(kind="image",
         eyebrow="WHO ACTUALLY HELD THE DESK",
         title="Time at the desk, by person.",
         image="report_desk.png",
         caption="Five of these are real names, matched to staff photographs "
                 "by face. The rest are anonymous ids — and some of them are "
                 "almost certainly the same people, which is exactly why the "
                 "page says so instead of printing a staff headcount."),

    dict(kind="statement",
         eyebrow="ANOMALIES",
         title="We do not set thresholds.\nIt learns the venue.",
         stats=[("per zone", "reception, dining, entry"),
                ("per hour", "20:00 is not 23:00"),
                ("per weekday", "Tuesday is not Saturday")],
         body=[
             "The system builds a picture of what normal looks like for this "
             "room, at this hour, on this day of the week — from its own "
             "history.",
             "Then it raises only what genuinely deviates from that. Nobody "
             "hand-tunes a number, and nobody re-tunes it per venue.",
             "And it reports in the floor's language — “host stand unstaffed "
             "during the peak ramp”, not “zone 3 count below threshold”.",
         ]),

    dict(kind="three",
         eyebrow="WHERE WE ACTUALLY STAND",
         title="The honest slide.",
         cols=[
             ("PROVEN, MEASURED", [
                 "A full hour at Delilah ran end to end and published a "
                 "report: 97% desk coverage, zero gaps.",
                 "The door line and the floor region both counted 8 guests. "
                 "Two methods, one answer.",
                 "Identity churn is down to 56% of tracks from 83% in July, "
                 "after a single tuning change.",
                 "Face matching put real names — sarah, staff1 — against "
                 "time at the desk.",
             ]),
             ("KNOWN GAPS", [
                 "Greet time is the weak number: 3.4 s came from one guest "
                 "out of eight. The other seven were never approached, and "
                 "we cannot yet tell a real service failure from a "
                 "measurement one.",
                 "No hand-labelled footage, so the system still grades its "
                 "own homework. We will not call that accuracy.",
                 "107 grid cells inside a counted zone never produced a "
                 "single detection — either nobody stood there, or we are "
                 "blind there.",
             ]),
             ("WHAT CLOSES THEM", [
                 "A labelling pass — a person marks a few hours of real "
                 "footage. The single biggest credibility unlock.",
                 "Re-train the detector on our own frames rather than "
                 "generic ones.",
                 "A second angle at the entrance, which is what moves "
                 "counting from a range to a number.",
                 "A language model for the greet question — a different "
                 "tool for something vision alone cannot answer.",
             ]),
         ]),

    dict(kind="ask",
         eyebrow="WHAT WE NEED",
         title="Three decisions.",
         asks=[
             ("01", "Approval to run on live footage",
              "Move off saved clips and onto the nightly stream from the "
              "venue's own recorder. This is the gate that turns a "
              "demonstration into a service."),
             ("02", "Infrastructure budget",
              "A GPU machine to process a night's footage before morning. "
              "One hour of one camera took roughly an hour to process — "
              "figure attached separately."),
             ("03", "A second camera angle at the entrance",
              "The one thing software cannot substitute for. It is what "
              "moves guest counting from a range to a number."),
         ],
         foot="What comes back: a nightly one-page report, and an annotated "
              "video that jumps straight to the moment worth watching."),
]


# ------------------------------------------------------------------ PPTX
def build_pptx():
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    def C(h):
        return RGBColor.from_string(h)

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    W, H = 13.333, 7.5
    blank = prs.slide_layouts[6]

    def new(bg=IVORY):
        s = prs.slides.add_slide(blank)
        r = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
        r.fill.solid()
        r.fill.fore_color.rgb = C(bg)
        r.line.fill.background()
        r.shadow.inherit = False
        return s

    def text(s, body, x, y, w, h, size=18, color=INK, font=SERIF,
             bold=False, italic=False, align=PP_ALIGN.LEFT, space=1.25,
             spacing=None):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, ln in enumerate(body.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = space
            if spacing:
                p.space_after = Pt(spacing)
            r = p.add_run()
            r.text = ln
            f = r.font
            f.size, f.name, f.bold, f.italic = Pt(size), font, bold, italic
            f.color.rgb = C(color)
        return tb

    def eyebrow(s, t):
        text(s, t.upper(), 0.85, 0.6, 11, 0.4, size=11, color=BRASS,
             font=MONO)

    def rule(s, y, x=0.85, w=11.6, col=LINE):
        ln = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(0.012))
        ln.fill.solid()
        ln.fill.fore_color.rgb = C(col)
        ln.line.fill.background()
        ln.shadow.inherit = False

    def statrow(s, stats, y, size=40):
        x = 0.85
        for v, lab in stats:
            text(s, v, x, y, 3.4, 0.7, size=size, color=BRASS, font=MONO)
            text(s, lab, x, y + 0.62, 3.4, 0.5, size=13, color=MUTED,
                 italic=True)
            x += 3.7

    def picture(s, name, x, y, w=None, h=None):
        return s.shapes.add_picture(str(HERE / name), Inches(x), Inches(y),
                                    Inches(w) if w else None,
                                    Inches(h) if h else None)

    def card(s, x, y, w, h, fill=PAPER, border=LINE):
        r = s.shapes.add_shape(5, Inches(x), Inches(y), Inches(w), Inches(h))
        r.fill.solid()
        r.fill.fore_color.rgb = C(fill)
        r.line.color.rgb = C(border)
        r.line.width = Pt(1)
        r.shadow.inherit = False
        r.adjustments[0] = 0.03
        return r

    for sl in SLIDES:
        k = sl["kind"]

        if k == "title":
            s = new()
            card(s, 0, 0, 0.42, H, BRASS, BRASS)      # copper spine
            text(s, sl["eyebrow"], 1.25, 2.35, 11, 0.4, size=12, color=BRASS,
                 font=MONO)
            text(s, sl["title"], 1.25, 2.85, 11, 1.4, size=54, color=INK)
            text(s, sl["sub"], 1.25, 4.45, 9.2, 1.0, size=20, color=MUTED)
            rule(s, 5.8, 1.25, 3.0, BRASS)

        elif k == "statement":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.15, 11.6, 1.7, size=42, space=1.15)
            statrow(s, sl["stats"], 3.15)
            rule(s, 4.3)
            text(s, "\n".join(sl["body"]), 0.85, 4.65, 11.6, 2.4, size=17,
                 color=MUTED, space=1.35, spacing=13)

        elif k == "image":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.05, 11.6, 0.7, size=34)
            pic = picture(s, sl["image"], 0.85, 1.85, w=11.6)
            if pic.height / 914400 > 4.75:     # too tall: fit by height instead
                pic._element.getparent().remove(pic._element)
                pic = picture(s, sl["image"], 0.85, 1.85, h=4.75)
                pic.left = Inches((W - pic.width / 914400) / 2)
            text(s, sl["caption"], 0.85, 6.78, 11.6, 0.6, size=13,
                 color=MUTED, italic=True)

        elif k == "imagelist":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.0, 11.6, 0.7, size=34)
            pic = picture(s, sl["image"], 0.85, 1.8, w=11.6)
            if pic.height / 914400 > 3.3:      # leave room for the commentary
                pic._element.getparent().remove(pic._element)
                pic = picture(s, sl["image"], 0.85, 1.8, h=3.3)
                pic.left = Inches((W - pic.width / 914400) / 2)
            y = 1.8 + pic.height / 914400 + 0.42
            for ln in sl["eye"]:
                text(s, "—", 0.85, y, 0.4, 0.4, size=14, color=BRASS,
                     font=MONO)
                text(s, ln, 1.35, y, 11.1, 0.7, size=14, color=MUTED,
                     space=1.3)
                y += 0.24 * (1 + len(ln) // 96) + 0.38

        elif k == "full":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 0.92, 11.6, 0.6, size=30)
            pic = picture(s, sl["image"], 0.6, 1.52, h=5.75)
            pic.left = Inches((W - pic.width / 914400) / 2)

        elif k == "table":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.05, 11.6, 0.7, size=34)
            cw = [3.5, 3.6, 4.5]
            x = 0.85
            for i, hd in enumerate(sl["head"]):
                text(s, hd.upper(), x, 2.05, cw[i], 0.3, size=10.5,
                     color=BRASS, font=MONO)
                x += cw[i]
            rule(s, 2.42)
            y = 2.62
            for row in sl["rows"]:
                x = 0.85
                for i, cell in enumerate(row):
                    text(s, cell, x, y, cw[i] - 0.35, 1.0,
                         size=15 if i == 0 else 13,
                         color=INK if i == 0 else MUTED,
                         italic=(i == 2))
                    x += cw[i]
                y += 1.02
                rule(s, y - 0.16)
            text(s, sl["foot"], 0.85, y + 0.1, 11.6, 0.5, size=12.5,
                 color=MUTED, italic=True)

        elif k == "video":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.05, 11.6, 0.9, size=42)
            card(s, 0.85, 2.15, 11.6, 0.95, TINT, BRASS)
            text(s, "▶  " + sl["file"], 1.15, 2.35, 11.0, 0.4, size=19,
                 color=BRASS, font=MONO)
            text(s, sl["meta"], 1.15, 2.76, 11.0, 0.4, size=13, color=MUTED,
                 italic=True)
            text(s, "WHAT TO LOOK FOR", 0.85, 3.45, 11.6, 0.3, size=11,
                 color=BRASS, font=MONO)
            rule(s, 3.75)
            y = 3.95
            for i, ln in enumerate(sl["look"], 1):
                text(s, f"{i:02d}", 0.85, y, 0.6, 0.4, size=14, color=BRASS,
                     font=MONO)
                text(s, ln, 1.55, y, 10.9, 0.8, size=15, color=INK)
                y += 0.82

        elif k == "split":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.05, 11.6, 0.7, size=34)
            picture(s, sl["image"], 0.85, 2.0, w=6.6)
            x = 7.85
            yy = 2.0
            for v, lab in sl["stats"]:
                text(s, v, x, yy, 1.5, 0.5, size=30, color=BRASS,
                     font=MONO)
                text(s, lab, x + 1.5, yy + 0.14, 3.2, 0.4, size=13.5,
                     color=MUTED, italic=True)
                yy += 0.62
            rule(s, yy + 0.15, 7.85, 4.6)
            text(s, "\n".join(sl["body"]), 7.85, yy + 0.5, 4.6, 3.0, size=14,
                 color=MUTED, space=1.35, spacing=11)

        elif k == "report":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.05, 11.6, 0.7, size=34)
            text(s, sl["note"], 0.85, 1.85, 11.6, 0.4, size=13, color=MUTED,
                 italic=True)
            x = 0.85
            for v, lab, tier in sl["tiles"]:
                card(s, x, 2.6, 2.72, 1.5)
                text(s, v, x + 0.28, 2.82, 2.2, 0.6, size=34, color=INK,
                     font=MONO)
                text(s, lab, x + 0.28, 3.42, 2.2, 0.3, size=13, color=MUTED)
                text(s, tier.upper(), x + 0.28, 3.72, 2.2, 0.25, size=9.5,
                     color=BRASS, font=MONO)
                x += 2.96
            text(s, "WHAT NEEDS A HUMAN EYE", 0.85, 4.42, 11.6, 0.3, size=11,
                 color=BRASS, font=MONO)
            y = 4.8
            for ln in sl["eye"]:
                text(s, "—  " + ln, 0.85, y, 11.6, 0.4, size=14.5, color=INK)
                y += 0.46
            text(s, sl["foot"], 0.85, y + 0.16, 11.6, 0.6, size=13,
                 color=MUTED, italic=True)

        elif k == "three":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.05, 11.6, 0.7, size=34)
            x = 0.85
            for i, (head, items) in enumerate(sl["cols"]):
                col = BRASS if i == 0 else (STOP if i == 1 else INK)
                card(s, x, 1.95, 3.68, 4.85,
                     "FFFFFF" if i != 1 else "FBF3F0", LINE)
                text(s, head, x + 0.28, 2.18, 3.2, 0.3, size=10.5, color=col,
                     font=MONO)
                rule(s, 2.52, x + 0.28, 3.1, col)
                yy = 2.7
                for it in items:
                    lines = -(-len(it) // 36)          # ~36 chars fit at 12.5pt Georgia
                    text(s, it, x + 0.28, yy, 3.15, lines * 0.23, size=12.5,
                         color=MUTED, space=1.28)
                    yy += lines * 0.23 + 0.30
                x += 3.9
            rule(s, 6.98)
            text(s, "Every figure on the left is measured and reproducible "
                    "from the repository. Nothing on this slide is a projection.",
                 0.85, 7.08, 11.6, 0.4, size=12, color=MUTED, italic=True)

        elif k == "ask":
            s = new()
            eyebrow(s, sl["eyebrow"])
            text(s, sl["title"], 0.85, 1.05, 11.6, 0.9, size=42)
            y = 2.3
            for n, head, body in sl["asks"]:
                text(s, n, 0.85, y + 0.05, 0.8, 0.6, size=30, color=BRASS,
                     font=MONO)
                text(s, head, 1.85, y, 10.6, 0.5, size=23, color=INK)
                text(s, body, 1.85, y + 0.52, 10.4, 0.8, size=14, color=MUTED,
                     space=1.3)
                y += 1.42
                rule(s, y - 0.22)
            text(s, sl["foot"], 0.85, y + 0.05, 11.6, 0.6, size=15,
                 color=INK, italic=True)

    out = HERE / "Keva_Vision_CTO_COO.pptx"
    if out.exists() and (HERE / ("~$" + out.name)).exists():
        out = HERE / "Keva_Vision_CTO_COO_NEW.pptx"   # the original is open in PowerPoint
    prs.save(str(out))
    return out, len(prs.slides.__iter__.__self__._sldIdLst)


if __name__ == "__main__":
    path, n = build_pptx()
    # check: a slide that silently lost its content is the failure mode here,
    # so assert every slide carries shapes and the deck is the length we wrote.
    from pptx import Presentation
    p = Presentation(str(path))
    assert len(p.slides) == len(SLIDES), \
        f"{len(p.slides)} slides built, {len(SLIDES)} written"
    for i, s in enumerate(p.slides, 1):
        assert len(s.shapes) > 1, f"slide {i} is empty"
    print(f"wrote {path.name} · {len(p.slides)} slides")
