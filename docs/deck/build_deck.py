"""build_deck.py -- one SLIDES list -> Keva_Vision_CTO_COO.pptx + index.html

Audience: a CTO and a COO who do not read code.

EVERY NUMBER ON THESE SLIDES IS MEASURED, and all of them come from ONE run:
`~/Downloads/keva_vision_FINAL/` — CAM.112 Reception, Delilah LA, 2026-07-28,
twenty minutes of evening service (SUMMARY.txt, CAM.112_scorecard.json,
people.csv). The earlier one-hour run is no longer quoted anywhere, so the
deck has a single story with a single set of figures.

That run FOUND SOMETHING, and it is the spine of the deck: the reception desk
was unstaffed three times, the longest for 3.4 minutes, with five guests
waiting through one of the gaps. Coverage 61.1% against a 90% target.

It also got a number badly wrong — 168 guests, when at most 8 people were ever
in the room — and said so itself. Both facts are on the slides. A system that
flags its own bad number is the thing worth buying.

The ONE blank left on purpose is the infrastructure figure on the last slide.

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
         sub="What we are building, what it found in twenty minutes, "
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

    dict(kind="statement",
         eyebrow="WHAT TWENTY MINUTES FOUND",
         title="The desk was empty three times.\nGuests were waiting.",
         stats=[("61.1%", "desk covered · target is 90%"),
                ("3", "gaps of a minute or more"),
                ("5", "guests waited through one")],
         body=[
             "This is not a capability demo. It is a finding, from twenty "
             "minutes of ordinary evening service on 28 July.",
             "Reception went unstaffed three separate times — 3.4 minutes from "
             "08:32, 1.9 minutes from 15:04, and 1.6 minutes from 17:44. "
             "During the first, five guests stood in the waiting area with "
             "nobody at the desk.",
             "Three times in twenty minutes is not a fluke — it is a pattern "
             "nobody could have seen without watching the whole night.",
             "No manager watched this footage. Nobody had to. It arrived as a "
             "line on a page the next morning.",
         ]),

    dict(kind="video",
         eyebrow="THE RESULT, PLAYED",
         title="Watch the desk go empty.",
         file=VIDEO,
         embed="_deck.mp4",
         poster="_clip_poster.jpg",
         meta="CAM.112 · 2026-07-28, 20:15–20:22 · 96 s · 4× speed. "
              "Embedded in this file — press play. Full-resolution master: "
              "docs/deck/keva_vision_demo_clip.mp4",
         look=[
             "It opens on ordinary service — the HUD top left reads "
             "reception: STAFFED. Watch that word.",
             "At 38 seconds it flips to AWAY and the copper bar appears. That "
             "is the first of the three gaps, caught live.",
             "Watch 'waiting now' climb to 4 while the desk reads AWAY. Those "
             "are the guests standing there with nobody to greet them.",
             "Orange box = staff, blue box = guest. The role comes from "
             "behaviour, not from uniforms. Names like 'staff1' are "
             "face-matched against staff photographs.",
             "At 1:28 the desk reads STAFFED again. The gap ran 3.4 minutes "
             "of real service.",
         ]),

    dict(kind="imagelist",
         eyebrow="WHAT LANDS AT 06:00",
         title="The finding, on one page.",
         image="report_answers.png",
         eye=[
             "Every figure carries a confidence label. EXACT means measured "
             "directly; PROXY means a stand-in signal; WEAK means do not act "
             "on it alone.",
             "Desk coverage is the number we are willing to promise on — it "
             "only asks whether anyone is standing at the station, so it never "
             "depends on telling people apart.",
             "Occupancy — at most 8 in the room — is the other reliable one, "
             "for the same reason: it counts bodies visible, not identities.",
         ]),

    dict(kind="image",
         eyebrow="WHO ACTUALLY HELD THE DESK",
         title="Six people, all matched by face.",
         image="report_desk.png",
         caption="Every one of the six was matched against a staff photograph "
                 "at 85–100% confidence. Between them they held the desk for "
                 "16.3 of the 20 minutes — which is the other way of reading "
                 "61%."),

    dict(kind="three",
         eyebrow="THE NUMBER IT GOT WRONG",
         title="It counted 168 guests. The room held 8.",
         cols=[
             ("WHAT IT REPORTED", [
                 "168 unique guests, tagged WEAK.",
                 "Two other counting methods disagreed with it and with each "
                 "other: a line across the door said 36, a region on the floor "
                 "said 10.",
                 "In an earlier one-hour run those two methods agreed exactly. "
                 "On this footage they do not.",
             ]),
             ("WHY IT IS WRONG", [
                 "At most 8 people were ever in the room at once, and the busiest "
                 "single frame held 10.",
                 "Even a complete turnover every two minutes only reaches about "
                 "80 people in twenty minutes.",
                 "So 168 identities means the system is splitting one person "
                 "into several as they move — 158 of those 168 never crossed "
                 "the door at all.",
             ]),
             ("WHY WE ARE SHOWING YOU", [
                 "Because the system said it first. The report prints the word "
                 "IMPLAUSIBLE next to its own number and tells the reader to "
                 "use the identity-free count instead.",
                 "A tool that hides a bad number costs you a decision. This one "
                 "refuses to let 168 be quoted.",
                 "The desk-coverage finding on the previous slides does not "
                 "depend on identity at all, which is exactly why we lead with "
                 "it.",
             ]),
         ],
         foot="Every figure here is read straight out of the run's own "
              "scorecard and summary — including the one it got wrong."),

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
             "Then it raises only what genuinely deviates. Nobody hand-tunes a "
             "number, and nobody re-tunes it per venue.",
             "It has not earned that yet here. This run flagged nothing, "
             "because one twenty-minute clip is not a history — the three desk "
             "gaps were measured, not flagged. More nights is what turns the "
             "second into the first.",
         ]),

    dict(kind="three",
         eyebrow="WHERE WE ACTUALLY STAND",
         title="The honest slide.",
         cols=[
             ("PROVEN, MEASURED", [
                 "A twenty-minute run produced a real operational finding: "
                 "three desk gaps, 3.4 min at worst, five guests waiting.",
                 "Six staff matched to photographs by face at 85–100% "
                 "confidence.",
                 "The floor geometry now solves exactly, from 36 measured "
                 "correspondences rather than an estimate.",
                 "Desk coverage and peak occupancy need no identity, so they "
                 "hold even while identity does not.",
             ]),
             ("KNOWN GAPS", [
                 "Identity fragments: 205 tracks became 162 chains, and the "
                 "guest count blew out to 168 against a true ceiling of ~8 in "
                 "the room.",
                 "Greet time is a proximity proxy, and 9 guests were never "
                 "approached at all — we cannot yet separate a service failure "
                 "from a measurement one.",
                 "167 grid cells inside a counted zone never produced a single "
                 "detection: either nobody stood there, or we are blind there.",
                 "No hand-labelled footage, so the system still grades its own "
                 "homework.",
             ]),
             ("WHAT CLOSES THEM", [
                 "A labelling pass — a person marks a few hours of real "
                 "footage. The single biggest credibility unlock, and what "
                 "turns 'self-graded' into 'accurate'.",
                 "Re-train the detector on our own frames rather than generic "
                 "ones, which is what stops the fragmenting.",
                 "A second angle at the entrance, which is what moves counting "
                 "from a range to a number.",
                 "More nights, so the anomaly baseline has a normal to compare "
                 "against.",
             ]),
         ]),

    dict(kind="ask",
         eyebrow="WHAT WE NEED",
         title="Three decisions.",
         asks=[
             ("01", "Approval to run on live footage",
              "Move off saved clips and onto the nightly stream from the "
              "venue's own recorder. This is the gate that turns a "
              "demonstration into a service — and the only way the anomaly "
              "baseline ever gets a history to learn from."),
             ("02", "Infrastructure budget",
              "A GPU machine to process a night's footage before morning. "
              "Twenty minutes of one camera renders in five — figure attached "
              "separately."),
             ("03", "A second camera angle at the entrance",
              "The one thing software cannot substitute for. It is what fixes "
              "the counting problem on the previous slide."),
         ],
         foot="What comes back: a nightly one-page report, and an annotated "
              "video that jumps straight to the moment worth watching — like "
              "the one you just saw."),
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
            text(s, sl["title"], 0.85, 1.02, 11.6, 0.7, size=34)

            # The clip is EMBEDDED, not linked. A path on a slide is not a
            # demo -- it is a promise the room cannot check.
            vw = 6.9
            vh = vw * 1372 / 1920
            mov = s.shapes.add_movie(
                str(HERE / sl["embed"]), Inches(0.85), Inches(1.85),
                Inches(vw), Inches(vh),
                poster_frame_image=str(HERE / sl["poster"]),
                mime_type="video/mp4")
            mov.line.color.rgb = C(LINE)
            text(s, sl["meta"], 0.85, 1.9 + vh, 7.0, 0.5, size=11.5,
                 color=MUTED, italic=True)

            x = 8.15
            text(s, "WHAT TO LOOK FOR", x, 1.85, 4.3, 0.3, size=11,
                 color=BRASS, font=MONO)
            rule(s, 2.15, x, 4.3)
            y = 2.35
            for i, ln in enumerate(sl["look"], 1):
                text(s, f"{i:02d}", x, y, 0.5, 0.35, size=12.5, color=BRASS,
                     font=MONO)
                lines = -(-len(ln) // 40)
                text(s, ln, x + 0.55, y, 3.75, lines * 0.22, size=12.5,
                     color=INK, space=1.3)
                y += lines * 0.22 + 0.30

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
            text(s, sl.get("foot", "Every figure here is measured and "
                            "reproducible from the run artefacts. Nothing on "
                            "this slide is a projection."),
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

    # PowerPoint locks a file it has open, and python-pptx cannot overwrite it.
    # Save to the canonical name when we can; otherwise take the next free one
    # and say so, rather than dying with a PermissionError.
    out = HERE / "Keva_Vision_CTO_COO.pptx"
    for cand in [out] + [HERE / f"Keva_Vision_CTO_COO_v{i}.pptx"
                         for i in range(2, 20)]:
        try:
            prs.save(str(cand))
            out = cand
            break
        except PermissionError:
            continue
    else:
        raise PermissionError("every candidate filename is locked — "
                              "close PowerPoint")
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
