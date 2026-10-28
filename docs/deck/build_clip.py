"""build_clip.py -- cut the demo clip out of the FINAL 20-minute render and
frame it with a Keva-liveried header, legend, and a gap alert.

SOURCE  ~/Downloads/keva_vision_FINAL/keva_vision_20min_FINAL.mp4
        18,000 frames covering 1,200 s of footage, played at 60 fps.
        A 4x time-lapse: video time x 4 = footage time. Verified, not assumed --
        the burned-in clock reads 20:17:48 at v118 and 20:22:20 at v186, so
        68 s of video carried 272 s of service.

        The file's own header lost its clock (SUMMARY.txt prints "None -> None"),
        so the wall times here are read off the burned-in timestamp instead:
        v0 = 20:09:56, so the run covers roughly 20:10-20:30.

WHY THIS WINDOW (video 90-186 s)
    The whole file was scanned for the HUD's `reception: STAFFED / AWAY` field
    by measuring the rendered width of that word every two seconds. The desk
    state over the full twenty minutes:

        v  4-128   STAFFED   8.3 min
        v128-178   AWAY      3.3 min   <- GAP 1, the "longest 3.4 min" in SUMMARY
        v186-196   AWAY      0.7 min
        v196-226   STAFFED   2.0 min
        v226-254   AWAY      1.9 min   <- GAP 2
        v266-290   AWAY      1.6 min   <- GAP 3

    Three gaps of a minute or more, which independently confirms SUMMARY.txt's
    "Desk unstaffed for 60s or more? 3".

    The clip covers GAP 1 with 38 s of ordinary staffed service in front of it,
    so the room sees normal before it sees the break. Starting later (v118) was
    too abrupt; starting earlier only adds uneventful footage, because v4-v128
    is one unbroken staffed run.

SPEED IS LEFT ALONE. The burned-in HUD says "TIME-LAPSE 4.0x"; re-timing would
leave that label lying on screen.

Box colours in the legend come from kevacv/helpers.py:57, not guesswork:
    customer #2a78d6   staff #eb6834   unknown #898781
"""
import pathlib
import subprocess

from PIL import Image, ImageDraw, ImageFont

import keva_theme as T

HERE = pathlib.Path(__file__).parent
SRC = pathlib.Path("/mnt/c/Users/prabh/Downloads/keva_vision_FINAL/"
                   "keva_vision_20min_FINAL.mp4")
FFMPEG = pathlib.Path.home() / "ffmpeg"

START, END = 90, 186               # video seconds
GAP_IN, GAP_OUT = 127, 178         # video seconds the desk was unstaffed
SPEED = 4
V0_CLOCK_S = 20 * 3600 + 9 * 60 + 56   # burned-in clock at video t=0

VW, VH = 1920, 1168
HEAD_H, FOOT_H = 108, 96
ALERT_H = 66

SLATE = T.rgb(T.SLATE)
FOG = T.rgb(T.FOG)
COPPER = T.rgb(T.COPPER)
SAGE = T.rgb(T.SAGE)
HAIR = (58, 52, 47)

STAFF_BOX = T.rgb(T.BOX_STAFF)
GUEST_BOX = T.rgb(T.BOX_GUEST)
FILTERED = T.rgb(T.BOX_FILTERED)
POSE = (240, 210, 60)
ENTRY = (220, 60, 50)

F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FMB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def track(d, xy, text, fnt, fill, sp=0):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += d.textlength(ch, font=fnt) + sp
    return x


def clock(video_s):
    t = V0_CLOCK_S + video_s * SPEED
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}"


def header():
    im = Image.new("RGB", (VW, HEAD_H), SLATE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, HEAD_H - 3, VW, HEAD_H], fill=COPPER)
    track(d, (44, 26), "KEVA VISION", font(FMB, 28), FOG, sp=4)
    d.text((44, 66), "Reception · CAM.112 · Delilah LA",
           font=font(F, 21), fill=SAGE)

    right = f"{clock(START)}–{clock(END)}  ·  2026-07-28  ·  4× SPEED"
    w = d.textlength(right, font=font(FM, 22))
    d.text((VW - 44 - w, 30), right, font=font(FM, 22), fill=COPPER)
    sub = "twenty minutes of service, rendered in five"
    w2 = d.textlength(sub, font=font(F, 19))
    d.text((VW - 44 - w2, 66), sub, font=font(F, 19), fill=SAGE)
    return im


def legend():
    im = Image.new("RGB", (VW, FOOT_H), SLATE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, VW, 2], fill=HAIR)
    items = [("STAFF", STAFF_BOX, "box"), ("GUEST", GUEST_BOX, "box"),
             ("FILTERED OUT", FILTERED, "dash"), ("BODY POSE", POSE, "line"),
             ("ENTRY LINE", ENTRY, "line")]
    x, fl, cy = 44, font(FM, 19), 34
    for label, col, shape in items:
        if shape == "box":
            d.rectangle([x, cy - 11, x + 26, cy + 11], outline=col, width=3)
        elif shape == "dash":
            for i in range(0, 26, 8):
                d.line([x + i, cy - 11, x + min(i + 4, 26), cy - 11],
                       fill=col, width=2)
                d.line([x + i, cy + 11, x + min(i + 4, 26), cy + 11],
                       fill=col, width=2)
        else:
            d.line([x, cy + 8, x + 26, cy - 8], fill=col, width=4)
        x = track(d, (x + 38, cy - 10), label, fl, FOG, sp=1.2) + 46

    # x now sits just past the last legend item; pick the longest note that
    # still clears it, rather than trusting an eyeballed length.
    notes = ["Top-left, the HUD prints whether the reception desk is STAFFED "
             "or AWAY — that is what this run is about.",
             "Top-left: the HUD prints STAFFED or AWAY for the desk.",
             "The HUD prints STAFFED or AWAY."]
    fn = font(F, 20)
    for note in notes:
        w = d.textlength(note, font=fn)
        if VW - 44 - w > x + 40:
            d.text((VW - 44 - w, 36), note, font=fn, fill=SAGE)
            break
    return im


def alert():
    """Shown only while the desk is unstaffed. The whole point of the clip."""
    im = Image.new("RGB", (VW, ALERT_H), T.rgb(T.COPPER_DK))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, VW, 3], fill=COPPER)
    track(d, (44, 21), "DESK UNSTAFFED", font(FMB, 26), (255, 245, 238), sp=3)
    d.text((392, 24), "guests are waiting and nobody is at the reception desk",
           font=font(F, 22), fill=(246, 216, 200))
    right = "longest gap this run · 3.4 min"
    w = d.textlength(right, font=font(FM, 21))
    d.text((VW - 44 - w, 24), right, font=font(FM, 21), fill=(246, 216, 200))
    return im


def build():
    header().save(HERE / "_clip_head.png")
    legend().save(HERE / "_clip_foot.png")
    alert().save(HERE / "_clip_alert.png")
    out = HERE / "keva_vision_demo_clip.mp4"

    gi, go = GAP_IN - START, GAP_OUT - START      # gap, in clip-relative seconds
    cmd = [
        str(FFMPEG), "-y", "-v", "error",
        "-ss", str(START), "-to", str(END), "-i", str(SRC),
        "-i", str(HERE / "_clip_head.png"),
        "-i", str(HERE / "_clip_foot.png"),
        "-i", str(HERE / "_clip_alert.png"),
        "-filter_complex",
        f"[0:v]pad={VW}:{VH + HEAD_H + FOOT_H}:0:{HEAD_H}:0x{T.SLATE}[p];"
        f"[p][1:v]overlay=0:0[h];"
        f"[h][2:v]overlay=0:{VH + HEAD_H}[g];"
        f"[g][3:v]overlay=0:{HEAD_H + VH - ALERT_H}:"
        f"enable='between(t,{gi},{go})'[v]",
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    out = build()
    probe = subprocess.run(
        [str(FFMPEG).replace("ffmpeg", "ffprobe"), "-v", "error",
         "-show_entries", "format=duration:stream=width,height",
         "-of", "default=nw=1", str(out)],
        capture_output=True, text=True, check=True).stdout
    got = dict(l.split("=") for l in probe.strip().splitlines())
    assert int(got["width"]) == VW, got
    assert int(got["height"]) == VH + HEAD_H + FOOT_H, got
    assert abs(float(got["duration"]) - (END - START)) < 1.5, got
    assert out.stat().st_size > 2_000_000, "clip suspiciously small"
    # the gap must sit inside the clip, or the alert never shows
    assert START < GAP_IN < GAP_OUT < END, "gap falls outside the clip"
    print(f"wrote {out.name} · {END - START}s · {got['width']}x{got['height']} · "
          f"gap alert {GAP_IN - START}-{GAP_OUT - START}s · "
          f"{out.stat().st_size / 1e6:.1f} MB")
