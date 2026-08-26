"""build_clip.py -- cut the demo clip out of the 1-hour annotated render and
frame it with a Keva-liveried header and legend.

WHY THIS WINDOW (video 198-260 s)
    The source render is a 4x time-lapse: 54,104 frames covering 3,601 s of
    footage, played at 60 fps, so 15 min of video == 1 h of service. Video
    time x4 = footage time.

    The window was picked off the engine's own HUD, not by taste. Across
    198-260 s it reads:

        13:59  people 5 (3 staff)  entered=4 exited=2
        14:39  people 3 (2 staff)  entered=6 exited=2
        16:18  people 4 (3 staff)  entered=6 exited=3
        16:58  people 8 (4 staff)  entered=7 exited=3

    Arrivals tick 4 -> 7 on screen and the room fills to 8 people. After 260 s
    it drops to 2 and stays there; before 198 s it is one person at a desk.

    The colour-lit stretch (127-171 s, house lights up) was REJECTED despite
    looking better: at t=155 a woman stands at the reception desk with no box
    on her. That is a miss, and a miss is not what a demo is for.

SPEED IS LEFT ALONE. The burned-in HUD says "TIME-LAPSE 4.0x". Re-timing the
clip would leave that label lying on screen, so the clip stays at native speed
and the header says 4x too.

Box colours in the legend are read from kevacv/helpers.py:57, not guessed:
    customer #2a78d6   staff #eb6834   unknown #898781
"""
import pathlib
import subprocess

from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).parent
SRC = pathlib.Path("/mnt/c/Users/prabh/Downloads/keva_delilah_1hr/"
                   "CAM.112_annotated_h264.mp4")
FFMPEG = pathlib.Path.home() / "ffmpeg"

START, END = 198, 260                 # video seconds
SPEED = 4                             # the render's own time-lapse factor
RUN_START_S = 19 * 3600 + 30 * 60     # the hour began at 19:30

VW, VH = 1920, 1168
HEAD_H, FOOT_H = 108, 96

# Keva restaurant-app tokens (RESTAURANT-APP/app/globals.css)
SLATE = (28, 25, 23)
FOG = (250, 248, 245)
COPPER = (212, 98, 43)
SAGE = (139, 126, 111)
HAIR = (58, 52, 47)

# kevacv/helpers.py:57 -- what the boxes in the video actually mean
STAFF_BOX = (235, 104, 52)
GUEST_BOX = (42, 120, 214)
FILTERED = (137, 135, 129)
POSE = (240, 210, 60)
ENTRY = (220, 60, 50)

F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FMB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def track(d, xy, text, fnt, fill, sp=0):
    """Draw text with letter-spacing, returning the x it ended at."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=fill)
        x += d.textlength(ch, font=fnt) + sp
    return x


def clock(video_s):
    """Video second -> wall clock in the venue."""
    t = RUN_START_S + video_s * SPEED
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
    sub = "one hour of service, rendered in fifteen minutes"
    w2 = d.textlength(sub, font=font(F, 19))
    d.text((VW - 44 - w2, 66), sub, font=font(F, 19), fill=SAGE)
    return im


def legend():
    im = Image.new("RGB", (VW, FOOT_H), SLATE)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, VW, 2], fill=HAIR)

    items = [("STAFF", STAFF_BOX, "box"),
             ("GUEST", GUEST_BOX, "box"),
             ("FILTERED OUT", FILTERED, "dash"),
             ("BODY POSE", POSE, "line"),
             ("ENTRY LINE", ENTRY, "line")]

    x = 44
    fl = font(FM, 19)
    for label, col, shape in items:
        cy = 34
        if shape == "box":
            d.rectangle([x, cy - 11, x + 26, cy + 11], outline=col, width=3)
        elif shape == "dash":
            for i in range(0, 26, 8):
                d.line([x + i, cy - 11, x + min(i + 4, 26), cy - 11], fill=col, width=2)
                d.line([x + i, cy + 11, x + min(i + 4, 26), cy + 11], fill=col, width=2)
        else:
            d.line([x, cy + 8, x + 26, cy - 8], fill=col, width=4)
        x += 38
        x = track(d, (x, cy - 10), label, fl, FOG, sp=1.2) + 46

    note = ("Crossing the entry line is what counts as an arrival — "
            "standing near the door does not.")
    w = d.textlength(note, font=font(F, 20))
    d.text((VW - 44 - w, 36), note, font=font(F, 20), fill=SAGE)
    return im


def build():
    header().save(HERE / "_clip_head.png")
    legend().save(HERE / "_clip_foot.png")
    out = HERE / "keva_vision_demo_clip.mp4"

    cmd = [
        str(FFMPEG), "-y", "-v", "error",
        "-ss", str(START), "-to", str(END), "-i", str(SRC),
        "-i", str(HERE / "_clip_head.png"),
        "-i", str(HERE / "_clip_foot.png"),
        "-filter_complex",
        f"[0:v]pad={VW}:{VH + HEAD_H + FOOT_H}:0:{HEAD_H}:0x1C1917[p];"
        f"[p][1:v]overlay=0:0[h];"
        f"[h][2:v]overlay=0:{VH + HEAD_H}[v]",
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    out = build()
    # check: the clip must be the window we asked for, at the padded size, and
    # big enough to actually contain picture -- a silent ffmpeg filter failure
    # produces a valid tiny file, which is the failure worth catching.
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
    print(f"wrote {out.name} · {END - START}s · "
          f"{got['width']}x{got['height']} · "
          f"{out.stat().st_size / 1e6:.1f} MB")
