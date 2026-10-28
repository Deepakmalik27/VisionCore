"""frame_ritual.py — the post-run frame review, made a ritual instead of an
accident.

The single highest-value bug list this project ever got came from a human
reviewing 64 sampled frames of an annotated video. This makes that repeatable:
extract N evenly-spaced frames from any annotated mp4 into a folder with a
one-page contact sheet you (or any vision model) can review in minutes, and
diff against the previous run's review.

Usage:
    python frame_ritual.py <annotated_video.mp4> [n_frames=64]

Output (next to the video):
    <video>_ritual/frame_001.jpg ... frame_NNN.jpg
    <video>_ritual/index.html          <- open in a browser, review, done
    <video>_ritual/REVIEW_TEMPLATE.md  <- the 9-issue checklist to fill in

# ponytail: cv2 only — the pipeline already requires it; no ffmpeg dependency.
"""
import sys
from pathlib import Path

import cv2


def main(video, n=64):
    video = Path(video)
    if not video.exists():
        sys.exit(f"not found: {video}")
    out = video.with_name(video.stem + "_ritual")
    out.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = total / fps_v
    n = min(n, max(total, 1))
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * total / n))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        if w > 960:
            fr = cv2.resize(fr, (960, int(h * 960 / w)))
        cv2.imwrite(str(out / f"frame_{k + 1:03d}.jpg"), fr,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()

    frames = sorted(out.glob("frame_*.jpg"))
    cells = "\n".join(
        f'<div class="c"><img src="{f.name}" loading="lazy">'
        f'<span>{f.stem} · t≈{i * dur / max(len(frames), 1):.0f}s</span></div>'
        for i, f in enumerate(frames))
    (out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>{video.stem} — frame ritual</title>
<style>body{{background:#111;color:#ccc;font:13px monospace;margin:12px}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.c img{{width:100%;display:block}} .c span{{opacity:.6}}</style>
<h2>{video.name} — {len(frames)} frames, {dur / 60:.1f} min</h2>
<p>Review against REVIEW_TEMPLATE.md. Anything you can SEE wrong here is a
bug report worth more than any metric.</p><div class=g>{cells}</div>""")

    (out / "REVIEW_TEMPLATE.md").write_text(f"""# Frame review — {video.name}
Build id (from the HUD band): ____________   Reviewer: ____________

Score each 0-10 and note frame numbers for anything wrong.

| # | Check | Score | Frames / notes |
|---|-------|-------|----------------|
| 1 | IN/OUT line fires when people cross            |   |   |
| 2 | No static object (plant/decor) boxed as person |   |   |
| 3 | Staff keeps ONE id (name label) throughout     |   |   |
| 4 | No new id after someone is briefly blocked     |   |   |
| 5 | Track survives occlusion (same id reappears)   |   |   |
| 6 | No duplicate boxes on one person               |   |   |
| 7 | No box flicker while a person stays in frame   |   |   |
| 8 | Colour↔IR transitions don't reset identities   |   |   |
| 9 | entered=/exited= counters look believable      |   |   |
| 10| Anything NEW not on this list                  |   |   |

Compare with the previous run's filled template before filing anything.
""")
    print(f"{len(frames)} frames -> {out}/index.html  (+ REVIEW_TEMPLATE.md)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64)
