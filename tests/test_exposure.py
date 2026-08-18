"""test_exposure.py — badly-lit frames must be corrected, not just IR ones.

WHY THIS EXISTS
    CLAHE fired on exactly one condition: "is this frame infrared", at a fixed
    clipLimit of 2.0. Exposure was never measured. So a daylight frame blown
    out by the window behind the desk, or a dim stretch before the IR
    cut-over that is still technically colour, went to the detector exactly as
    captured.

    That matters because detection confidence tracks contrast while the
    confidence floor is a fixed number — a badly exposed frame loses people
    silently, and the log shows nothing unusual. Symptom 19.

    The opposing risk is real and is why the clip limit is capped: CLAHE past
    ~4.0 amplifies sensor noise into edges, and the detector reads edges as
    people. Turning symptom 19 into symptoms 5/6 is not a fix.

Run: python tests/test_exposure.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kevacv.engine as E
from kevacv import config as CFG

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def flat(v):
    return np.full((90, 160, 3), v, np.uint8)


def test_verdicts():
    for value, want in ((8, "dark"), (40, "dark"), (120, "ok"),
                        (210, "bright"), (250, "bright")):
        got = E.frame_exposure(flat(value))["verdict"]
        check(got == want, f"luma {value} -> {want}", got)


def test_clipped_pixels_alone_trigger():
    """A frame can average 'fine' while half of it is pinned pure white — the
    blown-out-window case. Mean alone misses it."""
    img = flat(120)
    img[:, :80] = 255           # half the frame clipped high
    e = E.frame_exposure(img)
    check(e["clipped_high"] > CFG.EXPOSURE_CLIP_FRAC,
          "clipped-highlight share is measured", f"{e['clipped_high']:.2f}")
    check(e["verdict"] == "bright",
          "a half-blown frame is flagged despite an acceptable mean",
          e["verdict"])


def test_clip_limit_scales_with_severity():
    dim = E.exposure_clip_limit(E.frame_exposure(flat(40)), max_clip=4.0)
    dark = E.exposure_clip_limit(E.frame_exposure(flat(8)), max_clip=4.0)
    check(dark > dim, "a darker frame gets stronger equalisation",
          f"{dark:.2f} > {dim:.2f}")


def test_ok_frames_are_left_alone():
    c = E.exposure_clip_limit(E.frame_exposure(flat(120)), base=2.0)
    check(c == 2.0, "a well-exposed frame keeps the base clip limit", str(c))


def test_clip_limit_is_capped():
    """The cap is the whole reason this is safe to enable by default."""
    for v in (0, 4, 255):
        c = E.exposure_clip_limit(E.frame_exposure(flat(v)), max_clip=4.0)
        check(c <= 4.0, f"luma {v} stays within the cap", f"{c:.2f}")


def test_clahe_accepts_a_clip_limit():
    img = np.random.RandomState(0).randint(0, 60, (64, 64, 3), dtype=np.uint8)
    out = E.apply_clahe(img, clip_limit=3.5)
    check(out.shape == img.shape, "apply_clahe preserves shape")
    check(out.std() >= img.std(),
          "and increases contrast on a dark frame",
          f"{out.std():.1f} vs {img.std():.1f}")


def test_defaults_are_defined():
    for name in ("ENABLE_EXPOSURE_ADAPT", "EXPOSURE_DARK_MEAN",
                 "EXPOSURE_BRIGHT_MEAN", "EXPOSURE_CLIP_FRAC",
                 "EXPOSURE_MAX_CLIP"):
        check(hasattr(CFG, name), f"{name} is in config.py")


def main():
    print(__doc__.strip().splitlines()[0])
    for fn in (test_verdicts, test_clipped_pixels_alone_trigger,
               test_clip_limit_scales_with_severity,
               test_ok_frames_are_left_alone, test_clip_limit_is_capped,
               test_clahe_accepts_a_clip_limit, test_defaults_are_defined):
        print(f"\n{fn.__name__}")
        fn()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        return 1
    print("all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
