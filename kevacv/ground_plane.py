"""ground_plane.py — PHASE 3. Stop measuring the world in pixels.

THE PROBLEM
    Every spatial threshold in this pipeline is a single pixel number applied to
    a frame where perspective varies about 5x from the near corner to the far
    one:

        LIVE_REID_MAX_DIST_PX   MAX_PLAUSIBLE_SPEED_PX   REID_HANDOFF_PX
        GREET_PROXIMITY_PX      SHADOW_PX                CARRIED_* geometry

    One number cannot be right in both halves of the image. It is simultaneously
    too loose near the camera (merging different people) and too tight far away
    (failing to merge the same person). Every one of those thresholds has been
    hand-tuned against that contradiction for 14 versions.

THE FIX
    Convert image points to positions on the FLOOR, in metres. Then one
    threshold — "a person cannot walk faster than 2 m/s" — is correct
    everywhere, because it is a statement about the world rather than about
    pixels.

TWO WAYS TO GET THERE, and the cheap one needs nothing from you
    AUTO   (default, zero configuration)
        A vertical object of constant real height projects to a pixel height
        that is a LINEAR function of its foot's image row. That is exact for any
        camera pose viewing a plane, and Phase 1 already fits that line from the
        run's own isolated detections (_PerspectiveModel). From it we recover an
        approximate metric ground mapping with no clicks and no measurements.
        Assumes: flat floor, fixed camera, little roll, people standing.
    EXACT  (four clicks, when precision matters)
        Put four image points and their real-world floor coordinates in the
        zones JSON as "ground_points". cv2.findHomography then gives a true
        plane mapping with no assumptions at all. This is also the exact input
        NVIDIA-style multi-camera 3D tracking needs later, so it is not
        throwaway work.

HONESTY ABOUT THE AUTO MODE
    It assumes a level camera. A real camera is tilted down, which stretches the
    depth axis. test_ground_plane.py measures that error against a synthetic
    tilted camera instead of hand-waving it: lateral distance stays accurate,
    depth degrades gracefully with tilt, and BOTH are far better than pixels.
    describe() reports the implied camera height so an implausible value (a bad
    fit, or heavy tilt) is visible rather than silently wrong.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import cv2
except Exception:                                   # pragma: no cover
    cv2 = None

PERSON_H_M = 1.7          # median adult standing height, the reference ruler
DEFAULT_HFOV_DEG = 82.0   # typical wide fixed CCTV lens; only affects the DEPTH
                          # axis, and only in AUTO mode. Override in config if
                          # the real lens is known.


class GroundPlane:
    """Image pixels -> floor coordinates in metres.

    X is lateral (metres right of the optical centre), Z is depth (metres from
    the camera). Both measured on the floor, so distance between two people is
    the distance they would pace out, not the distance between their boxes.
    """

    def __init__(self, mode, *, a=None, b=None, cx=None, focal_px=None,
                 person_h=PERSON_H_M, H=None, frame_size=None, note=""):
        self.mode = mode                 # "auto" | "exact" | "none"
        self.a, self.b = a, b
        self.cx, self.focal_px = cx, focal_px
        self.person_h = person_h
        self.H = H                       # 3x3 homography, exact mode
        self.frame_size = frame_size
        self.note = note

    # -- constructors --------------------------------------------------------
    @classmethod
    def none(cls, why="no scene geometry available"):
        return cls("none", note=why)

    @classmethod
    def from_perspective(cls, a, b, frame_w, frame_h, person_h=PERSON_H_M,
                         hfov_deg=DEFAULT_HFOV_DEG, focal_px=None):
        """Build from the fitted line  h(y) = a*y + b  (pixel height of a
        standing person whose feet are at image row y).

        a <= 0 would mean people get SHORTER as they come closer, which is not a
        camera looking at a floor — it is a broken fit, and we refuse it rather
        than produce confident nonsense from it.
        """
        if a is None or a <= 1e-6:
            return cls.none(f"perspective fit unusable (a={a}) — not a floor view")
        if focal_px is None:
            focal_px = (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls("auto", a=float(a), b=float(b), cx=frame_w / 2.0,
                   focal_px=float(focal_px), person_h=person_h,
                   frame_size=(frame_w, frame_h),
                   note=f"auto from perspective fit (hfov~{hfov_deg:.0f} deg)")

    @classmethod
    def from_correspondences(cls, img_pts, world_pts, frame_size=None):
        """Exact: >=4 image points and their real floor coordinates in metres."""
        if cv2 is None:
            return cls.none("cv2 unavailable")
        img = np.asarray(img_pts, dtype=np.float32).reshape(-1, 1, 2)
        wld = np.asarray(world_pts, dtype=np.float32).reshape(-1, 1, 2)
        if len(img) < 4 or len(img) != len(wld):
            return cls.none(f"need >=4 matched points, got {len(img)}/{len(wld)}")
        H, _ = cv2.findHomography(img, wld, method=0)
        if H is None:
            return cls.none("findHomography failed — are the 4 points collinear?")
        return cls("exact", H=np.asarray(H, dtype=float), frame_size=frame_size,
                   note=f"exact homography from {len(img)} correspondences")

    @classmethod
    def from_zone_config(cls, cfg, frame_size, persp=None, **kw):
        """Prefer an exact homography if the zones JSON carries one:

            "ground_points": [
              {"image": [x, y], "world": [X, Z]},   x4 or more, metres
              ...
            ]

        Otherwise fall back to the automatic fit. This is the only place the two
        modes are chosen between, so the rest of the pipeline never cares which
        one it got.
        """
        gp = (cfg or {}).get("ground_points") or []
        if len(gp) >= 4:
            try:
                return cls.from_correspondences(
                    [p["image"] for p in gp], [p["world"] for p in gp], frame_size)
            except (KeyError, TypeError) as e:
                pass
        if persp is not None:
            fit = persp._refit() if hasattr(persp, "_refit") else None
            if fit:
                return cls.from_perspective(fit[0], fit[1], frame_size[0],
                                            frame_size[1], **kw)
        return cls.none("no ground_points in zones JSON and no usable perspective fit")

    # -- the mapping ---------------------------------------------------------
    @property
    def ok(self):
        return self.mode != "none"

    def expected_h(self, y):
        """Pixel height of a standing person with feet at row y (auto mode)."""
        if self.mode != "auto":
            return None
        h = self.a * float(y) + self.b
        return h if h > 1e-6 else None

    def to_ground(self, x, y):
        """Image point (x, y) — the FEET — to floor (X, Z) in metres.

        Returns None above the horizon, where the floor is not visible and any
        answer would be fabricated.
        """
        if self.mode == "exact":
            v = self.H @ np.array([float(x), float(y), 1.0])
            if abs(v[2]) < 1e-12:
                return None
            return (float(v[0] / v[2]), float(v[1] / v[2]))
        if self.mode == "auto":
            h = self.expected_h(y)
            if h is None:
                return None
            #  h = f*Hp/Z   =>   Z = f*Hp/h
            #  X = (x-cx)*Z/f = (x-cx)*Hp/h
            return (float((float(x) - self.cx) * self.person_h / h),
                    float(self.focal_px * self.person_h / h))
        return None

    def scale_at(self, y):
        """Metres per pixel at image row y. The number that was implicitly
        assumed constant by every px threshold in the pipeline."""
        if self.mode == "auto":
            h = self.expected_h(y)
            return None if h is None else self.person_h / h
        if self.mode == "exact":
            p0, p1 = self.to_ground(0.0, y), self.to_ground(1.0, y)
            if p0 is None or p1 is None:
                return None
            return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        return None

    def dist_m(self, p, q):
        """Floor distance in metres between two FOOT points. None if either is
        above the horizon."""
        a, b = self.to_ground(*p), self.to_ground(*q)
        if a is None or b is None:
            return None
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def speed_mps(self, p, q, dt):
        if dt <= 0:
            return None
        d = self.dist_m(p, q)
        return None if d is None else d / dt

    def px_for_metres(self, metres, y):
        """Convert a metric threshold back to pixels AT ROW y. Lets existing
        pixel-based code keep working while becoming perspective-correct."""
        s = self.scale_at(y)
        return None if not s else metres / s

    # -- self-report ---------------------------------------------------------
    def camera_height_m(self):
        """Implied camera height. For a level camera h = (Hp/Hc)*(y - y_horizon),
        so a == Hp/Hc. A tilted camera makes this an approximation, which is
        exactly why it is printed: 2-6 m is a reception camera, anything else
        means the fit is wrong or the tilt is severe."""
        if self.mode != "auto" or not self.a:
            return None
        return self.person_h / self.a

    def horizon_y(self):
        if self.mode != "auto" or not self.a:
            return None
        return -self.b / self.a

    def describe(self):
        if self.mode == "none":
            return f"ground plane: NOT AVAILABLE ({self.note}) — thresholds stay in pixels"
        if self.mode == "exact":
            return f"ground plane: EXACT ({self.note})"
        hc, hy = self.camera_height_m(), self.horizon_y()
        warn = ""
        if hc is not None and not (1.8 <= hc <= 7.0):
            warn = (f"  !! implied camera height {hc:.1f} m is implausible — the "
                    f"perspective fit or the camera tilt is off; treat metric "
                    f"numbers as indicative and supply ground_points for exact")
        return (f"ground plane: AUTO — implied camera height {hc:.2f} m, "
                f"horizon at row {hy:.0f}, {self.note}" + warn)

    def sanity(self, frame_h):
        """Cheap self-checks whose failure means 'do not trust the metres'."""
        out = []
        if self.mode != "auto":
            return out
        hc = self.camera_height_m()
        if hc is None or not (1.8 <= hc <= 7.0):
            out.append(f"implied camera height {hc} m outside 1.8-7.0 m")
        hy = self.horizon_y()
        if hy is None or hy > frame_h:
            out.append(f"horizon at row {hy} is below the frame ({frame_h})")
        s_near = self.scale_at(frame_h * 0.95)
        s_far = self.scale_at(frame_h * 0.35)
        if s_near and s_far and s_far / s_near < 1.15:
            out.append(f"near/far scale barely differs ({s_near:.4f} vs {s_far:.4f} "
                       f"m/px) — the camera may be near-overhead, in which case "
                       f"pixel thresholds were already fine")
        return out


# ---------------------------------------------------------------------------
# a synthetic camera, used by the tests — and by anyone who wants to check the
# maths without footage. Kept here on purpose: a calibration model you cannot
# generate known-answer data for is a calibration model you cannot trust.
# ---------------------------------------------------------------------------
def synth_camera(cam_h=3.0, focal_px=1200.0, frame=(1920, 1080), tilt_deg=0.0):
    """Return project(X, Z, Y) -> (x, y) for a camera at height cam_h looking
    down the +Z axis, optionally pitched down by tilt_deg."""
    cx, cy = frame[0] / 2.0, frame[1] / 2.0
    t = math.radians(tilt_deg)

    def project(X, Z, Y=0.0):
        # world -> camera (camera at (0, cam_h, 0), pitched down by t)
        yc = cam_h - Y
        zc = Z * math.cos(t) - yc * math.sin(t)
        yv = Z * math.sin(t) + yc * math.cos(t)
        if zc <= 1e-6:
            return None
        return (cx + focal_px * X / zc, cy + focal_px * yv / zc)

    return project
