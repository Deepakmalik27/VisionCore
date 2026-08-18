"""engine.py — video in, tracks and zone events out.

EXTRACTED FROM notebook Cell 7. This is the expensive half of the pipeline:
decode, detect, track, embed, recognise faces, and render the annotated video.

THE PASS STRUCTURE
    PASS 1  frame_source -> motion gate -> YOLO -> BoT-SORT(+CLIP) ->
            _IdentityMemory -> zone events, crossings, banked crops
    PASS 2  render_annotated -> ffmpeg libx264 -> *_annotated_h264.mp4

WHAT IS DELIBERATELY STILL DYNAMIC
    Cell 7 does 33 `globals().get(NAME, default)` lookups. Those are late
    binding on purpose: a later notebook cell (2e, the scale profile) rebinds
    some values AFTER this code is defined. Extraction keeps them, because
    turning them into import-time constants would freeze the pre-override
    value — which is exactly the bug V75 fixed in the config audit. Defaults
    come from kevacv.config via the star-import below, so `globals().get`
    resolves against this module.

RUNTIME BINDINGS
    The names under "injected at runtime" are not config: they are paths,
    devices and models that only exist once a run starts. They default to None
    so the module imports on a laptop with no GPU; the caller sets them.

NOT VERIFIED THE WAY analytics.py WAS
    tests/test_analytics_extraction.py proves Cell 5's extraction changed
    nothing, because Cell 5 is pure functions over data. Cell 7 needs a video,
    a GPU and model weights, so the equivalent proof needs a real run.
    tests/test_engine_extraction.py pins what CAN be checked without them:
    every pure helper, and that every name still resolves.
"""
from __future__ import annotations

from .log import get_logger, stage, banner

_log = get_logger("engine")


import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import *          # noqa: F401,F403  — defaults for globals().get
from .analytics import *       # noqa: F401,F403  — identity + metrics layer
# `import *` SKIPS underscore-prefixed names by language rule, so the four
# private helpers the engine actually calls were never imported. Nothing fails
# at import time — Python resolves globals at CALL time, so this would have
# raised NameError deep inside process_video, after the 3.4 GB download and
# the model load. Imported explicitly, and tests/test_no_missing_names.py now
# fails the suite if another one appears.
from .analytics import (_apply_global_tracklet_pass, _boxes_occluding,
                        _cosine, _IdentityMemory)
from .ground_plane import PERSON_H_M, GroundPlane
# Called by process_video but never imported — in the notebook these arrived
# via the shared cell namespace, so as a module they were NameErrors waiting
# for the video to finish decoding.
from .camera_health import CameraHealth, verdict_line
from .capabilities import CapabilityLedger, onnx_providers
from .detect_filters import (drop_tracks, implausible_size_mask, protected_ids,
                             static_min_life_by_id, static_track_ids)
from .dataset_collector import DatasetCollector
from .funnel import DetectionFunnel
from .phantoms import OnlineStaticSuppressor, in_phantom, phantom_regions
from .profiling import Profile
from .reid_calibration import LiveSeparability, describe_live
from .tiled import cost_estimate, height_roi, tiled_predict
from .helpers import (_safe_id, anchor_point, classify_zones, load_zone_config,
                      mmss, plot_reid_pair_audit, show_gallery,
                      uses_centre_anchor, wall, zone_color_map)

# ── injected at runtime by the caller (not configuration) ───────────────────
BASE = Path(".")               # working directory for the run
INPUT_ROOT = None              # where the video chunks live
OUTPUT_DIR = Path("output")    # where artefacts are written
DEVICE = "cpu"                 # "cuda" when a GPU is present
DETECTOR_MODEL = None          # path to best.pt
# per-venue settings, see kevacv.venue_profile. This used to default to {},
# which made VENUE_PROFILE["camera"] raise KeyError inside the camera-health
# block. That block catches Exception and logs "(camera-health check skipped)",
# so the view-drift guard silently never ran — a skipped check reads exactly
# like a passed one in the log. Defaulting to the real DEFAULTS means it runs
# unless a caller deliberately overrides it.
from .venue_profile import DEFAULTS as _VP_DEFAULTS  # noqa: E402
VENUE_PROFILE = {k: dict(v) if isinstance(v, dict) else v
                 for k, v in _VP_DEFAULTS.items()}
EVAL_WINDOW = None             # (start_s, end_s) when scoring a labelled slice
NATIVE_FPS_OVERRIDE = None     # force a source fps when the container lies

# ── model weight resolution ─────────────────────────────────────────────────
CLIP_REID_WEIGHTS = "clip_market1501.pt"
OSNET_WEIGHTS = None
OSNET_STOCK_VARIANT = "osnet_x0_25_msmt17.pt"
REID_BACKBONE_STOCK = "clip_market1501.pt"
FORCE_REID_BACKBONE = None

# ── zone roles used by the renderer and the staff heuristic ─────────────────
# ROLE_HEXES is PERSON roles (customer/staff/unknown) — the renderer colours
# bodies with it. My extraction header originally defined it as ZONE roles,
# which is a different concept entirely, and render_annotated died on
# ROLE_HEXES["customer"] after the whole analysis had finished. Imported from
# helpers so there is exactly one definition.
from .helpers import ROLE_HEXES  # noqa: E402

# render_annotated is the LAST stage, so a missing key here costs the entire
# run — detection, tracking, identity, all of it — and reports nothing. The
# static name sweep cannot see dict keys, so the keys the renderer indexes are
# asserted at import instead: a palette problem now fails in a second, before
# a frame is decoded, instead of an hour in.
_MISSING_ROLE_COLOURS = {"customer", "staff"} - set(ROLE_HEXES)
if _MISSING_ROLE_COLOURS:
    raise ImportError(
        f"ROLE_HEXES is missing {sorted(_MISSING_ROLE_COLOURS)}. render_annotated "
        f"indexes it by PERSON role (customer/staff/unknown); a zone palette "
        f"is a different thing and belongs in ZONE_HEXES.")
WAIT_ZONES = ("wait", "waiting_area")
STAFF_ANCHOR_ZONES = ("staff", "reception")

# ── locks: model loading is not thread-safe under the 2-GPU runner ──────────
_FACE_LOCK = threading.Lock()
_REID_LOCK = threading.Lock()
_OB = None                     # boxmot module handle, loaded lazily

_src_crop_h = []   # V76: source crop heights, pre-resize
# v42 mix track_id sort key (some are integers, some are string names like 'jane')
def track_sort_key(tid):
    if isinstance(tid, str) and not tid.isdigit():
        return (0, tid)
    try:
        return (1, int(tid))
    except ValueError:
        return (2, str(tid))

# Cell 7 — VIDEO ENGINE, two-pass architecture:
#   PASS 1 (analyze): detect -> track -> zones -> events. No drawing.
#   then: staff override -> Re-ID stitching -> ghost filter
#   PASS 2 (render): draw the video from FINAL stitched identities, so
#   on-screen ids are persistent, wait-clocks never reset, and roles are
#   correct from the first frame. (Industry-standard: analyze, then render.)
import bisect
import shutil
import supervision as sv
from collections import defaultdict, deque, Counter
from ultralytics import YOLO
from tqdm.auto import tqdm

# ── numpy>=2.5 compatibility shim ──────────────────────────────────────────
# numpy 2.5 removed 2-D np.cross; every released supervision (<=0.29) still
# calls it in exactly two places (verified against the 0.26.1 source). Rebind
# determinant-based equivalents at every import site — same fix supervision
# made on their develop branch. Verified numerically identical to np.cross.
from supervision.geometry.core import Point as _SvPoint
import supervision.geometry.utils as _sv_geo
try:
    import supervision.detection.utils.internal as _sv_internal
except ImportError:
    _sv_internal = None
import supervision.detection.tools.polygon_zone as _sv_pz
try:
    import supervision.detection.line_zone as _sv_lz
except ImportError:
    _sv_lz = None

def _cross2d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def _get_polygon_center(polygon):
    if len(polygon) == 0:
        raise ValueError("Polygon must have at least one vertex.")
    shift = np.roll(polygon, -1, axis=0)
    signed_areas = _cross2d(polygon, shift) / 2
    if signed_areas.sum() == 0:
        c = np.mean(polygon, axis=0).round()
        return _SvPoint(x=c[0], y=c[1])
    centroids = (polygon + shift) / 3.0
    c = np.average(centroids, axis=0, weights=signed_areas).round()
    return _SvPoint(x=c[0], y=c[1])

def _cross_product(anchors, vector):
    v0 = float(vector.end.x - vector.start.x)
    v1 = float(vector.end.y - vector.start.y)
    d = np.asarray(anchors, dtype=float) - np.array(
        [vector.start.x, vector.start.y], dtype=float)
    return v0 * d[..., 1] - v1 * d[..., 0]

_sv_geo.get_polygon_center = _get_polygon_center
_sv_pz.get_polygon_center = _get_polygon_center
if _sv_internal is not None:
    _sv_internal.cross_product = _cross_product
if _sv_lz is not None:
    _sv_lz.cross_product = _cross_product

_REID_STATE = {"embed": None, "failed": False, "method": None, "backend": None}
_FACE_STATE = {"app": None, "failed": False}

try:
    _FACE_LOCK
except NameError:
    import threading as _thr2
    _FACE_LOCK = _thr2.Lock()


def get_face_analyzer():
    with _FACE_LOCK:
        """InsightFace app (lazy, cached). Corroborating signal only — see
        ENABLE_FACE_CORROBORATION in config. Returns None (never raises) if
        insightface isn't installed or fails to load; caller just skips the
        face cross-check in that case."""
        if _FACE_STATE["failed"] or not ENABLE_FACE_CORROBORATION:
            return None
        if _FACE_STATE["app"] is None:
            try:
                from insightface.app import FaceAnalysis
                providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                            if "cuda" in str(DEVICE) else ["CPUExecutionProvider"])
                model_name = FACE_MODEL_NAME if 'FACE_MODEL_NAME' in globals() else "buffalo_sc"
                try:
                    app = FaceAnalysis(name=model_name, providers=providers)
                    app.prepare(ctx_id=0 if "cuda" in str(DEVICE) else -1, det_size=(320, 320))
                    _FACE_STATE["app"] = app
                    # V76b: the crop-resolution guard was mis-inserted here by
                    # V76b: the patch. _from_source and _med are not in scope,
                    # V76b: so it raised NameError, this try/except read that as
                    # V76b: "buffalo_l failed", and every run silently ran on
                    # V76b: the weaker buffalo_sc. It belongs with the crop
                    # V76b: diagnostic, where its variables actually exist.
                    _log.warning(f"✅ Face model ready (InsightFace {model_name})")
                    discover_and_load_staff_gallery(app)
                except Exception as e1:
                    if model_name != "buffalo_sc":
                        _log.warning(f"⚠️ Failed to load {model_name} ({e1}) - falling back to buffalo_sc...")
                        app = FaceAnalysis(name="buffalo_sc", providers=providers)
                        app.prepare(ctx_id=0 if "cuda" in str(DEVICE) else -1, det_size=(320, 320))
                        _FACE_STATE["app"] = app
                        _log.info("✅ Face model ready (InsightFace buffalo_sc)")
                        discover_and_load_staff_gallery(app)
                    else:
                        raise e1
            except Exception as exc:
                _FACE_STATE["failed"] = True
                _log.info(f"ℹ️ Face corroboration unavailable ({exc!r}) — "
                      f"CLIP/OSNet stitching is unaffected, this only disables "
                      f"the optional face cross-check.")
        return _FACE_STATE["app"]


_STAFF_FACE_GALLERY = {}
# A SECOND, higher-resolution face analyzer used ONLY to enrol the staff
# gallery. The shared one is prepared at 320px because it runs per video frame;
# enrolment runs once per photo, so it can afford 640px — and it has to, or a
# marginal photo silently fails to enrol and that person is never recognised
# for the whole chunk. Built lazily: runs whose photos all enrol never make it.
_ENROL_APP = None

_STAFF_SHOT_SUFFIX = re.compile(r"^(?P<base>.+?)[-_](?:\d+|same|alt|b|2nd)$",
                                re.IGNORECASE)


def staff_name_from_filename(path):
    """Which PERSON does this photo belong to?

    Two conventions, both supported:

        staff_gallery/priya.jpg              -> "priya"
        staff_gallery/priya-2.jpg            -> "priya"   (second shot)
        staff_gallery/priya_same.jpg         -> "priya"
        staff_gallery/priya/anything.jpg     -> "priya"   (a folder per person)

    WHY THIS EXISTS
        The loader used the whole filename as the identity. The real gallery
        contains "Staff2.jpg" AND "Staff2-same.jpg" — plainly two photos of
        one person — and that enrolled them as TWO different staff members.
        Every later "one body, one name" rule then had to arbitrate between
        two identities that were never distinct, and the person could be
        matched under either name from frame to frame.

        The folder form is unambiguous and is the one to prefer when a real
        name contains a hyphen (Anna-Maria), which the suffix rule would
        otherwise split.
    """
    p = Path(path)
    parent = p.parent.name.lower()
    # a folder per person — but not the gallery root itself
    if parent and parent not in ("staff_gallery", "gallery", ""):
        return parent
    stem = p.name[:-len(p.suffix)] if p.suffix else p.name
    m = _STAFF_SHOT_SUFFIX.match(stem)
    return (m.group("base") if m else stem).lower()


def _staff_gallery_sim(vec, entry):
    """Best similarity against any enrolled shot of one person.

    `entry` is a list of embeddings (multi-shot). Older callers may still hold
    a single flat vector, so that shape is accepted too rather than raising
    deep inside the frame loop.
    """
    if not entry:
        return -1.0
    if entry and not isinstance(entry[0], (list, tuple)):
        return _cosine(vec, entry)          # legacy single-vector entry
    return max(_cosine(vec, e) for e in entry)


def discover_and_load_staff_gallery(face_analyzer):
    """Load staff face images from STAFF_GALLERY_DIR if present."""
    global _STAFF_FACE_GALLERY
    if not _STAFF_FACE_GALLERY:
        _STAFF_FACE_GALLERY = {}
    if face_analyzer is None:
        return _STAFF_FACE_GALLERY
    
    gallery_dir = Path(STAFF_GALLERY_DIR if 'STAFF_GALLERY_DIR' in globals() else "staff_gallery")
    search_dirs = [BASE / gallery_dir, Path.cwd() / gallery_dir]
    if 'INPUT_ROOT' in globals() and INPUT_ROOT.exists():
        search_dirs.append(INPUT_ROOT / gallery_dir)
        try:
            for folder in INPUT_ROOT.rglob("*"):
                if folder.is_dir() and folder.name == gallery_dir.name:
                    search_dirs.append(folder)
        except Exception:
            pass
                
    found_dir = None
    for d in search_dirs:
        if d.exists() and d.is_dir():
            found_dir = d
            break
            
    # An empty or missing gallery used to return silently. That is the single
    # reason "staff are not consistently recognised as staff" (symptom 2): the
    # entire face path — enrolment, live matching, the C2 sweep — is gated on
    # _STAFF_FACE_GALLERY being non-empty, so with no photos NONE of it ever
    # executes, and staff identity falls back 100% to "stood in the reception
    # polygon long enough". A silent fallback is indistinguishable from a
    # working feature, which is how this survived unnoticed.
    def _no_gallery(reason):
        _log.error("=" * 74)
        _log.error(f"🚨 NO STAFF FACE GALLERY — {reason}")
        _log.error("   Every face-based staff mechanism is therefore INERT "
                   "this run: live staff matching, the C2 gallery sweep, and "
                   "the face veto. Staff identity comes ENTIRELY from zone "
                   "dwell, which cannot tell a receptionist from a guest "
                   "standing at the counter.")
        _log.error(f"   Fix: drop one clear face photo per staff member into "
                   f"{gallery_dir}/ — the FILENAME becomes their id "
                   f"(priya.jpg -> 'priya'). Crop from this camera's own "
                   f"daylight footage if you can; a phone selfie matches "
                   f"poorly against a 4K ceiling view at night.")
        _log.error("=" * 74)
        return _STAFF_FACE_GALLERY

    if not found_dir:
        return _no_gallery(f"no '{gallery_dir}' directory found in "
                           f"{[str(d) for d in search_dirs[:3]]}")

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_files = []
    try:
        img_files = [f for f in found_dir.iterdir() if f.suffix.lower() in img_exts]
    except Exception as _le:
        _log.error(f"   staff gallery unreadable at {found_dir}: {_le}")
    if not img_files:
        return _no_gallery(f"{found_dir} exists but contains no images "
                           f"(.jpg/.jpeg/.png/.webp/.bmp)")

    _log.warning(f"👥 Loading staff gallery from: {found_dir}")
    _shots = defaultdict(int)
    for f in img_files:
        name = staff_name_from_filename(f)
        try:
            img = cv2.imread(str(f))
            if img is None or img.size == 0:
                continue
            faces = face_analyzer.get(img)
            if not faces:
                # ENROLMENT DESERVES A BIGGER DETECTOR THAN PLAYBACK.
                #
                # The shared analyzer is prepared at det_size 320x320 because it
                # runs on tens of thousands of video frames and that cost is
                # per-frame. Enrolment runs FIVE TIMES, once per photo, so the
                # same economy makes no sense here — and it was silently costing
                # staff.
                #
                # Measured on this gallery: at 640x640 every one of the five
                # photos yields a face (det_score 0.66-0.82). At 320x320
                # Staff2.png does not — it is 297x390, so fitting it into 320
                # shrinks an already-marginal face below threshold. The run then
                # logged "No face detected in staff image Staff2.png", enrolled
                # 4 of 5, and reported staff2 and staff4 as "enrolled but NEVER
                # matched" for the rest of the chunk.
                #
                # Retrying larger can only ADD a face, never remove one, so this
                # cannot make matching worse.
                try:
                    import insightface  # noqa: F401
                    from insightface.app import FaceAnalysis
                    global _ENROL_APP
                    if _ENROL_APP is None:
                        _ENROL_APP = FaceAnalysis(
                            name=globals().get("FACE_MODEL_NAME", "buffalo_l"),
                            providers=["CUDAExecutionProvider",
                                       "CPUExecutionProvider"])
                        _ENROL_APP.prepare(
                            ctx_id=0 if "cuda" in str(DEVICE) else -1,
                            det_size=(640, 640))
                    faces = _ENROL_APP.get(img)
                    if faces:
                        _log.info(f"   🔍 {f.name}: no face at 320px, FOUND at "
                                  f"640px (det_score={faces[0].det_score:.2f}) "
                                  f"— enrolling it")
                except Exception as _ee:
                    _log.info(f"   (640px enrolment retry unavailable: {_ee!r})")
            if not faces:
                _log.warning(f"   ⚠️ No face detected in staff image {f.name} "
                             f"(tried 320px and 640px)")
                continue
            face = max(faces, key=lambda f_: (f_.bbox[2] - f_.bbox[0]) * (f_.bbox[3] - f_.bbox[1]))
            # A LIST per person, not one vector. Faces vary enormously with
            # pose and light, and this camera yields a face on ~2% of tracks —
            # so every extra reference shot is a real recall gain. Matching
            # takes the best shot (see _staff_gallery_sim), which is how a
            # multi-shot gallery is supposed to work.
            _STAFF_FACE_GALLERY.setdefault(name, []).append(
                face.normed_embedding.tolist())
            _shots[name] += 1
            _log.info(f"   ✅ Enrolled staff face: {name} "
                      f"(shot {_shots[name]}, det_score={face.det_score:.2f}, "
                      f"from {f.name})")
        except Exception as e:
            _log.info(f"   ❌ Error loading staff image {f.name}: {e}")

    # Enrolling zero faces from a folder that HAS images is a different and
    # worse failure than an empty folder: someone did the work and it silently
    # achieved nothing. Say which.
    if not _STAFF_FACE_GALLERY:
        _log.error(f"🚨 {len(img_files)} image(s) in {found_dir} but NO faces "
                   f"were enrolled — no detectable face in any of them. The "
                   f"face path is inert exactly as if the folder were empty. "
                   f"Use frontal, well-lit crops; the detector needs a face "
                   f"clearly larger than {FACE_MIN_FACE_PX}px.")
    else:
        _log.warning(f"👥 staff gallery: {len(_STAFF_FACE_GALLERY)} enrolled — "
                     f"{', '.join(sorted(_STAFF_FACE_GALLERY))}")
        if len(_STAFF_FACE_GALLERY) < len(img_files):
            _log.warning(f"   ({len(img_files) - len(_STAFF_FACE_GALLERY)} of "
                         f"{len(img_files)} image(s) had no detectable face "
                         f"and were skipped — listed above)")
    return _STAFF_FACE_GALLERY

def _blur_score(crop_bgr, reference_size=128):
    # v46-fix: module-level copy so embed_face_scored (module scope) can call it.
    # process_video keeps its own nested _blur_score for the frame loop; identical logic.
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return 0.0
    resized = cv2.resize(crop_bgr, (reference_size, reference_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def embed_face_scored(crop):
    """One crop -> (normalized 512-d embedding, detector confidence, face
    side px) for the largest face found, or (None, 0.0, 0) if no usable
    face. Person crops from CCTV are frequently too small/angled for a face
    to resolve at all — that's expected, not an error.

    Rejects on TWO independent grounds, not one:
      det_score  -- the detector's confidence a face exists here at all.
      face size  -- (v32) even a confidently-detected face can be too small/
                    blurry for the RECOGNITION embedding to carry real
                    identity signal — detection and recognition degrade at
                    different rates with resolution, so a size floor catches
                    failures a confidence floor alone does not.
    """
    app = get_face_analyzer()
    if app is None or crop is None or crop.size == 0:
        return None, 0.0, 0
    # v43 face crop blur gate
    ch, cw = crop.shape[:2]
    if min(ch, cw) >= MIN_CROP_PX_BLUR_GATE:
        if _blur_score(crop) < MIN_BLUR_VARIANCE:
            return None, 0.0, 0
    try:
        faces = app.get(crop)
    except Exception:
        return None, 0.0, 0
    if not faces:
        return None, 0.0, 0
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    side = min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
    if getattr(face, "det_score", 0.0) < FACE_MIN_DET_SCORE or side < FACE_MIN_FACE_PX:
        return None, 0.0, float(side)
    return face.normed_embedding.tolist(), float(face.det_score), float(side)

def get_track_face_embedding(crops, size_log=None):
    """Tries every banked crop for a track (not just crop[0], which was
    ranked by PERSON-detection confidence*height, not face visibility) and
    keeps whichever gives the highest-confidence face that also clears
    FACE_MIN_FACE_PX. If size_log is given, appends every rejected crop's
    face side (px) so callers can print a diagnostic of what's being
    filtered out and why (see the coverage print in the ReID stitch block)."""
    best_emb, best_score = None, 0.0
    for _, crop in crops:
        emb, score, side = embed_face_scored(crop)
        if emb is not None and score > best_score:
            best_emb, best_score = emb, score
        elif emb is None and size_log is not None and side > 0:
            size_log.append(side)
    return best_emb

def _recrop_face_from_source(video_path, entries, scale, step, native_fps,
                             start_seconds, k=4):
    """V65: retry a track's face at SOURCE resolution.

    entries: [(analysed_frame_idx, (x1, y1, x2, y2))] in ANALYSED coords.
    Picks the k largest boxes, seeks the source video at the matching native
    frame (src_frame = start*fps + idx*step), crops the head region (top 50%
    of the body + margin) at native scale, and returns the best-scoring
    embedding or None. A 15px face at 720p is a 45px face at 4K.
    """
    if not entries or scale <= 1.2:
        return None
    best = sorted(entries, key=lambda e: -(e[1][3] - e[1][1]))[:k]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    out, best_score = None, 0.0
    _f0 = int(round(float(start_seconds or 0.0) * native_fps))
    for fi, (x1, y1, x2, y2) in best:
        cap.set(cv2.CAP_PROP_POS_FRAMES, _f0 + int(fi) * int(step))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        X1, Y1, X2, Y2 = (int(v * scale) for v in (x1, y1, x2, y2))
        pad = max(4, int((X2 - X1) * 0.15))
        crop = fr[max(0, Y1 - pad):Y1 + max(1, (Y2 - Y1) // 2),
                  max(0, X1 - pad):min(fr.shape[1], X2 + pad)]
        if crop.size == 0:
            continue
        emb, score, _side = embed_face_scored(crop)
        if emb is not None and score > best_score:
            out, best_score = emb, score
    cap.release()
    return out


def _resolve_reid_weights(local_path, stock_name):
    """Single source of truth for a ReID weights path, generalized (v28) to
    work for ANY backbone family (CLIP-ReID, OSNet, ...) — used by BOTH the
    offline Re-ID stitcher (get_reid_embedder) and the online BotSORT tracker
    (build_online_tracker/_botsort_yaml), so they can never drift out of sync
    the way _botsort_yaml's old hardcoded model: auto silently did.
    boxmot creates a .lock file NEXT TO the weights — impossible on Kaggle's
    read-only /kaggle/input (Errno 30). Copy the file into the writable
    working dir and load from there. If no local file was shipped in the
    input dataset, return the bare stock filename — boxmot auto-downloads
    known filenames (clip_market1501.pt, osnet_x0_25_msmt17.pt, ...) itself.
    """
    if local_path and str(local_path).startswith("/kaggle/input"):
        _w_dst = BASE / Path(local_path).name
        if not _w_dst.exists():
            _w_dst.write_bytes(Path(local_path).read_bytes())
        return str(_w_dst)
    return str(local_path) if local_path else stock_name

def _resolve_osnet_weights():
    """Back-compat alias — OSNet is now the v28 FALLBACK backbone, kept as its
    own function since get_reid_embedder's fallback path still calls it by
    name for clarity in tracebacks/prints. (v35) stock filename now comes
    from OSNET_STOCK_VARIANT config, not a hardcoded string, so switching to
    a stronger variant (osnet_x1_0_msmt17.pt) is a one-line config change."""
    return _resolve_reid_weights(OSNET_WEIGHTS, OSNET_STOCK_VARIANT)

def _load_reid_backend(weights_path, dev):
    """Loads ANY boxmot-supported ReID checkpoint (CLIP-ReID, OSNet, ...) —
    the loader itself is filename/weights-driven, not backbone-specific, so
    the same three-import-path fallback works unchanged for either family.
    boxmot's internal module layout has moved more than once across
    releases — try every known location in turn instead of assuming pip
    installed a version matching just one path."""
    last_exc = None
    attempts = [
        ("boxmot.reid.core", "ReID",
         lambda cls: cls(weights=weights_path, device=dev, half=globals().get("REID_HALF", False)).model),
        ("boxmot.appearance.reid_auto_backend", "ReidAutoBackend",
         lambda cls: cls(weights=Path(weights_path), device=dev, half=globals().get("REID_HALF", False)).model),
        ("boxmot.appearance.backends.auto_backend", "AutoBackend",
         lambda cls: cls(weights=Path(weights_path), device=dev, half=globals().get("REID_HALF", False)).model),
    ]
    for module_name, cls_name, build in attempts:
        try:
            mod = __import__(module_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            return build(cls)
        except (ImportError, AttributeError) as e:
            last_exc = e
            continue
    raise ImportError(
        f"no working boxmot Re-ID import path found (last: {last_exc!r})"
    ) from last_exc

def _boxmot_dev(device):
    """boxmot wants '0'/'1'/'cpu' — NOT 'cuda' or 'cuda:1' (parse_device raises)."""
    d = str(device or globals().get("DEVICE", "cpu"))
    if "cuda" not in d:
        return "cpu"
    return d.split(":")[1] if ":" in d else "0"


_REID_CACHE = {}          # v55: device -> state, so 2 GPUs get 2 real embedders
try:
    _REID_LOCK
except NameError:
    import threading as _thr
    _REID_LOCK = _thr.Lock()


def get_reid_embedder(device=None):
    """Appearance embedder (lazy, cached), v28 chain: CLIP-ReID (heavy,
    stronger) -> OSNet (light, fallback) -> HSV color-histogram (last
    resort). Returns a function crops->list-of-vectors (dim depends on
    whichever backbone actually loaded — cosine similarity downstream
    doesn't care about dimensionality)."""
    _dev_key = _boxmot_dev(device)
    global _REID_STATE
    with _REID_LOCK:
        _REID_STATE = _REID_CACHE.setdefault(
            _dev_key, {"embed": None, "backend": None, "method": None, "failed": False})
        if _REID_STATE["failed"]:
            return None
        if _REID_STATE["embed"] is not None:
            return _REID_STATE["embed"]
        import os as _os
        _cvd = _os.environ.get("CUDA_VISIBLE_DEVICES")

        def _load_with_gpu_fallback(weights_path, label):
            try:
                return _load_reid_backend(weights_path, _dev_key)
            except Exception as gpu_exc:
                import traceback
                # fp16 is a SPEED setting and must never cost correctness or  # FP16RETRY
                # an order of magnitude of time. Without this retry the next  # FP16RETRY
                # fallback drops a CLIP ViT onto the CPU -- per detection,    # FP16RETRY
                # per frame -- instead of using the fp32 GPU path that was    # FP16RETRY
                # working yesterday, and says so only at INFO.                # FP16RETRY
                if globals().get("REID_HALF", False):  # FP16RETRY
                    _log.warning(f"   ReID fp16 load failed ({gpu_exc!r}) -- retrying fp32 on GPU")  # FP16RETRY
                    globals()["REID_HALF"] = False  # FP16RETRY
                    try:  # FP16RETRY
                        return _load_reid_backend(weights_path, _dev_key)  # FP16RETRY
                    except Exception as _fp32_exc:  # FP16RETRY
                        gpu_exc = _fp32_exc  # FP16RETRY
                _log.info(f"({label} on GPU failed: {gpu_exc!r} — retrying on CPU)")
                _log.info(traceback.format_exc()[-1500:])
                return _load_reid_backend(weights_path, "cpu")

        def _make_embed(backend):
            def _embed(crops):
                # v55: batch. get_features(boxes, img) already batches over boxes,
                # so tile the (uniform 128x256) crops side by side and ask once.
                # Was: one forward pass per crop, called per detection per FRAME.
                out = []
                B = int(globals().get("EMBED_BATCH", 24))
                for s in range(0, len(crops), B):
                    grp = [c if c.shape[:2] == (256, 128) else cv2.resize(c, (128, 256))
                           for c in crops[s:s + B]]
                    canvas = np.hstack(grp)
                    boxes = np.array([[k * 128, 0, (k + 1) * 128, 256]
                                      for k in range(len(grp))], dtype=np.float32)
                    feats = np.asarray(backend.get_features(boxes, canvas),
                                       dtype=np.float32).reshape(len(grp), -1)
                    for v in feats:
                        n = float(np.linalg.norm(v)) or 1e-9
                        out.append((v / n).tolist())
                return out
            return _embed

        clip_reason = None
        try:
            if FORCE_REID_BACKBONE == "osnet":
                raise RuntimeError(
                    "FORCE_REID_BACKBONE='osnet' — skipping CLIP-ReID "
                    "intentionally (not a real failure) for a clean "
                    "head-to-head comparison against OSNet on this footage")
            w_clip = _resolve_reid_weights(CLIP_REID_WEIGHTS, REID_BACKBONE_STOCK)
            try:
                backend = _load_with_gpu_fallback(w_clip, "CLIP-ReID")
            finally:
                # restore even on failure — a boxmot-mutated CUDA_VISIBLE_DEVICES
                # must never leak into later cells
                if _cvd is None:
                    _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    _os.environ["CUDA_VISIBLE_DEVICES"] = _cvd
            _REID_STATE["embed"] = _make_embed(backend)
            _REID_STATE["backend"] = backend
            _REID_STATE["method"] = "clip"
            _log.warning(f"✅ Re-ID model ready (CLIP-ReID, weights: {w_clip})")
        except Exception as clip_exc:
            import traceback
            clip_reason = f"{clip_exc!r}\n{traceback.format_exc()[-1500:]}"
            _log.warning("=" * 70)
            _log.warning(f"⚠️ CLIP-ReID unavailable: {clip_exc!r}")
            _log.info(traceback.format_exc()[-1500:])
            _log.info("   Trying OSNet next.")
            _log.info("=" * 70)

        if _REID_STATE["embed"] is None:
            try:
                w = _resolve_osnet_weights()
                try:
                    backend = _load_with_gpu_fallback(w, "OSNet")
                finally:
                    if _cvd is None:
                        _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        _os.environ["CUDA_VISIBLE_DEVICES"] = _cvd

                _REID_STATE["embed"] = _make_embed(backend)
                _REID_STATE["backend"] = backend
                _log.info(f"✅ Re-ID model ready (OSNet, weights: {w})")
                _REID_STATE["method"] = "osnet"
            except Exception as osnet_exc:
                import traceback
                osnet_reason = f"{osnet_exc!r}\n{traceback.format_exc()}"
                try:
                    import boxmot
                    boxmot_note = f"boxmot version installed: {boxmot.__version__}"
                except Exception:
                    boxmot_note = "boxmot import itself failed — package likely missing/broken"

                _REID_STATE["failed"] = True
                _log.error("=" * 70)
                _log.error("🚨 Re-ID backend FAILED — CLIP-ReID and OSNet both unavailable.")
                _log.info("   NOT falling back to the weak HSV matcher. Fix the cause below, then re-run.")
                _log.info("-" * 70)
                _log.info(f"CLIP-ReID failure:\n{clip_reason}")
                _log.info("-" * 70)
                _log.info(f"OSNet failure:\n{osnet_reason}")
                _log.info(f"   {boxmot_note}")
                _log.info("=" * 70)
                raise RuntimeError(
                    "Re-ID backend unavailable: both CLIP-ReID and OSNet failed to "
                    "load (see printed reasons above). Not falling back to HSV — "
                    "fix the root cause (likely a boxmot/package install issue, "
                    "missing weights, or the Kaggle Internet toggle being off) "
                    "and re-run."
                ) from osnet_exc
    return _REID_CACHE[_dev_key]["embed"]

def _ffmpeg_ok():
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=20)
        return True
    except Exception:
        return False


_HWACCEL_STATE = {}


def _hwaccel_args():
    """The T4 has a hardware video decoder sitting idle while the CPU grinds
    through 4K. Use it if this ffmpeg build was compiled with it — probe once,
    cache the answer, fall back silently to software if not.

    MEASURED AND REJECTED, 2026-08-15: keeping frames on the GPU for scaling
    too, i.e.

        -hwaccel cuda -hwaccel_output_format cuda
        -vf fps=8,scale_cuda=1280:720,hwdownload,format=nv12

    is the standard advice for 4K and it is SLOWER here. On 120 s of the 18:30
    chunk, L4:

        -hwaccel cuda, CPU scale   17.46 s     <- what we do
        scale_cuda + hwdownload    20.55 s

    hwdownload plus the nv12 -> bgr24 conversion costs more than the CPU scale
    it saves. Do not "optimise" this back without re-timing it on the box.
    """
    mode = str(globals().get("FFMPEG_HWACCEL", "auto")).lower()
    if mode in ("none", "off", "false"):
        return []
    if "args" in _HWACCEL_STATE:
        return _HWACCEL_STATE["args"]
    import subprocess
    args = []
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                             capture_output=True, timeout=30, text=True)
        have = (out.stdout or "") + (out.stderr or "")
        if "cuda" in have:
            # listed != working (no GPU visible to ffmpeg, driver mismatch...),
            # so actually decode one frame before trusting it.
            probe = subprocess.run(
                ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-f", "lavfi",
                 "-i", "testsrc=size=64x64:duration=0.1", "-frames:v", "1",
                 "-f", "null", "-"], capture_output=True, timeout=60)
            if probe.returncode == 0:
                args = ["-hwaccel", "cuda"]
    except Exception:
        args = []
    _HWACCEL_STATE["args"] = args
    _log.info(f"   ffmpeg decode: {'NVDEC (hardware)' if args else 'software (CPU)'}"
          f"{'' if args else ' — this build has no working cuda hwaccel'}")
    return args


def frame_source(video_path, fps, max_seconds=None, max_w=1920, start_seconds=0.0,
                 keyframes_only=False):
    """Yield (i, t, frame) at `fps`.

    ffmpeg drops and scales frames inside the decoder, so a 4K chunk costs one
    cheap decode instead of 108,240 full-size ones. t = i/fps: with the fps
    filter that is PTS-correct even on the VFR exports an NVR produces, which
    is what CLOCK_SOURCE was trying to work around by hand.
    cv2 fallback keeps the old behaviour when ffmpeg is unavailable.
    """
    import subprocess
    cap = cv2.VideoCapture(str(video_path))
    nat = cap.get(cv2.CAP_PROP_FPS) or NATIVE_FPS_OVERRIDE or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    ow = min(int(max_w), w) // 2 * 2
    oh = int(round(h * ow / max(w, 1))) // 2 * 2

    if globals().get("USE_FFMPEG_READER", True) and _ffmpeg_ok():
        cmd = ["ffmpeg", "-v", "error"] + _hwaccel_args()
        if keyframes_only:
            # decode I-frames only: for a coarse density scan we do not need
            # every frame, and this is ~5-10x cheaper on 4K. The fps filter
            # below still resamples on PTS, so the timestamps stay exact.
            cmd += ["-skip_frame", "nokey"]
        if start_seconds:
            cmd += ["-ss", str(float(start_seconds))]   # before -i = keyframe seek, instant
        cmd += ["-i", str(video_path)]
        if max_seconds:
            cmd += ["-t", str(float(max_seconds))]
        cmd += ["-vf", f"fps={fps},scale={ow}:{oh}", "-an",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=ow * oh * 3)
        nbytes, i, got_any = ow * oh * 3, 0, False
        try:
            while True:
                buf = proc.stdout.read(nbytes)
                if len(buf) < nbytes:
                    break
                got_any = True
                yield i, start_seconds + i / float(fps), np.frombuffer(buf, np.uint8).reshape(oh, ow, 3).copy()
                i += 1
        finally:
            try:
                proc.stdout.close(); proc.kill()
            except Exception:
                pass
        if got_any:
            return
        _log.error("   !! ffmpeg reader produced no frames — falling back to cv2")

    cap = cv2.VideoCapture(str(video_path))
    step = max(1, int(round(nat / float(fps))))
    idx = i = 0
    if start_seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_seconds * nat))
    limit = int(max_seconds * nat) if max_seconds else None
    while True:
        ok, fr = cap.read()
        if not ok or (limit and idx > limit):
            break
        if idx % step == 0:
            if fr.shape[1] > ow:
                fr = cv2.resize(fr, (ow, oh))
            yield i, start_seconds + idx / nat, fr
            i += 1
        idx += 1
    cap.release()


def apply_clahe(frame, clip_limit=2.0):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=float(clip_limit),
                        tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def frame_exposure(small_bgr):
    """How badly lit is this frame? -> {"mean", "clipped_low", "clipped_high",
    "verdict"} where verdict is one of dark / bright / ok.

    WHY THIS EXISTS
        CLAHE was applied on exactly one condition: the frame is infrared (or
        a global flag forced it always-on). Nothing looked at EXPOSURE. A
        daylight frame blown out by the window behind the desk, or a dim
        stretch before the IR cut-over that is still technically colour, went
        to the detector exactly as captured — and those are the frames where
        recall collapses.

        Detection confidence falls when contrast falls, and the conf floor is
        a fixed number, so a badly-exposed frame silently loses people while
        the log shows nothing unusual. Measuring exposure per frame is what
        makes that visible AND fixable.

    Deliberately computed on the 160x90 thumbnail the motion gate already
    builds, so this costs nothing per frame.
    """
    g = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(g.mean())
    # Share of pixels pinned at either end of the range. Clipped pixels carry
    # NO recoverable detail — CLAHE cannot invent it — so a high value is the
    # honest signal that this frame is partly unusable rather than just dim.
    low = float((g <= 16).mean())
    high = float((g >= 239).mean())
    if mean < EXPOSURE_DARK_MEAN or low > EXPOSURE_CLIP_FRAC:
        verdict = "dark"
    elif mean > EXPOSURE_BRIGHT_MEAN or high > EXPOSURE_CLIP_FRAC:
        verdict = "bright"
    else:
        verdict = "ok"
    return {"mean": mean, "clipped_low": low, "clipped_high": high,
            "verdict": verdict}


def exposure_clip_limit(exp, base=2.0, max_clip=4.0):
    """Stronger equalisation for a worse frame, bounded.

    A fixed clipLimit treats a slightly dim frame and a nearly-black one
    identically. Scaling with severity recovers contrast where it is actually
    missing; the cap exists because CLAHE past ~4.0 amplifies sensor noise into
    texture, and the detector reads that texture as edges — i.e. as phantom
    people. This trade is exactly symptom 19 against symptoms 5/6.
    """
    if exp["verdict"] == "ok":
        return base
    if exp["verdict"] == "dark":
        # 0 at the threshold, 1 at pitch black
        sev = min(1.0, max(0.0, (EXPOSURE_DARK_MEAN - exp["mean"])
                           / max(EXPOSURE_DARK_MEAN, 1e-6)))
    else:
        sev = min(1.0, max(0.0, (exp["mean"] - EXPOSURE_BRIGHT_MEAN)
                           / max(255.0 - EXPOSURE_BRIGHT_MEAN, 1e-6)))
    return float(base + (max_clip - base) * sev)

def _botsort_yaml(track_buffer_frames):
    # model: auto — Ultralytics native with_reid does NOT accept torchreid/boxmot
    # OSNet .pt checkpoints (wrong format for its YOLO()-based .pt loader; confirmed
    # via TypeError on real run). "auto" reuses the detector backbone's own features
    # as a lightweight appearance proxy instead — weaker than dedicated OSNet but a
    # real, working signal with zero extra load. Actual OSNet stays in the OFFLINE
    # stitcher (get_reid_embedder), which loads it via boxmot's own loader — a
    # different code path with no Ultralytics checkpoint-format constraint.
    #
    # proximity_thresh lowered 0.5 -> 0.3 (2026-07-12): diagnostic run found 77 ID
    # swaps split into two mechanisms — (A) live Hungarian swaps during close
    # encounters/crossing (no fix here, needs real online appearance embeddings),
    # and (B) dead-recovery errors where a track lost for 100-230+ frames snaps
    # onto a nearby but wrong person once the Kalman search window has expanded.
    # Tightening proximity_thresh targets (B) specifically — stricter spatial gate
    # on lost-track reactivation, fewer long-gap false recoveries. Does not fix (A).
    p = BASE / "botsort_reid.yaml"
    _hi = NEW_TRACK_CONF if ENABLE_CONF_HYSTERESIS else CONF_THRESHOLD
    _lo = KEEP_TRACK_CONF if ENABLE_CONF_HYSTERESIS else 0.1
    p.write_text(f"""tracker_type: botsort
track_high_thresh: {_hi}
track_low_thresh: {_lo}
new_track_thresh: {_hi}
track_buffer: {int(track_buffer_frames)}
match_thresh: {BOTSORT_MATCH_THRESH}
fuse_score: True
gmc_method: {"sparseOptFlow" if ENABLE_GMC else "none"}
proximity_thresh: 0.3
appearance_thresh: 0.25
with_reid: True
model: auto
""")
    return str(p)

def build_online_tracker(fps, device=None):
    """Real online Re-ID — supports BotSort and BoostTrack. No silent
    fallback: if the real backend or the tracker construction fails, this
    raises with the reason instead of quietly downgrading to the weaker
    with_reid=auto proxy.
    """
    if not USE_REAL_ONLINE_REID:
        return None
    get_reid_embedder(device=device)  # raises on failure — no silent HSV/None fallback
    _st = _REID_CACHE.get(_boxmot_dev(device), _REID_STATE)   # v55: per-GPU state
    backend = _st.get("backend")
    method = _st.get("method")
    _hi = NEW_TRACK_CONF if ENABLE_CONF_HYSTERESIS else CONF_THRESHOLD
    _lo = KEEP_TRACK_CONF if ENABLE_CONF_HYSTERESIS else 0.1
    _cmc = GMC_METHOD if ENABLE_GMC else None
    if TRACKER_MODE == "boosttrack":
        from boxmot.trackers.boosttrack.boosttrack import BoostTrack
        tracker = BoostTrack(
            reid_model=backend,
            device=_boxmot_dev(device),
            half=globals().get("REID_HALF", False),
            track_high_thresh=_hi,
            track_low_thresh=_lo,
            new_track_thresh=_hi,
            track_buffer=int(30 * LOST_TRACK_BUFFER_S),  # see note below
            match_thresh=BOTSORT_MATCH_THRESH,
            frame_rate=int(round(fps)),
        )
        _log.info(f"✅ tracker: BoostTrack with REAL {method.upper()} embeddings driving live association")
    elif TRACKER_MODE == "occluboost":
        # C3: BoostTrack + occlusion-aware Kalman damping + graveyard
        # re-association (GTA). Needs a boxmot newer than 19.x — raise loudly,
        # never silently fall back. Constructor args are signature-filtered so
        # minor upstream API drift doesn't crash the run.
        import inspect as _insp
        try:
            from boxmot import OccluBoost as _OB
        except ImportError:
            from boxmot.trackers.occluboost.occluboost import OccluBoost as _OB
        _want = dict(reid_model=backend, device=_boxmot_dev(device), half=globals().get("REID_HALF", False),
                     track_high_thresh=_hi, track_low_thresh=_lo,
                     new_track_thresh=_hi,
                     track_buffer=int(30 * LOST_TRACK_BUFFER_S),  # see note below
                     match_thresh=BOTSORT_MATCH_THRESH,
                     cmc_method=_cmc,
                     frame_rate=int(round(fps)))
        _sig = _insp.signature(_OB.__init__).parameters
        tracker = _OB(**{k: v for k, v in _want.items() if k in _sig})
        _log.info(f"✅ tracker: OccluBoost with REAL {method.upper()} embeddings "
              f"(occlusion-damped Kalman + GTA graveyard re-association)")
    else:
        from boxmot.trackers.botsort.botsort import BotSort
        tracker = BotSort(
            reid_model=backend,
            track_high_thresh=_hi,
            track_low_thresh=_lo,
            new_track_thresh=_hi,
            track_buffer=int(30 * LOST_TRACK_BUFFER_S),  # see note below
            match_thresh=BOTSORT_MATCH_THRESH,
            proximity_thresh=0.3,
            appearance_thresh=LIVE_APPEARANCE_THRESH,
            cmc_method=_cmc,
            frame_rate=int(round(fps)),
            with_reid=True,
        )
        _log.info(f"✅ tracker: BotSORT with REAL {method.upper()} embeddings driving live association")
    return tracker

def _dets_to_boxmot(dets):
    """sv.Detections -> boxmot's expected (N,6) x1,y1,x2,y2,conf,cls array."""
    if len(dets) == 0:
        return np.empty((0, 6), dtype=np.float32)
    conf = dets.confidence if dets.confidence is not None else np.ones(len(dets))
    cls = dets.class_id if dets.class_id is not None else np.zeros(len(dets))
    return np.column_stack([dets.xyxy.astype(np.float32),
                            np.asarray(conf, dtype=np.float32),
                            np.asarray(cls, dtype=np.float32)])


def _boxmot_to_dets(tracked):
    """boxmot TrackResults -> sv.Detections, so the rest of the pipeline
    (zones, events, crop banking) doesn't need to know which tracker ran.
    NOTE: sv.Detections.empty()'s tracker_id defaults to None, not an empty
    array, which breaks the downstream `zip(..., dets.tracker_id, ...)` — so
    the empty case is built explicitly rather than via .empty()."""
    if len(tracked) == 0:
        d = sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32))
        d.confidence = np.array([], dtype=np.float32)
        d.class_id = np.array([], dtype=int)
        d.tracker_id = np.array([], dtype=int)
        return d
    return sv.Detections(
        xyxy=np.asarray(tracked.xyxy, dtype=np.float32),
        confidence=np.asarray(tracked.conf, dtype=np.float32),
        class_id=np.asarray(tracked.cls, dtype=int),
        tracker_id=np.asarray(tracked.id, dtype=int),
    )


class _PerspectiveModel:
    """Scene-geometry prior, learned from this run's own detections.

    On a FIXED camera looking at a FLAT floor, a standing person's pixel height
    is a near-linear function of the y of their feet: people low in the frame
    are near and tall, people high in the frame are far and short. Fitting that
    line costs nothing (we already have every box) and gives us the one signal
    box-containment cannot provide — whether a detection is standing on the
    floor or floating above it.

    Robust by construction: y is binned and the MEDIAN height per bin is fitted,
    so a handful of bad boxes cannot tilt the line. Only isolated, confident
    detections are fed in, so a merged double-box does not teach it that people
    are twice as tall as they are.

    ponytail: least-squares over bin medians, not RANSAC. Bin medians already
    kill the outliers RANSAC would; upgrade only if a real fit is seen to drift.
    Phase 3 replaces this with a true 4-point homography and keeps this as the
    cross-check.
    """

    def __init__(self, n_bins=12, min_samples=None):
        self.n_bins = n_bins
        self.min_samples = (CARRIED_MIN_FIT_SAMPLES if min_samples is None
                            else min_samples)
        self.samples = []          # (foot_y, height)
        self._fit = None           # (a, b)  ->  h ~= a*foot_y + b
        self._n_at_fit = 0

    def add(self, foot_y, height):
        if height > 0:
            self.samples.append((float(foot_y), float(height)))

    @property
    def ready(self):
        return self._refit() is not None

    def _refit(self):
        n = len(self.samples)
        if n < self.min_samples:
            return None
        # refit every 25% growth — cheap, and the fit stabilises fast
        if self._fit is not None and n < self._n_at_fit * 1.25:
            return self._fit
        ys = [s[0] for s in self.samples]
        lo, hi = min(ys), max(ys)
        if hi - lo < 1e-6:
            return None
        buckets = [[] for _ in range(self.n_bins)]
        for y, h in self.samples:
            k = min(self.n_bins - 1, int((y - lo) / (hi - lo) * self.n_bins))
            buckets[k].append(h)
        pts = []
        for k, hs in enumerate(buckets):
            if len(hs) < 5:        # a bin with almost nothing in it is noise
                continue
            hs.sort()
            y_mid = lo + (k + 0.5) * (hi - lo) / self.n_bins
            pts.append((y_mid, hs[len(hs) // 2]))
        if len(pts) < 3:
            return None
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        den = sum((p[0] - mx) ** 2 for p in pts)
        if den <= 1e-9:
            return None
        a = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
        self._fit = (a, my - a * mx)
        self._n_at_fit = n
        return self._fit

    def expected_h(self, foot_y):
        f = self._refit()
        if f is None:
            return None
        h = f[0] * float(foot_y) + f[1]
        return h if h > 1.0 else None

    def describe(self):
        f = self._refit()
        if f is None:
            return (f"scene geometry NOT fitted ({len(self.samples)}/"
                    f"{self.min_samples} samples) — carried-suppression stays "
                    f"OFF, nothing is being deleted")
        return (f"scene geometry fitted from {len(self.samples)} isolated "
                f"detections: expected height = {f[0]:.3f}*foot_y + {f[1]:.0f}px")


def _feed_perspective(dets, model, min_conf=0.5):
    """Only ISOLATED, confident boxes teach the model — an overlapping pair may
    be one merged double-box, which would teach it the wrong scale."""
    if model is None or len(dets) == 0:
        return
    xy = dets.xyxy
    conf = dets.confidence
    for i in range(len(xy)):
        if conf is not None and float(conf[i]) < min_conf:
            continue
        if any(_boxes_occluding(xy[i], xy[j]) for j in range(len(xy)) if j != i):
            continue
        model.add(xy[i][3], xy[i][3] - xy[i][1])


def _suppress_carried(dets, persp=None, stats=None):
    """Drop a detection that is a CARRIED person (baby in arms), and only that.

    THREE independent conditions must all hold, because each one alone deletes
    real guests on an oblique camera:

      1. containment   >=CARRIED_CONTAIN inside a bigger box, <=CARRIED_MAX_AREA_RATIO
         of its area. This alone was the old rule, and it also describes a guest
         standing further back — verified deleting real people.
      2. off the floor  height < CARRIED_HEIGHT_TOL x the height the fitted scene
         geometry predicts for a person standing at that footline. A guest
         further back matches the prediction and survives.
      3. head is low    the box top is >=CARRIED_MIN_HEAD_DROP down the carrier's
         box. Condition 2 alone still deleted a guest whose legs were hidden by
         the door frame (their box bottom is the OCCLUDER's edge, not their
         feet, so they look "off the floor" too). But their HEAD is at full
         height, level with or above the person in front — a carried child's
         head is down at the carrier's chest. This is what separates them.

    With no fitted geometry, nothing is suppressed at all. Deleting a real guest
    is far more expensive than briefly double-counting a baby.

    stats: optional dict, incremented so suppression is never silent.
    """
    if not ENABLE_CARRIED_SUPPRESS or len(dets) < 2:
        return dets
    if persp is None or not persp.ready:
        return dets
    xy = dets.xyxy
    area = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
    keep = np.ones(len(dets), dtype=bool)
    for i in range(len(dets)):
        h_i = xy[i, 3] - xy[i, 1]
        exp_h = persp.expected_h(xy[i, 3])
        if exp_h is None or h_i >= CARRIED_HEIGHT_TOL * exp_h:
            continue        # (2) consistent with standing on the floor
        for j in range(len(dets)):
            if i == j or area[j] <= 0 or area[i] <= 0:
                continue
            h_j = xy[j, 3] - xy[j, 1]
            if h_j <= 0:
                continue
            head_drop = (xy[i, 1] - xy[j, 1]) / h_j
            if head_drop < CARRIED_MIN_HEAD_DROP:
                continue    # (3) head is up at full height -> occluded guest
            ix1, iy1 = max(xy[i, 0], xy[j, 0]), max(xy[i, 1], xy[j, 1])
            ix2, iy2 = min(xy[i, 2], xy[j, 2]), min(xy[i, 3], xy[j, 3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if (inter / area[i] >= CARRIED_CONTAIN            # (1)
                    and area[i] <= CARRIED_MAX_AREA_RATIO * area[j]):
                keep[i] = False
                if stats is not None:
                    stats["carried_suppressed"] = stats.get("carried_suppressed", 0) + 1
                break
    return dets[keep]


def _drop_implausible(dets, persp, stats=None):
    """D1: remove boxes too large to be a person standing at their own footline.

    Uses the scene-geometry fit already maintained for carried-person
    suppression, so there is no new model and no new calibration. With no fit
    yet, nothing is dropped — a filter that fires on missing information is
    worse than no filter.
    """
    if not ENABLE_SIZE_FILTER or len(dets) == 0 or persp is None or not persp.ready:
        return dets
    # A count with no denominator cannot be judged. "D1 dropped 29,124" reads
    # alarming or fine depending entirely on whether that is 2% or 60% of the
    # detections, and the first real run printed it without the total.
    if stats is not None:
        stats["size_seen"] = stats.get("size_seen", 0) + len(dets)
    # V62a: self-correcting tolerance. The first real run dropped 26.8% of ALL
    # detections on an unstable ground fit (implied camera height flapping
    # 1.3m<->3.7m). A quarter of detections being non-people is not a plausible
    # scene; a bad fit is. Past D1_MAX_DROP_FRAC the tolerance doubles, once,
    # loudly — and the drop rate keeps being tracked against the new bar.
    _tol = SIZE_FILTER_TOL
    if stats is not None:
        _seen = stats.get("size_seen", 0)
        _drop = stats.get("size_dropped", 0)
        if stats.get("d1_relaxed"):
            _tol = SIZE_FILTER_TOL * 2.0
        elif (_seen > globals().get("D1_GUARD_WARMUP", 2000)
                and _drop / max(_seen, 1) > globals().get("D1_MAX_DROP_FRAC", 0.12)):
            stats["d1_relaxed"] = True
            _tol = SIZE_FILTER_TOL * 2.0
            _log.info(f"📏 D1 GUARD: drop rate {_drop / max(_seen, 1):.0%} exceeds "
                  f"{globals().get('D1_MAX_DROP_FRAC', 0.12):.0%} — the geometry "
                  f"fit is not trustworthy; tolerance relaxed "
                  f"{SIZE_FILTER_TOL}x -> {_tol}x for the rest of this video. "
                  f"Supply ground_points in the venue profile for the real fix.")
    mask = implausible_size_mask(dets.xyxy, persp.expected_h, tol=_tol)
    if not any(mask):
        return dets
    if stats is not None:
        stats["size_dropped"] = stats.get("size_dropped", 0) + sum(mask)
    return dets[np.array([not m for m in mask], dtype=bool)]


def _detector_has_head_class(model):
    """TRUE only for a real 2-class person/head detector. Hard gate: on stock
    COCO weights class 1 is 'bicycle', and treating that as a head would invent
    people out of parked bikes."""
    try:
        names = {int(k): str(v).lower() for k, v in model.names.items()}
    except Exception:
        return False
    return names.get(0) == "person" and names.get(1) == "head" and len(names) == 2


def _split_person_head(dets, has_head):
    """-> (person_dets, head_dets). Heads are an OCCLUSION SIGNAL: a head with
    no person box around it is a body the detector lost behind someone else."""
    if len(dets) == 0 or dets.class_id is None:
        return dets, dets[np.zeros(len(dets), dtype=bool)]
    persons = dets[dets.class_id == 0]
    heads = dets[dets.class_id == 1] if has_head else dets[np.zeros(len(dets), dtype=bool)]
    return persons, heads


def _heads_without_person(heads, persons):
    """Heads whose centre falls in no person box — each one is a missed person."""
    if len(heads) == 0:
        return heads
    if len(persons) == 0:
        return heads
    px = persons.xyxy
    keep = []
    for hx1, hy1, hx2, hy2 in heads.xyxy:
        cx, cy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
        keep.append(not any(px[j, 0] <= cx <= px[j, 2] and px[j, 1] <= cy <= px[j, 3]
                            for j in range(len(px))))
    return heads[np.array(keep, dtype=bool)]


def _body_from_head(head_box, persp, frame_w, frame_h,
                    ratio=None, aspect=None):
    """An orphan head -> the person box the detector failed to draw.

    Two independent estimates of body height, combined:

      1. ANTHROPOMETRY. Head height is a stable fraction of stature — the
         classical figure-drawing canon is ~7.5 heads, and a detector's head
         box (crown to chin, no neck) sits near 7. This works with no scene
         knowledge at all, which matters because it is available on frame one.

      2. SCENE GEOMETRY. _PerspectiveModel already learns expected body height
         as a function of foot position from thousands of real detections on
         THIS camera. It is the better estimate once fitted, but it needs the
         feet — which is what we are solving for. So iterate: guess the height,
         place the feet, ask the plane what height that implies, repeat. Damped
         so it converges rather than oscillates; three passes is plenty.

    Returns xyxy clamped to the frame.
    """
    ratio = HEAD_TO_BODY_RATIO if ratio is None else ratio
    aspect = HEAD_RECOVERY_ASPECT if aspect is None else aspect
    hx1, hy1, hx2, hy2 = (float(v) for v in head_box)
    head_h = max(1.0, hy2 - hy1)
    cx = (hx1 + hx2) / 2.0
    body_h = ratio * head_h
    if persp is not None and getattr(persp, "ready", False):
        for _ in range(3):
            foot_y = min(hy1 + body_h, frame_h - 1.0)
            eh = persp.expected_h(foot_y)
            if not eh or eh <= 0:
                break
            body_h = 0.5 * body_h + 0.5 * float(eh)
    w = max(1.0, aspect * body_h)
    x1 = max(0.0, cx - w / 2.0)
    x2 = min(float(frame_w), cx + w / 2.0)
    y2 = min(float(frame_h), hy1 + body_h)
    return (x1, max(0.0, hy1), x2, y2)


def _split_merged_persons(persons, heads, min_head_conf=None):
    """One person box containing TWO heads is two people. Split it.

    WHY THIS IS PROVABLE, NOT HEURISTIC
        Symptom 9 — "multiple people can get merged into one detection" — is
        normally undiagnosable from a single box: nothing in a person box says
        how many bodies are inside it. But this detector emits a HEAD class as
        well, and a person cannot have two heads. Two head boxes inside one
        person box is therefore direct evidence of a merged detection, not an
        inference from size or aspect.

        A merged box is worse than a missed one. It has ONE id, so two people
        share an identity for as long as they overlap; its centre sits between
        them, so it triggers the wrong zone; and when it finally splits, both
        people are re-born as new ids. That is symptom 9 producing symptoms 4,
        15 and 11 at once.

    HOW THE SPLIT IS DRAWN
        Each head keeps its own horizontal band — the person box is divided at
        the midpoints between adjacent head centres — while every part keeps
        the original box's vertical extent, because the feet of both people
        are genuinely somewhere in that region and we cannot tell whose are
        whose. Confidence is inherited: this is a re-interpretation of a real
        detection, not a new one, so it is not penalised the way a
        head-recovered box is.
    """
    min_head_conf = (HEAD_RECOVERY_MIN_CONF if min_head_conf is None
                     else min_head_conf)
    if len(persons) == 0 or len(heads) == 0:
        return persons, 0
    h_conf = (heads.confidence if heads.confidence is not None
              else np.ones(len(heads), dtype=float))
    p_conf = (persons.confidence if persons.confidence is not None
              else np.ones(len(persons), dtype=float))
    p_cls = (persons.class_id if persons.class_id is not None
             else np.zeros(len(persons), dtype=int))

    out_boxes, out_conf, out_cls, n_split = [], [], [], 0
    for i, (px1, py1, px2, py2) in enumerate(persons.xyxy):
        inside = []
        for j, (hx1, hy1, hx2, hy2) in enumerate(heads.xyxy):
            if float(h_conf[j]) < min_head_conf:
                continue
            hcx, hcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
            if px1 <= hcx <= px2 and py1 <= hcy <= py2:
                inside.append(hcx)
        if len(inside) < 2:
            out_boxes.append((px1, py1, px2, py2))
            out_conf.append(float(p_conf[i])); out_cls.append(int(p_cls[i]))
            continue
        inside.sort()
        # cut at the midpoints between adjacent heads
        edges = [float(px1)]
        edges += [(inside[k] + inside[k + 1]) / 2.0
                  for k in range(len(inside) - 1)]
        edges.append(float(px2))
        for k in range(len(inside)):
            x1, x2 = edges[k], edges[k + 1]
            if (x2 - x1) < 4:
                continue
            out_boxes.append((x1, float(py1), x2, float(py2)))
            out_conf.append(float(p_conf[i]))
            out_cls.append(int(p_cls[i]))
        n_split += 1

    if not out_boxes:
        return persons, 0
    return sv.Detections(xyxy=np.array(out_boxes, dtype=float),
                         confidence=np.array(out_conf, dtype=float),
                         class_id=np.array(out_cls, dtype=int)), n_split


def _recover_bodies_from_heads(persons, heads, persp, frame_w, frame_h,
                               min_conf=None, conf_penalty=None):
    """Add a person box for every head the detector left orphaned.

    WHY THIS IS THE OCCLUSION FIX
        When two people overlap, the detector often returns ONE person box and
        two heads. The rear body is not "low confidence" — it is absent, so no
        threshold recovers it. The track dies, and when the person steps clear
        they are born as a new id. That single mechanism produces symptom 12
        (occlusion breaks tracking) and a large share of symptom 11 (flicker
        then a new id) and symptom 14 (unique counts inflate).

        _heads_without_person() has computed exactly this signal all along and
        the result was only ever COUNTED into a log line. ENABLE_HEAD_RECOVERY
        existed as a flag referenced in one f-string and nowhere else — there
        was no implementation behind it.

    Recovered boxes are marked by a confidence penalty: they are inferred, not
    observed, so they should lose to a real detection in any comparison and be
    visible as inferred in the funnel.
    """
    min_conf = HEAD_RECOVERY_MIN_CONF if min_conf is None else min_conf
    conf_penalty = (HEAD_RECOVERY_CONF_PENALTY if conf_penalty is None
                    else conf_penalty)
    orphans = _heads_without_person(heads, persons)
    if len(orphans) == 0:
        return persons, 0
    boxes, confs = [], []
    o_conf = (orphans.confidence if orphans.confidence is not None
              else np.ones(len(orphans), dtype=float))
    for i, hb in enumerate(orphans.xyxy):
        c = float(o_conf[i])
        if c < min_conf:
            continue
        bb = _body_from_head(hb, persp, frame_w, frame_h)
        if (bb[2] - bb[0]) < 4 or (bb[3] - bb[1]) < 8:
            continue        # degenerate; a head that size is noise
        boxes.append(bb)
        confs.append(max(0.01, c * (1.0 - conf_penalty)))
    if not boxes:
        return persons, 0
    new_xyxy = np.array(boxes, dtype=float)
    new_conf = np.array(confs, dtype=float)
    new_cls = np.zeros(len(boxes), dtype=int)      # class 0 == person
    if len(persons) == 0:
        merged = sv.Detections(xyxy=new_xyxy, confidence=new_conf,
                               class_id=new_cls)
    else:
        p_conf = (persons.confidence if persons.confidence is not None
                  else np.ones(len(persons), dtype=float))
        p_cls = (persons.class_id if persons.class_id is not None
                 else np.zeros(len(persons), dtype=int))
        merged = sv.Detections(
            xyxy=np.vstack([persons.xyxy, new_xyxy]),
            confidence=np.concatenate([p_conf, new_conf]),
            class_id=np.concatenate([p_cls, new_cls]))
    return merged, len(boxes)


def _frame_chroma(frame_bgr, small=None):
    """Mean |R-G| + |G-B| — how far apart the colour channels are.

    Brightness-independent on purpose: HSV saturation explodes on dark pixels
    because it divides by brightness, so it calls a dark COLOUR frame infrared.
    True greyscale stays near zero however dark the image is. This is the same
    test probe_environment() already uses; F1 just runs it every frame."""
    s = small if small is not None else cv2.resize(frame_bgr, (160, 90))
    b = s[:, :, 0].astype(np.int16)
    g = s[:, :, 1].astype(np.int16)
    r = s[:, :, 2].astype(np.int16)
    return float((np.abs(r - g) + np.abs(g - b)).mean())


def detection_sanity(video_path, label, n=3, device=None):
    """Visual checkpoint BEFORE the long run: raw detections on sample frames."""
    if not Path(video_path).exists():
        _log.info(f"(skip sanity check — {video_path} missing)")
        return
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or NATIVE_FPS_OVERRIDE or 25
    device_str = str(device or globals().get('DEVICE', 'cuda'))
    model = YOLO(DETECTOR_MODEL)
    shots, n_found = [], 0
    for frac in [0.1, 0.5, 0.85][:n]:
        idx = max(0, int(total * frac))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            continue
        dets = sv.Detections.from_ultralytics(
            model(fr, conf=CONF_THRESHOLD, iou=0.45, imgsz=YOLO_IMGSZ,
                  verbose=False, device=device_str)[0])
        dets = dets[dets.class_id == 0]
        fr = sv.BoxAnnotator(thickness=2).annotate(fr, dets)
        fr = cv2.resize(fr, (720, int(720 * fr.shape[0] / fr.shape[1])))
        shots.append((idx / fps, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        n_found = len(dets)
    cap.release()
    if not shots:
        _log.info(f"❌ {label}: no frame could be read — re-download the file.")
        return
    show_gallery(shots, f"STEP CHECK · person detections · {label} "
                        f"({n_found} people in last sample)", ncols=3)

def _hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

def _dim(bgr, f=0.55):
    """PHASE11_RENDER_LEGIBILITY: push a colour back so it reads as map, not
    as a claim about a person."""
    return tuple(int(max(0, min(255, c * f))) for c in bgr)


def _dashed_poly(frame, poly, colour, dash=11, gap=8, thick=1):
    """A dashed outline. Zones and person boxes were both solid rectangles, so
    a zone covering half the frame read as a giant detection. Dashed vs solid
    is a difference the eye resolves instantly, at any size, in greyscale IR
    footage too — unlike hue, which IR night frames wash out."""
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
    n = len(pts)
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        seg = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        if seg < 1:
            continue
        steps = max(1, int(seg // (dash + gap)))
        for k in range(steps + 1):
            a = min(1.0, (k * (dash + gap)) / seg)
            b = min(1.0, (k * (dash + gap) + dash) / seg)
            if a >= 1.0:
                break
            cv2.line(frame,
                     (int(p[0] + (q[0] - p[0]) * a), int(p[1] + (q[1] - p[1]) * a)),
                     (int(p[0] + (q[0] - p[0]) * b), int(p[1] + (q[1] - p[1]) * b)),
                     colour, thick, cv2.LINE_AA)


def _in_poly(poly, pt):
    return cv2.pointPolygonTest(poly.astype(np.float32), pt, False) >= 0

def render_annotated(video_path, out_path, frame_log, canon, roles, events,
                     crossings, polygons, zcolors, entry_line, eff_fps, step,
                     native_fps, zone_roles=None, proxy_dir=None, duration_s=None,
                     phantoms=None):
    """PASS 2: draw everything from FINAL identities. Returns snapshots."""
    # role-based zone lookup — NOT hardcoded zone names, so this works on any
    # venue (cafe wait_zone, store queue_zone, restaurant waiting, ...).
    zone_roles = zone_roles or classify_zones(list(polygons.keys()))
    WAIT_ZONES = {z for z, rs in zone_roles.items() if "wait" in rs}
    STAFF_ANCHOR_ZONES = {z for z, rs in zone_roles.items() if "staff" in rs}
    wait_ivs = defaultdict(list)          # canonical -> waiting intervals
    for e in events:
        if e["zone"] in WAIT_ZONES:
            wait_ivs[e["track_id"]].append((e["t_in"], e["t_out"]))
    for ivs in wait_ivs.values():
        ivs.sort()
    rec_staff_ivs = sorted((e["t_in"], e["t_out"]) for e in events
                           if e["zone"] in STAFF_ANCHOR_ZONES and e["role"] == "staff")
    # per-door crossing times, so each line can show ITS OWN count
    _in_by_line, _out_by_line = {}, {}
    for _c in crossings or []:
        _ln = _c.get("line") or "entry"
        (_in_by_line if _c.get("direction") == "in" else _out_by_line) \
            .setdefault(_ln, []).append(float(_c.get("t", 0.0)))
    for _d in (_in_by_line, _out_by_line):
        for _k in _d:
            _d[_k].sort()
    try:
        from .analytics import venue_entry_lines
        _venue_doors = venue_entry_lines(list(_in_by_line) + list(_out_by_line)
                                         + list(entry_line or {}))
    except Exception:
        _venue_doors = set()
    ins, outs = {}, {}                    # canonical -> first crossing t
    for c in crossings:
        cid = c["track_id"]
        if roles.get(cid) == "staff":
            continue
        d = ins if c["direction"] == "in" else outs
        d.setdefault(cid, c["t"])
    in_times = sorted(ins.values())
    out_times = sorted(outs.values())
    # v48: P1..PN display numbers by first appearance (staff keep their names)
    _pnum = {}
    if ENABLE_DISPLAY_RENUMBER:
        for _fi, _t, _boxes in frame_log:
            for _tid, *_ in _boxes:
                _c = canon.get(_tid, _tid)
                if _c not in _pnum:
                    _pnum[_c] = len(_pnum) + 1
    role_bgr = {"customer": _hex2bgr(ROLE_HEXES["customer"]),
                "staff": _hex2bgr(ROLE_HEXES["staff"]),
                "unknown": (150, 150, 150)}
    zone_bgr = {n: _hex2bgr(c) for n, c in zcolors.items()}
    trails = defaultdict(lambda: deque(maxlen=int(eff_fps * 2)))
    _trail_gap = {}

    def _claim_label_spot(placed, x, y, w, h, step=22, tries=8):
        """v53: collision-free labels — if this spot overlaps an already-drawn
        label, slide down until free, so every id stays readable."""
        for _ in range(tries):
            r = (x, y, x + w, y + h)
            if all(r[2] <= q[0] or r[0] >= q[2] or r[3] <= q[1] or r[1] >= q[3]
                   for q in placed):
                placed.append(r)
                return int(y)
            y += step
        placed.append((x, y, x + w, y + h))
        return int(y)

    def wait_clock(cid, t):
        total, inside = 0.0, False
        for s, e2 in wait_ivs.get(cid, ()):
            if t >= s:
                total += min(t, e2) - s
                if s <= t <= e2 + 0.6:
                    inside = True
        return total if inside else None

    cap = None if proxy_dir is not None else cv2.VideoCapture(str(video_path))
    writer = None
    _ff_pipe = None   # V68: init ONCE — per-frame reset respawned ffmpeg every frame
    snapshots, next_snap = [], 0.0
    render_index = []   # v55: night-time of every frame written
    # RENDER_WINDOW: draw only [start, end] seconds instead of the whole chunk.
    #
    # Rendering was all-or-nothing, and on an hour that is 11 minutes of wall
    # time and ~8 GB. So every A/B ran with --no-render for speed, which left
    # nothing to WATCH — and "is that box actually on a person?" is a question
    # no counter can answer. The result was numeric verdicts on runs nobody
    # could see, while the open visual questions stayed open.
    #
    # A window costs a fortieth of that and answers the same visual question:
    # the failures cluster (a crowded stretch, an IR switch, a blink-out), so
    # two chosen minutes are worth more than sixty unwatched ones. Filtering
    # here windows the ENTIRE pass — coasting, HUD smoothing, decode and
    # encode all key off `frames`.
    _rw = globals().get("RENDER_WINDOW") or None
    if _rw:
        _r0, _r1 = float(_rw[0]), float(_rw[1])
        _keep = {idx for idx, t, _ in frame_log if _r0 <= float(t) <= _r1}
        if not _keep:
            _log.warning(f"   RENDER_WINDOW {_r0:.0f}-{_r1:.0f}s selected NO "
                         f"frames (chunk spans "
                         f"{frame_log[0][1]:.0f}-{frame_log[-1][1]:.0f}s) — "
                         f"rendering the whole chunk instead")
        else:
            _log.info(f"   🎬 RENDER_WINDOW {_r0:.0f}-{_r1:.0f}s: drawing "
                      f"{len(_keep)} of {len(frame_log)} analysed frames")
            frame_log = [r for r in frame_log if r[0] in _keep]
    frames = {idx: list(boxes) for idx, t, boxes in frame_log}
    times = {idx: t for idx, t, boxes in frame_log}

    # A6: coast a briefly-missed box (<= RENDER_COAST_S) so a person whose
    # detection blinks for a frame or two keeps a box ON SCREEN. Conf
    # hysteresis keeps the TRACK alive but nothing drew a box on the missed
    # frames, and HUD smoothing hid the blink in the counter while the boxes
    # still flashed. Display-only: `frames` holds copies, frame_log untouched.
    _coast_max = max(1, int(round(eff_fps * globals().get("RENDER_COAST_S", 0.5))))
    _cord = sorted(frames)
    _last_seen = {}
    _n_coasted = 0
    for _n, _idx in enumerate(_cord):
        for _b in list(frames[_idx]):
            _tid = _b[0]
            if _tid in _last_seen:
                _pn, _pb = _last_seen[_tid]
                _g = _n - _pn - 1
                if 1 <= _g <= _coast_max:
                    for _m in range(1, _g + 1):
                        _f = _m / (_g + 1.0)
                        _ib = tuple(int(round(_pb[c] + (_b[c] - _pb[c]) * _f))
                                    for c in range(1, 5))
                        frames[_cord[_pn + _m]].append((_tid,) + _ib)
                        _n_coasted += 1
            _last_seen[_tid] = (_n, _b)
    if _n_coasted:
        _log.info(f"render coasting: filled {_n_coasted} blinked box-frame(s) "
              f"(gap <= {_coast_max} analysed frames)")

    # displayed-count smoothing: rolling median over HUD_SMOOTH_S so a person
    # whose detection blinks for a few frames doesn't wobble the dashboard
    ordered = sorted(frames)
    half = max(1, int(round(HUD_SMOOTH_S * eff_fps / 2)))
    raw_counts, raw_zone = [], []
    for idx in ordered:
        zc = defaultdict(int)
        for tid, x1, y1, x2, y2 in frames[idx]:
            for name, poly in polygons.items():
                # one rule, one definition — see helpers.uses_centre_anchor
                _c = uses_centre_anchor(name, STAFF_ANCHOR_ZONES)
                if _in_poly(poly, anchor_point((x1, y1, x2, y2), _c)):
                    zc[name] += 1
        raw_counts.append(len(frames[idx]))
        raw_zone.append(zc)

    def _med(vals):
        s = sorted(vals)
        return s[len(s) // 2]

    smooth_total, smooth_zone = {}, {}
    for k, idx in enumerate(ordered):
        lo, hi = max(0, k - half), min(len(ordered), k + half + 1)
        smooth_total[idx] = _med(raw_counts[lo:hi])
        smooth_zone[idx] = {name: _med([raw_zone[j].get(name, 0)
                                        for j in range(lo, hi)])
                            for name in polygons}
    max_idx = max(frames) if frames else -1
    _span_s = float(duration_s or ((max_idx + 1) / max(eff_fps, 1e-6)))
    if proxy_dir is not None:
        _idxs = sorted(int(p.stem) for p in Path(proxy_dir).glob("*.jpg"))
        _log.info(f"   rendering {len(_idxs)} frames that actually had someone in them "
              f"(of {len(frames)} analysed)")
    else:
        _idxs = list(range(max_idx + 1))
    _prev_rendered_t = None
    _native_pos = 0   # cv2 fallback: next native frame index the cap will read
    try:
        for frame_idx in tqdm(_idxs, desc="render"):
            if proxy_dir is not None:
                frame = cv2.imread(str(Path(proxy_dir) / f"{frame_idx:07d}.jpg"))
                if frame is None:
                    continue
            else:
                # cv2 fallback: analysed index k lives at NATIVE frame k*step.
                # Sequential cap.read() drew native frames 0,1,2,... — i.e.
                # analysed frame k's boxes were drawn on the frame at t=k/30
                # instead of t=k*step/30 (4x temporal skew), at 4K while boxes
                # are in analysis coordinates. Skip-ahead and downscale.
                _tgt = frame_idx * step
                while _native_pos < _tgt:
                    if not cap.grab():
                        break
                    _native_pos += 1
                ok, frame = cap.read()
                _native_pos += 1
                if not ok:
                    break
                _amw = int(globals().get("ANALYSIS_MAX_W", 1280))
                if frame.shape[1] > _amw:
                    frame = cv2.resize(frame, (_amw,
                                       int(frame.shape[0] * _amw / frame.shape[1])))
            if frame_idx not in frames:
                continue
            t = times[frame_idx]
            if writer is None and _ff_pipe is None:
                h, w = frame.shape[:2]
                # v53: +62px letterbox band on top for the HUD, so the HUD never
                # covers people or labels inside the actual video area
                # v54: the output plays at PLAYBACK_FPS while the analysis ran
                # at eff_fps. Same frames, same numbers — only the playback clock
                # changes, so 10 h of footage becomes ~60-80 min to review.
                # Falls back to eff_fps when Cell 2e hasn't run (short clips).
                _play_fps = float(globals().get("PLAYBACK_FPS", eff_fps) or eff_fps)
                if globals().get("RENDER_DIRECT_H264") and _ffmpeg_ok():
                    # V64a: straight to h264 — the ~1.5GB raw mp4v intermediate
                    # never exists and the runner's re-encode pass is skipped.
                    import subprocess as _sp
                    if not Path(out_path).stem.endswith("_h264"):
                        out_path = Path(out_path).with_name(
                            Path(out_path).stem + "_h264.mp4")
                    _ff_pipe = _sp.Popen(
                        ["ffmpeg", "-y", "-loglevel", "error",
                         "-f", "rawvideo", "-pix_fmt", "bgr24",
                         "-s", f"{w}x{h + 88}", "-r", f"{_play_fps:.3f}",
                         "-i", "-",
                         "-c:v", "libx264", "-preset", "veryfast",
                         "-pix_fmt", "yuv420p",
                         "-crf", str(globals().get("RENDER_CRF", 28)),
                         "-movflags", "+faststart",   # browser can stream it
                         str(out_path)], stdin=_sp.PIPE,
                        stderr=_sp.PIPE)   # V73: never lose ffmpeg's reason
                    writer = None
                    _log.info(f"   render -> h264 directly (no raw intermediate)")
                else:
                    writer = cv2.VideoWriter(str(out_path),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             _play_fps, (w, h + 88))
                    if not writer.isOpened():
                        # a missing codec makes writer.write() a silent no-op
                        # for 27k frames and leaves a 0-byte file behind
                        raise RuntimeError(
                            f"VideoWriter failed to open {out_path} "
                            f"(mp4v codec missing?)")
                if _play_fps > eff_fps:
                    _log.info(f"   output plays {_play_fps/eff_fps:.1f}x faster than "
                          f"real time ({_play_fps:.0f} fps out, {eff_fps:.1f} analysed)")
            _lbl_placed = []   # v53: label rects drawn this frame (collision check)
            # PHASE11: zones are the MAP. Thin, dashed and dimmed, with a small name
            # tag at the polygon's top corner — never a solid rectangle, because a
            # solid rectangle now means exactly one thing: "this is a person".
            for name, poly in polygons.items():
                _zc = _dim(zone_bgr.get(name, (200,) * 3))
                _dashed_poly(frame, poly, _zc)
                _pt = min(poly.tolist(), key=lambda p: (p[1], p[0]))
                # clamp into frame: a polygon whose top corner sits at the right
                # edge had its name run off the side and vanish, which is exactly
                # the zone you most need named
                (_ztw, _zth), _ = cv2.getTextSize(str(name),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                _zx = min(max(int(_pt[0]) + 3, 2), max(2, frame.shape[1] - _ztw - 2))
                _zy = min(max(int(_pt[1]) - 4, _zth + 2), frame.shape[0] - 2)
                cv2.putText(frame, str(name), (_zx, _zy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, _zc, 1, cv2.LINE_AA)
            # anything D3 removed, shown struck through: a filter you cannot see is
            # a filter you cannot trust or debug
            for _ph in (phantoms or []):
                _a, _b, _c, _d = [int(v) for v in _ph["box"]]
                cv2.rectangle(frame, (_a, _b), (_c, _d), (90, 90, 90), 1)
                cv2.line(frame, (_a, _b), (_c, _d), (90, 90, 90), 1)
                cv2.line(frame, (_a, _d), (_c, _b), (90, 90, 90), 1)
                cv2.putText(frame, "IGNORED (static phantom)", (_a + 3, _b + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
            n_in = bisect.bisect_right(in_times, t)
            n_out = bisect.bisect_right(out_times, t)
            # EVERY DOOR, EACH WITH ITS OWN COUNT.
            #
            # This drew ONE line — list(drawn_lines.values())[0], whichever the
            # zone file happened to list first — and labelled it with the TOTAL
            # of all doors. On CAM.112 that was "dining entry", so the video
            # showed the DINING door wearing the street entrance's numbers, and
            # every review of the render concluded the entrance line was in the
            # wrong place. It was not; the label was.
            _doors = (entry_line if isinstance(entry_line, dict)
                      else ({"entry": entry_line} if entry_line else {}))
            for _dname, _dpts in _doors.items():
                (ex1, ey1), (ex2, ey2) = _dpts
                _di = bisect.bisect_right(_in_by_line.get(_dname, []), t)
                _do = bisect.bisect_right(_out_by_line.get(_dname, []), t)
                # the venue entrance is the number that matters — draw it red,
                # interior thresholds grey, so one glance says which is which
                _isv = _dname in _venue_doors
                cv2.line(frame, (ex1, ey1), (ex2, ey2),
                         (60, 60, 230) if _isv else (120, 120, 120), 3 if _isv else 2)
                mx, my = (ex1 + ex2) // 2, (ey1 + ey2) // 2
                tag = f"{_dname}  IN {_di} | OUT {_do}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                _ty = _claim_label_spot(_lbl_placed, mx - 6, my - th - 12,
                                        tw + 12, th + 12)
                cv2.rectangle(frame, (mx - 6, _ty), (mx + tw + 6, _ty + th + 12),
                              (25, 25, 25), -1)
                cv2.putText(frame, tag, (mx, _ty + th + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (90, 90, 255), 2, cv2.LINE_AA)
            zone_counts = defaultdict(int)
            n_role = {"customer": 0, "staff": 0}
            boxes = frames[frame_idx]
            for tid, x1, y1, x2, y2 in boxes:
                cid = canon.get(tid, tid)
                role = roles.get(cid, "customer")
                n_role[role if role in n_role else "customer"] += 1
                bc = anchor_point((x1, y1, x2, y2), False)
                trails[cid].append((int(bc[0]), int(bc[1])))
                for name, poly in polygons.items():
                    anchor = anchor_point(
                        (x1, y1, x2, y2),
                        uses_centre_anchor(name, STAFF_ANCHOR_ZONES))
                    if _in_poly(poly, anchor):
                        zone_counts[name] += 1
                color = role_bgr.get(role, role_bgr["unknown"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                if isinstance(cid, str) and not str(cid).isdigit():
                    lbl = f"{cid} {role}"          # enrolled staff keep their name
                elif ENABLE_DISPLAY_RENUMBER and cid in _pnum:
                    # v53: "customer" is the default and just adds width in a
                    # crowded queue — show it only when the role is notable
                    lbl = (f"P{_pnum[cid]}" if role == "customer"
                           else f"P{_pnum[cid]} {role}")
                else:
                    lbl = f"#{cid} {role}"
                wc = wait_clock(cid, t)
                if wc is not None and role != "staff":
                    lbl += f" {mmss(wc)}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                _ly0 = max(y1 - th - 10, 0)
                _ly = _claim_label_spot(_lbl_placed, x1, _ly0, tw + 8, th + 10)
                # PHASE11: collision avoidance can slide a label up to 176px away.
                # Without a leader line it then reads as belonging to whoever it
                # landed on — in a queue that is worse than an overlapping label.
                if _ly - _ly0 > 6:
                    cv2.line(frame, (x1 + 3, _ly + th + 10), (x1 + 3, y1),
                             color, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, _ly), (x1 + tw + 8, _ly + th + 10),
                              color, -1)
                cv2.putText(frame, lbl, (x1 + 4, _ly + th + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                            cv2.LINE_AA)
            present_now = {canon.get(tid, tid) for tid, *_ in boxes}
            for cid in list(trails):                 # prune departed people's
                if cid not in present_now:           # trails after ~1.5s
                    _trail_gap[cid] = _trail_gap.get(cid, 0) + 1
                    if _trail_gap[cid] > int(eff_fps * 1.5):
                        trails.pop(cid, None)
                        _trail_gap.pop(cid, None)
                else:
                    _trail_gap.pop(cid, None)
            if TRAIL_MODE != "off":
                for cid in present_now:
                    pts = trails.get(cid)
                    if not pts or len(pts) < 2:
                        continue
                    if TRAIL_MODE == "moving":
                        dx = pts[-1][0] - pts[0][0]
                        dy = pts[-1][1] - pts[0][1]
                        if dx * dx + dy * dy < 1600:  # <40px net = not walking
                            continue
                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False,
                                  (200, 200, 60), 2)
            # v53: zone counts moved OFF the video onto the HUD band (3rd line) —
            # nothing is ever drawn over people any more; polygons stay outlined
            # ── #18 STALE STATUS ────────────────────────────────────────────
            # `staffed` used to read rec_staff_ivs, which are the GAP-MERGED
            # event intervals: OccupancyRecorder bridges absences up to
            # GAP_MERGE_S (15 s) so a dwell is not shredded by one missed
            # detection. Correct for measuring dwell, WRONG for a live badge —
            # it displayed STAFFED through a 14-second absence you can watch
            # happening. The HUD describes THIS FRAME, so ask this frame.
            staffed = any(
                roles.get(canon.get(_tid, _tid)) == "staff"
                and any(_in_poly(polygons[_z],
                                 anchor_point((_x1, _y1, _x2, _y2),
                                              uses_centre_anchor(_z, STAFF_ANCHOR_ZONES)))
                        for _z in (STAFF_ANCHOR_ZONES & set(polygons)))
                for _tid, _x1, _y1, _x2, _y2 in boxes)

            # ── #17 IMPOSSIBLE COUNTS ───────────────────────────────────────
            # "5 people (6 staff)". The old comment blamed "two different
            # sources on one line" and patched it with max(), which HIDES the
            # contradiction instead of removing it: the total was still the
            # temporally-smoothed median while the staff count was raw, so the
            # two numbers described different instants and max() just stopped
            # the lie being arithmetically obvious.
            #
            # A HUD line that says "in frame" must be THIS frame. Both numbers
            # now come from `boxes`, so the total cannot be smaller than its
            # part — not because it is clamped, but because they are counted
            # from the same list. The smoothed series stays for the zone
            # occupancy line below, where a rolling median is the right thing.
            _hud_total = n_role["customer"] + n_role["staff"]
            hud = (f"t={mmss(t)}  people in frame: {_hud_total}"
                   f" ({n_role['staff']} staff)  "
                   f"entered={n_in} exited={n_out} (unique)")
            wait_now = sum(smooth_zone[frame_idx].get(z, 0) for z in WAIT_ZONES)
            hud2 = f"waiting now: {wait_now}"
            if _prev_rendered_t is not None and (t - _prev_rendered_t) > 30:
                hud2 += f"   [skipped {mmss(t - _prev_rendered_t)} — nobody present]"
            _prev_rendered_t = t
            if STAFF_ANCHOR_ZONES & set(polygons):
                _label = sorted(STAFF_ANCHOR_ZONES & set(polygons))[0]
                hud2 += f"   {_label}: {'STAFFED' if staffed else 'AWAY'}"
            # A7: the build that made this video, burned in — we could not tell
            # which build a reviewed video came from, so fixes looked like no-ops
            hud2 += f"   build {str(globals().get('_BUILD_ID', '?'))[:12]}"
            # v53: HUD lives on a band ADDED ABOVE the frame (letterbox), not
            # painted over the video — the full camera view stays visible
            frame = cv2.copyMakeBorder(frame, 88, 0, 0, 0,
                                       cv2.BORDER_CONSTANT, value=(25, 25, 25))
            cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, hud2, (10, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (170, 220, 170), 1, cv2.LINE_AA)
            # PHASE11: a mask zone exists to DELETE detections. Reporting "how many
            # people are in it" is meaningless and invites the reader to treat a
            # suppression region as a place guests stand.
            _mask_z = {z for z, rs in (zone_roles or {}).items() if "mask" in rs}
            hud3 = "   ".join(f"{n}: {smooth_zone[frame_idx].get(n, 0)}"
                              for n in polygons if n not in _mask_z)
            cv2.putText(frame, hud3, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 200, 255), 1, cv2.LINE_AA)
            # v54: real wall clock + where we are in the true timeline. Without
            # these a fast-forwarded video is actively misleading — every gap
            # looks continuous and an abandoned desk reads as two seconds.
            _clock = wall(t) if globals().get("VIDEO_START_CLOCK") else mmss(t)
            _tz = globals().get("DRIVE_TZ", "")
            cv2.putText(frame, f"{_clock} {_tz}", (frame.shape[1] - 240, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 235, 190), 2, cv2.LINE_AA)
            _span = max(_span_s, 1e-6)
            _fill = int((frame.shape[1] - 20) * min(1.0, t / _span))
            cv2.rectangle(frame, (10, 84), (frame.shape[1] - 10, 87), (70, 70, 70), -1)
            cv2.rectangle(frame, (10, 84), (10 + _fill, 87), (255, 190, 90), -1)
            cv2.putText(frame, f"{mmss(t)} / {mmss(_span)} real",
                        (frame.shape[1] - 240, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (170, 170, 170), 1, cv2.LINE_AA)
            if _ff_pipe is not None:
                try:
                    _ff_pipe.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError) as _pe:
                    _log.error(f"   !! h264 pipe died ({_pe}) — video stops here; "
                          f"analysis results are unaffected")
                    _ff_pipe = None
                    # stop the render entirely: looping on would keep growing
                    # render_index past the last written frame, and every
                    # moment clip cut from the .index.json would then land on
                    # the wrong seconds.
                    break
            elif writer is not None:
                writer.write(frame)
            render_index.append(round(float(t), 3))
            if t >= next_snap:
                small = cv2.resize(frame, (720, int(720 * frame.shape[0]
                                                    / frame.shape[1])))
                snapshots.append((t, cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                next_snap += SNAPSHOT_EVERY_S
    finally:
        # finalize even on a crash: an unfinalized mp4 (no moov atom) is
        # unplayable. The pipe close was previously DEAD CODE nested under
        # `if writer:` — writer and _ff_pipe are mutually exclusive, so the
        # h264 file was only ever finalized by garbage collection.
        if cap is not None:
            cap.release()
        if _ff_pipe is not None:
            try:
                _ff_pipe.stdin.close()
                _ff_pipe.wait(timeout=600)
            except Exception as _fe:
                _log.error(f"   !! ffmpeg finalize: {_fe}")
        if writer is not None:
            writer.release()
    try:
        # absolute night time, always: the consumer (moment clips) works on the
        # night's clock, and a chunk-local index silently mis-cuts every clip.
        _shift = float(globals().get("_RENDER_INDEX_SHIFT", 0.0))
        Path(str(out_path)).with_suffix(".index.json").write_text(
            json.dumps([round(v + _shift, 3) for v in render_index]))
    except Exception as _ix:
        _log.info(f"(render index not written: {_ix})")
    return snapshots

def probe_environment(video_path, n_probes=10, start_s=0.0, span_s=None):
    """Is THIS chunk infrared? (Cell 2e answers it for chunk 1 only, and a
    night shift crosses dusk halfway through.) Colour-based identity signals
    are meaningless on a greyscale image, so the attire tier has to be
    decided per chunk, not per run."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    _fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lo = int(start_s * _fps)
    hi = min(n, int((start_s + span_s) * _fps)) if span_s else n
    sats, chromas = [], []
    for k in range(n_probes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(lo + (hi - lo) * k / n_probes))
        ok, fr = cap.read()
        if not ok:
            continue
        sats.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()))
        # brightness-independent: how far apart the colour channels are. True
        # greyscale stays ~0-3 however dark the image is, where saturation
        # explodes on dark pixels because it divides by brightness.
        b, g, r = (fr[:, :, i].astype(np.int16) for i in range(3))
        chromas.append(float((np.abs(r - g) + np.abs(g - b)).mean()))
    cap.release()
    if not sats:
        return {"is_ir": False, "switches": False, "sat_min": None,
                "sat_max": None, "chroma_min": None, "chroma_max": None}
    thr = float(globals().get("IR_SAT_THRESHOLD", 12.0))
    cthr = float(globals().get("IR_CHROMA_THRESHOLD", 6.0))
    is_ir = (max(sats) < thr) or (max(chromas) < cthr)
    switches = (not is_ir) and (max(sats) > thr > min(sats)
                                or max(chromas) > cthr > min(chromas))
    return {"is_ir": is_ir, "switches": switches,
            "sat_min": min(sats), "sat_max": max(sats),
            "chroma_min": min(chromas), "chroma_max": max(chromas)}


def process_video(camera_id, video_path, zones_path, max_seconds=None, device=None,
                  chunk_tag="", start_seconds=0.0):
    # Open the video FIRST so we know its real frame size before loading
    # zones — zone JSON files are often drawn against a different reference
    # resolution (a screenshot, a different export of the same footage),
    # so load_zone_config auto-scales polygon coordinates to whatever this
    # video's actual frame_w x frame_h turns out to be.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps != native_fps or native_fps < 1:
        native_fps = NATIVE_FPS_OVERRIDE or 25
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(native_fps / FPS_TARGET))
    frame_step_s = step / native_fps
    eff_fps = 1.0 / frame_step_s

    # v55 CRITICAL: frames are analysed at ANALYSIS width (<=1920), not at the
    # camera's native 4K. Zones must be scaled to the frame the detector
    # actually sees, or every polygon sits at twice the coordinates of every
    # box and no zone ever triggers correctly. Must match frame_source()'s
    # rounding exactly.
    ANALYSIS_MAX_W = int(globals().get("ANALYSIS_MAX_W", 1920))
    an_w = min(ANALYSIS_MAX_W, frame_w) // 2 * 2
    an_h = int(round(frame_h * an_w / max(frame_w, 1))) // 2 * 2
    if (an_w, an_h) != (frame_w, frame_h):
        _log.info(f"\U0001f4d0 analysing at {an_w}x{an_h} (source {frame_w}x{frame_h}) — "
              f"zones scaled to the analysed frame, not the source")
    _src_w, _src_h = frame_w, frame_h   # V65: native dims, for face re-crop
    frame_w, frame_h = an_w, an_h
    polygons, entry_lines = load_zone_config(zones_path, frame_size=(frame_w, frame_h))
    zcolors = zone_color_map(polygons)
    # zone ROLES (wait / staff / seating / entry / service / other) are
    # derived from each zone's NAME via keyword matching (see classify_zones
    # in the ZONES cell) — nothing here is tied to "restaurant" vocabulary,
    # so the exact same process_video() runs on a cafe, a store, or anything
    # else whose zones_<stem>.json uses recognizable zone names.
    # Read ONCE, here, before the first use. This used to be loaded 240 lines
    # further down, so every per-door setting below raised UnboundLocalError
    # the moment a run reached it — a runtime-only failure that compiles clean
    # and passes every test that cannot import torch.
    try:
        _zone_cfg_raw = json.loads(Path(zones_path).read_text())
    except Exception:
        _zone_cfg_raw = {}
    zcfg_flip = (_zone_cfg_raw or {}).get("entry_lines_flip") or {}
    zcfg_band = (_zone_cfg_raw or {}).get("entry_lines_band") or {}
    zcfg_confirm = (_zone_cfg_raw or {}).get("entry_lines_confirm_s") or {}
    if zcfg_flip:
        _log.info(f"\U0001f6aa per-door IN/OUT direction: {zcfg_flip}")
    zone_roles = classify_zones(list(polygons.keys()))
    staff_zones_here = {z for z, rs in zone_roles.items() if "staff" in rs}
    _mask_names = {z for z, rs in zone_roles.items() if "mask" in rs}
    if _mask_names:
        _log.info(f"DEAD AREAS active (reflection/poster suppression): "
              f"{sorted(_mask_names)}")

    def _drop_masked(dets):
        """v53: detections whose feet land inside a mask/ignore zone are
        phantoms (mirror, glass reflection, poster, TV) - dropped before
        they can mint ids or pollute counts."""
        if not _mask_names or len(dets) == 0:
            return dets
        _keep = []
        for (_mx1, _my1, _mx2, _my2) in dets.xyxy:
            _bc = ((_mx1 + _mx2) / 2.0, _my2)
            _keep.append(not any(_in_poly(polygons[_n], _bc)
                                 for _n in _mask_names))
        return dets[np.array(_keep, dtype=bool)]

    # ── ZONE-COMPLETENESS GUARD ─────────────────────────────────────────────
    # Silent failure mode discovered 2026-07-12: cafe_004 zones file only had
    # "customers"/"staff" polygons and no entry_line — Q1 (people entered)
    # came back as raw in=None/out=None (line_zone never built at all, not a
    # detection failure), and Q2/Q3/Q5/Q6/Q7 were silently ABSENT from
    # answers.json (each gated behind "if seating_zones:" / "if wait_zones:"
    # etc., which just skip quietly when nothing matches that role). Nothing
    # anywhere printed a warning — the run looked clean (self-audit green)
    # while 5 of 7 business questions were never computed. This guard makes
    # that loud instead of silent, at the one place both facts are known:
    # right after roles are classified and before any answer is computed.
    _present_roles = {r for rs in zone_roles.values() for r in rs}
    _missing_roles = [r for r in ("wait", "seating") if r not in _present_roles]
    if not entry_lines:
        _missing_roles = ["entry_line (no line-crossing at all)"] + _missing_roles
    if _missing_roles:
        _log.error(f"🚨 ZONE GAP · {camera_id}: missing {_missing_roles} — the questions "
              f"depending on these will be silently ABSENT from answers.json, not "
              f"zero. Draw the missing zone(s)/entry_line in this venue's "
              f"zones_<stem>.json before treating any answer here as complete.")
    else:
        _log.info(f"✅ {camera_id}: entry_line + entry/wait/seating zones all present")

    device_str = str(device or globals().get('DEVICE', 'cuda'))
    model = YOLO(DETECTOR_MODEL)
    _dk = ("fine-tuned dense-scene"
           if ("crowdhuman" in Path(DETECTOR_MODEL).name.lower()
               or Path(DETECTOR_MODEL).name.lower() == "best.pt") else "stock")
    _log.info(f"ℹ️  {_dk} detector (person only) — staff vs customer comes "
          f"entirely from the staff-zone override (this video's own zones file)")

    det_conf = DETECT_CONF_FLOOR if ENABLE_CONF_HYSTERESIS else CONF_THRESHOLD
    if probe_environment(video_path, start_s=start_seconds,
                         span_s=max_seconds)["is_ir"]:
        # IR/greyscale: every body scores lower than it would in colour, so the
        # same floor silently drops people at night. CLAHE (below) restores
        # contrast; this restores recall.
        det_conf = min(det_conf, IR_DETECT_CONF_FLOOR)

    use_online_tracker = TRACKER_MODE in ("botsort-reid", "boosttrack",
                                          "occluboost")   # F6b: without
    # this the ablation variant built OccluBoost and then fell through to
    # BYTETRACK — the A/B would have measured nothing.
    online_tracker = None
    tracker = None

    def _fresh_tracker():
        """v55: a tracker counts FRAMES. After the motion gate skips a long
        empty stretch, its idea of 'recently lost' is minutes stale, and it
        will happily reattach an abandoned id to whoever walks in next. Start
        it clean instead — bridging a gap that long is the offline stitcher's
        job, and it has appearance evidence the live tracker does not."""
        if use_online_tracker:
            return build_online_tracker(eff_fps, device=device_str)
        return sv.ByteTrack(
            track_activation_threshold=max(0.1, CONF_THRESHOLD - 0.1),
            lost_track_buffer=int(30 * LOST_TRACK_BUFFER_S),
            minimum_matching_threshold=0.8,
            frame_rate=int(round(eff_fps)),
            minimum_consecutive_frames=2,
        )

    if use_online_tracker:
        online_tracker = build_online_tracker(eff_fps, device=device_str)
        if online_tracker is None:
            # fallback: previous path (Ultralytics-wrapped, detector-feature
            # proxy for with_reid — kept working, just weaker on overlap)
            tracker_yaml = _botsort_yaml(eff_fps * LOST_TRACK_BUFFER_S)
            tracker = None
            _log.info(f"tracker: BotSORT + inline ReID ({tracker_yaml}, model=auto — "
                  f"detector-feature proxy; real CLIP-ReID/OSNet runs in the "
                  f"offline stitcher pass only)")
    else:
        tracker = _fresh_tracker()

    # ── v45 RUN DIAGNOSTIC: make the LIVE-association appearance source
    # (real ReID model vs weak auto-proxy) unmissable in every Kaggle log ─────
    if use_online_tracker and online_tracker is not None:
        _appear = f"REAL {(_REID_STATE.get('method') or '?').upper()} embeddings (online tracker)"
    elif use_online_tracker:
        _appear = "WEAK detector-feature proxy (with_reid=auto) — real ReID only runs OFFLINE!"
    else:
        _appear = "ByteTrack (motion only, no appearance)"
    _sc = ((frame_w**2 + frame_h**2) ** 0.5) / REF_DIAGONAL_PX
    _log.info("+" + "-" * 68)
    _log.info(f"| RUN DIAGNOSTIC · {camera_id}")
    _log.info(f"|   detector        : {DETECTOR_MODEL} @ imgsz {YOLO_IMGSZ}")
    _log.info(f"|   tracker          : {TRACKER_MODE}")
    _log.info(f"|   live appearance  : {_appear}")
    _log.info(f"|   sample fps       : {FPS_TARGET} (native {native_fps:.1f}, step {step})")
    _log.info(f"|   conf hysteresis  : {'ON' if ENABLE_CONF_HYSTERESIS else 'off'} "
          f"(detect>={DETECT_CONF_FLOOR}, new>={NEW_TRACK_CONF}, keep>={KEEP_TRACK_CONF})")
    _log.info(f"|   occlusion guard  : {'ON' if ENABLE_OCCLUSION_GUARD else 'off'}  "
          f"co-visibility: {'ON' if ENABLE_COVISIBILITY_BLOCK else 'off'}")
    _log.info(f"|   GMC              : {GMC_METHOD if ENABLE_GMC else 'none'}")
    _log.info(f"|   res-scaling      : {'ON' if ENABLE_RESOLUTION_SCALING else 'off'} "
          f"(frame {frame_w}x{frame_h}, scale {_sc:.2f}x)")
    _log.info("+" + "-" * 68)

    # zone anchors: feet everywhere EXCEPT desk zones (desk-clipped people's
    # feet land outside the polygon — run-2 QA)
    zones = {}
    for name, poly in polygons.items():
        if name in _mask_names:
            continue
        anchor = (sv.Position.CENTER
                  if uses_centre_anchor(name, staff_zones_here)
                  else sv.Position.BOTTOM_CENTER)
        zones[name] = sv.PolygonZone(polygon=poly, triggering_anchors=(anchor,))
    # U3: one LineZone per door. Crossings carry which door, so a two-entrance
    # venue gets per-door counts and a correct total instead of nothing.
    line_zones, drawn_lines = {}, {}
    for _lname, _lpts in entry_lines.items():
        (x1, y1), (x2, y2) = _lpts
        # WHICH SIDE IS "IN" IS PER DOOR, NOT GLOBAL.
        #
        # A LineZone's in/out depends on the ORDER of its two points, i.e. on
        # the direction the operator happened to drag the mouse. ENTRY_LINE_FLIP
        # was one boolean applied to EVERY line, so with three hand-drawn doors
        # it is arithmetically impossible for it to be right for all of them:
        # flipping to fix the entrance inverts dining and the staff gate.
        #
        # The zone file may now carry  "entry_lines_flip": {"<door>": true/false}
        # per door; anything unlisted falls back to the global default, so an
        # existing single-door venue behaves exactly as before.
        _flip = (zcfg_flip.get(_lname, ENTRY_LINE_FLIP)
                 if isinstance(zcfg_flip, dict) else ENTRY_LINE_FLIP)
        if _flip:
            (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
        # BUFFER BAND, per door. minimum_crossing_threshold is how many
        # consecutive frames a track must sit on the far side before the
        # crossing is admitted — the "person shifting their weight on the line"
        # guard. 2 is thin where guests pause (a host stand); brisk interior
        # thresholds want it low or real transits get eaten.
        _band = int((zcfg_band or {}).get(_lname, LINE_CROSS_FRAMES))
        # WHICH POINT ON THE BODY DECIDES A CROSSING.
        #
        # BOTTOM_CENTER is the conventional choice, and it is right when the box
        # ends at the feet: the line is drawn on the floor, so the feet should
        # cross it. On this camera the box does NOT end at the feet. The
        # detector is CrowdHuman-trained (street-level, h/w ~2.5) looking at a
        # ceiling view where a foreshortened person is h/w ~1.14, so the box
        # runs well past them. Measured on the one frame whose hand labels were
        # not carried forward (n=5):
        #
        #     BOTTOM_CENTER   262 px from the true feet
        #     CENTER          173 px from the true centre    (1.5x closer)
        #
        # The count damage is worst exactly where it matters. A constant offset
        # mostly shifts WHEN a crossing fires, which preserves the count — but
        # near the frame edge the box bottom is clipped, or already beyond the
        # line before the person arrives, and then the crossing never fires at
        # all. Our entrance is at the bottom-right. That is consistent with
        # main_entrance having recorded 0 hits in 8,219 boxes and with the entry
        # lines needing a 60% extension to catch anything.
        #
        # Published practice agrees: centroid trajectories, not a per-frame foot
        # test. Left at BOTTOM_CENTER by default because switching the anchor
        # also changes WHERE a line should be drawn (a floor line is placed for
        # feet), and the zone lines were drawn for the current anchor. Flip it
        # with analysis.entry_line_anchor and compare crossings on one chunk.
        # WHAT THE LIBRARY ALREADY DOES — checked, 2026-08-18, so nobody rebuilds
        # it. supervision 0.26.1's LineZone:
        #   * compares each track's position ACROSS TWO FRAMES, i.e. it is
        #     already trajectory-based. Published practice for counting is
        #     "intersect the trajectory with the line, not a per-frame side
        #     test", and that is what this is. Do not write a segment-
        #     intersection layer on top; it exists.
        #   * minimum_crossing_threshold is the jitter guard, and we pass it as
        #     _band above.
        #   * triggering_anchors DEFAULTS TO ALL FOUR CORNERS, which demands the
        #     whole box clear the line. We deliberately override to ONE anchor:
        #     with boxes 1.6x too tall, requiring four corners would lose far
        #     more crossings than it prevents.
        #   * max_linger does not exist in 0.26.1 (it is newer). Nothing to set.
        #
        # So the crossing LOGIC is not the defect. The defect is the box we hand
        # it — see the anchor note below.
        _anchor_name = str(globals().get("ENTRY_LINE_ANCHOR", "bottom_center")).lower()
        _anchor = (sv.Position.CENTER if _anchor_name == "center"
                   else sv.Position.BOTTOM_CENTER)
        line_zones[_lname] = sv.LineZone(
            start=sv.Point(x1, y1), end=sv.Point(x2, y2),
            minimum_crossing_threshold=_band,
            triggering_anchors=[_anchor])
        drawn_lines[_lname] = [[x1, y1], [x2, y2]]
    if len(line_zones) > 1:
        _log.info(f"\U0001f6aa {len(line_zones)} entry line(s): {sorted(line_zones)}")

    # U1/U2 CAMERA HEALTH — one extra decoded frame, before anything expensive.
    # Chunk 1 lays down the reference view; every later chunk is compared to it,
    # so a camera knocked at 21:00 is caught even though chunk 7 is internally
    # consistent with itself.
    _view = {"valid": True, "reasons": [], "checked": False}
    try:
        _ref_path = OUTPUT_DIR / f"viewref_{camera_id}"
        _probe = next((f for _i, _t, f in frame_source(
            video_path, 1.0, max_seconds=2.0, max_w=ANALYSIS_MAX_W,
            start_seconds=start_seconds)), None)
        if _probe is not None:
            _hc = CameraHealth.load(_ref_path)
            if _hc is None:
                _hc = CameraHealth.from_frame(
                    _probe, zone_tol_frac=VENUE_PROFILE["camera"]["zone_tol_frac"])
                _hc.save(_ref_path)
                _log.info(f"\U0001f4f7 reference view saved for {camera_id} — later "
                      f"chunks are checked against THIS frame")
            _view = _hc.check(_probe, polygons)
            _view["checked"] = True
            _log.info(f"\U0001f4f7 {verdict_line(_view)}")
    except Exception as _hex:
        _log.info(f"(camera-health check skipped: {_hex})")

    rec = OccupancyRecorder(frame_step_s=frame_step_s,
                            gap_merge_s=GAP_MERGE_S, min_event_s=MIN_EVENT_S)
    crossings = []
    track_crops = defaultdict(list)     # tid -> best REID_CROPS_PER_TRACK (score, crop)
    track_pos = {}                      # tid -> [start_bc, last_bc]
    track_seated_frames = defaultdict(int)  # v42: tid -> count of frames where bbox looks seated
    track_total_frames = defaultdict(int)   # v42: tid -> total frames seen
    track_time = {}                     # tid -> [first_t, last_t]  (ALL tracks, not just zone-event ones)
    frame_log = []                      # (frame_idx, t, [(tid,x1,y1,x2,y2)])

    # ── LIVE IDENTITY MEMORY (v27) ──────────────────────────────────────────
    # Set up ONCE per video, before the frame loop. `_embed_one_live` reuses
    # the same lazy-cached embedder the offline stitcher uses — CLIP-ReID if
    # it loaded, else OSNet, else HSV (get_reid_embedder(device=device) is a no-op after
    # the first real call), so there's no separate model load — just an
    # extra call for each newly-BORN raw track_id, which is rare relative
    # to total frames.
    def _blur_score(crop_bgr, reference_size=128):
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0
        resized = cv2.resize(crop_bgr, (reference_size, reference_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _embed_one_live(crop):
        fn = get_reid_embedder(device=device)
        if fn is None or crop is None:
            return None
        try:
            vecs = fn([crop])
            return vecs[0] if vecs else None
        except Exception:
            return None

    _dist_scale = (((frame_w**2 + frame_h**2) ** 0.5) / REF_DIAGONAL_PX) if ENABLE_RESOLUTION_SCALING else 1.0
    # F4: the live re-id gate is a budget for how far someone can move in ONE
    # frame, so it only means anything relative to eff_fps. At 4 fps this is
    # 560/4 = 140px, exactly the old hardcoded value; at 8 fps it correctly
    # halves instead of staying twice as loose as it should be.
    _live_max_dist = max(40.0, LIVE_REID_MAX_SPEED_PX_S / max(eff_fps, 1e-6))
    _log.info(f"   live re-id gate: {_live_max_dist:.0f}px @ {eff_fps:.1f} fps "
          f"(derived from {LIVE_REID_MAX_SPEED_PX_S:.0f} px/s)")
    # G1: the plane is built lazily — the perspective fit needs samples, so
    # early frames run on pixel gates and everything after runs on metres. Same
    # pattern as carried-suppression, and for the same reason: a wrong metric
    # answer is worse than an honest pixel one.
    # _zone_cfg_raw already loaded above, before its first use.
    _ground = GroundPlane.none("perspective fit not ready yet")

    _ground_n = [0]

    def _refresh_ground():
        """B2: the plane used to be built once, the moment 200 samples existed,
        and never revisited — the first real run locked it to 200 of the 13,335
        detections eventually available and reported an implied camera height of
        1.12 m where the final fit implies 3.35 m. Now it is rebuilt whenever the
        evidence has doubled. An EXACT homography from ground_points never needs
        refitting and is left alone."""
        nonlocal _ground
        # MEASURED POINTS DO NOT WAIT FOR A GUESS.
        #
        # This used to return early on `not _persp.ready`, which gated the
        # EXACT homography behind the automatic perspective fit it exists to
        # replace. _persp needs 200 samples; the 2026-08-13 smoke run reached
        # 90. So a venue that had measured its floor and written ground_points
        # into the zones file got the "not fitted" fallback anyway, and nothing
        # said the measurement was being ignored -- the one failure mode that
        # makes an operator stop measuring.
        #
        # A homography from >=4 correspondences needs no samples, no warmup and
        # no video. Build it on the first call and keep it.
        if _ground.mode != "exact" and (_zone_cfg_raw or {}).get("ground_points"):
            _exact = GroundPlane.from_zone_config(
                _zone_cfg_raw, (frame_w, frame_h), persp=None,
                person_h=VENUE_PROFILE["camera"]["person_height_m"],
                hfov_deg=VENUE_PROFILE["camera"]["hfov_deg"])
            if _exact.ok and _exact.mode == "exact":
                _ground = _exact
                _log.info(f"\U0001f4d0 G1 EXACT plane from measured ground_points "
                          f"— {_ground.describe()}")
                for _w in _ground.sanity(frame_h):
                    _log.error(f"      !! {_w}")
                if _identity_memory is not None:
                    _identity_memory.plane = _ground
                return _ground
            _log.error("\U0001f4d0 ground_points ARE PRESENT but did not produce an "
                       "exact plane — check they are >=4, not collinear, and in "
                       "this zone file's coordinate space. Falling back to the "
                       "automatic fit, which is what they were meant to replace.")
        if not _persp.ready:
            return _ground
        _n = len(_persp.samples)
        if _ground.ok and (_ground.mode == "exact" or _n < _ground_n[0] * 2):
            return _ground
        _ground_n[0] = _n
        _ground = GroundPlane.from_zone_config(
            _zone_cfg_raw, (frame_w, frame_h), persp=_persp,
            person_h=VENUE_PROFILE["camera"]["person_height_m"],
            hfov_deg=VENUE_PROFILE["camera"]["hfov_deg"])
        if _ground.ok:
            _log.info(f"\U0001f4d0 G1 {_ground.describe()}")
            for _w in _ground.sanity(frame_h):
                _log.error(f"      !! {_w}")
            if _identity_memory is not None:
                _identity_memory.plane = _ground
        return _ground

    _identity_memory = (
        _IdentityMemory(embed_fn=_embed_one_live,
                        sim_threshold=LIVE_REID_SIM_THRESHOLD,
                        max_dist_px=_live_max_dist * _dist_scale,
                        memory_ttl_s=LIVE_REID_MEMORY_TTL_S,
                        dist_scale=_dist_scale,
                        max_speed_mps=MAX_WALK_SPEED_MPS)
        if ENABLE_LIVE_IDENTITY_MEMORY else None
    )
    _prev_occ_raw = set()   # v53: raw ids that were occluded LAST frame

    cap.release()   # v55: probing done; frame_source owns decoding from here
    duration_s = (n_frames / native_fps) if native_fps else 0.0
    if max_seconds:
        duration_s = min(duration_s, float(max_seconds))
    n_expected = int(duration_s * eff_fps) or None

    proxy_dir = None
    if PROXY_RENDER:
        proxy_dir = OUTPUT_DIR / f"_proxy_{camera_id}{chunk_tag}"
        if proxy_dir.exists():
            shutil.rmtree(proxy_dir, ignore_errors=True)
        proxy_dir.mkdir(parents=True, exist_ok=True)

    _env = probe_environment(video_path, start_s=start_seconds, span_s=max_seconds)
    _is_ir = bool(_env["is_ir"])
    if _is_ir:
        _log.info(f"\U0001f319 INFRARED / NIGHT VISION (saturation "
              f"{_env['sat_min']:.0f}-{_env['sat_max']:.0f}, colour spread "
              f"{_env['chroma_min']:.1f}-{_env['chroma_max']:.1f}) — attire/HSV "
              f"merging disabled, CLAHE on, detection floor lowered to "
              f"{IR_DETECT_CONF_FLOOR}. Colour evidence does not exist here.")
    elif _env["switches"]:
        _log.info(f"\u26a0\ufe0f  IR SWITCH INSIDE THIS CHUNK (saturation "
              f"{_env['sat_min']:.0f}-{_env['sat_max']:.0f}) — identities are NOT "
              f"merged on colour across it.")
    _clahe_on = bool(ENABLE_CLAHE or _is_ir)
    _attire_on = bool(ENABLE_ATTIRE_MERGE_TIER and not _is_ir)
    _staff_seen_names = set()

    _prev_small = None          # motion gate state
    # F1 per-frame IR state / F2 scene geometry / F3 head accounting
    _frame_ir = {}              # frame_idx -> bool
    _ir_state = [None]          # last seen IR state, for switch logging
    _ir_prev_main = [None]      # C4: last modality the MAIN loop acted on
    _ir_cut_last = [-1e9]       # C4: debounce clock for the hard cut
    _ir_switches = []           # (t, is_ir_now)
    _ir_pend = [0, None]        # V68b debounce: [streak, candidate]
    _exp_counts = {}            # exposure verdict -> frames (symptom 19)
    _track_ir_frames = defaultdict(int)
    _persp = _PerspectiveModel()
    _supp_stats = {}
    _has_head = _detector_has_head_class(model)
    _n_head_only = 0
    _n_recovered = [0]          # bodies rebuilt from orphan heads (symptom 12)
    _n_split = [0]              # merged person boxes split apart (symptom 9)
    _deadband = [0, 0]          # [dets in CONF..NEW_TRACK band, dets seen]
    _absurd = [0]               # D0: boxes too big to be a person, any geometry
    _absurd_why = [0, 0]        # [height-capped, area-capped]
    _absurd_h = []              # heights of what D0 threw away, in pixels
    _absurd_footy = []          # and where their feet were
    _prof = Profile(enabled=bool(globals().get('ENABLE_PROFILING', True)))
    _prof.start_wall()
    _tile_stats = [0, 0]        # [tiles run, frames tiled]
    # Non-circular separability measurement (symptoms 3/4/11/14). Costs one
    # cosine per co-visible pair per frame over vectors that already exist.
    # The pairing window must exceed REEMBED_EVERY_S or there are no pairs to
    # find: a track is re-embedded every REEMBED_EVERY_S seconds, so at 0.5s
    # the first real run collected same=3, diff=142 and could report nothing.
    #
    # HONEST CAVEAT, since this measurement's whole value is that it is not
    # circular: over a 6-second window the tracker's own association used
    # appearance (BoT-SORT runs with_reid=True), so "same raw id" is no longer
    # purely a motion fact. The SAME-FRAME negatives stay unimpeachable — a
    # person cannot be two boxes — but treat the positives as a lower bound.
    # kevacv/reid_calibration.calibrate() remains the authoritative same-person
    # number; this is the cheap per-run sanity check beside it.
    _sep = (LiveSeparability(max_same_gap_s=max(1.0, REEMBED_EVERY_S * 1.5))
            if ENABLE_LIVE_SEPARABILITY else None)

    # ── DETECTION FUNNEL ────────────────────────────────────────────────────
    # The chain below was copy-pasted into all three tracker branches, which is
    # how the ByteTrack and BotSORT paths silently ran DIFFERENT filters for a
    # while (see the A3 comments: _refresh_ground was missing from two of them,
    # so the metre gates were dead there and nobody could tell from the log).
    # One definition, three call sites, and every stage counted on the way past.
    _funnel = DetectionFunnel(label=f"{camera_id}{chunk_tag}")

    # Live phantom suppression shares the per-zone patience the end-of-chunk
    # static filter uses: 30 s of rigidity is furniture in a doorway, but the
    # reception desk needs minutes because people legitimately stand still
    # there. Suppressing the receptionist is far worse than keeping a plant.
    _zone_life = dict(globals().get("STATIC_MIN_LIFE_BY_ROLE") or {})

    def _life_at(px, py):
        best = None
        for _zn, _zp in polygons.items():
            if _zn in _mask_names or len(_zp) < 3:
                continue
            if _in_poly(_zp, (float(px), float(py))):
                for _r in zone_roles.get(_zn, ()):
                    _v = _zone_life.get(_r)
                    if _v is not None and (best is None or _v > best):
                        best = _v
        return STATIC_MIN_LIFE_S if best is None else best

    _live_phantoms = (
        OnlineStaticSuppressor((frame_w, frame_h), min_life_for=_life_at,
                               default_life_s=STATIC_MIN_LIFE_S)
        if (ENABLE_PHANTOM_FILTER and ENABLE_LIVE_PHANTOM_SUPPRESS) else None)
    if _live_phantoms is None:
        _log.info("live phantom suppression OFF — phantoms are removed only by "
                  "the end-of-chunk pass, after they have already consumed "
                  "canonical ids for the length of the chunk")

    def _filter_chain(dets, t):
        """Raw detector output -> what the tracker is allowed to see."""
        nonlocal _n_head_only
        _funnel.record_first("yolo raw", len(dets))

        # D0 ABSURD-SIZE CAP — runs FIRST, and the D1 guard cannot relax it.
        #
        # The audit's "huge box around the doorway", "P3 over the whole
        # right-side background" and "P11 over a large mostly empty area" all
        # pass every existing filter: aspect is inside MIN/MAX_BODY_ASPECT
        # (a 610x1070 box is 1.75:1), and D1 is geometry-based, so a bad ground
        # fit makes it DOUBLE its own tolerance and wave them through -- the
        # 2.5x -> 5.0x relax that made the giant boxes worse, not better.
        #
        # This cap needs no geometry at all. On a ceiling camera nobody is 90%
        # of the frame tall or a third of its area, whatever the plane says, so
        # it is safe to apply before any fit exists and wrong to make it
        # relaxable. Deliberately generous: it is a sanity bound on the
        # physically impossible, not a person-size model.
        _n = len(dets)
        if globals().get("ENABLE_ABSURD_SIZE_CAP", True) and len(dets):
            _hf = float(globals().get("MAX_BOX_HEIGHT_FRAC", 0.70))
            _af = float(globals().get("MAX_BOX_AREA_FRAC", 0.30))
            _fa = float(frame_w * frame_h)
            _ok = []
            for _x1, _y1, _x2, _y2 in dets.xyxy:
                _h = (_y2 - _y1) / max(frame_h, 1)
                _a = ((_x2 - _x1) * (_y2 - _y1)) / max(_fa, 1.0)
                _hbad, _abad = _h > _hf, _a > _af
                _ok.append(not (_hbad or _abad))
                if _hbad or _abad:
                    # WHICH bound fired, and how tall the reject was in pixels.
                    # D0 removed 26% of every detection in the 18:30 run and the
                    # funnel could only say "26%" — not whether it was killing
                    # frame-wide garbage or the near-field guests that F2, in
                    # the SAME run, measures as 804px tall at the frame bottom
                    # (0.745 of frame — above a 0.70 cap). One number that big
                    # with no breakdown is not evidence, it is a rumour.
                    _absurd_why[0 if _hbad else 1] += 1
                    _absurd_h.append(_y2 - _y1)
                    _absurd_footy.append(_y2)
            if not all(_ok):
                _absurd[0] += len(_ok) - sum(_ok)
                dets = dets[np.array(_ok, dtype=bool)]
        _funnel.record("D0 absurd size", _n, len(dets))

        _n = len(dets)
        dets, _heads = _split_person_head(dets, _has_head)
        _n_head_only += len(_heads_without_person(_heads, dets))
        _funnel.record("person/head split", _n, len(dets))
        # Occlusion recovery: a head with no body around it is a person the
        # detector lost behind someone else. ADDS detections, so the funnel
        # shows it as a stage whose out exceeds its in — the only one that
        # should, and worth seeing as such.
        if ENABLE_MERGED_SPLIT and _has_head:
            # Two heads in one person box is two people (symptom 9). Runs
            # BEFORE head recovery so the split heads are then correctly seen
            # as belonging to a body, not as orphans needing a new one.
            _n = len(dets)
            dets, _n_sp = _split_merged_persons(dets, _heads)
            _n_split[0] += _n_sp
            _funnel.record("split merged (+)", _n, len(dets))
        if ENABLE_HEAD_RECOVERY and _has_head:
            _n = len(dets)
            dets, _n_rec = _recover_bodies_from_heads(
                dets, _heads, _persp, frame_w, frame_h)
            _n_recovered[0] += _n_rec
            _funnel.record("head recovery (+)", _n, len(dets))
        _feed_perspective(dets, _persp)
        _refresh_ground()
        _n = len(dets)
        dets = _suppress_carried(dets, _persp, _supp_stats)
        _funnel.record("carried-object suppress", _n, len(dets))
        _n = len(dets)
        dets = _drop_implausible(dets, _persp, _supp_stats)
        _funnel.record("implausible size", _n, len(dets))
        _n = len(dets)
        dets = _drop_masked(dets)
        _funnel.record("dead-area mask", _n, len(dets))
        # LIVE phantom suppression (symptoms 5/6/7). The end-of-chunk pass
        # still runs and still has the last word; this stops a plant consuming
        # a canonical id — and, through co-visibility, pushing a real person
        # onto a NEW id — for the minutes before that pass exists.
        if _live_phantoms is not None and len(dets):
            _n = len(dets)
            _keep = _live_phantoms.filter_boxes(t, dets.xyxy)
            dets = dets[np.array(_keep, dtype=bool)]
            _funnel.record("live phantom suppress", _n, len(dets))
        return dets

    if ENABLE_TILED_DETECT:
        # Printed BEFORE the loop. Tiling multiplies detector calls, and a 3x
        # multiplier on a 30-minute run is 90 minutes — a fact worth knowing
        # while cancelling is still cheap.
        _tc = cost_estimate(frame_w, frame_h, TILE_PX, TILE_OVERLAP)
        _log.warning(
            f"\U0001f9e9 TILED DETECT ON — {_tc['calls_per_frame']} detector "
            f"call(s) per frame ({_tc['vs_baseline']} baseline) before the ROI "
            f"band narrows it. Small people gain ~{max(1, YOLO_IMGSZ // TILE_PX)}x "
            f"effective resolution; the run costs proportionally more. Compare "
            f"the funnel against a non-tiled run before keeping it on.")

    def _tile_roi():
        """The band where a person is predicted shorter than TILE_TARGET_MIN_PX.

        Tiling the WHOLE frame multiplies detector cost by the tile count for
        no benefit near the camera, where people are already large. The
        perspective model has fitted `height = slope*foot_y + intercept` from
        this run's own detections, so the band is derived, not guessed.

        Returns None until the fit exists — before that, tile everything,
        because "we do not know where the small people are" is not a reason to
        skip the small people.
        """
        fit = _persp._refit() if _persp.ready else None
        if not fit:
            return None
        return height_roi(frame_w, frame_h, fit[0], fit[1],
                          TILE_TARGET_MIN_PX)

    def _detect_tiled(frame):
        """Whole-frame pass + overlapping tiles -> sv.Detections.

        Kept separate from _filter_chain because it REPLACES the detector call,
        rather than filtering its output. Everything downstream is unchanged:
        the boxes come back in full-frame coordinates.
        """
        def _predict(img):
            r = model(img, conf=det_conf, iou=0.45, imgsz=YOLO_IMGSZ,
                      verbose=False, device=device_str,
                      quantize=("fp16" if globals().get("DETECTOR_HALF", False)
                                else None))[0]
            d = sv.Detections.from_ultralytics(r)
            if len(d) == 0:
                return []
            cid = (d.class_id if d.class_id is not None
                   else np.zeros(len(d), dtype=int))
            cf = (d.confidence if d.confidence is not None
                  else np.ones(len(d), dtype=float))
            return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                     float(cf[i]), int(cid[i])) for i, b in enumerate(d.xyxy)]

        # FISHEYE: rectify -> detect -> map boxes BACK to source pixels, so
        # every zone, door line and stored coordinate keeps working untouched.
        # FISHEYE_K defaults to 0.0 and kevacv.fisheye short-circuits that to a
        # plain predict, so this costs nothing and changes nothing until a k has
        # actually been MEASURED by tools/autocalib_fisheye.py and written to
        # config/cam112.yaml. Wired at 0.0 on purpose: a switch that exists and
        # is off beats a yaml key that nothing reads.
        _fk = float(globals().get("FISHEYE_K", 0.0) or 0.0)
        _pred = _predict
        if abs(_fk) > 1e-9:
            from .fisheye import dewarped_predict
            def _pred(img):                      # noqa: E306
                return dewarped_predict(img, _predict, _fk)

        def _predict_batch(imgs):
            """All tiles + the full frame in ONE detector call.

            Identical boxes to calling _predict per image -- ultralytics
            letterboxes and scores each image of a list independently -- but
            the GPU sees one batch of ~9 instead of nine batches of 1.
            """
            rs = model(imgs, conf=det_conf, iou=0.45, imgsz=YOLO_IMGSZ,
                       verbose=False, device=device_str,
                       quantize=("fp16" if globals().get("DETECTOR_HALF", False)
                                 else None))
            outs = []
            for r in rs:
                d = sv.Detections.from_ultralytics(r)
                if len(d) == 0:
                    outs.append([])
                    continue
                cid = (d.class_id if d.class_id is not None
                       else np.zeros(len(d), dtype=int))
                cf = (d.confidence if d.confidence is not None
                      else np.ones(len(d), dtype=float))
                outs.append([(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                              float(cf[i]), int(cid[i]))
                             for i, b in enumerate(d.xyxy)])
            return outs

        # Batching is off only if a fisheye k is set, because dewarped_predict
        # wraps ONE image at a time; correctness first, speed second.
        _batch = None if abs(_fk) > 1e-9 else _predict_batch
        boxes, stats = tiled_predict(
            frame, _pred, tile=TILE_PX, overlap=TILE_OVERLAP,
            roi=_tile_roi(), iou_thresh=TILE_NMS_IOU, full_frame=True,
            predict_batch_fn=_batch)
        _tile_stats[0] += stats.get("tiles", 0)
        _tile_stats[1] += 1
        if not boxes:
            return sv.Detections(xyxy=np.empty((0, 4), dtype=float),
                                 confidence=np.empty(0, dtype=float),
                                 class_id=np.empty(0, dtype=int))
        return sv.Detections(
            xyxy=np.array([b[:4] for b in boxes], dtype=float),
            confidence=np.array([b[4] for b in boxes], dtype=float),
            class_id=np.array([b[5] for b in boxes], dtype=int))

    def _dedup_nms(dets):
        """A1: two same-size boxes on one body -> one box. Deliberately NOT on
        the model.track() branch, where ids are minted before this point."""
        _n = len(dets)
        if len(dets):
            dets = dets.with_nms(threshold=DEDUP_NMS_IOU, class_agnostic=True)
        _funnel.record("dedup NMS", _n, len(dets))
        return dets

    _last_det_t = -1e9
    _gate_since = None          # when the current gated silence began
    _n_skipped = _n_resets = 0
    _face_state = {}            # raw tid -> [tries, last_try_t, has_face]
    t = 0.0
    _eval_dir = None
    _dataset = None
    if globals().get("ENABLE_DATASET_EXPORT"):
        _dataset = DatasetCollector(OUTPUT_DIR / "venue_dataset")
        _log.info(f"📦 DATASET EXPORT ON -> {OUTPUT_DIR / 'venue_dataset'} "
                  f"(one frame every {DATASET_EXPORT_EVERY_S:.0f}s). These are "
                  f"PSEUDO-labels — the pipeline's own predictions. Correct "
                  f"them before fine-tuning or you teach the model its own "
                  f"mistakes.")
    _dataset_last = [-1e9]
    _eval_n = [0]

    if globals().get("EVAL_EXPORT"):
        _eval_dir = OUTPUT_DIR / "eval_frames" / f"{camera_id}{chunk_tag}"
        _eval_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"\U0001f4cf EVAL EXPORT ON -> {_eval_dir} "
                  f"(window: {'whole chunk' if EVAL_WINDOW is None else EVAL_WINDOW})"
                  f" — these are the frames a human corrects into gt.txt, "
                  f"which is what unblocks HOTA, threshold calibration and "
                  f"detector retraining. Nothing else produces them.")
    else:
        _log.warning("EVAL EXPORT OFF — this run produces NO labelling frames, "
                     "so it cannot be scored and cannot feed a retrain. Pass "
                     "--eval-export to tools/run_pipeline.py if that is what "
                     "you wanted from it.")

    # v55 #9: detection runs on a BATCH of frames per call. Same model, same
    # frames, same results — Ultralytics batches internally and returns one
    # result per image — but 14,430 individual launches become ~3,600. The
    # motion gate lives in here too, so gated frames never enter a batch.
    _batchable = (use_online_tracker and online_tracker is not None) or (
        not use_online_tracker)

    def _analysis_stream():
        # _eval_dir is REBOUND below when the disk fills, and a bare
        # assignment inside a nested function makes the name local to
        # it — so every earlier READ becomes an UnboundLocalError.
        nonlocal _prev_small, _gate_since, _n_skipped, _eval_dir
        buf, gaps = [], []

        def _flush():
            if not buf:
                return []
            # F1: if ANY frame in this batch is infrared, the whole batch runs
            # at the IR floor. Erring low costs a few extra weak detections the
            # tracker will drop; erring high loses people at night permanently.
            _bconf = (min(det_conf, IR_DETECT_CONF_FLOOR)
                      if any(_frame_ir.get(_i0) for _i0, _, _ in buf) else det_conf)
            if _batchable and DET_BATCH > 1:
                with _prof.stage("detect"):
                 res = model([f for _, _, f in buf], conf=_bconf, iou=0.45,
                            imgsz=YOLO_IMGSZ, verbose=False, device=device_str,
                            quantize=("fp16" if globals().get("DETECTOR_HALF", False) else None))
            elif _batchable:
                res = [model(f, conf=_bconf, iou=0.45, imgsz=YOLO_IMGSZ,
                             verbose=False, device=device_str,
                             quantize=("fp16" if globals().get("DETECTOR_HALF", False) else None))[0]
                       for _, _, f in buf]
            else:
                res = [None] * len(buf)     # model.track() path: cannot batch
            out = list(zip(list(buf), res, list(gaps)))
            buf.clear(); gaps.clear()
            return out

        # TIME THE DECODE. Only detect/filters/track+reid were instrumented, so
        # on the 18:30 hour the profile read
        #     detect 503s · track+reid 509s · filters 20s · UNACCOUNTED 962s
        # and 45% of the run — the single largest line — had no owner. Pulling
        # frames out of ffmpeg is most of it, and until it is measured there is
        # no way to know whether the 20-minute target needs a faster model or a
        # faster reader.
        def _timed_frames(_src):
            while True:
                with _prof.stage("decode"):
                    try:
                        _item = next(_src)
                    except StopIteration:
                        return
                yield _item

        for _fi, _t, _fr in _timed_frames(iter(frame_source(
                video_path, eff_fps,
                max_seconds=max_seconds,
                max_w=ANALYSIS_MAX_W,
                start_seconds=start_seconds))):
            # ── MOTION GATE ────────────────────────────────────────────────
            # A reception at 02:00 is empty most of the night and the detector
            # is the most expensive thing here. Skip it only when BOTH hold:
            # nothing detected for MOTION_IDLE_S, and the frame is static — so
            # a person standing perfectly still is never dropped.
            # F1: chroma on EVERY frame, measured BEFORE CLAHE (CLAHE moves L,
            # which would shift the channel spread we are testing). One 160x90
            # resize is shared with the motion gate, so this is ~free.
            _smallc = cv2.resize(_fr, (160, 90))
            _chroma71 = _frame_chroma(None, small=_smallc)
            # V71c: hysteresis — LEAVING infrared needs clearly-colour chroma
            # (1.35x), so a value hovering at the threshold cannot oscillate.
            _ir_raw = _chroma71 < (IR_CHROMA_THRESHOLD
                                   * (1.35 if _ir_state[0] else 1.0))
            # V68b: debounce — dusk chroma hovers at the threshold (43
            # flaps/hr measured). A flip must hold IR_DEBOUNCE_FRAMES
            # consecutive frames before the modality actually changes.
            if _ir_state[0] is None:
                _ir_state[0] = _ir_raw
                _ir_switches.append((_t, _ir_raw))
                _ir_pend[:] = [0, _ir_raw]
            elif _ir_raw != _ir_state[0]:
                _ir_pend[0] = _ir_pend[0] + 1 if _ir_pend[1] == _ir_raw else 1
                _ir_pend[1] = _ir_raw
                if _ir_pend[0] >= int(globals().get("IR_DEBOUNCE_FRAMES", 24)):
                    _ir_state[0] = _ir_raw
                    _ir_switches.append((_t, _ir_raw))
                    _ir_pend[0] = 0
            else:
                _ir_pend[0] = 0
            _ir_now = _ir_state[0]
            _frame_ir[_fi] = _ir_now
            # ── EXPOSURE ADAPTATION (symptom 19) ──────────────────────────
            # Measured on the thumbnail the motion gate already built, so it
            # is free. IR still always gets CLAHE; the new part is that a
            # badly-exposed COLOUR frame does too, at a strength scaled to how
            # bad it is instead of a fixed 2.0.
            _exp = frame_exposure(_smallc) if ENABLE_EXPOSURE_ADAPT else None
            if _exp is not None:
                _exp_counts[_exp["verdict"]] = _exp_counts.get(_exp["verdict"], 0) + 1
            if ENABLE_CLAHE or _ir_now or (_exp and _exp["verdict"] != "ok"):
                _clip = (exposure_clip_limit(_exp, max_clip=EXPOSURE_MAX_CLIP)
                         if _exp is not None else 2.0)
                _fr = apply_clahe(_fr, clip_limit=_clip)

            if MOTION_GATE:
                _small = cv2.cvtColor(_smallc, cv2.COLOR_BGR2GRAY)
                if _prev_small is not None and (_t - _last_det_t) > MOTION_IDLE_S:
                    _chg = float((cv2.absdiff(_small, _prev_small) > 18).mean())
                    if _chg < MOTION_MIN_FRAC:
                        _prev_small = _small
                        if _gate_since is None:
                            _gate_since = _t
                        # full-length render: gated (still, empty) frames must
                        # still reach the proxy store or they vanish from the
                        # annotated video even with RENDER_ONLY_OCCUPIED=False
                        if proxy_dir is not None and not RENDER_ONLY_OCCUPIED:
                            cv2.imwrite(str(proxy_dir / f"{_fi:07d}.jpg"), _fr,
                                        [cv2.IMWRITE_JPEG_QUALITY,
                                         PROXY_JPEG_QUALITY])
                        frame_log.append((_fi, _t, []))
                        _n_skipped += 1
                        continue
                _prev_small = _small

            _gap = 0.0
            if _gate_since is not None:
                _gap = _t - _gate_since
                _gate_since = None
            # EVAL_WINDOW None = export the whole analysed span. Indexing it
            # unconditionally crashed any run that enabled the export without
            # also setting a window — an hour in, after the download.
            if (_eval_dir is not None and _eval_n[0] < EVAL_MAX_FRAMES and (
                    EVAL_WINDOW is None
                    or EVAL_WINDOW[0] <= _t <= EVAL_WINDOW[1])):
                try:
                    if cv2.imwrite(str(_eval_dir / f"{_fi:07d}.jpg"), _fr,
                                   [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        _eval_n[0] += 1
                        if _eval_n[0] == EVAL_MAX_FRAMES:
                            _log.warning(
                                f"eval export reached its cap of "
                                f"{EVAL_MAX_FRAMES} frames at t={_t:.0f}s and "
                                f"stopped. That is far more than anyone will "
                                f"hand-label; use --eval-window to choose WHICH "
                                f"minutes you want.")
                except OSError as _ese:
                    _log.error(f"!! eval export disabled at t={_t:.0f}s: {_ese}. "
                               f"The RUN CONTINUES — the labelling frames are a "
                               f"side artifact, not the analysis.")
                    _eval_dir = None
            buf.append((_fi, _t, _fr)); gaps.append(_gap)
            if len(buf) >= max(1, int(DET_BATCH)):
                for item in _flush():
                    yield item
        for item in _flush():
            yield item

    for (frame_idx, t, frame), _pre_result, _gap_s in tqdm(
            _analysis_stream(), total=n_expected, desc=camera_id + " · analyze"):
        # C4: a colour<->IR flip is a modality change — every live appearance
        # anchor is garbage on the other side. Same rebuild as the gated-
        # silence path below, debounced against dusk flicker. OFF by default;
        # the ablation run decides whether it earns its place.
        _irf = _frame_ir.get(frame_idx)
        _ir_cut = (ENABLE_IR_HARD_CUT and _irf is not None
                   and _ir_prev_main[0] is not None
                   and _irf != _ir_prev_main[0]
                   and (t - _ir_cut_last[0]) >= IR_CUT_MIN_GAP_S)
        if _irf is not None:
            _ir_prev_main[0] = _irf
        if _ir_cut:
            _ir_cut_last[0] = t
            _log.info(f"✂️  C4 IR hard-cut at t={t:.0f}s: tracker + live identity "
                  f"reset ({'-> IR' if _irf else '-> colour'})")
        # coming back from a long gated silence: the tracker has been frozen in
        # frame-time while real time moved on. Rebuild it (see _fresh_tracker).
        if _gap_s > LOST_TRACK_BUFFER_S or _ir_cut:
            if use_online_tracker and online_tracker is not None:
                online_tracker = _fresh_tracker()
            elif tracker is not None:
                tracker = _fresh_tracker()
            _n_resets += 1
            if _identity_memory is not None:
                _identity_memory.raw_to_canon.clear()

        if use_online_tracker and online_tracker is not None:
            # Real-OSNet path: detect only (no Ultralytics tracker wrapper),
            # then hand raw detections + the frame to boxmot's BotSort so its
            # live Hungarian match gets genuine appearance vectors.
            _raw = (_detect_tiled(frame) if ENABLE_TILED_DETECT
                    else sv.Detections.from_ultralytics(_pre_result))
            with _prof.stage("filters"):
                dets = _dedup_nms(_filter_chain(_raw, t))
            _n_pre_track = len(dets)
            # THE DEAD BAND. The detector emits everything >= CONF_THRESHOLD,
            # but BoT-SORT's new_track_thresh (= NEW_TRACK_CONF) decides what
            # may START a track. Detections in between are found, filtered,
            # handed to the tracker, and silently dropped — the single largest
            # loss in this pipeline (14,614 detections, 818 emptied frames on
            # the 18:30 hour) and the one nothing reported.
            try:
                _cf = getattr(dets, "confidence", None)
                if _cf is not None and len(_cf):
                    _lo = float(globals().get("CONF_THRESHOLD", 0.35))
                    _hi = float(globals().get("NEW_TRACK_CONF", 0.45))
                    _deadband[0] += int(sum(1 for c in _cf if _lo <= c < _hi))
                    _deadband[1] += len(_cf)
            except Exception:
                pass
            with _prof.stage("track+reid"):
                tracked = online_tracker.update(_dets_to_boxmot(dets), frame)
            dets = _boxmot_to_dets(tracked)
            _funnel.record("tracker (boxmot)", _n_pre_track, len(dets))
        elif TRACKER_MODE == "botsort-reid":
            result = model.track(frame, persist=True, conf=det_conf,
                                 iou=0.45, imgsz=YOLO_IMGSZ,
                                 tracker=tracker_yaml, verbose=False,
                                 device=device_str,
                                 quantize=("fp16" if globals().get("DETECTOR_HALF", False) else None))[0]
            # F6: A1/dedup-NMS deliberately NOT applied on this branch — ids
            # were already minted inside model.track(persist=True); post-hoc
            # NMS makes the SURVIVING id flip with per-frame confidence rank,
            # which feeds the live identity memory an oscillating id: worse
            # than the duplicate. The other two branches NMS BEFORE the tracker.
            dets = _filter_chain(sv.Detections.from_ultralytics(result), t)
            if dets.tracker_id is None:
                dets = dets[np.zeros(len(dets), dtype=bool)]
                dets.tracker_id = np.array([], dtype=object)
        else:
            _raw = (_detect_tiled(frame) if ENABLE_TILED_DETECT
                    else sv.Detections.from_ultralytics(_pre_result))
            with _prof.stage("filters"):
                dets = _dedup_nms(_filter_chain(_raw, t))
            _n_pre_track = len(dets)
            dets = tracker.update_with_detections(dets)
            _funnel.record("tracker (bytetrack)", _n_pre_track, len(dets))

        # ── v44 OCCLUSION GUARD (computed once; used by identity memory
        # AND crop banking): detections whose boxes mutually overlap this frame
        # have crops contaminated by the neighbouring body. ──────────────────
        _occluded_idx = set()
        if ENABLE_OCCLUSION_GUARD and len(dets.xyxy) > 1:
            _bx = dets.xyxy
            for _a in range(len(_bx)):
                for _b in range(_a + 1, len(_bx)):
                    if _boxes_occluding(_bx[_a], _bx[_b]):
                        _occluded_idx.add(_a); _occluded_idx.add(_b)

        # ── LIVE IDENTITY MEMORY: resolve/remap BEFORE anything downstream
        # (positions, zones, crossings, crop banking, render log) ever sees
        # this frame's ids, so a corrected id is the ONLY id those consumers
        # ever observe -- no separate patch-up needed in each of them.
        if (_identity_memory is not None and len(dets.tracker_id)
                and dets.confidence is not None):
            remapped = []
            assigned_in_frame = set()
            _row_info = []   # v53: (det idx, raw id, clean crop, was_occluded)
            # Evict any stale duplicate binding BEFORE the seed below reads
            # raw_to_canon, so co-visibility starts from a set in which every
            # canonical id is claimed by exactly one raw track.
            if ENABLE_COVISIBILITY_BLOCK:
                for _ev_raw, _ev_canon in _identity_memory.split_duplicate_raws(
                        [_safe_id(_t) for _t in dets.tracker_id], t):
                    _log.debug(f"co-visibility: raw {_ev_raw} evicted from "
                               f"canonical {_ev_canon} (already held by an "
                               f"earlier track visible in this frame)")
            # v44 CO-VISIBILITY: canonical ids already on-screen this frame.
            # Seeded here from raw ids the memory ALREADY knows, then kept live
            # by PASS D as each detection is resolved.
            #
            # It used to be a snapshot taken only here, and that was a real
            # duplicate-id bug: two raw ids BORN on the same frame are both
            # absent from raw_to_canon, so neither appears in the seed. The
            # first resolved into some banked identity C; the second was still
            # handed the stale (C-free) blocked set and could resolve into C
            # as well — one canonical id drawn on two boxes in the same frame,
            # which is precisely what co-visibility exists to forbid. Updating
            # the set as we go costs nothing and closes the hole.
            _present_canons = set()
            if ENABLE_COVISIBILITY_BLOCK:
                for _t in dets.tracker_id:
                    _tc = _identity_memory.raw_to_canon.get(_safe_id(_t))
                    if _tc is not None:
                        _present_canons.add(_tc)
            # ── v55 PASS A: build the clean crops. No model calls here. ────
            _crops, _rawids, _confs, _boxes_i = [], [], [], []
            for _i, (bx1, by1, bx2, by2) in enumerate(dets.xyxy):
                raw_tid = _safe_id(dets.tracker_id[_i])
                bx1i, by1i, bx2i, by2i = int(bx1), int(by1), int(bx2), int(by2)
                conf_i = float(dets.confidence[_i])
                crop = None
                if conf_i >= 0.35 and (by2i - by1i) >= REID_MIN_CROP_H:
                    # v43 aspect ratio gate — v53: a body clipped by the
                    # frame border is legitimately short/wide, so skip the
                    # gate for edge boxes instead of discarding the person
                    aspect = float(by2i - by1i) / max(1, bx2i - bx1i)
                    _at_edge = (bx1i <= 2 or by1i <= 2
                                or bx2i >= frame.shape[1] - 3
                                or by2i >= frame.shape[0] - 3)
                    if _at_edge or MIN_BODY_ASPECT <= aspect <= MAX_BODY_ASPECT:
                        _c = frame[max(0, by1i):by2i, max(0, bx1i):bx2i]
                        if _c.size:
                            ch, cw = _c.shape[:2]
                            is_blurry = False
                            if min(ch, cw) >= MIN_CROP_PX_BLUR_GATE:
                                if _blur_score(_c) < MIN_BLUR_VARIANCE:
                                    is_blurry = True
                            if not is_blurry:
                                crop = cv2.resize(_c, (128, 256))
                _crops.append(crop); _rawids.append(raw_tid)
                _confs.append(conf_i); _boxes_i.append((bx1i, by1i, bx2i, by2i))

            # ── v55 PASS B: ONE batched Re-ID forward pass for the crops that
            # actually need a fresh vector. A track whose appearance we
            # embedded < REEMBED_EVERY_S ago does not need it again — it was
            # being recomputed for every person on every frame, at batch 1.
            _need = []
            for _i, _cr in enumerate(_crops):
                if _cr is None or _i in _occluded_idx:
                    continue
                _own_c = _identity_memory.raw_to_canon.get(_rawids[_i])
                if _own_c is None:
                    _need.append(_i)                       # a birth must have one
                else:
                    _rc = _identity_memory.bank.get(_own_c)
                    if _rc is None or (t - _rc.get("t_emb", -1e9)) >= REEMBED_EVERY_S:
                        _need.append(_i)
            _vecs = {}
            if _need:
                try:
                    _efn = get_reid_embedder(device=device)
                    if _efn is not None:
                        for _k, _v in zip(_need, _efn([_crops[k] for k in _need])):
                            _vecs[_k] = _v
                except Exception as _emb_exc:
                    if not globals().get("_EMB_WARNED"):
                        globals()["_EMB_WARNED"] = True
                        _log.error(f"!! appearance embedding unavailable ({_emb_exc}) — "
                              f"tracking continues on motion only, identity will "
                              f"fragment more. Fix before trusting per-person numbers.")

            # Separability evidence, free: these vectors were computed for
            # tracking anyway. Same-frame pairs are different people by
            # construction; the same raw id a fraction of a second apart is
            # the same person by motion, not by appearance. Neither fact
            # depends on what the embedding says, so measuring appearance
            # against them is not circular. See reid_calibration.LiveSeparability.
            if _sep is not None and _vecs:
                _sep.observe(t, [(_rawids[_i2], _vecs.get(_i2))
                                 for _i2 in range(len(_crops))])

            # ── v55 PASS C: faces, throttled. InsightFace on every crop on
            # every frame cost more than the detector did. A track only needs
            # ONE good face; after FACE_MAX_TRIES we stop paying for it.
            _faces = {}
            if ENABLE_FACE_CORROBORATION:
                for _i, _cr in enumerate(_crops):
                    if _cr is None:
                        continue
                    _fs = _face_state.setdefault(_rawids[_i], [0, -1e9, False])
                    if _fs[2] or _fs[0] >= FACE_MAX_TRIES or (t - _fs[1]) < FACE_RETRY_EVERY_S:
                        continue
                    _fs[0] += 1; _fs[1] = t
                    _emb, _dsc, _side = embed_face_scored(_cr)
                    if _emb is not None:
                        _fs[2] = True
                        _faces[_i] = _emb

            # ── v55 PASS D: resolve identities (no model calls left) ────────
            for _i in range(len(_crops)):
                raw_tid, conf_i, crop = _rawids[_i], _confs[_i], _crops[_i]
                bx1i, by1i, bx2i, by2i = _boxes_i[_i]
                face_emb = _faces.get(_i)

                # Check staff face match
                staff_match_name = None
                if face_emb is not None and _STAFF_FACE_GALLERY:
                    best_sim = -1
                    for sname, semb in _STAFF_FACE_GALLERY.items():
                        sim = _staff_gallery_sim(face_emb, semb)
                        if sim > best_sim:
                            best_sim = sim
                            if sim >= STAFF_MATCH_THRESHOLD:
                                staff_match_name = sname

                # If matched staff face and not yet assigned to another in this frame
                if staff_match_name and staff_match_name not in assigned_in_frame:
                    assigned_in_frame.add(staff_match_name)
                    if _identity_memory:
                        _identity_memory.raw_to_canon[raw_tid] = staff_match_name
                        _identity_memory._remember(staff_match_name, face_emb,
                                                   ((bx1i + bx2i) / 2, (by1i + by2i) / 2),
                                                   t, conf_i)
                    remapped.append(staff_match_name)
                    if ENABLE_COVISIBILITY_BLOCK:
                        _present_canons.add(staff_match_name)
                    rec.vote_role(staff_match_name, "staff")
                    _staff_seen_names.add(staff_match_name)
                else:
                    _is_occ = _i in _occluded_idx
                    _own = _identity_memory.raw_to_canon.get(raw_tid)
                    _blocked = (_present_canons - {_own}) if _own is not None else _present_canons
                    resolved_id = _identity_memory.resolve(
                        raw_tid, crop, conf_i, (bx1i, by1i, bx2i, by2i), t,
                        frozen=_is_occ, blocked_canons=_blocked,
                        vec=_vecs.get(_i))
                    remapped.append(resolved_id)
                    if ENABLE_COVISIBILITY_BLOCK:
                        _present_canons.add(resolved_id)
                    _row_info.append((_i, raw_tid, crop, _is_occ))

            # v53 SWAP RE-VALIDATION: tracks that were overlapping last frame
            # and are clean now get their identities cross-checked - a trade
            # made mid-occlusion is undone here, on the first clean evidence.
            if ENABLE_SWAP_REVALIDATION:
                _exited = [(ix, r, c) for (ix, r, c, o) in _row_info
                           if (not o) and r in _prev_occ_raw and c is not None]
                for _ea in range(len(_exited)):
                    for _eb in range(_ea + 1, len(_exited)):
                        _ia, _ra, _ca = _exited[_ea]
                        _ib, _rb, _cb = _exited[_eb]
                        if _identity_memory.try_swap(_ra, _ca, _rb, _cb,
                                                     SWAP_MARGIN):
                            remapped[_ia] = _identity_memory.raw_to_canon[_ra]
                            remapped[_ib] = _identity_memory.raw_to_canon[_rb]
                _prev_occ_raw = {r for (_ix, r, _c, o) in _row_info if o}
            dets.tracker_id = np.array(remapped, dtype=object)

        boxes_this = []
        if len(dets.xyxy):
            _last_det_t = t
            if proxy_dir is not None:
                try:
                    cv2.imwrite(str(proxy_dir / f"{frame_idx:07d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, PROXY_JPEG_QUALITY])
                except OSError as _pxe:
                    _log.error(f"!! render proxy disabled at t={t:.0f}s: {_pxe}. "
                               f"The RUN CONTINUES and every number is still "
                               f"computed; only the annotated VIDEO is lost.")
                    proxy_dir = None
        elif proxy_dir is not None and (
                not RENDER_ONLY_OCCUPIED
                # F4: a solo person blinking out leaves an EMPTY frame —
                # without a proxy JPEG the A6 coasted box was silently
                # dropped at render. Keep proxies within the coast window.
                or (t - _last_det_t) <= globals().get("RENDER_COAST_S", 0.5)):
            cv2.imwrite(str(proxy_dir / f"{frame_idx:07d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, PROXY_JPEG_QUALITY])
        for (bx1, by1, bx2, by2), tid, cid_ in zip(dets.xyxy, dets.tracker_id,
                                                   dets.class_id):
            tid = _safe_id(tid)
            bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
            boxes_this.append((tid, bx1, by1, bx2, by2))
            # v42: seated/standing detection via bbox aspect ratio
            _bw, _bh = max(bx2 - bx1, 1), max(by2 - by1, 1)
            track_total_frames[tid] += 1
            if _frame_ir.get(frame_idx):
                _track_ir_frames[tid] += 1      # F1: colour evidence is invalid
            if _bw / _bh > 0.45:  # seated people have wider/shorter bboxes
                track_seated_frames[tid] += 1
            bc = ((bx1 + bx2) / 2.0, float(by2))
            if tid not in track_pos:
                track_pos[tid] = [bc, bc]
                track_time[tid] = [t, t]
            else:
                track_pos[tid][1] = bc
                track_time[tid][1] = t
            role_vote = "staff" if (isinstance(tid, str) and not tid.isdigit()) else "customer"
            rec.vote_role(tid, role_vote)
        frame_log.append((frame_idx, t, boxes_this))

        # Venue dataset (symptom 8). Sampled, and written from the FILTERED
        # detections — phantoms, merged boxes and masked areas are already
        # resolved, so the pseudo-labels start closer to correct. They are
        # still predictions and still need correcting; dataset.yaml says so.
        if (_dataset is not None and len(boxes_this) >= DATASET_MIN_BOXES
                and (t - _dataset_last[0]) >= DATASET_EXPORT_EVERY_S):
            _dataset_last[0] = t
            try:
                _dataset.save_frame_pseudo_labels(
                    frame, [(b[1], b[2], b[3], b[4]) for b in boxes_this])
            except OSError as _dse:
                # An OPTIONAL side artifact must never destroy the analysis.
                # A full disk at 95% of a 28-minute run killed everything —
                # detection, tracking, identity, all of it — for the sake of a
                # training-set export nobody was waiting on. Disable it and
                # keep going: the numbers are the product, this is a bonus.
                _log.error(f"!! dataset export disabled at t={t:.0f}s: {_dse}. "
                           f"The RUN CONTINUES — every number is still being "
                           f"computed. Free space and re-run with "
                           f"--dataset-export to collect a training set.")
                _dataset = None

        for name, zone in zones.items():
            for tid in dets.tracker_id[zone.trigger(dets)]:
                rec.add(t, name, _safe_id(tid))
        for _lname, line_zone in line_zones.items():
            crossed_in, crossed_out = line_zone.trigger(dets)
            # G3: record WHERE the crossing happened. Tier A de-duplicates
            # in space and time, so a crossing without a position is a crossing
            # that can only fall back to trusting the track id.
            _bc_of = {}
            for (_cx1, _cy1, _cx2, _cy2), _ctid in zip(dets.xyxy, dets.tracker_id):
                _bc_of[_safe_id(_ctid)] = ((float(_cx1) + float(_cx2)) / 2.0,
                                           float(_cy2))
            for tid in dets.tracker_id[crossed_in]:
                _sid = _safe_id(tid)
                crossings.append({"t": round(t, 2), "track_id": _sid,
                                  "direction": "in", "line": _lname,
                                  "pos": _bc_of.get(_sid)})
            for tid in dets.tracker_id[crossed_out]:
                _sid = _safe_id(tid)
                crossings.append({"t": round(t, 2), "track_id": _sid,
                                  "direction": "out", "line": _lname,
                                  "pos": _bc_of.get(_sid)})
        if ENABLE_REID_STITCH and dets.confidence is not None:
            for _bi, (box, conf, tid) in enumerate(zip(dets.xyxy, dets.confidence,
                                      dets.tracker_id)):
                if _bi in _occluded_idx:
                    continue  # v44: don't bank a crop contaminated by overlap
                bx1, by1, bx2, by2 = [int(v) for v in box]
                if conf >= 0.35 and (by2 - by1) >= REID_MIN_CROP_H:
                    # v43 aspect ratio gate
                    aspect = float(by2 - by1) / max(1, bx2 - bx1)
                    if MIN_BODY_ASPECT <= aspect <= MAX_BODY_ASPECT:
                        crop = frame[max(0, by1):by2, max(0, bx1):bx2]
                        if crop.size:
                            # v43 blur gate
                            ch, cw = crop.shape[:2]
                            is_blurry = False
                            if min(ch, cw) >= MIN_CROP_PX_BLUR_GATE:
                                if _blur_score(crop) < MIN_BLUR_VARIANCE:
                                    is_blurry = True
                            if not is_blurry:
                                lst = track_crops[_safe_id(tid)]
                                lst.append((float(conf) * (by2 - by1),
                                            cv2.resize(crop, (128, 256))))
                                lst.sort(key=lambda sc: -sc[0])
                                del lst[REID_CROPS_PER_TRACK:]
                                # V76: the SOURCE height, before the resize to
                                # 256. Banked crops are all exactly 256 tall,
                                # so measuring them can only ever report 256.
                                try:
                                    _src_crop_h.append(int(by2 - by1))
                                except NameError:
                                    pass
    if start_seconds:
        _log.info(f"\u23e9 analysed {mmss(start_seconds)} -> "
              f"{mmss(start_seconds + (max_seconds or 0))} of this chunk only")

    if MOTION_GATE and _n_skipped:
        _log.info(f"\u26a1 motion gate: skipped the detector on {_n_skipped} empty, "
              f"static frames ({100.0 * _n_skipped / max(1, len(frame_log)):.0f}% "
              f"of this chunk); tracker restarted {_n_resets} time(s) after a "
              f"silence longer than {LOST_TRACK_BUFFER_S}s")

    # ── THE FUNNEL: where every detection went, stage by stage ──────────────
    # Printed before the individual diagnostics below, because it is the map
    # they are details of: if 40% of detections died at one stage, that is the
    # thing to look at, not the per-filter footnotes.
    # ── CAPABILITY LEDGER ───────────────────────────────────────────────────
    # Built HERE, not before the loop. The first version was assembled early so
    # the operator could read it before committing an hour — and it crashed on
    # _has_head, which is set later, while the staff-gallery row would have
    # reported "no faces enrolled" because the gallery loads lazily on the
    # first face embed. Both are the same mistake: describing subsystems before
    # they exist. The disk check is the early guard; this is the summary.
    _caps = CapabilityLedger()
    _caps.ok("detector", f"{Path(str(DETECTOR_MODEL)).name} @ imgsz {YOLO_IMGSZ}, "
                         f"{'person+head' if _has_head else 'person only'}")
    if not _has_head:
        _caps.degraded("head class", "detector has no head class",
                       "occlusion recovery (#12) and merged-box splitting (#9) "
                       "cannot run — both need heads")
    _caps.ok("tracker", f"{TRACKER_MODE} @ {eff_fps:.1f} fps (step {step})")
    if ENABLE_TILED_DETECT:
        _avg_t = _tile_stats[0] / max(1, _tile_stats[1])
        _caps.ok("tiled detect (SAHI)",
                 f"{_avg_t:.1f} tiles/frame + 1 full pass, tile={TILE_PX}px")
    else:
        _caps.degraded(
            "tiled detect (SAHI)", "off — detecting on the downscaled frame only",
            f"a person at the far wall is ~60px tall at {frame_w}x{frame_h}. "
            f"That is small-object detection, which is what bad boxes (#8), "
            f"merged people (#9) and detection flicker (#11) are. Enable "
            f"ENABLE_TILED_DETECT to give them {YOLO_IMGSZ // TILE_PX}x "
            f"effective resolution.")

    _onnx_avail, _onnx_gpu = onnx_providers()
    if ENABLE_FACE_CORROBORATION:
        if not _onnx_gpu:
            _caps.degraded(
                "face (onnxruntime)",
                f"no CUDA provider; running on CPU ({', '.join(_onnx_avail)})",
                "InsightFace is ~20x slower, so far fewer tracks get a face. "
                "Staff recognition leans almost entirely on zone dwell. "
                "Fix: pip install --force-reinstall onnxruntime-gpu")
        else:
            _caps.ok("face (onnxruntime)", "CUDA provider available")
    else:
        _caps.missing("face", "ENABLE_FACE_CORROBORATION is off",
                      "no face evidence at all; staff = zone dwell only")

    _n_enrolled = len(_STAFF_FACE_GALLERY or {})
    if _n_enrolled:
        _caps.ok("staff gallery", f"{_n_enrolled} enrolled: "
                                  f"{', '.join(sorted(_STAFF_FACE_GALLERY))}")
    else:
        _caps.missing("staff gallery", "no faces enrolled",
                      "staff identity is 100% zone dwell, which cannot "
                      "separate a receptionist from a guest at the counter")

    _caps.ok("zones", f"{len(polygons)} polygon(s), {len(line_zones)} entry line(s)")
    if not entry_lines:
        _caps.missing("entry line", "none drawn",
                      "line-crossing counts are impossible; arrivals fall back "
                      "to the region method alone with no cross-check")

    # Rows that can only be known AFTER the loop: the plane is fitted from the
    # run's own detections, and the embedder is resolved lazily.
    if _ground.ok:
        _gsan = _ground.sanity(frame_h)
        if _gsan:
            _caps.degraded("ground plane", f"{_ground.mode}: {_gsan[0]}",
                           "every METRE gate (walk-speed, hand-off distance) is "
                           "computed in these units. Add ground_points to the "
                           "zones JSON for an exact homography.")
        else:
            _caps.ok("ground plane", _ground.describe()[:60])
    else:
        _caps.degraded("ground plane", "not fitted",
                       "distance gates fall back to PIXELS, which mean "
                       "different things near and far from the camera")
    _rmethod = (_REID_STATE.get("method") or "none")
    if _rmethod in ("clip", "osnet"):
        _caps.ok("re-id embeddings", _rmethod.upper())
    else:
        _caps.degraded("re-id embeddings", f"{_rmethod} (colour histogram)",
                       "appearance cannot survive the infrared boundary, and "
                       "58% of this camera's frames are infrared")
    if _exp_counts:
        _bad_e = sum(v for k, v in _exp_counts.items() if k != "ok")
        if _bad_e > 0.5 * sum(_exp_counts.values()):
            _caps.degraded("exposure", f"{_bad_e} badly-exposed frame(s)",
                           "CLAHE recovers contrast, not detail — clipped "
                           "pixels carry nothing. This is a camera setting.")
        else:
            _caps.ok("exposure", f"{_bad_e} frame(s) corrected")
    for _cl in _caps.describe().splitlines():
        (_log.warning if not _caps.trustworthy() else _log.info)(_cl)

    # WHERE THE TIME WENT — unconditional, next to the funnel, because both
    # answer "which stage is hurting me" and reading them apart is the bug.
    for _pl in _prof.describe().splitlines():
        _log.info(_pl)
    _pv = _prof.verdict()
    if _pv:
        _log.warning(f"\U0001f9ed PHASE C GATE: {_pv}")
    for _fl in _funnel.describe().splitlines():
        _log.info(_fl)
    for _lvl, _msg in _funnel.findings():
        (_log.warning if _lvl == "WARN" else _log.info)(f"   funnel: {_msg}")

    # F1/F2/F3 run diagnostics — these were the silent failures.
    _n_ir = sum(1 for v in _frame_ir.values() if v)
    _log.info(f"\U0001f319 F1 infrared: {_n_ir}/{len(_frame_ir)} analysed frames "
          f"({100.0 * _n_ir / max(1, len(_frame_ir)):.0f}%), "
          f"{max(0, len(_ir_switches) - 1)} switch(es) mid-chunk"
          + (" — colour/IR handled per frame" if _ir_switches else ""))
    for _st, _sir in _ir_switches[1:6]:
        _log.info(f"      switch at {mmss(_st)} -> {'INFRARED' if _sir else 'colour'}")
    if _sep is not None:
        _sep_rep = _sep.report(current_thresholds={
            "LIVE_REID_SIM_THRESHOLD": LIVE_REID_SIM_THRESHOLD,
            "REID_SIM_THRESHOLD": REID_SIM_THRESHOLD,
            "ANCHOR_SIM_THRESHOLD": ANCHOR_SIM_THRESHOLD})
        for _sl2 in describe_live(_sep_rep).splitlines():
            _log.info(_sl2)
    if _dataset is not None and _dataset.counter:
        _dsy = _dataset.export_dataset_yaml(
            venue_name=str(VENUE_PROFILE.get("venue", {}).get("name", camera_id)))
        _log.info(f"\U0001f4e6 dataset: {_dataset.counter} frame(s) exported -> "
                  f"{_dsy}. NEXT STEP IS CORRECTION, not training: these boxes "
                  f"are what the pipeline already believes, so fine-tuning on "
                  f"them raw makes it more confidently wrong. Import into CVAT "
                  f"(see tools/gt_kit.py), delete phantoms, add missed people, "
                  f"split boxes covering two bodies — then train.")
    if _live_phantoms is not None:
        for _pl in _live_phantoms.describe().splitlines():
            _log.info(f"\U0001fab4 {_pl}")
    if _exp_counts:
        _tot_e = sum(_exp_counts.values()) or 1
        _bad = _tot_e - _exp_counts.get("ok", 0)
        _log.info(f"\U0001f4a1 exposure: "
                  + ", ".join(f"{k}={v} ({100.0 * v / _tot_e:.0f}%)"
                              for k, v in sorted(_exp_counts.items())))
        if _bad:
            _log.info(f"      {_bad} frame(s) got contrast-scaled CLAHE. Before "
                      f"this, only INFRARED frames were corrected — a badly "
                      f"exposed colour frame went to the detector as captured "
                      f"and lost people silently.")
        if _bad / _tot_e > 0.5:
            _log.warning(f"      over half this chunk is badly exposed. CLAHE "
                         f"recovers contrast, not detail: pixels clipped to "
                         f"pure black or white carry nothing to recover, and "
                         f"no threshold fixes that. This is a camera exposure "
                         f"setting, not a code problem.")
    if _deadband[1]:
        _lo = float(globals().get("CONF_THRESHOLD", 0.35))
        _hi = float(globals().get("NEW_TRACK_CONF", 0.45))
        _pc = 100.0 * _deadband[0] / max(1, _deadband[1])
        _log.info(f"\U0001f3af tracker dead band: {_deadband[0]} of "
                  f"{_deadband[1]} detections ({_pc:.1f}%) had confidence in "
                  f"[{_lo:.2f}, {_hi:.2f}) — detected, filtered, handed to the "
                  f"tracker, and unable to START a track.")
        if _pc > 5.0:
            _log.warning(f"      A person only ever seen at this confidence — "
                         f"dim, infrared, or half-occluded behind the counter — "
                         f"is detected on every frame and tracked on none. "
                         f"Narrow the gap with analysis.new_track_conf and A/B "
                         f"it; do not close it entirely, the high bar is what "
                         f"stops spurious tracks.")
    _log.info(f"\U0001f4d0 F2 {_persp.describe()}")
    # D0 vs F2, side by side, because they answer the same question and can
    # DISAGREE. F2 measures how tall a person actually is at a given foot
    # position, from thousands of isolated detections in this very chunk. D0
    # applies a flat fraction-of-frame cap decided before any of that existed.
    # If the flat cap is below what F2 says a near-field person measures, D0 is
    # deleting real guests at the bottom of the frame — where the door is — and
    # the funnel reports it only as an anonymous percentage.
    if _absurd[0]:
        _hf = float(globals().get("MAX_BOX_HEIGHT_FRAC", 0.70))
        _cap_px = _hf * frame_h
        _log.info(f"\U0001f4cf D0 cap audit: dropped {_absurd[0]} box(es) — "
                  f"{_absurd_why[0]} on HEIGHT (> {_cap_px:.0f}px = "
                  f"{_hf:.2f} of frame), {_absurd_why[1]} on AREA")
        _exp_bottom = _persp.expected_h(float(frame_h)) if _persp.ready else None
        if _exp_bottom and _absurd_h:
            # expected_h returns None outside the fitted domain — treat that as
            # "no opinion" rather than letting it become a 0 that calls every
            # dropped box implausible.
            def _exp_at(_fy):
                _e = _persp.expected_h(float(_fy))
                return _e if _e else None
            _judged = [(_h, _exp_at(_fy)) for _h, _fy in zip(_absurd_h, _absurd_footy)]
            _judged = [(_h, _e) for _h, _e in _judged if _e]
            _plausible = sum(1 for _h, _e in _judged if _h <= 1.35 * _e)
            _log.info(f"      F2 says a person standing at the frame bottom is "
                      f"{_exp_bottom:.0f}px tall = "
                      f"{_exp_bottom / max(1, frame_h):.3f} of frame")
            if _exp_bottom > _cap_px:
                _log.warning(f"      !! THE CAP IS BELOW A REAL PERSON. Anyone "
                             f"whose feet land near the bottom of the frame is "
                             f"taller than D0 allows and is being deleted before "
                             f"the tracker ever sees them. Raise "
                             f"MAX_BOX_HEIGHT_FRAC above "
                             f"{_exp_bottom / max(1, frame_h):.2f}.")
            _log.info(f"      {_plausible} of {len(_judged)} dropped box(es) "
                      f"were within 1.35x the height F2 expects at their own "
                      f"foot position — i.e. person-shaped, not garbage.")
    _log.info(f"      carried-suppression removed "
          f"{_supp_stats.get('carried_suppressed', 0)} detection(s) this chunk "
          f"(needs containment AND off-floor height AND a low head — an "
          f"occluded guest fails the head test and is kept)")
    if _has_head:
        _log.info(f"\U0001f9e0 F3 head class ACTIVE: {_n_head_only} head detection(s) "
              f"had no person box around them — each is a body the detector lost "
              f"behind someone else"
              + (f"; {_n_recovered[0]} body box(es) REBUILT from them "
                 f"(occlusion recovery, symptom 12)"
                 if ENABLE_HEAD_RECOVERY else
                 " (recovery OFF — set ENABLE_HEAD_RECOVERY=True)"))
    else:
        _log.info("\U0001f9e0 F3 detector is not 2-class person/head — head signal "
              "unavailable (this is correct for stock COCO weights)")
    if _identity_memory is not None and _identity_memory.reassignments:
        _log.info(f"\U0001f9e0 Live identity memory: "
              f"{_identity_memory.reassignments} raw track-id birth(s) "
              f"redirected to an existing identity in real time "
              f"(id would otherwise have changed abruptly)")
    if _identity_memory is not None and _identity_memory.duplicate_splits:
        _log.info(f"\U0001f9e0 Co-visibility: "
              f"{_identity_memory.duplicate_splits} raw track(s) evicted from a "
              f"canonical id another visible track already held — each one was "
              f"the same id drawn on two bodies at once")

    # entry zones keep sub-2s doorway transits: a walking-pace pass through a
    # shallow main_entrance polygon lasts ~1s, and MIN_EVENT_S=2.0 was deleting
    # every one of them — which is why the region arrival count read 0 even
    # with a perfectly drawn entry polygon.
    events = rec.events(min_event_by_zone={z: 0.5 for z, rs in zone_roles.items()
                                           if "entry" in rs})
    if staff_zones_here:
        events, _staff_ev = apply_staff_zone_override(
            events, staff_zones_here, STAFF_OVERRIDE_MIN_S,
            min_share_dwell_s=STAFF_OVERRIDE_SHARE_MIN_S,
            observation_s=duration_s, min_visits=STAFF_MIN_VISITS,
            min_spread=STAFF_MIN_SPREAD, sole_dwell_s=STAFF_SOLE_DWELL_S,
            return_evidence=True)
        # Printed in full, every run. A wrong staff label is invisible in every
        # downstream number — it looks exactly like a right one — so the only
        # place it can be caught is here, next to the evidence it was made on.
        for _sl in describe_staff_decision(_staff_ev).splitlines():
            _log.info(_sl)

    # ── Re-ID stitching (appearance + spatio-temporal hand-off) ────────────
    id_merges, mapping = {}, {}
    _sweep_hits = {}   # F3: defined for EVERY flag combination — the
    #    protection call sites below reference it even when stitch is off
    calibration_report, face_validation_report = None, None
    face_veto_report, face_veto_flagged = None, None
    merge_diagnostics, identity_dossiers = None, {}
    if ENABLE_REID_STITCH and (events or track_time):
        try:
            embed = get_reid_embedder(device=device)
            windows = {tid: (tt[0], tt[1]) for tid, tt in track_time.items()}
            for e in events:   # widen using zone dwell bounds too, doesn't hurt
                w0 = windows.get(e["track_id"])
                lo = min(w0[0], e["t_in"]) if w0 else e["t_in"]
                hi = max(w0[1], e["t_out"]) if w0 else e["t_out"]
                windows[e["track_id"]] = (lo, hi)
            embeddings = {}
            anchor_embeddings = {}
            if embed is not None:
                embeddings = {tid: embed([c for _, c in crops])
                              for tid, crops in track_crops.items()
                              if tid in windows and crops}
                # ANCHOR SNAPSHOT (v25): crops in track_crops[tid] are kept
                # re-sorted best-first every time one is banked (see the
                # `lst.sort(key=lambda sc: -sc[0])` above), so crops[0] IS the
                # single highest-quality snapshot for that track — the
                # canonical reference image an operator would point to and
                # say "that's this person." Embed it separately (not just as
                # the first item pulled from the full gallery) so
                # merge_fragmented_tracks can do a clean 1:1 anchor-to-anchor
                # comparison, which is the strongest single piece of identity
                # evidence available and immune to a lucky/unlucky pairing
                # among the OTHER banked crops skewing the best-of-gallery
                # score in _gallery_sim.
                anchor_embeddings = {tid: embed([crops[0][1]])[0]
                                     for tid, crops in track_crops.items()
                                     if tid in windows and crops}
            # (v37) same banked crop IMAGES (not embeddings) — feeds the new
            # HSV attire merge tier in merge_fragmented_tracks, reusing the
            # exact crops already banked for appearance, no extra work.
            # F1: a track that lived under infrared has NO colour information,
            # so its HSV attire similarity is noise. Feeding it to the attire
            # merge tier and the HSV vetoes does not just add nothing — it
            # invents matches. Those tracks are withheld from every colour-based
            # signal; their face/appearance/hand-off evidence is untouched.
            _ir_tracks = {tid for tid, n in _track_ir_frames.items()
                          if n >= IR_TRACK_FRAC * max(1, track_total_frames.get(tid, 1))}
            raw_crops = {tid: [c for _, c in crops]
                        for tid, crops in track_crops.items()
                        if tid in windows and crops and tid not in _ir_tracks}
            if _ir_tracks:
                # DENOMINATOR MUST BE THE POPULATION _ir_tracks WAS DRAWN FROM.
                # It was len(track_crops) -- only the tracks that banked a crop --
                # while _ir_tracks comes from _track_ir_frames, which is every
                # track. That printed "9 of 4 tracks lived under infrared", an
                # impossible ratio, in the 2026-08-13 smoke run. A number that
                # cannot be true teaches you to skim the line it is on.
                _ir_total = max(len(_track_ir_frames), len(_ir_tracks))
                _withheld = len(_ir_tracks & set(track_crops))
                _log.info(f"\U0001f319 F1: {len(_ir_tracks)} of {_ir_total} tracks "
                      f"lived under infrared ({_withheld} of {len(track_crops)} "
                      f"with banked crops withheld from attire/HSV evidence — "
                      f"colour does not exist in those frames)")
            positions = {tid: (track_pos[tid][0], track_pos[tid][1])
                         for tid in windows if tid in track_pos}
            # A4: fraction of each track's life spent under IR — feeds the
            # cross-modality guard inside merge_fragmented_tracks
            _ir_frac = {tid: _track_ir_frames.get(tid, 0)
                        / max(1, track_total_frames.get(tid, 1))
                        for tid in windows}
            # (v36) hard number on crop resolution -- a common, boring, and
            # easily-missed reason appearance embeddings don't separate:
            # crops just too small for ANY model (or a human) to make out
            # real detail. Cheap to compute, worth printing every run.
            if track_crops:
                # V76: source heights if we collected them, else fall back
                # to the old (always-256) measurement rather than crashing.
                _sizes = list(globals().get("_src_crop_h") or [])
                _from_source = bool(_sizes)
                if not _sizes:
                    _sizes = [c.shape[0] for crops in track_crops.values()
                              for _, c in crops]
                if _sizes:
                    _sizes.sort()
                    _med = _sizes[len(_sizes)//2]
                    _p10 = _sizes[int(len(_sizes)*0.1)]
                    _log.info(f"📐 {'SOURCE' if _from_source else 'Banked'} crop height: median {_med}px, p10 {_p10}px "
                          f"across {len(_sizes)} crops — ReID models are "
                          f"typically trained on ~256px-tall crops; well "
                          f"below that (say <100px) means the embeddings may "
                          f"not have enough real detail to separate people "
                          f"regardless of backbone or threshold")
            method = (_REID_CACHE.get(_boxmot_dev(device), _REID_STATE)
                      .get("method")) or "none"
            thresh = (REID_SIM_THRESHOLD if method in ("osnet", "clip")
                      else max(REID_SIM_THRESHOLD, 0.93))

            # PHASE8FIX_ROLE_HINT_HOISTED: hoisted above the FACE_SCOPE block, which
            # reads role_hint. It used to be defined 55 lines LATER, so the
            # read raised UnboundLocalError, the surrounding try/except
            # swallowed it, and the entire Re-ID stitching stage silently did
            # nothing — 95 fragmented 'people' instead of 22.
            # Role-boundary guard: only trust a role if the fragment actually
            # earned it — real dwell time, not a single frame's classification.
            # (v38) moved BEFORE calibration so calibrate_appearance_threshold
            # can exclude staff-vs-customer pairs from its same-person ground
            # truth too — see that function's role_hint docstring for why a
            # busy counter needs this guard on the ground truth, not just on
            # real merges.
            _zone_time = defaultdict(lambda: {"staff": 0.0, "other": 0.0})
            for e in events:
                bucket = "staff" if e["zone"] in staff_zones_here else "other"
                _zone_time[e["track_id"]][bucket] += e["duration"]
            role_hint = {}
            _vid_dur = max(1.0, float(t))          # seconds of footage seen
            for tid, dwell in _zone_time.items():
                _long_enough = dwell["staff"] >= STAFF_OVERRIDE_MIN_S
                _big_share   = dwell["staff"] >= STAFF_MIN_VIDEO_SHARE * _vid_dur
                _dominates   = dwell["staff"] >= STAFF_DOMINANCE_RATIO * dwell["other"]
                if _long_enough and _big_share and _dominates:
                    role_hint[tid] = "staff"
                elif dwell["other"] >= MIN_SEATED_S:
                    role_hint[tid] = "customer"

            # ── FACE EMBEDDINGS (v30, optional, corroborating only) ────────
            # PHASE 8: FACE_SCOPE gates whose face is computed post-tracking.
            # "staff_only" = only tracks known to be staff (gallery match or zone
            # dwell). Customer faces are never embedded in this phase.
            # "all" = every track (requires explicit consent in venue profile).
            face_embeddings = {}
            _face_rejected_sizes = []
            _sweep_hits = {}   # C2: {track_id: enrolled staff name}
            _sweep_scores = {}  # V68c: face score per sweep hit
            # V65: per-track box index for source-resolution face re-crops
            _v65_scale = _src_w / max(frame_w, 1)
            _v65_budget = [int(globals().get("FACE_RECROP_MAX_TRACKS", 120))]
            _v65_rescued = [0, 0]     # attempts, successes
            _v65_boxes = {}
            if globals().get("FACE_SOURCE_RECROP") and _v65_scale > 1.2:
                for _fi, _t65, _bs in frame_log:
                    for _b in _bs:
                        _v65_boxes.setdefault(_b[0], []).append(
                            (_fi, (_b[1], _b[2], _b[3], _b[4])))

            def _v65_retry(tid):
                if (not _v65_boxes or _v65_budget[0] <= 0
                        or tid not in _v65_boxes):
                    return None
                _v65_budget[0] -= 1
                _v65_rescued[0] += 1
                _fe = _recrop_face_from_source(
                    video_path, _v65_boxes[tid], _v65_scale, step, native_fps,
                    start_seconds, k=int(globals().get("FACE_RECROP_FRAMES", 4)))
                if _fe is not None:
                    _v65_rescued[1] += 1
                return _fe
            _face_scope = FACE_SCOPE if 'FACE_SCOPE' in dir() else "staff_only"
            if ENABLE_FACE_CORROBORATION:
                # Determine which track IDs are eligible for face embedding
                if _face_scope == "all":
                    _face_eligible = set(track_crops.keys())
                else:
                    # staff_only: gallery-matched IDs + zone-dwell staff IDs
                    _face_eligible = set()
                    # Staff identified by gallery match (string IDs from _identity_memory)
                    if _identity_memory:
                        for _raw, _canon in _identity_memory.raw_to_canon.items():
                            if isinstance(_canon, str) and _canon in _STAFF_FACE_GALLERY:
                                _face_eligible.add(_canon)
                                _face_eligible.add(_raw)
                    # Staff identified by zone dwell (role_hint computed above)
                    for _tid, _role in role_hint.items():
                        if _role == "staff":
                            _face_eligible.add(_tid)
                    if _face_eligible:
                        _log.info(f"🔒 FACE_SCOPE={_face_scope!r}: face embeddings "
                              f"restricted to {len(_face_eligible)} staff track(s)")
                    else:
                        _log.info(f"🔒 FACE_SCOPE={_face_scope!r}: no staff tracks "
                              f"identified — no post-processing face embeddings computed")

                for tid, crops in track_crops.items():
                    if tid not in windows or not crops:
                        continue
                    if tid not in _face_eligible:
                        continue
                    fe = get_track_face_embedding(crops, size_log=_face_rejected_sizes)
                    if fe is None:
                        fe = _v65_retry(tid)   # V65: 4K source re-crop
                    if fe is not None:
                        face_embeddings[tid] = fe

                # C2: STAFF-GALLERY SWEEP — breaks the staff_only circularity.
                # The eligibility gate above only embeds tracks ALREADY
                # believed staff, so a staff member the live gallery match
                # missed (fragmented at the door, face turned away for the
                # first minutes) could never be face-merged at all. Here every
                # remaining track's best crops get ONE chance against the
                # ENROLLED gallery. A match is the strongest identity evidence
                # this pipeline has; a non-match is discarded on the spot —
                # customer faces are still never stored.
                if ENABLE_STAFF_GALLERY_SWEEP and _STAFF_FACE_GALLERY:
                    for tid, crops in track_crops.items():
                        if (tid in face_embeddings or tid not in windows
                                or not crops):
                            continue
                        fe = get_track_face_embedding(crops)
                        if fe is None:
                            fe = _v65_retry(tid)   # V65: 4K source re-crop
                        if fe is None:
                            continue
                        _best, _bname = -1.0, None
                        for _sn, _se in _STAFF_FACE_GALLERY.items():
                            _s = _staff_gallery_sim(fe, _se)
                            if _s > _best:
                                _best, _bname = _s, _sn
                        if _best >= STAFF_MATCH_THRESHOLD:
                            _sweep_hits[tid] = _bname
                            _sweep_scores[tid] = _best
                            face_embeddings[tid] = fe   # a staff face — kept
                            role_hint[tid] = "staff"
                        # else: fe goes out of scope — nothing stored
                    # V68c: one body, one name — two tracks that OVERLAP in
                    # time cannot both be this staff member. Keep the best
                    # face score; losers stay role=staff but unnamed.
                    _by_name = {}
                    for _v68t in sorted(_sweep_hits,
                                        key=lambda k: -_sweep_scores.get(k, 0)):
                        _v68n = _sweep_hits[_v68t]
                        _v68w = windows.get(_v68t)
                        _kept = _by_name.setdefault(_v68n, [])
                        if _v68w and any(
                                _v68w[0] < windows[_k][1]
                                and windows[_k][0] < _v68w[1]
                                for _k in _kept if _k in windows):
                            del _sweep_hits[_v68t]
                            _log.info(f"   V68c: track {_v68t} face-matched "
                                  f"{_v68n} but overlaps a stronger "
                                  f"{_v68n} track — name not applied")
                        else:
                            _kept.append(_v68t)
                    if _sweep_hits:
                        _log.info(f"🧲 C2 gallery sweep: {len(_sweep_hits)} "
                              f"track(s) matched enrolled staff by face: "
                              + ", ".join(f"{t}->{n}" for t, n in
                                          list(_sweep_hits.items())[:6]))
                    # Enrolled but never found. Silence here reads as "they
                    # weren't working tonight" when it usually means the photo
                    # does not match this camera, or the threshold is too high
                    # — and the difference is only visible if we say it.
                    _never = sorted(set(_STAFF_FACE_GALLERY)
                                    - set(_sweep_hits.values()))
                    if _never:
                        _best_seen = max(_sweep_scores.values(), default=0.0)
                        _log.warning(
                            f"👤 enrolled but NEVER matched: {', '.join(_never)}. "
                            f"Either they were not on shift, or their photo "
                            f"does not match this camera. Best face score "
                            f"anywhere this chunk was {_best_seen:.2f} against "
                            f"a threshold of {STAFF_MATCH_THRESHOLD} — if that "
                            f"is close, lower STAFF_MATCH_THRESHOLD; if it is "
                            f"far, re-take the photos from this camera's own "
                            f"footage.")
                if _v65_rescued[0]:
                    _log.info(f"👓 V65 source re-crop: {_v65_rescued[1]} face(s) "
                          f"rescued from {_v65_rescued[0]} 4K retr"
                          f"{'y' if _v65_rescued[0] == 1 else 'ies'} "
                          f"(scale {_v65_scale:.1f}x, budget left "
                          f"{_v65_budget[0]})")
                if track_crops:
                    _scope_label = ("STAFF ONLY — customer faces not processed"
                                    if _face_scope != "all"
                                    else "ALL TRACKS — customer consent required")
                    _log.info(f"🔒 Face scope: {_scope_label}")
                    cov_pct = 100.0 * len(face_embeddings) / max(1, len(track_crops))
                    _log.info(f"👤 Face corroboration: {len(face_embeddings)}/"
                          f"{len(track_crops)} tracks ({cov_pct:.0f}%) had a "
                          f"usable face crop (>= {FACE_MIN_FACE_PX}px, "
                          f"det_score >= {FACE_MIN_DET_SCORE})")
                    # (v32) most CCTV face misses are SIZE misses, not
                    # confidence misses -- surface that split so a low
                    # coverage number doesn't get misread as "faces rarely
                    # visible" when it's actually "faces visible but too
                    # small to trust," which is a different, more fixable
                    # problem (camera angle/zoom, not camera absence).
                    if _face_rejected_sizes:
                        _face_rejected_sizes.sort()
                        _med = _face_rejected_sizes[len(_face_rejected_sizes)//2]
                        _log.info(f"   ℹ️  {len(_face_rejected_sizes)} additional "
                              f"crop(s) found A face but it was < "
                              f"{FACE_MIN_FACE_PX}px (median rejected size "
                              f"{_med:.0f}px) — too small to trust as identity "
                              f"evidence, excluded rather than risking a "
                              f"noisy embedding")

            # ── APPEARANCE-THRESHOLD CALIBRATION (v30, auto-applied v37) ────
            # Measures THIS run's actual same/different-person cosine
            # distribution for whichever backbone loaded, instead of trusting
            # OSNet's carried-over numbers. When CALIBRATION_AUTO_APPLY is on,
            # suggest_appearance_thresholds() turns that measurement into the
            # threshold actually used a few lines down — no more copying a
            # printed suggestion into the config cell by hand between runs.
            run_anchor_thresh = ANCHOR_SIM_THRESHOLD
            if ENABLE_REID_CALIBRATION and anchor_embeddings:
                cal = calibrate_appearance_threshold(
                    windows, positions, anchor_embeddings,
                    handoff_gap_s=REID_HANDOFF_GAP_S,
                    handoff_px=REID_HANDOFF_PX,
                    stationary_px=REID_STATIONARY_PX,
                    role_hint=role_hint)
                if cal["same_n"] and cal["diff_n"]:
                    sep_ok = (cal["same_p10"] is not None and cal["diff_p90"] is not None
                             and cal["same_p10"] > cal["diff_p90"])
                    _log.info(f"📏 Calibration ({method}): same-person "
                          f"p10={cal['same_p10']:.3f} p50={cal['same_p50']:.3f} "
                          f"(n={cal['same_n']}) | different-person "
                          f"p50={cal['diff_p50']:.3f} p90={cal['diff_p90']:.3f} "
                          f"(n={cal['diff_n']})")
                    if cal.get("role_conflicts_excluded"):
                        _log.info(f"   ℹ️  excluded {cal['role_conflicts_excluded']} "
                              f"spatially-close pair(s) from the same-person "
                              f"ground truth — they'd earned opposite staff/"
                              f"customer roles, so proximity alone doesn't "
                              f"make them one person (busy-counter guard)")
                    if cal.get("duplicate_excluded"):
                        _log.info(f"   ℹ️  excluded {cal['duplicate_excluded']} "
                              f"co-visible pair(s) from the different-person "
                              f"ground truth — they stayed glued to the same "
                              f"spot the whole time, so they're one body "
                              f"wearing two track IDs, not two people "
                              f"(duplicate-track guard)")
                    if CALIBRATION_AUTO_APPLY:
                        thresh, run_anchor_thresh = suggest_appearance_thresholds(
                            cal, thresh, ANCHOR_SIM_THRESHOLD)
                        if sep_ok:
                            _log.info(f"   ✅ clean separation — AUTO-APPLIED this "
                                  f"run: appearance/anchor thresholds = "
                                  f"{thresh}/{run_anchor_thresh} (config "
                                  f"defaults {REID_SIM_THRESHOLD}/"
                                  f"{ANCHOR_SIM_THRESHOLD} left untouched)")
                        else:
                            _log.warning(f"   ⚠️  distributions overlap — {method}'s "
                                  f"appearance signal alone can't cleanly "
                                  f"separate same/different person on this "
                                  f"footage; AUTO-APPLIED a conservative "
                                  f"appearance/anchor bar of {thresh}/"
                                  f"{run_anchor_thresh} for this run and "
                                  f"leaning on the hand-off/stationary, "
                                  f"attire (HSV), and face merge tiers to "
                                  f"cover what appearance alone is missing")
                    else:
                        suggestion = (round((cal["same_p10"] + cal["diff_p90"]) / 2, 3)
                                     if sep_ok else None)
                        if sep_ok:
                            _log.info(f"   ✅ clean separation — suggested "
                                  f"REID_SIM_THRESHOLD/ANCHOR_SIM_THRESHOLD "
                                  f"≈ {suggestion} (currently "
                                  f"{REID_SIM_THRESHOLD}/{ANCHOR_SIM_THRESHOLD}, "
                                  f"CALIBRATION_AUTO_APPLY is off)")
                        else:
                            _log.warning(f"   ⚠️  distributions overlap — {method}'s "
                                  f"appearance signal is not cleanly separating "
                                  f"same/different person on this footage at any "
                                  f"single threshold; current thresholds "
                                  f"(REID_SIM_THRESHOLD={REID_SIM_THRESHOLD}, "
                                  f"ANCHOR_SIM_THRESHOLD={ANCHOR_SIM_THRESHOLD}) "
                                  f"are a best-effort compromise, not a clean "
                                  f"cut (CALIBRATION_AUTO_APPLY is off)")
                else:
                    _log.info(f"📏 Calibration: not enough hand-off/stationary "
                          f"({cal['same_n']}) or co-visible ({cal['diff_n']}) "
                          f"pairs this run to measure — skipped")
                calibration_report = cal
                # F5: persist the suggestion so it can be reviewed and PINNED in
                # Cell 2, instead of silently changing between runs.
                try:
                    _sg, _sa = suggest_appearance_thresholds(
                        cal, REID_SIM_THRESHOLD, ANCHOR_SIM_THRESHOLD)
                    _cp = OUTPUT_DIR / f"calibration_{camera_id}{chunk_tag}.json"
                    _cp.write_text(json.dumps({
                        "backbone": method, "applied_this_run": bool(CALIBRATION_AUTO_APPLY),
                        "in_use": {"REID_SIM_THRESHOLD": thresh,
                                   "ANCHOR_SIM_THRESHOLD": run_anchor_thresh},
                        "suggested": {"REID_SIM_THRESHOLD": _sg,
                                      "ANCHOR_SIM_THRESHOLD": _sa},
                        "same_n": cal["same_n"], "diff_n": cal["diff_n"],
                        "same_p10": cal["same_p10"], "diff_p90": cal["diff_p90"],
                    }, indent=2, default=str))
                    if not CALIBRATION_AUTO_APPLY:
                        _log.info(f"   \U0001f4cc F5: thresholds FROZEN at "
                              f"{thresh}/{run_anchor_thresh}. This run suggests "
                              f"{_sg}/{_sa} -> {_cp.name}. Pin it in Cell 2 only "
                              f"after a scored A/B, never automatically.")
                except Exception as _cex:
                    _log.info(f"   (calibration not persisted: {_cex})")
                # (v36) show the ACTUAL crops behind the worst-separating
                # pairs -- a percentile number can't tell you WHY same/
                # different-person similarity overlaps; the images usually
                # can (too small/blurry to see detail, near-identical
                # uniforms, bad angle, heavy occlusion...).
                if cal.get("same_pairs_worst"):
                    plot_reid_pair_audit(
                        cal["same_pairs_worst"], track_crops,
                        f"{camera_id}: spatially-certain SAME person, "
                        f"LOWEST appearance similarity (should be high)")
                if cal.get("diff_pairs_worst"):
                    plot_reid_pair_audit(
                        cal["diff_pairs_worst"], track_crops,
                        f"{camera_id}: certainly DIFFERENT people, "
                        f"HIGHEST appearance similarity (should be low)")

            
            mapping, merge_edges, merge_diagnostics = merge_fragmented_tracks(
                windows, embeddings, thresh, REID_MAX_GAP_S,
                positions=positions,
                handoff_gap_s=REID_HANDOFF_GAP_S,
                handoff_px=REID_HANDOFF_PX,
                role_hint=role_hint,
                stationary_px=REID_STATIONARY_PX,
                anchor_embeddings=anchor_embeddings,
                anchor_sim_threshold=run_anchor_thresh,
                raw_crops=(raw_crops if _attire_on else None),
                plane=_ground, handoff_m=REID_HANDOFF_M,
                stationary_m=REID_STATIONARY_M,
                hsv_sim_threshold=HSV_MERGE_SIM_THRESHOLD,
                face_embeddings=(face_embeddings if ENABLE_FACE_MERGE_TIER else None),
                face_sim_threshold=FACE_MERGE_SIM_THRESHOLD,
                ir_hint=_ir_frac)
            _n_ident = len(set(mapping.values()))
            if ENABLE_GLOBAL_TRACKLET and _n_ident > GLOBAL_TRACKLET_MAX_IDS:
                _log.info(f"\U0001F310 global tracklet pass SKIPPED: {_n_ident} identities "
                      f"> GLOBAL_TRACKLET_MAX_IDS={GLOBAL_TRACKLET_MAX_IDS} (the linker "
                      f"builds a dense {2*_n_ident}x{2*_n_ident} cost matrix — it would "
                      f"eat the RAM and never finish). Greedy tier merges still ran.")
            elif ENABLE_GLOBAL_TRACKLET:
                _gbefore = len(set(mapping.values()))
                mapping = _apply_global_tracklet_pass(
                    mapping, windows, positions, anchor_embeddings, thresh)
                _gafter = len(set(mapping.values()))
                _log.info(f"\U0001F310 global tracklet pass: {_gbefore} -> {_gafter} "
                      f"identities ({_gbefore - _gafter} extra long-range merge(s))")
            if ENABLE_ATTIRE_MERGE_TIER or ENABLE_FACE_MERGE_TIER:
                _tiers_on = ", ".join(
                    t for t, on in (("attire/HSV", ENABLE_ATTIRE_MERGE_TIER),
                                    ("face", ENABLE_FACE_MERGE_TIER)) if on)
                _log.info(f"🧷 merge tiers active this run: appearance/anchor "
                      f"(thresh={thresh}/{run_anchor_thresh}), "
                      f"hand-off/stationary, {_tiers_on} "
                      f"(hsv>={HSV_MERGE_SIM_THRESHOLD}, "
                      f"face>={FACE_MERGE_SIM_THRESHOLD})")

            # (v38) per-tier accepted-merge breakdown + rejection diagnostics
            # — answers "which lever is actually pulling weight" directly,
            # instead of inferring it from before/after fragment counts.
            if merge_diagnostics:
                _tc = merge_diagnostics["tier_counts"]
                if _tc:
                    _tc_str = ", ".join(f"{t}={n}" for t, n in
                                        sorted(_tc.items(), key=lambda kv: -kv[1]))
                    _log.info(f"   📊 merges by tier: {_tc_str}")
                _rc = merge_diagnostics["role_conflicts_blocked"]
                _oc = merge_diagnostics["overlap_blocked_count"]
                if _rc or _oc:
                    _ot = merge_diagnostics.get("overlap_blocked_transitive", 0)
                    _log.info(f"   🧾 blocked candidates: {_rc} by role conflict "
                          f"(staff vs customer), {_oc} by time-window overlap "
                          f"(evidence said match, but groups were already "
                          f"co-visible under a different id)")
                    if _oc:
                        _log.info(
                            f"        of those {_oc}: {_ot} were TRANSITIVE "
                            f"({100*_ot/max(1,_oc):.0f}%) — the two tracks never "
                            f"co-exist, only their greedily-merged groups do. "
                            f"That share is an artefact of merge ORDER and is "
                            f"what a global solver would recover; the rest are "
                            f"genuinely two people and must stay separate.")
                # F11: the colour veto is waived across the colour<->IR
                # boundary, where a hue histogram compares the SENSOR rather
                # than the clothing. 'rescued' is the only number that says
                # whether that changed anything: pairs the veto would have
                # blocked and no longer does. If it is 0, the fix is inert on
                # this footage and must not be credited with any improvement.
                _sk = merge_diagnostics.get("hsv_veto_skipped_ir", 0)
                _rs = merge_diagnostics.get("hsv_veto_rescued", 0)
                if _sk or _rs:
                    _log.info(f"   🌗 colour veto waived across the IR boundary "
                              f"on {_sk} spatial pair(s); {_rs} of those it "
                              f"WOULD have blocked (that is the fix's effect)")
                if merge_diagnostics["overlap_blocked_samples"]:
                    _log.info(f"   ℹ️  strongest overlap-blocked candidates "
                          f"(evidence was good, order/window said no):")
                    for sim, a, b, tier in merge_diagnostics["overlap_blocked_samples"][:5]:
                        _log.info(f"      ID {a} <-> ID {b}: tier={tier} score={sim:.3f}")

            # ── FACE VETO (v32) ─────────────────────────────────────────────
            # Unlike the corroboration/cross-validation below (print-only),
            # this ACTIVELY reverses a merge when face evidence confidently
            # contradicts it — but only on the SPECIFIC direct edge that
            # caused the merge, and only when that edge's own evidence was
            # appearance-only (not hand-off/stationary/anchor tier, which is
            # independent of appearance and shouldn't be overridden by a
            # single face signal — see apply_face_veto docstring for why
            # v31's group-wide check caused visible ID flicker on evidence
            # that was actually correct). Runs before n_merged/remap_events
            # so every downstream count (id_merges, self-audit, exported
            # events) reflects the post-veto mapping.
            face_veto_flagged = None
            if ENABLE_FACE_VETO and face_embeddings:
                try:
                    mapping, face_veto_report, face_veto_flagged = apply_face_veto(
                        mapping, merge_edges, face_embeddings, windows,
                        sim_threshold=FACE_SIM_THRESHOLD,
                        veto_margin=FACE_VETO_MARGIN,
                        max_edge_score_for_veto=FACE_VETO_MAX_EDGE_SCORE)
                    if face_veto_report:
                        _log.info(f"🚫 Face veto: reversed {len(face_veto_report)} "
                              f"merge(s) on a confident face mismatch "
                              f"(appearance-only evidence, no hand-off/"
                              f"stationary/anchor backing):")
                        for a, b, sim, escore in face_veto_report:
                            _log.info(f"      ID {a} <-> ID {b}: face_sim={sim:.3f} "
                                  f"(edge evidence score {escore:.3f})")
                    if face_veto_flagged:
                        _log.info(f"   ℹ️  {len(face_veto_flagged)} merge(s) had a "
                              f"confident face mismatch but were kept — the "
                              f"merge evidence was hand-off/stationary/anchor "
                              f"tier (independent of appearance), which "
                              f"outranks a single face signal; still worth a "
                              f"manual look:")
                        for a, b, sim, escore in face_veto_flagged:
                            _log.info(f"      ID {a} <-> ID {b}: face_sim={sim:.3f} "
                                  f"(edge evidence score {escore:.3f})")
                except Exception as veto_exc:
                    _log.info(f"(Face veto skipped: {veto_exc})")

            # C2: gallery-sweep pins — a fragment whose face matched an
            # enrolled staff member IS that staff member, overriding whatever
            # the appearance tiers concluded; its whole merge-group follows.
            for _tid, _sn in _sweep_hits.items():
                for _k, _v in list(mapping.items()):
                    if _v == _tid:
                        mapping[_k] = _sn
                mapping[_tid] = _sn
            n_merged = sum(1 for k, v in mapping.items() if k != v)

            # (v38) persistent identity memory — one dossier per confirmed
            # person, pooling every evidence snapshot (face / hand-off /
            # stationary / anchor / attire / plain-gallery) collected across
            # their merged fragments. This is what keeps an id FIXED: it's
            # the memory match_track_to_dossiers cross-verifies future
            # sightings against, instead of trusting a single signal.
            if mapping:
                try:
                    identity_dossiers = build_identity_dossiers(
                        mapping, merge_edges, windows,
                        embeddings=embeddings,
                        anchor_embeddings=anchor_embeddings,
                        face_embeddings=face_embeddings,
                        raw_crops=raw_crops, positions=positions)
                    _eq = Counter(t for d in identity_dossiers.values()
                                 for t in d["evidence_tiers_used"])
                    _eq_str = (", ".join(f"{t}={n}" for t, n in
                                         sorted(_eq.items(), key=lambda kv: -kv[1]))
                              if _eq else "none (all single-fragment people)")
                    _log.info(f"🗂️  identity memory: {len(identity_dossiers)} "
                          f"person dossier(s) built (confirming evidence — "
                          f"{_eq_str})")
                except Exception as dossier_exc:
                    import traceback
                    _log.info(f"(Identity dossier build skipped: {dossier_exc})")
                    traceback.print_exc()
                    identity_dossiers = {}

            # DEBUG — self-selecting: find close-but-unmerged fragment pairs
            _unmerged_near = []
            _dbg_ids = sorted(windows, key=lambda k: windows[k][0])[:400]  # v55: was every pair
            for a in _dbg_ids:
                for b in _dbg_ids:
                    if str(a) >= str(b) or mapping.get(a) == mapping.get(b, b):
                        continue
                    pa, pb = positions.get(a), positions.get(b)
                    if not pa or not pb:
                        continue
                    d = math.hypot(pa[1][0] - pb[0][0], pa[1][1] - pb[0][1])
                    g = windows[b][0] - windows[a][1]
                    if 0 <= g <= REID_MAX_GAP_S and d <= 150:
                        _unmerged_near.append((d, g, a, b))
            # F12: say WHY. This report has listed pairs 2.7px apart as
            # "unmerged" for several sessions, and every attempt to explain them
            # was an inference from aggregate counters — five of which were
            # wrong. merge_diagnostics["pair_reject"] carries the first gate
            # that actually rejected each pair.
            _prj = (merge_diagnostics or {}).get("pair_reject") or {}
            _why_hist = defaultdict(int)
            for d, g, a, b in sorted(_unmerged_near,
                                     key=lambda x: (x[0], x[1], str(x[2]), str(x[3])))[:12]:
                _why = _prj.get((a, b)) or _prj.get((b, a)) or (
                    "no gate recorded — the pair was never even considered "
                    "(check the tier entry conditions)")
                _log.info(f"DEBUG unmerged near-pair: {a}->{b}  "
                          f"dist={d:.1f}px  gap={g:.1f}s")
                _log.info(f"        rejected by: {_why}")
            # The histogram is the actionable half: one gate accounting for most
            # of the fragmentation is a fix; an even spread means the merger is
            # not where the problem is.
            for d, g, a, b in _unmerged_near:
                _w = _prj.get((a, b)) or _prj.get((b, a)) or "never considered"
                _why_hist[_w.split("(")[0].strip().rstrip(",")] += 1
            if _why_hist:
                _log.info(f"   🧭 WHY {len(_unmerged_near)} obvious pairs stayed "
                          f"unmerged:")
                for _w, _n in sorted(_why_hist.items(), key=lambda kv: -kv[1])[:8]:
                    _log.info(f"        {_n:>5}x  {_w[:96]}")

            if n_merged:
                events, crossings, _ = remap_events(events, crossings, mapping)
                # F2: remap_events recomputes role from fragment majority-dwell
                # — which voted "customer" during the live loop. A face match
                # against the ENROLLED gallery outranks dwell: force it, or the
                # staff member C2 just rescued is counted as a guest all night.
                _pinned = set(_sweep_hits.values())
                if _pinned:
                    events = [dict(e, role="staff")
                              if e["track_id"] in _pinned else e
                              for e in events]
                # roles may change after merge (fragments pool their dwell)
                if staff_zones_here:
                    # Re-run AFTER the merge: fragments pool their dwell, and
                    # more importantly their VISITS — two fragments of one
                    # receptionist are two visits only once they are known to
                    # be the same person, which is precisely what spread needs.
                    events, _staff_ev = apply_staff_zone_override(
                        events, staff_zones_here, STAFF_OVERRIDE_MIN_S,
                        min_share_dwell_s=STAFF_OVERRIDE_SHARE_MIN_S,
                        observation_s=duration_s,
                        min_visits=STAFF_MIN_VISITS,
                        min_spread=STAFF_MIN_SPREAD,
                        sole_dwell_s=STAFF_SOLE_DWELL_S,
                        return_evidence=True)
                    for _sl in describe_staff_decision(
                            _staff_ev, ).splitlines():
                        _log.info(_sl)
                id_merges = {k: v for k, v in mapping.items() if k != v}
                _log.info(f"🔗 Re-ID stitching ({method} + hand-off, "
                      f"sim>={thresh}): {len(windows)} track fragments -> "
                      f"{len(set(mapping.values()))} people "
                      f"({n_merged} merges)")

                # ── AUTOMATIC CROSS-VALIDATION (v26) ────────────────────────
                # Independently re-check every OSNet merge with a second,
                # unrelated feature space (HSV histogram) on the SAME banked
                # crops from this run — track IDs line up exactly, no manual
                # cross-referencing against a separately re-tracked video.
                # Diagnostic only: never touches `mapping` or downstream
                # events, just flags what's worth a manual look.
                if ENABLE_CROSS_VALIDATION:
                    try:
                        xval = cross_validate_reid(
                            mapping, track_crops, windows,
                            gallery_sim_threshold=CROSS_VAL_GALLERY_SIM,
                            anchor_sim_threshold=CROSS_VAL_ANCHOR_SIM,
                            max_gap_s=REID_MAX_GAP_S)
                        n_agree = len(xval["agree"])
                        n_disagree = len(xval["disagree"])
                        n_hsv_only = len(xval["hsv_only"])
                        n_checked = n_agree + n_disagree
                        if n_checked:
                            _log.warning(f"🔍 Cross-validation (independent HSV check): "
                                  f"{n_agree}/{n_checked} {method.upper()} merges corroborated")
                        if n_disagree:
                            _log.warning(f"   ⚠️  {n_disagree} {method.upper()} merge(s) NOT "
                                  f"corroborated by HSV — review these:")
                            for a, b, gs, asim in sorted(
                                    xval["disagree"],
                                    key=lambda r: -max(r[2], r[3]))[:5]:
                                _log.info(f"      ID {a} <-> ID {b}: "
                                      f"gallery_sim={gs:.3f} anchor_sim={asim:.3f}")
                        if n_hsv_only:
                            _log.info(f"   ℹ️  {n_hsv_only} additional candidate(s) "
                                  f"HSV flagged that {method.upper()} did NOT merge:")
                            for a, b, gs, asim in sorted(
                                    xval["hsv_only"],
                                    key=lambda r: -max(r[2], r[3]))[:5]:
                                _log.info(f"      ID {a} <-> ID {b}: "
                                      f"gallery_sim={gs:.3f} anchor_sim={asim:.3f}")
                    except Exception as xval_exc:
                        import traceback
                        _log.info(f"(Cross-validation skipped: {xval_exc})")
                        traceback.print_exc()

                # ── FACE CROSS-VALIDATION (v30) ─────────────────────────────
                # Independent of both the appearance backbone AND HSV — where
                # a face resolved for both sides of a merge, this is the
                # strongest available check. Sparse by nature; print-only,
                # never touches `mapping`.
                if ENABLE_FACE_CORROBORATION and face_embeddings:
                    try:
                        fval = cross_validate_faces(
                            mapping, face_embeddings, windows,
                            sim_threshold=FACE_SIM_THRESHOLD,
                            max_gap_s=REID_MAX_GAP_S)
                        f_agree, f_disagree = len(fval["agree"]), len(fval["disagree"])
                        f_checked = f_agree + f_disagree
                        if f_checked:
                            _log.warning(f"👤 Face cross-validation: {f_agree}/"
                                  f"{f_checked} merges had a face on both "
                                  f"sides and matched")
                        if f_disagree:
                            _log.warning(f"   ⚠️  {f_disagree} merge(s) had a face on "
                                  f"both sides that did NOT match — strongest "
                                  f"available contradiction signal, review "
                                  f"these first:")
                            for a, b, sim in sorted(fval["disagree"],
                                                    key=lambda r: r[2])[:5]:
                                _log.info(f"      ID {a} <-> ID {b}: face_sim={sim:.3f}")
                        if fval["face_only"]:
                            _log.info(f"   ℹ️  {len(fval['face_only'])} unmerged "
                                  f"pair(s) had matching faces but were never "
                                  f"merged by {method} — possible missed "
                                  f"matches (e.g. person changed clothes):")
                            for a, b, sim in sorted(fval["face_only"],
                                                    key=lambda r: -r[2])[:5]:
                                _log.info(f"      ID {a} <-> ID {b}: face_sim={sim:.3f}")
                        face_validation_report = fval
                    except Exception as fval_exc:
                        import traceback
                        _log.error(f"(Face cross-validation skipped: {fval_exc})")
                        traceback.print_exc()
            else:
                _log.info("Re-ID stitching: no fragments needed merging")
        except Exception as exc:
            import traceback
            _log.error("=" * 78)
            _log.error(f"\U0001f6a8 Re-ID STITCHING FAILED: {exc}")
            _log.info("   Identity fragments were NOT merged. Every unique-people count "
                  "below is inflated — a previous run of this footage produced 22 "
                  "people with stitching working and 95 without it.")
            _log.info("   Treat this run's Tier B numbers as INVALID until this is fixed.")
            _log.info("=" * 78)
            traceback.print_exc()

    # D2: furniture does not fidget. The potted plant held a track for the whole
    # chunk at identical pixels and was labelled "staff" purely by zone dwell.
    _static = {}
    if ENABLE_STATIC_FILTER:
        try:
            _prot = protected_ids(crossings=crossings,
                                  face_ids=(list(_staff_seen_names)
                                            + list(_sweep_hits.values())),  # F3 protect
                                  canon=(mapping or {}))
            # Per-zone patience: a doorway phantom dies in 30 s, a person
            # standing at the desk gets 240. See STATIC_MIN_LIFE_BY_ROLE.
            _static_life = static_min_life_by_id(
                frame_log, polygons, zone_roles, canon=(mapping or {}),
                default_s=STATIC_MIN_LIFE_S,
                by_role=globals().get("STATIC_MIN_LIFE_BY_ROLE") or {})
            if _static_life:
                _lives = sorted(set(_static_life.values()))
                _log.info(f"\U0001fab4 D2 static-filter patience by zone: "
                          f"{_lives} s across {len(_static_life)} track(s) "
                          f"(one global bar would have used "
                          f"{STATIC_MIN_LIFE_S:.0f}s everywhere)")
            _static = static_track_ids(
                frame_log, canon=(mapping or {}), protected=_prot,
                min_life_s=STATIC_MIN_LIFE_S,
                max_centre_jitter=STATIC_CENTRE_JITTER,
                max_size_jitter=STATIC_SIZE_JITTER,
                min_life_by_id=_static_life)
            if _static:
                events, crossings, frame_log = drop_tracks(
                    events, crossings, frame_log, _static,
                    canon=(mapping or {}))
                _log.info(f"\U0001fab4 D2 dropped {len(_static)} static track(s) — "
                      f"furniture, not people:")
                for _cid, _d in sorted(_static.items(),
                                       key=lambda kv: -kv[1]["seconds"])[:5]:
                    _log.info(f"      id {_cid} at {_d['at']} for {_d['seconds']:.0f}s, "
                          f"centre jitter {_d['centre_jitter']:.4f} of body height")
        except Exception as _sf:
            _log.info(f"(static filter skipped: {_sf})")
    if _supp_stats.get("size_dropped"):
        _d1n, _d1s = _supp_stats["size_dropped"], _supp_stats.get("size_seen", 0)
        _d1pct = (100.0 * _d1n / _d1s) if _d1s else float("nan")
        _log.info(f"\U0001f4cf D1 dropped {_d1n} of {_d1s} detection(s) "
              f"({_d1pct:.1f}%) too large to be a person at their own footline "
              f"(tol {SIZE_FILTER_TOL}x the predicted area)")
        if _d1s and _d1pct > 25:
            _log.info(f"   \u26a0\ufe0f  D1 is dropping more than a quarter of every "
                  f"detection. Either the scene-geometry fit is wrong or this "
                  f"filter is eating real people — check the fit above before "
                  f"trusting any count from this run.")

    # ── D3: static phantoms whose IDS CHURN ────────────────────────────────
    # PHASE10_PHANTOM_AND_REGION_ARRIVALS. D2 above is per-track-id and is
    # structurally blind to this: the plant and the mirror re-mint a fresh id
    # every few seconds, so no single id ever lives long enough to be tested.
    # This asks the question of the LOCATION instead, and reuses drop_tracks so
    # events, crossings and frame_log stay consistent with each other.
    _phantoms = []
    if globals().get("ENABLE_PHANTOM_FILTER", True):
        try:
            _pprot = protected_ids(crossings=crossings,
                                   face_ids=(list(_staff_seen_names)
                                            + list(_sweep_hits.values())),  # F3 protect
                                   canon=(mapping or {}))
            _phantoms = phantom_regions(
                frame_log, frame_wh=(frame_w, frame_h), protected=_pprot,
                min_span_s=PHANTOM_MIN_SPAN_S,
                max_centre_jitter=PHANTOM_CENTRE_JITTER,
                max_size_cv=PHANTOM_SIZE_CV)
            if _phantoms:
                _pbox = {}
                for _fi, _t, _bs in frame_log:
                    for _tid, _a, _b, _c2, _d2 in _bs:
                        _pbox.setdefault(mapping.get(_tid, _tid) if mapping
                                         else _tid, []).append((_a, _b, _c2, _d2))
                _pdrop = {}
                for _cid, _bxs in _pbox.items():
                    if _cid in _pprot:
                        continue
                    _n = len(_bxs)
                    _med = tuple(sorted(v[k] for v in _bxs)[_n // 2]
                                 for k in range(4))
                    if in_phantom(_med, _phantoms):
                        _pdrop[_cid] = {"at": (round((_med[0] + _med[2]) / 2),
                                               round((_med[1] + _med[3]) / 2)),
                                        "sightings": _n}
                if _pdrop:
                    events, crossings, frame_log = drop_tracks(
                        events, crossings, frame_log, _pdrop,
                        canon=(mapping or {}))
                _log.info(f"\U0001faa9 D3 dropped {len(_pdrop)} id(s) sitting on "
                      f"{len(_phantoms)} static phantom region(s) — the plant/"
                      f"mirror class of false positive that D2 cannot see "
                      f"because their ids churn:")
                for _r in _phantoms[:4]:
                    _log.info(f"      at {_r['centre']}: {_r['why']}")
        except Exception as _pf:
            _log.info(f"(phantom filter skipped: {_pf})")

    event_ids = {e["track_id"] for e in events}
    # a genuinely-tracked person whose only zone presence was a sub-threshold
    # doorway transit has NO zone event — discarding their crossing here would
    # delete a real arrival, so tracked ids are exempt from the ghost filter.
    ghosts = sum(1 for c in crossings if c["track_id"] not in event_ids
                 and c["track_id"] not in track_time)
    if ghosts:
        crossings = [c for c in crossings if c["track_id"] in event_ids
                     or c["track_id"] in track_time]
        _log.info(f"filtered {ghosts} ghost line-crossings (flicker tracks with "
              f"no real zone presence)")

    # v55 #6: if the entry line's endpoints are the wrong way round, IN and
    # OUT swap and nothing complains — the headline number is simply backwards.
    # Over a whole chunk of a venue that is filling or steady, inward crossings
    # should not be dwarfed by outward ones.
    _n_in = sum(1 for c in crossings if c["direction"] == "in")
    _n_out = sum(1 for c in crossings if c["direction"] == "out")

    # A line that NEVER fires is indistinguishable from an empty venue in every
    # number downstream — "0 people entered" reads like a fact. On the first full
    # hour of CAM.112 the line triggered zero times while 46 people passed through
    # the waiting zone and reception was visited 74 times. Nothing said a word.
    _movers = len({e["track_id"] for e in events})
    if (_n_in + _n_out) == 0 and _movers >= 5:
        _log.error("=" * 78)
        _log.error(f"\U0001f6a8 ENTRY LINE NEVER TRIGGERED \u00b7 {camera_id}")
        _log.error(f"   {_movers} people moved through the zones and NOT ONE crossed the "
              f"entry line.")
        _log.error("   Every arrival/exit number is therefore 0 — that is a BROKEN LINE, "
              "not a quiet venue.")
        _log.info("   Most likely, in order:")
        _log.info("     1. the line is too SHORT and people walk around its ends —")
        _log.info("        it must span the full width of the doorway, end to end")
        _log.info("     2. it is drawn in the wrong place (e.g. across a corridor "
              "people never actually cross)")
        _log.info("     3. it is positioned where feet are occluded — the trigger "
              "anchor is the BOTTOM of the box")
        _log.info("   Fix: redraw entry_line in the zones JSON across the real "
              "threshold, on the floor, wall to wall.")
        _log.error("=" * 78)
    elif (_n_in + _n_out) > 0 and _movers >= 20 and (_n_in + _n_out) < _movers * 0.2:
        _log.error(f"\u26a0\ufe0f  entry line fired only {_n_in + _n_out} time(s) for "
              f"{_movers} people in the zones — it may be clipping most arrivals.")
    if (_n_in + _n_out) >= 20 and _n_out > _n_in * 1.6:
        _log.error("=" * 78)
        _log.error(f"🚨 ENTRY LINE LOOKS BACKWARDS · {camera_id}: {_n_out} people "
              f"crossed OUT vs {_n_in} IN over {(t or 1) / 60:.0f} min.")
        _log.info(f"   Unless this camera really did watch a room empty out, set "
              f"ENTRY_LINE_FLIP = {not ENTRY_LINE_FLIP} in Cell 2 and re-run. "
              f"Every 'people entered' number below is the wrong direction "
              f"until you do.")
        _log.info("=" * 78)

    roles = {**rec.final_roles(),
             **{e["track_id"]: e["role"] for e in events}}

    # A5: TIER-A crossing dedupe — one person crossing once, wearing several
    # ids while doing it, collapses to ONE event BEFORE anything counts it.
    # tier_a_crossings existed for exactly this and was wired into nothing the
    # HUD or the report reads; both count unique ids, which is what churn
    # inflates ("entered=5" for one re-minted person). Staff crossings pass
    # through untouched (tier A is a guest counter by design).
    _crossings_raw = list(crossings)   # F1: Tier B + the 1c alarm need the
    #    UN-deduped events, or the two estimators stop being independent and
    #    the disagreement alarm goes quiet exactly when identity fragments.
    if crossings:
        _staff_c = [c for c in crossings
                    if roles.get(c["track_id"]) == "staff"]
        _, _in_k = tier_a_crossings(crossings, plane=_ground,
                                    direction="in", roles=roles)
        _, _out_k = tier_a_crossings(crossings, plane=_ground,
                                     direction="out", roles=roles)
        _deduped = sorted(_staff_c + _in_k + _out_k, key=lambda c: c["t"])
        if len(_deduped) < len(crossings):
            _log.info(f"tier-A crossing dedupe: {len(crossings)} raw crossing "
                      f"event(s) -> {len(_deduped)} (id churn at the line "
                      f"collapsed before counting)")
        crossings = _deduped

    # CONFIRMATION WAIT / U-TURN FILTER. Runs AFTER the tier-A dedupe because
    # the two fix different errors and compose in this order: dedupe collapses
    # one person crossing once under several ids (SAME direction), then this
    # cancels IN/OUT PAIRS where the person crossed and came straight back.
    # A dedupe can never catch that pair — the events point opposite ways.
    #
    # Matters here because CAM.112's entry line sits exactly where guests pause
    # to be greeted, which is the worst placement for a bare line counter.
    if ENABLE_CROSSING_CONFIRM and crossings:
        from .analytics import confirm_crossings
        _n_before = len(crossings)
        crossings, _uturn = confirm_crossings(
            crossings, confirm_s=CROSSING_CONFIRM_S, per_line=zcfg_confirm)
        if _uturn:
            _byline = {}
            for _u in _uturn:
                _k = _u.get("line") or "entry"
                _byline[_k] = _byline.get(_k, 0) + 1
            _log.info(f"\U0001f6ac U-turn / confirmation: {_n_before} -> "
                      f"{len(crossings)} crossing(s); dropped {len(_uturn)} where "
                      f"someone crossed and came back within {CROSSING_CONFIRM_S}s "
                      f"— nobody actually arrived or left. per door: {_byline}")
    _log.info(f"{camera_id}: {len(events)} zone events, "
          f"{len(event_ids)} tracked people")

    # ── PASS 2: render the video from final identities ─────────────────────
    out_path = OUTPUT_DIR / f"{camera_id}{chunk_tag}_annotated.mp4"
    canon = dict(mapping) if mapping else {}
    _skip_render = not globals().get("RENDER_VIDEO", True)  # NORENDER
    if _skip_render:  # NORENDER
        _log.info("   render SKIPPED (RENDER_VIDEO=False) — PASS 2 decodes the "  # NORENDER
                  "whole video a second time purely to draw boxes on it. Every "  # NORENDER
                  "number is already computed; the video is for a human to "  # NORENDER
                  "audit WHY, so it is worth its ~20% only when someone will "  # NORENDER
                  "watch it.")  # NORENDER
    snapshots = [] if _skip_render else render_annotated(
        video_path, out_path, frame_log, canon, roles, events, crossings,
        polygons, zcolors,
        (drawn_lines or None),      # ALL doors, not just the first
        eff_fps, step, native_fps, zone_roles=zone_roles,
        proxy_dir=proxy_dir, duration_s=duration_s, phantoms=_phantoms)
    if proxy_dir is not None:
        shutil.rmtree(proxy_dir, ignore_errors=True)   # ~1 GB per chunk
    # RENDER_DIRECT_H264 renames the file to *_annotated_h264.mp4 INSIDE
    # render_annotated; without picking that up here the run reported
    # "annotated_video": None and the export cell shipped no video at all.
    _h264_alt = Path(out_path).with_name(Path(out_path).stem + "_h264.mp4")
    if _h264_alt.exists():
        out_path = _h264_alt
    if not Path(out_path).exists():
        # nobody was in shot for this whole chunk, so no frame was written and
        # no file exists. Say so instead of handing downstream a dead path.
        _log.info(f"(no annotated video for {camera_id}{chunk_tag}: nobody appeared "
              f"in any analysed frame of this chunk)")
        out_path = None
    else:
        _log.info(f"annotated video (persistent ids) -> {out_path}")

    return {
        "camera_id": camera_id, "events": events, "roles": roles,
        "is_ir": _is_ir, "duration_s": duration_s,
        "staff_matched_names": sorted(_staff_seen_names),
        "frames_skipped_idle": _n_skipped, "tracker_resets": _n_resets,
        # B1: the labelling export re-decodes at THIS rate. Falling back to
        # FPS_TARGET (8) while analysis ran at eff_fps (7.5) drifted 6.7% and
        # tripped the 5% alignment guard, so no package was ever built.
        "eval_fps": eff_fps,
        "size_dropped": _supp_stats.get("size_dropped", 0),
        "size_seen": _supp_stats.get("size_seen", 0),
        # The funnel travels with the run, not just the log — comparing two
        # runs stage by stage is how you tell a threshold change from a
        # detector change, and a log you have to re-read by eye is not that.
        "detection_funnel": _funnel.as_dict(),
        "capabilities": _caps.as_dict(),
        "profile_ms": _prof.counters(),
        "phantom_regions": [{k: v for k, v in _r.items() if k != "box"}
                            for _r in _phantoms],
        "zone_roles": dict(zone_roles),
        "static_dropped": len(_static),
        "ir_frames": sum(1 for v in _frame_ir.values() if v),
        "ir_frames_total": len(_frame_ir),
        "ir_switches": [(round(s, 1), bool(v)) for s, v in _ir_switches],
        "perspective_fit": _persp.describe(),
        "ground_plane": _ground.describe(),
        "ground_mode": _ground.mode,
        "perspective_coeffs": (list(_persp._refit()) if _persp.ready else None),
        "frame_size_analysed": [frame_w, frame_h],
        "carried_suppressed": _supp_stats.get("carried_suppressed", 0),
        "heads_without_person": _n_head_only,
        "live_max_dist_px": round(_live_max_dist, 1),
        "crossings": crossings,
        "crossings_raw": _crossings_raw,   # F1
        "t_end": t + frame_step_s,
        "custom_model": False,   # kept for downstream compat; always False now
        "line_in": (sum(lz.in_count for lz in line_zones.values())
                    if line_zones else None),
        "line_out": (sum(lz.out_count for lz in line_zones.values())
                     if line_zones else None),
        "per_line": {n: {"in": lz.in_count, "out": lz.out_count}
                     for n, lz in line_zones.items()},
        # U1: the verdict travels WITH the run, so the report cannot forget it
        "view_valid": bool(_view.get("valid", True)),
        "view_reasons": list(_view.get("reasons", [])),
        "view_checked": bool(_view.get("checked", False)),
        "snapshots": snapshots, "zcolors": zcolors,
        "id_merges": id_merges,
            "seated_ratios": {tid: round(track_seated_frames.get(tid, 0) / max(track_total_frames.get(tid, 1), 1), 2) for tid in set(list(track_seated_frames.keys()) + list(track_total_frames.keys()))},
        "online_reid": (_REID_STATE.get("method") if online_tracker is not None
                        else "auto-proxy" if use_online_tracker else None),
        "annotated_video": (str(out_path) if out_path else None),
        # v53: raw per-frame boxes + final identity mapping, so an offline
        # audit can reconstruct exactly who held which id in every frame
        "frame_log": frame_log,
        "canon_map": dict(mapping) if mapping else {},
        # (v31) diagnostic/veto reports, kept per-run so the self-audit and
        # any manual review can look back at exactly what fired without
        # re-running the pipeline. None when the corresponding feature was
        # off, skipped, or had nothing to report.
        "calibration_report": calibration_report,
        "face_validation_report": face_validation_report,
        "face_veto_report": face_veto_report,
        "face_veto_flagged": face_veto_flagged,
        # (v38) persistent per-person identity memory + the diagnostics that
        # explain which evidence tier proved/blocked each merge this run.
        "merge_diagnostics": merge_diagnostics,
        "identity_dossiers": identity_dossiers,
        # (v34) raw per-frame box centers -- already computed for rendering,
        # kept here too so a heatmap can be built without re-processing the
        # video. NOTE: this is a dwell-density VISUALIZATION built from the
        # same track positions the pipeline already has -- it does not, and
        # cannot, improve identity tracking itself (see plot_occupancy_heatmap).
        "frame_log": frame_log,
    }
