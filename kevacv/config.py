"""config.py — every tracking-critical constant, in one importable place.

WHY THIS EXISTS
    These lived as bare globals in notebook Cell 2, and Cell 2e overrode three
    of them AFTERWARDS. The CONFIG AUDIT printed the Cell 2 values, so the log
    said REID_MAX_GAP_S = 7200 while the run actually used 900. A block whose
    stated job is "all tracking-critical parameters" was reporting a stale
    version of them (patched in V75; this module removes the possibility).

    Two values genuinely have two lives:

        REID_MAX_GAP_S   Cell 2: 7200   Cell 2e: 900
        MAX_BODY_GAP_S   Cell 2:  480   Cell 2e: 300

    Cell 2e is the 10-hour scale profile and its values are the ones that run.
    They are the defaults here; SCALE_PROFILE_OVERRIDES records what changed so
    the difference stays visible instead of being folded away silently.

HOW TO CHANGE A VALUE
    Not by editing this file for one run. Use tools/run_night.py:

        KV_REID_MAX_GAP_S=1200 python tools/run_night.py

    which injects an override cell after the config cell — so a run's settings
    are recorded in that run's own log rather than in an edit nobody sees.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field, fields

# ── appearance / re-identification ──────────────────────────────────────────
# MEASURED, not chosen. From the 2026-08-13 run on CAM.112, over 723
# same-person and 6535 different-person pairs:
#
#     same-person       p10 0.332   p50 0.461
#     different-person  p50 0.352   p90 0.538
#     EER sweep         optimal 0.370   (balanced accuracy 0.673)
#
# The old bars were 0.60 and 0.75 — ABOVE the median true match. The stitcher
# was therefore rejecting more than half of all correct merges by the
# project's own measurement, which is why one hour produced 194 "people" from
# roughly 22 bodies. The earlier note here said the same thing and put the p50
# at 0.435; it was dismissed as circular, and re-measuring only confirmed it.
#
# THE HONEST PART: balanced accuracy 0.673 means these distributions OVERLAP
# badly. 0.37 is the best single cut available, not a clean separation — it
# will merge some strangers. That is the trade, made deliberately: on this
# footage over-merging is recoverable (the face veto and co-visibility can
# split them) while under-merging silently inflates every count.
#
# The LIVE matcher no longer uses a bar at all — see REID_RATIO. These two
# remain because the offline stitcher still decides pairwise.
REID_SIM_THRESHOLD = 0.37      # gallery merge bar
ANCHOR_SIM_THRESHOLD = 0.45    # best-crop 1:1, stricter than the gallery

# ── time gates ──────────────────────────────────────────────────────────────
REID_MAX_GAP_S = 900.0         # Cell 2e value (Cell 2 said 7200)
MAX_BODY_GAP_S = 300.0         # Cell 2e value (Cell 2 said 480)
NEAR_GAP_S = 15.0
FAR_GAP_S = 180.0
NEAR_GAP_BONUS = 0.04          # short gap -> threshold relaxed by this
FAR_GAP_PENALTY = 0.05         # long gap  -> threshold raised by this
# 0.05 is small next to what a long gap costs in discriminative power: the
# metric gate allows max_speed * gap, which at 900 s is ~1980 m — i.e. no
# constraint at all. kevacv/topology.py exists to hold that line instead.

# ── spatial plausibility ────────────────────────────────────────────────────
MAX_PLAUSIBLE_SPEED_PX = 220.0
SPATIAL_PENALTY_SCALE = 0.15
MAX_SPATIAL_PENALTY = 0.30

# ── occlusion ───────────────────────────────────────────────────────────────
OCCLUSION_IOU = 0.30           # boxes at/above this mutually occlude
OCCLUSION_CONTAIN = 0.60       # ...or one is this fraction inside the other

# ── vetoes (a second opinion may block a merge, never create one) ───────────
ENABLE_HANDOFF_APPEARANCE_VETO = True
HANDOFF_VETO_SIM = 0.30
ENABLE_HANDOFF_HSV_VETO = True
HANDOFF_HSV_VETO_SIM = 0.50
ENABLE_APPEARANCE_HSV_VETO = True
APPEARANCE_HSV_VETO_SIM = 0.45

# ── infrared ────────────────────────────────────────────────────────────────
IR_TRACK_FRAC = 0.5            # a track this IR-heavy gets no colour evidence
# 58% of CAM.112's frames are infrared and 67 of 94 tracks were withheld from
# colour evidence entirely. Published SOTA for visible-infrared re-id is ~77%
# Rank-1, so nothing here should depend on appearance surviving the boundary.

# ── geometry ────────────────────────────────────────────────────────────────
DEFAULT_HFOV_DEG = 90.0        # overridden per venue by venue_profile
PERSON_H_M = 1.70

SCALE_PROFILE_OVERRIDES = {
    # what Cell 2e changes, and why. Kept as data so the audit can print the
    # difference rather than one value silently winning.
    "REID_MAX_GAP_S": (7200.0, 900.0,
                       "7200 s was 'the whole video' when a video was 10 min. "
                       "Over 10 h it invites two strangers in black shirts, "
                       "hours apart, to become one person."),
    "MAX_BODY_GAP_S": (480.0, 300.0,
                       "appearance-only tiers get a tighter cap than face."),
}


@dataclass
class TrackingConfig:
    """All of the above as one object, so a run can carry its settings around
    instead of reading module globals that something else may have rebound."""
    reid_sim_threshold: float = REID_SIM_THRESHOLD
    anchor_sim_threshold: float = ANCHOR_SIM_THRESHOLD
    reid_max_gap_s: float = REID_MAX_GAP_S
    max_body_gap_s: float = MAX_BODY_GAP_S
    near_gap_s: float = NEAR_GAP_S
    far_gap_s: float = FAR_GAP_S
    near_gap_bonus: float = NEAR_GAP_BONUS
    far_gap_penalty: float = FAR_GAP_PENALTY
    max_plausible_speed_px: float = MAX_PLAUSIBLE_SPEED_PX
    spatial_penalty_scale: float = SPATIAL_PENALTY_SCALE
    max_spatial_penalty: float = MAX_SPATIAL_PENALTY
    occlusion_iou: float = OCCLUSION_IOU
    occlusion_contain: float = OCCLUSION_CONTAIN
    handoff_veto_sim: float = HANDOFF_VETO_SIM
    handoff_hsv_veto_sim: float = HANDOFF_HSV_VETO_SIM
    appearance_hsv_veto_sim: float = APPEARANCE_HSV_VETO_SIM
    enable_handoff_appearance_veto: bool = ENABLE_HANDOFF_APPEARANCE_VETO
    enable_handoff_hsv_veto: bool = ENABLE_HANDOFF_HSV_VETO
    enable_appearance_hsv_veto: bool = ENABLE_APPEARANCE_HSV_VETO
    ir_track_frac: float = IR_TRACK_FRAC
    default_hfov_deg: float = DEFAULT_HFOV_DEG

    def as_dict(self):
        return asdict(self)

    def describe(self):
        """The audit, printed from the object that will actually be used —
        so it cannot describe a value something later overrode."""
        L = ["=" * 78, "  TRACKING CONFIG — the values this run will use", "=" * 78]
        for f in fields(self):
            L.append(f"  {f.name:<34} = {getattr(self, f.name)}")
        if SCALE_PROFILE_OVERRIDES:
            L.append("  " + "-" * 74)
            L.append("  scale-profile overrides applied (Cell 2 value -> used):")
            for k, (old, new, why) in SCALE_PROFILE_OVERRIDES.items():
                L.append(f"    {k:<20} {old} -> {new}")
                L.append(f"      {why}")
        L.append("=" * 78)
        return "\n".join(L)


DEFAULT = TrackingConfig()


# ============================================================================
#  ENGINE — extracted from notebook Cell 7 (132 bare globals -> named config)
#
#  Cell 7 also performs 33 `globals().get(...)` lookups at runtime. Those are
#  deliberate late-binding hooks (a later cell may rebind a value), so they
#  are NOT replaced here — engine.py keeps reading them, and this module just
#  supplies the defaults they fall back to.
# ============================================================================

# ── detector — what counts as a person at all ───────────────────────────────
DETECT_CONF_FLOOR = 0.25
CONF_THRESHOLD = 0.35
NEW_TRACK_CONF = 0.45
KEEP_TRACK_CONF = 0.2
ENABLE_CONF_HYSTERESIS = True
YOLO_IMGSZ = 1280
# Frames per detector forward. Larger batches keep the GPU busy across the
# Python bookkeeping between flushes — the `sm` trace showed bursts to 100%
# then long zeros, which is a GPU waiting for its next batch. Pure scheduling:
# the same frames go through the same model and the outputs are unchanged.
# 24 fits an L4's 22GB at imgsz 1280 with room to spare.
DET_BATCH = 24
DEDUP_NMS_IOU = 0.7

# ── tiled inference (SAHI) — the small-person problem ───────────────────────
# The camera is 3840x2160 and analysis runs at 1280x720, so a guest at the
# door is ~60 px tall. A detector sees that as a handful of pixels, which is
# what "bad bounding boxes" (#8), "two people merged into one box" (#9) and
# much of the detection flicker (#11) actually ARE — small-object failure.
#
# Slicing Aided Hyper Inference (Akyon et al., arXiv:2202.06934) is the
# standard fix: cut the frame into overlapping tiles, run the detector on each
# at the model's full imgsz — so a 640 px tile at imgsz 1280 gives everything
# in it 2x effective resolution — then map back and NMS.
#
# kevacv/tiled.py implemented this, with tests, and was wired to nothing.
#
# HONEST ABOUT THE TRADE: this multiplies detector calls per frame. The ROI
# band keeps that bounded — people are only small where the ground plane says
# they are small, i.e. the far part of the frame — so tiles cover that band
# rather than the whole image. cost_estimate() is logged at startup so the
# multiplier is visible BEFORE an hour is spent on it.
#
# OFF by default: it changes both cost and detections, and should be turned on
# against a measured baseline, not inherited silently.
ENABLE_TILED_DETECT = False
TILE_PX = 640            # tile edge; fed to the detector at YOLO_IMGSZ
TILE_OVERLAP = 0.2       # fraction — a body on a seam must appear whole in one
TILE_NMS_IOU = 0.55      # merging boxes from overlapping tiles
TILE_TARGET_MIN_PX = 110  # tile only where a person is predicted SHORTER than
                          # this. Above it the whole-frame pass already sees
                          # them and a tile adds cost for nothing.
IR_DETECT_CONF_FLOOR = 0.2

# ── fisheye ─────────────────────────────────────────────────────────────────
# One-parameter radial lens model; see kevacv/fisheye.py. 0.0 = no correction,
# and fisheye.dewarped_predict short-circuits that to a plain predict, so the
# default costs nothing.
#
# DO NOT SET THIS BY HAND. tools/autocalib_fisheye.py measures it from straight
# edges in the frame and REFUSES to write a k that fails its hold-out check --
# a wrong k moves every box silently, which is worse than no correction.
FISHEYE_K = 0.0

# ── sampling & motion gate — the frames we bother to look at ────────────────
# THE DEFAULT ONLY. config/cam112.yaml is authoritative — see apply_run_config
# at the bottom of this file. Three different values were live at once (15
# here, 8 in the yaml, 7 in the notebook) and the yaml was never read by
# anything, so the value nobody had chosen is the one that ran.
#
# It is not a cosmetic setting: eff_fps derives the live re-id distance gate
# (engine._live_max_dist), the lost-track buffer, and the frame step every
# zone dwell is measured in. Getting it wrong loosens or tightens every motion
# gate in the pipeline simultaneously.
FPS_TARGET = 8

# ── analysis resolution — the ceiling on re-id quality ──────────────────────
# THIS IS THE INPUT TO EVERY APPEARANCE DECISION, and it was the tightest
# constraint in the pipeline.
#
# The camera is 3840x2160. At 1280 a guest at the far wall is ~60 px tall, and
# BOTH re-id crop sites read from this frame:
#
#   engine.py  _c   = frame[...]   live crop  -> _IdentityMemory -> live match
#   engine.py  crop = frame[...]   banked     -> offline fragment merging
#
# Each is then resized to 128x256 before the backbone sees it, so a 60 px body
# is upscaled ~4x and the embedding is built mostly from interpolation.
# Measured separability was 0.658 balanced accuracy — and as kevacv/tiled.py
# puts it, that is a number "no threshold repairs". Raising the source pixels
# is the only thing that moves it.
#
# 1920 makes that ~90 px. Not free, but cheap in the right places:
#   detector   UNCHANGED — YOLO_IMGSZ=1280 resizes the input either way; it now
#              resamples from 1920 rather than 3840, which is marginally better
#   decode     2.25x the pixels; an L4 is nowhere near saturated
#   gates      handled automatically. REF_DIAGONAL_PX (1468.6) IS the 1280x720
#              diagonal, so ENABLE_RESOLUTION_SCALING multiplies every pixel
#              threshold by exactly 1.50x. This is what that machinery is for.
#   disk       proxy JPEGs grow; the preflight disk check reports it up front
#
# VERIFY, DO NOT ASSUME: the RE-ID SEPARABILITY block printed by every run
# measures same-person vs different-person similarity without labels. Compare
# same_p50 at 1280 and at 1920. If it moves, resolution was the bottleneck and
# native-resolution re-crop (3x, one extra decode pass) is worth building. If
# it does not, the limit is elsewhere — infrared, or CLIP-ReID on this domain —
# and that pass would have been wasted work.
ANALYSIS_MAX_W = 1920
MOTION_GATE = True
MOTION_IDLE_S = 10.0
MOTION_MIN_FRAC = 0.002
ENABLE_RESOLUTION_SCALING = True
REF_DIAGONAL_PX = 1468.6
ENABLE_CLAHE = False
# ── exposure adaptation (symptom 19: dark / overexposed frames) ─────────────
# CLAHE used to fire on exactly one condition — "is this frame infrared" — at
# a fixed clipLimit of 2.0. Exposure was never measured at all, so a frame
# blown out by the window behind the desk, or a dim stretch before the IR
# cut-over that is still technically colour, went to the detector as captured.
# Detection confidence tracks contrast while the conf floor is a fixed number,
# so those frames lose people silently.
ENABLE_EXPOSURE_ADAPT = True
EXPOSURE_DARK_MEAN = 60.0      # mean luma below this = underexposed
EXPOSURE_BRIGHT_MEAN = 195.0   # mean luma above this = overexposed
EXPOSURE_CLIP_FRAC = 0.20      # or this share of pixels pinned at an extreme
EXPOSURE_MAX_CLIP = 4.0        # CLAHE ceiling — past ~4 it amplifies sensor
                               # noise into edges the detector reads as people,
                               # which trades symptom 19 for symptoms 5/6

# ── tracker ─────────────────────────────────────────────────────────────────
TRACKER_MODE = 'botsort-reid'
USE_REAL_ONLINE_REID = True
BOTSORT_MATCH_THRESH = 0.75
# How long a lost track is remembered before its identity is retired.
#
# 2026-08-16 BUG: this was silently a quarter of what it says. boxmot's BotSort
# does
#     self.buffer_size = int(frame_rate / 30.0 * track_buffer)
#     self.max_time_lost = self.buffer_size
# i.e. track_buffer is expressed in FRAMES AT 30 FPS and rescaled internally.
# Three call sites passed int(fps * LOST_TRACK_BUFFER_S), so the fps scaling was
# applied TWICE:
#     passed   8 * 60 = 480
#     boxmot   int(8/30 * 480) = 128 frames = 16 s at 8 fps
#     intended                              = 60 s
# A guest occluded, turned away, or behind the plant for more than 16 s came
# back as a NEW PERSON. On the 18:30 hour that is 408 track fragments resolving
# to 31 "people" at a desk that sees a handful, and it is the mechanical half of
# the operator's "P4 becomes someone else when they cross" report.
# Correct form is int(30 * LOST_TRACK_BUFFER_S) — which engine.py:2193 was
# already using for a different tracker, so the codebase held both formulas.
LOST_TRACK_BUFFER_S = 60
# GMC OFF BY DEFAULT, 2026-08-20, on measurement.
#
# Global motion compensation estimates how the CAMERA moved between frames.
# CAM.112 is a G6 turret bolted to a ceiling. Logged as bug G4 on 2026-08-19
# and untestable until the yaml key existed -- the first attempt reported
# "identical to baseline" only because a duplicated key meant the flag never
# applied.
#
# The real A/B (m_gmcoff_v2, GMC "sof" -> "none", single variable):
#     arrivals   line 5 region 6   vs baseline line 5 region 5
#     tracks     64                vs 64
#     fragments  1.12 ids/person   vs 1.12
# i.e. no measurable effect on any count, so it is pure cost on a fixed camera.
# Set analysis.enable_gmc: true to put it back for a moving/PTZ install.
ENABLE_GMC = False
GMC_METHOD = 'sof'

# ── live identity memory — one canonical id per body, decided per frame ─────
ENABLE_LIVE_IDENTITY_MEMORY = True
LIVE_REID_SIM_THRESHOLD = 0.62   # legacy; the ratio test decides now
# ── RELATIVE MATCHING (Lowe's ratio) — replaces the fixed accept bar ────────
# The run measured same-person p50 = 0.461 and different-person p90 = 0.538 on
# this camera. Those distributions OVERLAP, so no single threshold separates
# them: 0.60 rejects most true matches, 0.37 admits strangers. Choosing a
# number only chooses which error to make.
#
# And 45% of the footage is infrared, where every cosine is globally lower. An
# absolute bar is therefore too strict at night and too loose by day AT THE
# SAME TIME — which is precisely the "ids change too easily" symptom.
#
# Lowe's ratio test asks a relative question instead: is the best candidate
# decisively better than the runner-up? A constant factor applied to every
# similarity — what a modality change effectively does — leaves that unchanged.
#
# REID_RATIO: the runner-up may be at most this fraction of the winner. Lower
# is stricter. 0.90 means the winner must be >=11% better than second place.
REID_RATIO = 0.90
# The absolute bar, demoted: it now only decides the case where there is no
# runner-up to compare against. Set just under the measured same-person p10
# (0.332), so nine in ten true matches clear it — the ratio test and the
# physical gates do the discriminating.
LIVE_REID_ABS_FLOOR = 0.32
LIVE_REID_MEMORY_TTL_S = 1800.0
LIVE_REID_MAX_SPEED_PX_S = 560.0
LIVE_APPEARANCE_THRESH = 0.5
ENABLE_COVISIBILITY_BLOCK = True
ENABLE_OCCLUSION_GUARD = True
ENABLE_SWAP_REVALIDATION = True
SWAP_MARGIN = 0.1
# ROBUST GROUND PLANE — use the RANSAC fitter instead of bin-median LSQ.
#
# kevacv/geometry_calibration.fit_robust_ground_plane has existed, been
# exported from __init__, and been called by NOTHING. The fitter actually in
# use (engine's bin-median least squares) says in its own comment: "upgrade
# only if a real fit is seen to drift." It has drifted.
#
# MEASURED on output/p0v4 (9,516 detections, CAM.112):
#     in use (bin-median LSQ)   camera 4.04 m, horizon row -261   impossible
#     fit_robust_ground_plane   camera 2.50 m, horizon row  +32   plausible
#
# Validated against the FLOOR, not against an opinion. The checkerboard's
# horizontal period is 128.6 px over rows 700-860; converted through the
# RANSAC plane that is a tile edge of 0.34 / 0.30 / 0.28 m. The operator
# states the tiles are 30 cm, and 12x12 in (30.48 cm) is the standard
# checkerboard size. The in-use plane would render the same tiles at ~0.49 m.
#
# The RANSAC fitter keeps only boxes with aspect 2.0-4.8, which is 17% of
# detections on this camera. That looked wrong -- hand labels here measure
# h/w 1.14 -- but the narrow band is doing the work: it selects UPRIGHT,
# FULLY-VISIBLE people and rejects the crouched, seated and desk-occluded
# boxes that were tilting the line. Widening it to 0.8-2.2 was tried and gave
# camera 3.93 m, horizon -266, i.e. worse.
#
# Default False so this file changes nothing by itself. A/B it, read the
# scorecard's ground-plane verdict, then flip it.
# A RANSAC line through 2 of 12 points fits anything. Measured on the first
# m_plane run, the robust fit wandered 4.85 -> 9.0 -> 1.83 -> 2.96 m as samples
# accumulated, while the SAME fitter over the run's full 9,516 detections was a
# stable 2.50 m / horizon +32. The fitter's own floor is 10 points, which is a
# guard against crashing, not against being wrong.
ROBUST_PLANE_MIN_SAMPLES = 300
# ── event queue (P7) ────────────────────────────────────────────────────────
# Stream events to disk DURING the run instead of buffering everything and
# dumping at the end. On a 600s chunk that is irrelevant; on the 24h runs this
# is built for, a crash at hour 23 currently loses the whole night.
# The queue never blocks the CV loop and counts anything it drops.
ENABLE_EVENT_QUEUE = False
EVENT_QUEUE_MAXSIZE = 10000

# ── pose / activity (P9) ────────────────────────────────────────────────────
# A layer ON TOP of identity, never inside it. Off by default and deliberately
# so: pose answers "what is this body doing", not "who is this" or "did they
# come in", so it cannot improve any count this pipeline is currently wrong
# about. Wired anyway, because a module nothing imports is how this codebase
# accumulated fit_robust_ground_plane, validate_entry_line, CLAHE and tiled.py
# -- all built, all correct, all changing nothing.
ENABLE_POSE = False
POSE_MODEL = "yolo11n-pose.pt"
POSE_MAX_TRACKS = 8            # compute budget: pose on the longest-lived only
POSE_MIN_TRACK_S = 2.0         # standing vs walking needs >= 2 samples
POSE_STRIDE = 8                # sample every Nth analysed frame

# ── observation layer (per-frame rows for the intelligence layer) ───────────
# One row per person per frame -> EventQueue -> observations.jsonl ->
# tools/ingest_obs.py -> Postgres. OFF by default: it is an output nobody
# needs on a tuning run, and at 8 fps it is ~145k rows/camera-hour.
ENABLE_OBSERVATIONS = False
OBS_QUEUE_MAXSIZE = 20000      # ~2.5 s of backlog at 8 fps x 5 people
OBS_STATIONARY_PX_S = 8.0      # ~1 px/frame at 8 fps
OBS_MOTION_WINDOW = 4          # anchors kept per track for the median
OBS_EMB_STRIDE = 8             # embeddings per track: 1/s, not 8/s

ENABLE_ROBUST_GROUND_PLANE = False
MAX_WALK_SPEED_MPS = 2.2

# ── offline re-id — crops, gaps and the merge tiers ─────────────────────────
ENABLE_REID_STITCH = True
REID_CROPS_PER_TRACK = 6
REID_MIN_CROP_H = 50
REEMBED_EVERY_S = 4.0
REID_HANDOFF_PX = 160
REID_HANDOFF_M = 1.6
REID_HANDOFF_GAP_S = 4.0
# STATIONARY TIER GAP CAP — None = no cap (behaviour before 2026-08-20).
#
# The stationary tier merges two fragments on PROXIMITY ALONE. Unlike the
# hand-off tier it has never had a time bound: its comment reads "near-zero
# displacement across a long gap = same seat, same person". That is sound for a
# SEAT and precisely wrong for a RECEPTION DESK, whose whole function is
# different people occupying the same spot in succession.
#
# Measured on output/p0classfix2 (600s, CAM.112):
#     merges by tier: stationary=33 of 54 (61%)
#     71->82  dist 14.0px  gap  62.3s  accepted, score 0.919
#     55->88  dist 17.1px  gap 125.1s  accepted, score 0.914
#     result: 5 of 7 "customers" span 7-10 minutes of a 10-minute chunk
#
# Neither veto can catch this here. The CLIP anchor cannot separate same from
# different on this footage at ANY threshold (same-person p10 0.318 vs
# different-person p90 0.627 -- see reid_calibration), and the HSV veto
# deliberately abstains across an IR boundary, which is 66% of these frames.
# So proximity is effectively the only evidence, and proximity at a desk is
# evidence of a DESK, not of a person.
#
# Left at None so this file changes nothing by itself. Set
# analysis.reid_stationary_max_gap_s and A/B it -- three constants have
# already been shipped here on reasoning alone and all three were reverted.
REID_STATIONARY_MAX_GAP_S = None
REID_STATIONARY_PX = 60.0
REID_STATIONARY_M = 0.6
GAP_MERGE_S = 15
ENABLE_ATTIRE_MERGE_TIER = True
HSV_MERGE_SIM_THRESHOLD = 0.75
ENABLE_GLOBAL_TRACKLET = True
GLOBAL_TRACKLET_MAX_IDS = 900
# Measure appearance separability on every run, from facts that do not depend
# on appearance: two boxes in one frame are two people; the same raw id 0.1s
# apart is one person by motion. The existing 0.435 same-person p50 was
# measured circularly and this file says so — this replaces it with a number
# that can be trusted, at no extra GPU cost. Recommends only; nothing is
# applied automatically.
ENABLE_LIVE_SEPARABILITY = True
ENABLE_REID_CALIBRATION = True
CALIBRATION_AUTO_APPLY = False
ENABLE_CROSS_VALIDATION = True
CROSS_VAL_GALLERY_SIM = 0.6
CROSS_VAL_ANCHOR_SIM = 0.75

# ── faces — staff only, by policy (DPDP Act; see docs/DELILAH_CV_MASTER_DOC.tex) 
FACE_MODEL_NAME = 'buffalo_l'
FACE_SCOPE = 'staff_only'
FACE_MIN_DET_SCORE = 0.55
FACE_MIN_FACE_PX = 45
FACE_SIM_THRESHOLD = 0.35
FACE_MERGE_SIM_THRESHOLD = 0.45
ENABLE_FACE_MERGE_TIER = True
ENABLE_FACE_CORROBORATION = True
ENABLE_FACE_VETO = True
FACE_VETO_MARGIN = 0.15
FACE_VETO_MAX_EDGE_SCORE = 0.8
FACE_MAX_TRIES = 6
FACE_RETRY_EVERY_S = 3.0
STAFF_GALLERY_DIR = 'staff_gallery'
STAFF_MATCH_THRESHOLD = 0.4
ENABLE_STAFF_GALLERY_SWEEP = True
STAFF_DOMINANCE_RATIO = 3.0
STAFF_MIN_VIDEO_SHARE = 0.35
STAFF_OVERRIDE_MIN_S = 60
# The SHARE rule catches a desk-anchored person who fragmented into short
# tracks: each fragment is ~100% reception but none reaches 60s alone.
#
# SET BACK TO 60 AFTER MEASURING IT. Dropping it to 15 labelled 31 of 45
# identities "staff" at a one-receptionist desk, because a GUEST standing at
# the counter also spends ~100% of a short track there. The rule cannot tell
# them apart on dwell-share alone.
#
# It also turned out to be unnecessary: staff were invisible because
# window_mask covered the desk, not because the threshold was too high. Two
# changes went in together and only one was needed — this is that lesson
# written into the file. If the mask fix alone does not produce a sensible
# staff count, the discriminator to add is SPREAD (staff return to the desk
# across the whole hour; a guest visits once), not a lower dwell.
STAFF_OVERRIDE_SHARE_MIN_S = 60

# ── staff discriminator: SPREAD ─────────────────────────────────────────────
# The rule the note above asks for. Dwell and share both describe ONE visit,
# and one visit is exactly what a customer makes — which is why 31 of 45
# identities came back staff at a one-receptionist desk. Staff RETURN: their
# desk time is scattered across the whole window; a guest's is one blob.
#
# Confirmed with the operator 2026-08-12: behaviour at this desk VARIES (some
# staff stay put, some come and go) and MORE THAN THREE people work the shift.
# So the two rules are OR-ed rather than one replacing the other, and with 3+
# staff each individual makes fewer visits than a lone receptionist would —
# hence min_visits stays low. These are starting points chosen to be tuned
# against the evidence table (describe_staff_decision), not measured values.
STAFF_MIN_VISITS = 2      # separate returns to a staff zone
STAFF_MIN_SPREAD = 0.25   # those visits must span >=25% of the analysed window
# The backstop for a window too short for spread to have evidence. Deliberately
# far above any plausible customer interaction — at 10 minutes continuous, a
# person standing at a reception counter is not a guest being served.
STAFF_SOLE_DWELL_S = 600.0

# ── phantom & implausible-detection filters ─────────────────────────────────
# ── D0: absurd-size cap ─────────────────────────────────────────────────────
# A sanity bound on the physically impossible, applied BEFORE any geometry and
# NOT relaxable by the D1 guard. The audit's giant boxes -- "P3 over the whole
# right-side background", "huge box around the doorway" -- pass the aspect
# filter (610x1070 is 1.75:1, inside MIN/MAX_BODY_ASPECT) and then D1 doubles
# its own tolerance on a bad ground fit and waves them through.
#
# Generous on purpose. It is not a person-size model; D1 is, once ground_points
# exist.
#
# 2026-08-15 -- HEIGHT RAISED 0.70 -> 0.95 ON MEASUREMENT, NOT PREFERENCE.
#   "nobody is 70% of the frame tall" was written before anything measured how
#   tall somebody actually is here, and it was wrong. On the 18:30 chunk the F2
#   scene fit, from 5,499 isolated detections IN THAT SAME RUN, gives
#       expected height = 0.807 * foot_y - 68px
#   so a person standing at the bottom of a 1080px analysis frame measures
#   804px = 0.744 of frame -- ABOVE the 0.70 cap. The bottom of this frame is
#   where the main entrance is.
#
#   The cost was not subtle. In 600 seconds D0 dropped 8,246 boxes; 8,246 of
#   them on HEIGHT and ZERO on area; and 7,999 (97%) were within 1.35x the
#   height F2 expects at their own foot position -- person-shaped, not garbage.
#   That is the "the system only sees one person when three are standing there"
#   complaint, upstream of the tracker, caused by this constant.
#
#   Why 0.95 and not "delete the cap": the two bounds do different jobs and the
#   AREA bound is the one that actually catches what this filter was written
#   for. A near-field guest is TALL and NARROW -- 1026px x ~460px = 0.23 of
#   frame area, so it survives. A frame-wide phantom over the doorway is tall
#   AND wide -- ~0.59 of area, so it is still killed. Height alone never
#   separated those two; area does.
# ── where the foot actually is inside a predicted box ───────────────────────
# 1.0 = the bottom edge, which is what every zone test, entry-line crossing and
# ground-plane fit assumed until 2026-08-17.
#
# It is wrong on this camera, and measurably so. Against 600 hand-labelled
# boxes the detector's boxes are 1.6x taller than the person in them, so the
# bottom edge lands a median 260 px BELOW the real feet. The true foot sits at
#     p25 0.501   MEDIAN 0.590   p75 0.951
# of the way down the box. See kevacv/helpers.py:anchor_point for the full
# working and why (CrowdHuman street-level h/w ~2.5 vs overhead h/w ~1.14).
#
# Shipping default stays 1.0 — identical behaviour — until an A/B against
# label_pkg/quick100 + gt.txt says the change helps. The p25-p75 spread is
# wide, so 0.59 is a better estimate rather than a correct one, and this is
# exactly the kind of change that has looked like an improvement and been a
# regression before.
FOOT_ANCHOR_FRAC = 1.0

# Which point on the body decides an entry-line crossing.
#   "bottom_center"  the feet — correct when the box ends at the feet
#   "center"         the middle — 1.5x closer to truth on THIS camera, because
#                    the box overshoots downward and the error is concentrated
#                    at the bottom edge
# Measured n=5 on the one frame whose labels were not carried forward:
#   BOTTOM_CENTER 262px from the true feet · CENTER 173px from the true centre
# Default unchanged: switching the anchor also changes where a line SHOULD be
# drawn, and the current lines were placed for feet. A/B it on one chunk.
ENTRY_LINE_ANCHOR = "bottom_center"

# ── mask regions: require MOTION rather than deleting everything ────────────
# A mask polygon is drawn around a static distractor (plant, mirror, poster),
# but it inevitably also covers floor that people walk on. On CAM.112 the
# "plant area mask" is 18.7% of the frame and its lower half is the floor
# beside the MAIN ENTRANCE. Measured against hand labels: person 5's feet were
# inside it in 100 of 100 frames, detected 29% of the time. The stage removed
# 18,501 detections and emptied 212 frames.
#
# Location cannot separate a plant from a person standing in front of it. Motion
# can, and it is the same discriminator phantoms.py already relies on: a statue
# holds size cv ~0.004 on this footage while a standing person holds ~0.080.
#
# MASK_REQUIRE_MOTION is how many PIXELS a detection at that spot must have
# moved within MASK_MOTION_WINDOW_S to survive a mask region.
#   0.0  = OFF, delete everything in the mask (previous behaviour)
#   ~40  = a person crossing a 64px cell survives; a plant does not
# Default OFF: A/B it on one chunk. Two constants have already been shipped
# here on reasoning alone and both had to be reverted.
# MASK_REQUIRE_MOTION is the SPREAD in pixels below which a spot counts as
# motionless; MASK_STATIC_S is how long it must stay that way before the mask
# deletes it. A plant sits still permanently; a guest pausing at the entrance
# does not last 30 s within 40 px.
#   0.0 = OFF, delete everything inside a mask (previous behaviour)
MASK_REQUIRE_MOTION = 0.0
MASK_STATIC_S = 30.0
MASK_MOTION_WINDOW_S = 60.0

ENABLE_ABSURD_SIZE_CAP = True
MAX_BOX_HEIGHT_FRAC = 0.70
MAX_BOX_AREA_FRAC = 0.30

ENABLE_SIZE_FILTER = True
# D1 tolerance. 2.5 -> 4.0 on measurement, 2026-08-19.
#
# D1 rejects a box as "too large to be a person standing there" by comparing
# its area to expected_h(foot_y) from the ground-plane fit. On the full-frame
# run it deleted 1,147 of 3,042 detections -- 37.7% -- and the fit it judges
# against is measurably broken:
#
#     G1 ground plane: implied camera height 5.51m -> 4.34m -> 4.20m -> 4.06m
#                      horizon at row -429, -207, -173, -142
#
# A restaurant ceiling is ~3m and a NEGATIVE horizon row means the vanishing
# line sits above the image entirely. The fit is unstable frame to frame and
# physically impossible, so expected_h is wrong, so D1's verdict is wrong --
# and it is now the largest single deleter in the pipeline.
#
# Widening the tolerance keeps D1's real job (the frame-wide phantom boxes it
# was written for measured 4.34x expected area) while stopping it from
# rejecting people on the strength of a bad room model. The proper fix is
# ground_points in the zones file via tools/calibrate_plane.py -- until that
# exists this filter cannot be trusted to be tight.
SIZE_FILTER_TOL = 4.0
MIN_BODY_ASPECT = 0.75
MAX_BODY_ASPECT = 4.0
MIN_BLUR_VARIANCE = 15.0
MIN_CROP_PX_BLUR_GATE = 40
ENABLE_STATIC_FILTER = True
STATIC_MIN_LIFE_S = 120.0
# Per-zone patience for the static filter, by ZONE ROLE. One global number has
# to be conservative enough for the most legitimately-still place in the frame,
# which means everywhere else pays for it: a plant in the corridor mints ids
# and pollutes zone events for two full minutes before anything stops it.
#
# Operator, 2026-08-12: people stand still at the reception desk and in the
# waiting area; the corridors and the entrance are a thoroughfare where nothing
# human holds position. So those two get MORE patience than the global bar and
# the through-routes get much less. The most conservative value wins where
# zones overlap — an over-long wait leaves a phantom alive, an over-short one
# deletes a person, and only one of those is recoverable.
STATIC_MIN_LIFE_BY_ROLE = {
    "staff": 240.0,     # reception desk — staff legitimately stand still
    "seating": 240.0,
    "wait": 240.0,      # waiting area — a guest can hold position for minutes
    "service": 180.0,
    "entry": 30.0,      # doorway — nobody stands in a doorway for 30 s
    "walkway": 30.0,
    "other": 45.0,
}
STATIC_CENTRE_JITTER = 0.02
STATIC_SIZE_JITTER = 0.03
# ── phantom gates, RE-MEASURED 2026-08-18 against real tracks ──────────────
# These were set from a SYNTHETIC test that claimed a statue holds size cv
# ~0.004. Real tracks from run10 say otherwise:
#
#     REAL static objects (tracks 145, 49)   size cv 0.034-0.045
#                                            centre move 1.2 px
#                                            span 129-146 s
#     REAL moving people                     size cv 0.116-0.299
#                                            centre move 50-310 px
#
# So both gates in front of the good discriminator were too strict to ever
# reach it:
#     PHANTOM_SIZE_CV 0.015  is BELOW the real statue's 0.034  -> never matches
#     PHANTOM_MIN_SPAN_S 240 is ABOVE the statue's 129-146 s   -> never matches
# which is why the statue survived a whole chunk and was only removed by the
# end-of-chunk pass, 44 minutes too late, after holding a canonical id
# throughout.
#
# CENTRE MOVEMENT is the strong signal — 1.2 px vs 50-310 px is roughly 40x
# separation, against 7.6x for size cv. PHANTOM_CENTRE_JITTER 0.02 (as a
# fraction of body height) already sits correctly between them: the statue
# measures ~0.0024, real people ~0.10. That gate was never wrong; it was just
# unreachable.
#
# So: relax the two blocking gates to sit in the MEASURED gap, and let centre
# jitter do the discriminating it was designed for.
#     size_cv  0.015 -> 0.07   (static 0.045 | 0.07 | 0.116 moving)
#     span     240 s -> 120 s  (statue tracks span 129-146 s)
PHANTOM_MIN_SPAN_S = 120.0
PHANTOM_CENTRE_JITTER = 0.02
PHANTOM_SIZE_CV = 0.07

# ── evidence-scaled patience for the LIVE suppressor ────────────────────────
# On the 18:30 chunk the live stage reported "removed nothing all chunk". Not a
# bug: the plant and the mirror sit inside reception/seating, which get 240 s of
# patience because people legitimately stand still there. On a 600 s run the
# phantom therefore survives to the end, holding a canonical id and blocking
# real people from resolving to it the whole time.
#
# PHANTOM_FAST_CV_RATIO lets a location skip the CLOCK — never the evidence —
# when its box is far more rigid than a person's can be. Measured on this
# footage:
#     statue / mirror   size cv ~0.004
#     person standing   size cv ~0.080      19x apart
# At 0.30 the fast path needs cv <= 0.30 * 0.015 = 0.0045, which the statue
# clears and a standing person misses by an order of magnitude.
#
# 0.0 = OFF (previous behaviour). Turn it on as a single A/B knob and read the
# ledger: 'live phantom suppress' should stop removing nothing, and the tracker
# should mint fewer ids. Do not ship it unmeasured.
# Re-based on the real numbers above. 0.30 * 0.07 = 0.021, which the real
# statue (0.034-0.045) still misses — so the fast path needs ~0.70 to be
# reachable at all. The old 0.30 * 0.015 = 0.0045 was 8-10x stricter than the
# thing it was hunting, which is why it measured zero effect.
PHANTOM_FAST_CV_RATIO = 0.0
PHANTOM_FAST_MIN_S = 30.0      # never suppress on less than this, whatever cv
ENABLE_CARRIED_SUPPRESS = True
CARRIED_CONTAIN = 0.7
CARRIED_HEIGHT_TOL = 0.55
CARRIED_MAX_AREA_RATIO = 0.45
CARRIED_MIN_FIT_SAMPLES = 200
CARRIED_MIN_HEAD_DROP = 0.15
# ── head recovery (symptoms 12, 11, 14) ─────────────────────────────────────
# When two people overlap the detector often returns ONE person box and TWO
# heads. The rear body is ABSENT, not low-confidence, so no threshold recovers
# it: the track dies and the person is reborn with a new id on stepping clear.
#
# _heads_without_person() has computed that signal since the beginning and the
# result was only ever counted into a log line. This flag existed, referenced
# in exactly one f-string, with NO implementation behind it — "recovery OFF
# until Phase 2 scores it" described a feature that was never written.
#
# Now implemented. It only does anything when the detector actually has a head
# class (_detector_has_head_class), so it is inert on stock COCO weights.
# Two heads inside one person box is two people — the detector merged them.
# Provable rather than heuristic: nothing else about a box says how many
# bodies are in it, but a person cannot have two heads. A merged box is worse
# than a missed one — it carries ONE id, so two people share an identity while
# they overlap; its centre sits between them, so it triggers the wrong zone;
# and when it splits, both are re-born as new ids. Symptom 9, driving 4/11/15.
ENABLE_MERGED_SPLIT = True
ENABLE_HEAD_RECOVERY = True
# ── head->body geometry: THE SOURCE OF THE TALL BOXES ───────────────────────
# These two constants manufactured the defect that was blamed on the detector
# for a week. Head recovery rebuilds 734-749 boxes per run, and it built them
# at 1/0.42 = h/w 2.38 — almost exactly the 2.51 measured in the pipeline's
# output. Meanwhile the DETECTOR itself outputs h/w 1.33, and the hand-labelled
# truth is 1.14. The detector was never the problem; this reconstruction was.
#
# Both values are standing-figure anthropometry: the figure-drawing canon of
# ~7.5 heads, and a body roughly 0.42 as wide as it is tall. Correct for a
# person seen from the SIDE. From a ceiling camera the body is foreshortened —
# you see head, shoulders and a compressed torso.
#
# MEASURED on CAM.112 from real head boxes matched to hand-labelled bodies:
#     HEAD_TO_BODY_RATIO    7.0  -> 2.52
#     HEAD_RECOVERY_ASPECT  0.42 -> 0.82
#     implied h/w           2.38 -> 1.22   (hand labels say 1.14)
#
# n=2 pairs on the one frame whose labels were not carried forward, so treat
# the exact figures as provisional — but the OLD values are provably wrong
# (they produce 2.38 where truth is 1.14-1.22) and two independent derivations
# agree on the new ones.
#
# These are CAMERA GEOMETRY, not universal constants: a wall-mounted camera
# really does see ~7 heads. The canon stays the default for unknown cameras;
# per-camera values belong in config/<cam>.yaml.
# TODO (deferred by the operator until IN/OUT reaches ~98%): THESE SHOULD NOT
# BE TYPED AT ALL. Both are measurable from the run itself, with no labels:
# every frame where a head box sits inside a person box yields one
# ratio = person_h / head_h and one aspect = person_w / person_h sample. A few
# thousand such pairs accumulate in minutes, and the median is a better answer
# than any constant — and it is per-camera automatically, which is the whole
# problem here (7.0 is right for a wall camera and wrong for a ceiling one).
#
# The same argument applies to MAX_BOX_HEIGHT_FRAC, REID_SIM_THRESHOLD,
# NEW_TRACK_CONF and the phantom gates: this file is full of numbers that the
# footage could measure. F2 (_PerspectiveModel) already does exactly this for
# expected body height, so the pattern exists in the codebase — it just was
# never applied to these.
HEAD_TO_BODY_RATIO = 7.0        # overridden per camera — see above
HEAD_RECOVERY_ASPECT = 0.42     # body width as a fraction of body height
HEAD_RECOVERY_MIN_CONF = 0.35   # ignore weak heads — a recovered box from a
                                # noisy head is a phantom with extra steps
HEAD_RECOVERY_CONF_PENALTY = 0.3  # recovered boxes are INFERRED, not observed,
                                # so they must lose to a real detection and be
                                # identifiable as inferred downstream

# ── infrared ────────────────────────────────────────────────────────────────
IR_CHROMA_THRESHOLD = 6.0
ENABLE_IR_HARD_CUT = False
IR_CUT_MIN_GAP_S = 30.0

# ── settings that existed only as inline fallbacks ──────────────────────────
# engine.py reads all of these through globals().get(NAME, <literal>). The
# NAMES were defined nowhere — not here, not in engine.py — so every run used
# the inline literal, and no config file could change any of them. They were
# invisible: you cannot grep a value you do not know exists, and the CONFIG
# AUDIT printed a tidy table that omitted all of them.
#
# EVAL_EXPORT was the same bug with teeth (no default, so the whole labelling
# export silently never ran). These have defaults, so nothing was BROKEN — but
# several of them govern the exact symptoms under investigation, and "not
# broken, just untunable and unlisted" is how a knob stays at the wrong value
# for months.
#
# Values below are EXACTLY the inline literals, so behaviour is unchanged by
# surfacing them. Now they can be tuned, and they appear in the audit.
ENABLE_PHANTOM_FILTER = True   # static/mirror phantom stage (symptoms 5,6,7)
# Suppress a static location DURING the chunk, not only at the end of it. The
# end-of-chunk pass still runs and still has the last word — but deleting a
# phantom afterwards cannot undo the id break it caused while it was alive:
# it held a canonical id, and co-visibility then blocked a real person from
# resolving to it, pushing that person onto a fresh id.
ENABLE_LIVE_PHANTOM_SUPPRESS = True
IR_DEBOUNCE_FRAMES = 24        # frames a colour<->IR flip must hold (symptom 19)
IR_SAT_THRESHOLD = 12.0        # saturation bar for the IR probe (symptom 19)
D1_GUARD_WARMUP = 2000         # detections before the drop-rate guard arms
D1_MAX_DROP_FRAC = 0.12        # guard trips above this share dropped
EMBED_BATCH = 24               # crops per Re-ID forward pass
RENDER_COAST_S = 0.5           # how long a box survives a detection gap on video
USE_FFMPEG_READER = True       # ffmpeg decode instead of cv2
FFMPEG_HWACCEL = "auto"        # 'auto' | 'none' | an explicit hwaccel
FACE_SOURCE_RECROP = False     # re-crop faces from native-res source
FACE_RECROP_FRAMES = 4         # frames per track when re-cropping
FACE_RECROP_MAX_TRACKS = 120   # cap on tracks re-cropped
DRIVE_TZ = ""                  # timezone for Drive chunk names ("" = naive)

# ── labelling / evaluation export ───────────────────────────────────────────
# The frames a human corrects to produce gt.txt. Without them there is no
# HOTA, no threshold calibration, and no training set — i.e. no way to fix
# detection or association at all, only to guess at them.
#
# engine.py has always read EVAL_EXPORT through globals().get(), and the name
# was never defined ANYWHERE — not here, not in the runner. So it read None on
# every run and the export silently never happened. A missing global is
# indistinguishable from a disabled feature, which is the same failure mode
# that made RENDER_DIRECT_H264 quietly produce 4.2 GB mp4v files.
# ── venue dataset export (symptom 8: retrain the detector) ──────────────────
# Writes frame + predicted boxes in YOLO format, for fine-tuning best.pt on
# THIS camera's angles, lighting and occlusion profile.
#
# The labels are the pipeline's OWN predictions. Trained on raw they teach the
# model to reproduce its current errors — a starting point for CORRECTION, not
# a training set. dataset.yaml says so at the top of the file.
ENABLE_DATASET_EXPORT = False
DATASET_EXPORT_EVERY_S = 5.0   # sampling interval; every frame of an hour at
                               # 8fps is 28,800 near-duplicate images, which is
                               # a slow way to train on one moment
DATASET_MIN_BOXES = 1          # skip empty frames — a label file with no boxes
                               # teaches "there is nobody here", and an hour of
                               # empty corridor would dominate the set

EVAL_EXPORT = False
# (start_s, end_s) of the slice to export. None means the WHOLE analysed span:
# engine.py used to index EVAL_WINDOW[0] unconditionally once the export was
# on, so turning the feature on with the default value crashed the run an hour
# in. Pick a window deliberately — labelling is the expensive step, and two
# representative minutes beat an unlabelled hour.
EVAL_WINDOW = None
# Hard cap on exported frames, whatever the window says. Exporting a whole
# hour wrote ~27,000 JPEGs and filled the volume at 95% of a 28-minute run.
# Nobody hand-labels 27,000 frames; a representative 2-4 minutes is the real
# working set, and this is the ceiling that makes forgetting --eval-window
# survivable rather than fatal.
EVAL_MAX_FRAMES = 2000

# ── zones & events ──────────────────────────────────────────────────────────
MIN_EVENT_S = 2.0
MIN_SEATED_S = 60
ENTRY_LINE_FLIP = True

# ── crossing quality: the three guards the industry uses ───────────────────
# 1 BUFFER BAND      frames a track must hold the far side before a crossing
#                    counts. Stops "standing on the line shifting your weight"
#                    being counted in and out repeatedly.
# 2 CONFIRMATION     a crossing counts only if the person was STILL there after
#   WAIT / U-TURN    this long. Someone who steps in, hesitates and steps back
#                    produces IN then OUT — both real crossings, both wrong.
#                    tier_a_crossings cannot catch it: it collapses events in
#                    the SAME direction, so the pair survives as one arrival
#                    plus one departure.
# Both are per-door overridable from the zone file, because a street entrance
# where guests pause and a dining threshold people stride through need opposite
# settings — entry_lines_band and entry_lines_confirm_s.
LINE_CROSS_FRAMES = 2
ENABLE_CROSSING_CONFIRM = True

# PRIOR STABILITY — the first of the six published conditions for a valid
# crossing, and the one this pipeline was missing.
#
# A crossing only counts if the track existed for this many seconds BEFORE it.
# Without it, a track BORN near a line counts as a transit though nobody
# walked anywhere. On this camera that is common: 16.3% of detections cannot
# start a track, and ids churn. The operator's frame-by-frame audit found the
# exact signature — "dining entry IN 2" with one staff member visible, and
# "staff entry OUT 6" while those staff were still in the room.
#
# 0.0 = OFF (previous behaviour). ~1.5s is roughly 12 frames at 8 fps: long
# enough that a real approach qualifies, short enough that a brisk transit
# through an interior threshold is not eaten. A/B it on one chunk.
# D4 (2026-08-19): KEEP THIS AT 0.0 FOR A STREET DOOR, and make the reason
# explicit so nobody "fixes" it upward.
#
# The rule requires a track to have existed for min_prior_s BEFORE it crosses,
# on the argument that a real transit has a history of approaching while a
# track invented ON the line does not. That is right for an INTERIOR threshold
# and wrong for the venue entrance: at a street door every genuine guest's
# track is BORN at or just outside the door, by definition. Turning this on
# globally deletes real arrivals at the one door that matters, while working
# as intended everywhere else.
#
# confirm_crossings already takes per_line for exactly this kind of split
# (see zcfg_confirm). If this is ever raised, it must be raised PER LINE --
# never globally.
CROSSING_MIN_PRIOR_S = 0.0
CROSSING_CONFIRM_S = 5.0

# ── render ──────────────────────────────────────────────────────────────────
# Pass 1 writes every analysed frame as a JPEG so pass 2 can re-read them
# instead of re-decoding the video. That trade made sense when decode was the
# expensive part. It is not, any more:
#
#   2026-08-14, tiling off: 11-18 it/s with the GPU at 14% UTILISATION. The
#   pipeline was CPU-bound, and the biggest CPU job per frame is encoding a
#   JPEG. An hour writes ~27,000 of them (8.4 GB) -- and a crashed run leaves
#   the whole 8.4 GB behind, which is how the box filled to 5.9 GB free.
#
# ACCURACY-NEUTRAL either way: the proxy is a cached copy of the same frames
# the analysis already saw, at the same resolution. Off = re-decode in pass 2.
PROXY_RENDER = False

# ── profiling ───────────────────────────────────────────────────────────────
# Per-stage wall time, printed every run and recorded into the run ledger.
# ON by default: a profiler you have to remember to enable is a profiler that is
# off on the run where you needed it. The bottleneck here was misdiagnosed three
# times (detector, tiling, then finally the proxy JPEG writer) and each wrong
# guess cost a run.
#
# Cost is one perf_counter pair per stage per frame — microseconds against a
# 20-minute run. It also decides Phase C: selective ReID is a 3-5x win if ReID
# dominates and a wasted week if it does not. See kevacv/profiling.py.
ENABLE_PROFILING = True
PROXY_JPEG_QUALITY = 72
RENDER_ONLY_OCCUPIED = False
SNAPSHOT_EVERY_S = 30
HUD_SMOOTH_S = 2.0
TRAIL_MODE = 'moving'
ENABLE_DISPLAY_RENUMBER = True

# ── annotated-video encoding ────────────────────────────────────────────────
# These three were set in notebook Cell 2e and never made it into the package,
# so engine.py's globals().get("RENDER_DIRECT_H264") was always None and every
# codebase run silently took the mp4v branch. The result: a 4.2 GB file in a
# 1990s codec that browsers and QuickTime refuse to play, where the notebook
# produced 60 MB of h264 from the same hour. Nothing failed — the flag simply
# did not exist, and a missing global reads exactly like a disabled feature.
RENDER_DIRECT_H264 = True   # pipe frames straight to libx264, no raw intermediate
RENDER_CRF = 28             # visually lossless enough for review at this size
# Review clock. The analysis samples at 15 fps; writing the file at 15 too
# makes it play at exactly real time, so an hour of footage is an hour of
# watching — which is why nobody watches it. 60 gives 4x: the same frames, the
# same numbers, an hour reviewable in 15 minutes, and you can still pause.
PLAYBACK_FPS = 60

# ── precision ───────────────────────────────────────────────────────────────
# fp16 halves the memory traffic of every GPU forward pass. On an L4 that is
# roughly 1.5-2x on the detector and on the CLIP embedder, which together are
# the GPU work in the frame loop.
#
# THE HONEST PART: this changes numerics. Detection confidences move by ~1e-3,
# so a box sitting exactly on the 0.25/0.45 hysteresis can flip; cosine
# similarities move by ~1e-3 against merge thresholds of 0.6/0.75, which is
# orders of magnitude above the noise. Expected impact is nil — but "expected"
# is not "measured", and nothing measures accuracy here yet. Turn these off to
# reproduce a pre-fp16 run exactly.
DETECTOR_HALF = True
REID_HALF = True

# ── GPU math modes ──────────────────────────────────────────────────────────
# ALLOW_TF32 targets the CLIP embedder specifically. A ViT is almost entirely
# matmul, and PyTorch ships matmul TF32 OFF by default while cuDNN TF32 is on —
# so the single most expensive thing in the frame loop was running in full
# fp32 for no reason anyone chose. TF32 keeps fp32's exponent range and drops
# mantissa bits; on similarity scores compared against 0.6/0.75 thresholds the
# difference is far below the noise floor.
#
# CUDNN_BENCHMARK autotunes convolution algorithms for a fixed input size. It
# changes which kernel runs, never what it computes. Only worth it because
# every frame here is exactly 1280x720 — with varying sizes it would re-tune
# constantly and lose.
#
# Both are OFF-switchable to reproduce an earlier run bit-for-bit.
ALLOW_TF32 = True
CUDNN_BENCHMARK = True

# ============================================================================
#  RUN CONFIG — config/cam112.yaml, actually applied
#
#  ARCHITECTURE_DECISION.md says "profile/config is data, not notebook cells",
#  and config/cam112.yaml was written to be that data. Nothing read it. run.sh
#  `sed`s it into the log, which made it look wired while the run used the
#  literals in this module — the yaml's own header claimed "run.sh reads this
#  and exports it into the environment", and that was simply not true.
#
#  A config file that is printed but not obeyed is worse than no config file:
#  every reader, human or otherwise, trusts it.
# ============================================================================

# yaml key -> the engine global it sets. Deliberately explicit: an unrecognised
# key is REPORTED, never silently dropped, because silent-drop is the failure
# this whole block exists to end.
RUN_CONFIG_KEYS = {
    "analysis.fps": "FPS_TARGET",
    "analysis.imgsz": "YOLO_IMGSZ",
    "analysis.tracker": "TRACKER_MODE",
    "analysis.max_width": "ANALYSIS_MAX_W",
    # ── scene geometry, exposed 2026-08-20 ──────────────────────────────────
    # Neither of these was reachable from yaml, and ground_plane.py carried a
    # SHADOW copy of the hfov (82.0 vs this file's 90.0) that no config change
    # could have corrected. focal_px = (frame_w/2)/tan(hfov/2), so the two
    # differed by 15% and every metre threshold inherited whichever module did
    # the arithmetic. The shadow is deleted; these make the real value tunable
    # per camera, which is the point -- 82 deg is a plausible lens and so is 90,
    # and the only way to know is to MEASURE it per venue.
    "analysis.hfov_deg": "DEFAULT_HFOV_DEG",
    "analysis.person_h_m": "PERSON_H_M",
    # ── the detection knobs, added 2026-08-13 ───────────────────────────────
    # These are the three settings an accuracy A/B actually turns, and until
    # now the only way to turn them was to EDIT THIS FILE. That makes every
    # experiment an untracked source change you can forget to revert, and it is
    # why "we tried tiling once" exists as a memory rather than as a number.
    #
    # As config they are printed by describe_run_config, land in the run ledger
    # beside the counters they moved, and get diffed against the previous run
    # automatically. The comparison stops being an argument.
    "analysis.tiled": "ENABLE_TILED_DETECT",
    "analysis.conf": "CONF_THRESHOLD",
    "analysis.dedup_nms_iou": "DEDUP_NMS_IOU",
    "analysis.fisheye_k": "FISHEYE_K",
    "analysis.max_box_height_frac": "MAX_BOX_HEIGHT_FRAC",
    "analysis.max_box_area_frac": "MAX_BOX_AREA_FRAC",
    # The merge bar for offline stitching. Promoted to a knob on 2026-08-15,
    # because it is the single number behind the "customers are labelled staff"
    # complaint and there was no way to move it except editing source.
    #
    # The chain, measured on profC: at 0.37 the stitcher made 98 merges and the
    # independent HSV check corroborated only 58. The 40 it disputed sit at
    # sims 0.46-0.56 -- e.g. ID 41 <-> ID 208 at gallery_sim 0.544 with
    # anchor_sim 0.098, which is two different people. Fusing strangers builds
    # identities whose desk visits are scattered across the whole window, and
    # scatter is exactly what apply_staff_zone_override reads as STAFF. So a
    # too-low merge bar manufactures staff: 8 of 10 identities were called
    # staff at a desk with one receptionist.
    "analysis.reid_sim_threshold": "REID_SIM_THRESHOLD",
    # POC PARITY (2026-08-18). The notebook the operator confirms produced
    # GOOD output on camera1.mp4 (notebook20627ee753 "v43") had these values;
    # the shipped pipeline had drifted from every one of them, and every drift
    # moved towards "accept more, merge more":
    #     reid_sim_threshold   0.60 -> 0.37     anchor_sim  0.75 -> 0.45
    #     reid_max_gap_s       7200 -> 900      min_crop_h    70 -> 50
    #     min_body_aspect       1.0 -> 0.75     fps           15 -> 8
    # They were unreachable from a venue profile, so an experiment meant
    # editing source -- which is how the drift went unnoticed.
    # The filter stack that has NO equivalent in the notebook the operator
    # confirms worked. Every one can only delete detections or alter identity,
    # and none was reachable from a venue profile -- so "turn it off and
    # measure" required a source edit, which is how the drift went unnoticed.
    # ENABLE_HEAD_RECOVERY in particular was explicitly OFF in the notebook
    # ("stays OFF until Phase 2 can score it against ground truth") and was
    # switched on here without ever being scored.
    # B3: the live re-identification distance gate.
    #
    # The notebook passed a FIXED LIVE_REID_MAX_DIST_PX = 140 px. The package
    # renamed it to a SPEED and divides by fps:
    #     _live_max_dist = LIVE_REID_MAX_SPEED_PX_S / eff_fps
    #         560 / 8  =  70 px      560 / 15 = 37 px
    # i.e. ~4x tighter than the notebook at the same frame rate, and raising
    # fps tightens it FURTHER. A person moving faster than the gate cannot be
    # rebound to their existing identity and gets a new id -- a direct,
    # mechanical fragmentation source. Because the constant was RENAMED rather
    # than retuned, it is invisible in any notebook-vs-package value table.
    # The notebook's 140px at 15fps is equivalent to ~2100 px/s here.
    #
    # Exposed so it can be A/B'd instead of argued about.
    "analysis.live_reid_max_speed_px_s": "LIVE_REID_MAX_SPEED_PX_S",
    "analysis.reid_ratio": "REID_RATIO",
    "analysis.live_reid_abs_floor": "LIVE_REID_ABS_FLOOR",
    "analysis.lost_track_buffer_s": "LOST_TRACK_BUFFER_S",
    # B6 note: analysis.max_width was ALREADY mapped above. 1280 (every
    # notebook) -> 1920 (package): with ENABLE_RESOLUTION_SCALING that
    # multiplies EVERY pixel gate by 1.5x, silently rescaling thresholds the
    # notebook tuned at 1280 and confounding any A/B on them.
    "analysis.enable_head_recovery": "ENABLE_HEAD_RECOVERY",
    "analysis.enable_merged_split": "ENABLE_MERGED_SPLIT",
    "analysis.enable_absurd_size_cap": "ENABLE_ABSURD_SIZE_CAP",
    "analysis.enable_size_filter": "ENABLE_SIZE_FILTER",
    "analysis.size_filter_tol": "SIZE_FILTER_TOL",
    "analysis.enable_static_filter": "ENABLE_STATIC_FILTER",
    "analysis.enable_phantom_filter": "ENABLE_PHANTOM_FILTER",
    "analysis.gmc_method": "GMC_METHOD",
    "analysis.anchor_sim_threshold": "ANCHOR_SIM_THRESHOLD",
    "analysis.reid_max_gap_s": "REID_MAX_GAP_S",
    "analysis.max_body_gap_s": "MAX_BODY_GAP_S",
    "analysis.reid_min_crop_h": "REID_MIN_CROP_H",
    "analysis.min_body_aspect": "MIN_BODY_ASPECT",
    "analysis.live_reid_memory_ttl_s": "LIVE_REID_MEMORY_TTL_S",
    "analysis.calibration_auto_apply": "CALIBRATION_AUTO_APPLY",
    "analysis.phantom_fast_cv_ratio": "PHANTOM_FAST_CV_RATIO",
    "analysis.phantom_fast_min_s": "PHANTOM_FAST_MIN_S",
    # The bar a detection must clear to START a new track (BoT-SORT's
    # new_track_thresh). Promoted to a knob 2026-08-16 because it is the
    # prime suspect for the largest unexplained loss in the pipeline:
    #
    #   tracker (boxmot)   66,872 in -> 52,258 out   14,614 dropped (21.9%)
    #                      and it EMPTIED 818 frames that had detections
    #
    # The detector emits everything >= CONF_THRESHOLD 0.35, but nothing below
    # NEW_TRACK_CONF 0.45 may begin an identity. A person detected only in the
    # 0.35-0.45 band — dim, infrared, half-occluded behind the reception
    # counter — is therefore detected on every frame and tracked on none. In
    # the 18:30 chunk a large, unoccluded woman at the desk is visibly missed
    # while a statue on the shelf is tracked.
    #
    # Hysteresis itself is right (high bar to start, low bar to continue). What
    # was never measured is whether a 0.10 gap is the correct width.
    # The floor the DETECTOR is asked for. BYTE (ECCV 2022) is explicit that
    # discarding low-score boxes "inadvertently eliminates legitimate objects
    # that are partially occluded or experiencing motion blur, leading to
    # fragmented trajectories and identity switches" -- which is this camera's
    # exact symptom (69 fragments for 18 people). Measured at the backlit door
    # on 2026-08-18: conf 0.25 yields 1 box on a frame with ~5 arriving
    # guests; conf 0.10 yields 7. The people are there; the bar deletes them.
    "analysis.detect_conf_floor": "DETECT_CONF_FLOOR",
    "analysis.new_track_conf": "NEW_TRACK_CONF",
    "analysis.keep_track_conf": "KEEP_TRACK_CONF",
    # Frames per detector call. MUST NOT EXCEED the batch a TensorRT engine was
    # compiled for — an .engine has a fixed maximum and feeding it more does not
    # fail cleanly, it raises
    #     cudaError 700: an illegal memory access was encountered
    # and takes the whole run with it. The engine built 2026-08-16 was exported
    # at batch 4 (batch 24 OOM'd the 30 GB box during ONNX conversion), so a
    # run using it needs det_batch <= 4. With models/best.pt any value is fine.
    "analysis.det_batch": "DET_BATCH",
    "analysis.foot_anchor_frac": "FOOT_ANCHOR_FRAC",
    "analysis.entry_line_anchor": "ENTRY_LINE_ANCHOR",
    # GMC / live identity memory — both were UNREACHABLE from yaml (2026-08-19).
    # ENABLE_GMC gates camera-motion compensation, which this camera (bolted to
    # a ceiling) should not need; the A/B that would prove it could not be run
    # from config at all. ENABLE_LIVE_IDENTITY_MEMORY gates the only consumer of
    # MAX_WALK_SPEED_MPS, i.e. the one metre-based gate riding on a ground plane
    # that reports a 4-5.5m camera height and a NEGATIVE horizon row. Turning it
    # off is how you stop trusting metres before the plane is calibrated.
    "analysis.reid_stationary_max_gap_s": "REID_STATIONARY_MAX_GAP_S",
    "analysis.enable_robust_ground_plane": "ENABLE_ROBUST_GROUND_PLANE",
    "analysis.robust_plane_min_samples": "ROBUST_PLANE_MIN_SAMPLES",
    "analysis.enable_clahe": "ENABLE_CLAHE",
    "analysis.enable_event_queue": "ENABLE_EVENT_QUEUE",
    "analysis.event_queue_maxsize": "EVENT_QUEUE_MAXSIZE",
    "analysis.enable_pose": "ENABLE_POSE",
    "analysis.pose_model": "POSE_MODEL",
    "analysis.pose_max_tracks": "POSE_MAX_TRACKS",
    "analysis.pose_min_track_s": "POSE_MIN_TRACK_S",
    "analysis.pose_stride": "POSE_STRIDE",
    "analysis.enable_gmc": "ENABLE_GMC",
    "analysis.enable_live_identity_memory": "ENABLE_LIVE_IDENTITY_MEMORY",
    "analysis.mask_require_motion": "MASK_REQUIRE_MOTION",
    "analysis.mask_motion_window_s": "MASK_MOTION_WINDOW_S",
    "analysis.mask_static_s": "MASK_STATIC_S",
    "analysis.proxy_render": "PROXY_RENDER",
    "analysis.profiling": "ENABLE_PROFILING",
    "analysis.crossing_confirm_s": "CROSSING_CONFIRM_S",
    "analysis.line_cross_frames": "LINE_CROSS_FRAMES",
    "analysis.crossing_min_prior_s": "CROSSING_MIN_PRIOR_S",
    "analysis.head_to_body_ratio": "HEAD_TO_BODY_RATIO",
    "analysis.head_recovery_aspect": "HEAD_RECOVERY_ASPECT",
    # Tile geometry as config, because "how expensive is tiling" is a question
    # you answer by TURNING THE KNOB, not by turning the feature off. At
    # 1920x1080: 640px = 9 calls/frame, 1280px = 3. The published guidance is
    # exactly this -- "choose slice size to reduce tile count while keeping
    # recall" (SAHI throughput practice) -- and recall only matters down to
    # TILE_TARGET_MIN_PX, which this camera never approaches.
    "analysis.tile_px": "TILE_PX",
    "analysis.tile_overlap": "TILE_OVERLAP",
}

# Keys that are real config but are consumed by the CALLER rather than by an
# engine global (paths resolved against the machine, not the venue). Listed so
# they are not reported as unknown.
RUN_CONFIG_CALLER_KEYS = {
    "analysis.detector",       # -> bind_runtime(detector=...)
    "analysis.reid_weights",   # -> engine's weight resolver
}


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def duplicate_yaml_keys(path):
    """Keys that appear TWICE under the same mapping. YAML keeps the LAST one.

    This is not hypothetical. Recorded in config/cam112_fullframe.yaml:
    "(duplicate max_box_height_frac removed: the LATER key silently won and
    reverted the fix)" and the same for reid_sim_threshold. It happened AGAIN
    on 2026-08-20: a matrix config PREPENDED `enable_gmc: false` to a file that
    already contained `enable_gmc: true`, so the later value won, the run used
    GMC anyway, and the A/B came back "identical to baseline" -- a result that
    looked like a finding and was an artefact. Ten minutes of GPU and a wrong
    conclusion, from a key nobody could see was duplicated.

    Deliberately a TEXT scan, not a yaml parse: by the time PyYAML returns a
    dict the duplicate is already gone. Indentation identifies the mapping,
    which is enough for these flat `analysis:` blocks and does not pretend to
    handle every YAML construct.
    """
    import collections as _c
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return []
    # Indentation ALONE is not the mapping: `camera: {id: A}` and
    # `analysis: {id: B}` both put `id` at indent 2 and are different keys.
    # An early version flagged that as a duplicate -- a false positive that
    # REFUSES to run, which is worse than no check at all. Track the parent
    # path with a stack.
    stack = []                      # [(indent, key), ...]
    seen = _c.Counter()
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw:
            continue
        indent = len(raw) - len(raw.lstrip())
        key = raw.strip().split(":", 1)[0].strip()
        if not key or key.startswith("-") or " " in key:
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path_key = ".".join([k for _i, k in stack] + [key])
        seen[(path_key, indent)] += 1
        stack.append((indent, key))
    dups = [(pk.split(".")[-1], n, ind)
            for (pk, ind), n in seen.items() if n > 1]
    return sorted(dups)


# Constants that are machinery, not knobs. Setting these from yaml would
# corrupt the config mechanism itself rather than tune a run.
_NOT_TUNABLE = {
    "RUN_CONFIG_KEYS", "RUN_CONFIG_CALLER_KEYS", "SCALE_PROFILE_OVERRIDES",
    "DEFAULT", "HOW", "WHY",
}


def _implicit_key(key):
    """Resolve `analysis.foo` -> `FOO` when this module defines a scalar FOO.

    WHY THIS EXISTS
        RUN_CONFIG_KEYS was a hand-maintained list, and 25 ENABLE_* flags had
        never been added to it -- ENABLE_FACE_VETO, ENABLE_REID_STITCH,
        ENABLE_GLOBAL_TRACKLET, ENABLE_CROSS_VALIDATION and the rest. The only
        way to A/B any of them was to EDIT THIS FILE, which makes every
        experiment an untracked source change you can forget to revert. That is
        the same failure this module's header already describes for values that
        print but do not apply, one step earlier: a knob you cannot turn from
        the run config is a knob whose result cannot be attributed to a config.

        Hand-adding 25 entries fixes today and drifts again at flag 26. This
        makes every scalar constant reachable by convention, so a flag is
        settable the moment it is defined.

    WHY IT IS STILL SAFE
        * Only names this module actually defines resolve; a typo'd key still
          lands in `result["unknown"]` exactly as before, so the guard against
          silently-ignored config survives.
        * Only bool/int/float/str constants resolve. Dicts, sets and dataclasses
          are machinery, and a yaml scalar cannot sensibly replace them.
        * Resolutions that did NOT come from the explicit table are reported in
          result["implicit"], so the log still shows every value that moved and
          on whose authority.
    """
    if not key.startswith("analysis."):
        return None
    name = key[len("analysis."):].upper()
    if name in _NOT_TUNABLE or name not in globals():
        return None
    return name if isinstance(globals()[name], (bool, int, float, str)) else None


def apply_run_config(path, target=None):
    """Read a run-config yaml and apply it to the engine module.

    `target` defaults to kevacv.engine — imported lazily so this module still
    imports on a machine with no torch.

    Returns {"applied": {...}, "unknown": [...], "source": str}. The caller is
    expected to LOG that: a setting that changed the run and was not printed is
    the same failure as a setting that was printed and did not change the run.

    Sections other than `analysis` (camera, chunk, measured_baseline, targets)
    are descriptive and are not applied here — measured_baseline in particular
    is a record of a past run, not an instruction for this one.
    """
    from pathlib import Path as _P
    result = {"applied": {}, "unknown": [], "duplicates": [], "source": str(path)}
    p = _P(path)
    # A duplicated key is UNRECOVERABLE by reading the parsed dict: PyYAML has
    # already thrown the first value away. Catch it in the text, and REFUSE --
    # a config whose stated value is not the value that will run is worse than
    # a missing file, because the run looks fine and the conclusion is wrong.
    if p.exists():
        _dups = duplicate_yaml_keys(p)
        if _dups:
            result["duplicates"] = [{"key": k, "times": n, "indent": i}
                                    for k, n, i in _dups]
            raise ValueError(
                "DUPLICATE KEY(S) in %s: %s. YAML keeps the LAST occurrence, "
                "so the value you can see at the top of the file is not the "
                "value that will run. This exact failure made a GMC A/B "
                "return 'identical to baseline' on 2026-08-20 -- an artefact "
                "that read like a finding. Remove the duplicate and re-run."
                % (p, ", ".join(f"{k} x{n}" for k, n, _ in _dups)))
    if not p.exists():
        result["unknown"].append(f"(file not found: {p})")
        return result
    try:
        import yaml
    except ImportError:
        result["unknown"].append("(pyyaml not installed — run config IGNORED)")
        return result
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    if target is None:
        from . import engine as target

    flat = _flatten({"analysis": cfg.get("analysis") or {}})
    for key, value in flat.items():
        name = RUN_CONFIG_KEYS.get(key) or _implicit_key(key)
        if name is None:
            if key not in RUN_CONFIG_CALLER_KEYS:
                result["unknown"].append(key)
            continue
        if key not in RUN_CONFIG_KEYS:
            result.setdefault("implicit", []).append(f"{key} -> {name}")
        before = getattr(target, name, None)
        # PROPAGATE TO EVERY MODULE THAT HOLDS A COPY.
        #
        # engine.py, analytics.py and helpers.py each do `from .config import
        # *`, which BINDS A COPY at import time. Setting the value on `engine`
        # alone left the others stale -- and the Re-ID merging that this knob
        # actually controls lives in analytics.py. Measured 2026-08-18:
        #
        #     after apply_run_config("...reid_sim_threshold: 0.60")
        #        engine    0.60      <- reported as applied
        #        analytics 0.37      <- what actually merged people
        #
        # So the run log honestly printed "0.37 -> 0.6" while the code doing
        # the work used 0.37. Every yaml knob has been half-applied. This
        # module's own header says a config that is printed but not obeyed is
        # worse than none; that was literally true here.
        _touched = []
        for _modname in ("engine", "analytics", "helpers", "pipeline",
                         "arrivals", "derive", "detect_filters", "answers",
                         # phantoms was MISSED in the first propagation fix:
                         # it reads globals().get("PHANTOM_FAST_CV_RATIO") and
                         # imports nothing from config, so the fast path was
                         # permanently off -- and a "MEASURED: NO EFFECT"
                         # conclusion was recorded on a knob that never
                         # reached the code.
                         "phantoms", "geometry_calibration",
                         # ground_plane and venue_profile were missed for the
                         # same reason phantoms was: they hold their own copies
                         # and imported nothing this list covered. ground_plane
                         # had gone further and REDEFINED DEFAULT_HFOV_DEG as
                         # 82.0 against this file's 90.0 -- a 15% focal-length
                         # split, so every metre in the run depended on which
                         # module computed it. The shadow is gone; this keeps
                         # the value reachable once it is set.
                         "ground_plane", "venue_profile"):
            try:
                _m = importlib.import_module("." + _modname, __package__)
            except Exception:
                continue
            if hasattr(_m, name):
                setattr(_m, name, value)
                _touched.append(_modname)
        setattr(target, name, value)
        globals()[name] = value          # this module too
        result["applied"][name] = {"from": before, "to": value, "key": key,
                                   "modules": _touched}
    return result


def describe_run_config(result):
    """One block a human can read in the log, showing what MOVED."""
    L = ["=" * 78,
         f"  RUN CONFIG — {result['source']}",
         "=" * 78]
    if not result["applied"]:
        L.append("  (nothing applied — module defaults are in force)")
    for name, ch in sorted(result["applied"].items()):
        arrow = "  (unchanged)" if ch["from"] == ch["to"] else ""
        L.append(f"  {name:<20} {ch['from']!r} -> {ch['to']!r}"
                 f"   [{ch['key']}]{arrow}")
    for k in result["unknown"]:
        L.append(f"  !! UNRECOGNISED, NOT APPLIED: {k}")
    # Say which values arrived by CONVENTION rather than from the explicit
    # table. Both apply identically; the distinction matters because an
    # explicit entry is a decision somebody made and an implicit one is a name
    # that happened to match, and the reader deserves to tell them apart.
    if result.get("implicit"):
        L.append(f"  ({len(result['implicit'])} resolved by name convention, "
                 f"not from RUN_CONFIG_KEYS:)")
        for m in result["implicit"]:
            L.append(f"     {m}")
    L.append("=" * 78)
    return "\n".join(L)


# PASS 2 re-decodes the entire chunk to draw identities onto it. It changes no
# number — every answer is computed in PASS 1 — so on an unattended nightly run
# it is ~20% of the wall clock spent on a file nobody opens. Leave it on while
# you are still debugging zones and identities; turn it off for production.
RENDER_VIDEO = True
# (start_s, end_s) to render only part of the chunk, or None for all of it.
# Set by --render-window. Exists because the two previous options were "spend
# 11 minutes and 8 GB rendering an hour" or "render nothing and have no way to
# answer whether a box is on a real person" — and every A/B took the second,
# which is how visual defects stayed open while the counters kept improving.
RENDER_WINDOW = None
