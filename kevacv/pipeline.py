"""pipeline.py — the root of the tree. One run, start to finish, in one place.

WHY THIS EXISTS
    Every capability had a module and nothing had a caller. preflight,
    topology, report_slim, tiled, threshold, merge_ab, graph_fusion — all
    built, all tested, none reachable. The orchestration lived in notebook
    cells, so the package was a library of parts with no machine.

    This is the machine. It is also the only place the stage timeline can be
    built, because only the caller knows where one phase ends and the next
    begins.

THE SHAPE OF A RUN

    run_camera
      ├─ preflight      can these zones answer the questions at all?
      ├─ analyse        engine.process_video: decode, detect, track, embed
      ├─ identity       merge fragments into people; optional topology veto
      ├─ answers        desk coverage, greet latency, guest count
      └─ report         SUMMARY.txt + people.csv + snaps/

    Each step is a `stage`, so the log reads root-to-leaf with timings and
    counters, and a failure names the phase it died in.

DESIGN
    Every heavy dependency is injected. `analyse_fn` defaults to
    engine.process_video but can be a stub, which is how this is tested on a
    laptop with no GPU, no weights and no video. A pipeline you cannot
    exercise without a GPU is a pipeline nobody exercises.

    Nothing here computes an analytic. It calls the modules that do, in order,
    and records what happened. If a number is wrong, it is wrong in a module
    with its own tests — not in a thousand-line orchestrator.
"""
from __future__ import annotations

import json
from pathlib import Path

from .answers import answer_set, to_report_rows
from .arrivals import arrivals_from_regions, cross_check, entry_zone_coverage
from .clock import (check_dst_span, check_frame_clock, parse_start,
                    verify_provenance)
from .config import DEFAULT as TRACKING_DEFAULTS
from .derive import enrich, report_rows
from .detect_filters import (
    drop_tracks, mirrored_pair_ids, protected_ids, rigid_track_ids,
                             static_track_ids)
from .log import banner, get_logger, stage
from .report_slim import describe_video, write_slim_outputs
from .topology import doors_from_zones, veto_pairs

_log = get_logger("pipeline")

# Everything the run produced, plus what it refused to claim.
DEBUG_SUBDIR = "debug"


def _default_analyse(video_path, zones_path, camera_id="CAM", **kw):
    """engine.process_video, imported only when actually needed — engine pulls
    torch/ultralytics/boxmot, and this module must import on a laptop.

    process_video's first positional is camera_id; passing only video/zones
    raised TypeError, which meant the codebase path had never actually reached
    the engine. The notebook always called it directly, so nothing noticed.
    """
    from .engine import process_video
    return process_video(camera_id=camera_id, video_path=str(video_path),
                         zones_path=str(zones_path), **kw)


def bind_runtime(*, base=None, output_dir=None, device=None, detector=None,
                 input_root=None, venue_profile=None, staff_gallery=None):
    """Point engine.py's runtime bindings at real paths for this machine.

    engine.py declares BASE/OUTPUT_DIR/DEVICE/DETECTOR_MODEL as None-ish
    module globals so it imports on a laptop with no GPU. They are not
    configuration — they are per-machine facts — so the caller supplies them
    once, here, instead of the module guessing.
    """
    from . import engine as E
    if base is not None:
        E.BASE = Path(base)
    if output_dir is not None:
        E.OUTPUT_DIR = Path(output_dir)
        E.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if input_root is not None:
        E.INPUT_ROOT = Path(input_root)
    if detector is not None:
        E.DETECTOR_MODEL = str(detector)
    if venue_profile is not None:
        # MERGE, do not replace. A caller supplying only {"venue": {...}} would
        # otherwise delete the "camera" section, and the code that reads it
        # swallows the KeyError as a skipped check rather than an error.
        merged = {k: dict(v) if isinstance(v, dict) else v
                  for k, v in E.VENUE_PROFILE.items()}
        for sec, val in venue_profile.items():
            if isinstance(val, dict) and isinstance(merged.get(sec), dict):
                merged[sec].update(val)
            else:
                merged[sec] = val
        E.VENUE_PROFILE = merged
    if staff_gallery is not None:
        E.STAFF_GALLERY_DIR = str(staff_gallery)
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    E.DEVICE = device
    _tune_gpu(E.DEVICE)
    _log.info(f"runtime bound — device={E.DEVICE} detector={E.DETECTOR_MODEL} "
              f"output={E.OUTPUT_DIR}")
    return {"device": E.DEVICE, "detector": E.DETECTOR_MODEL,
            "output_dir": str(E.OUTPUT_DIR), "base": str(E.BASE)}


def _tune_gpu(device):
    """Scheduling and math-mode settings. Same model, same frames, same maths.

    These are the speedups that cost nothing, as opposed to the ones that trade
    accuracy for time:

      TF32 matmul   PyTorch defaults matmul TF32 OFF and cuDNN TF32 ON, so the
                    CLIP ViT — nearly pure matmul, and the single most
                    expensive thing in the frame loop — was running full fp32
                    by default rather than by decision.
      cudnn.benchmark  autotunes conv algorithms. Valid here only because every
                    frame is exactly the same size; with varying input it
                    re-tunes forever and loses.

    Neither changes what is computed in any way that reaches a threshold: TF32
    keeps fp32's exponent range, and similarity scores are compared against
    0.6/0.75. Both are config-switchable so an earlier run can be reproduced.
    """
    from . import config as CFG
    if "cuda" not in str(device):
        return
    try:
        import torch
        if getattr(CFG, "ALLOW_TF32", False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if getattr(CFG, "CUDNN_BENCHMARK", False):
            torch.backends.cudnn.benchmark = True
        _log.info(f"   gpu math: TF32={torch.backends.cuda.matmul.allow_tf32} "
                  f"cudnn.benchmark={torch.backends.cudnn.benchmark} "
                  f"det_batch={getattr(CFG, 'DET_BATCH', '?')} "
                  f"fp16(det/reid)={getattr(CFG, 'DETECTOR_HALF', False)}/"
                  f"{getattr(CFG, 'REID_HALF', False)}")
    except Exception as exc:                      # noqa: BLE001
        _log.warning(f"(gpu tuning skipped: {exc})")


def _mask_swallows_zone(zones, zone_roles, warn_frac=0.25, error_frac=0.60):
    """A DEAD-AREA mask must not cover the zone it is meant to sit beside.

    WHY THIS IS ITS OWN CHECK
        On CAM.112 the window_mask polygon covered 100% of the reception zone.

        Be precise about what that does, because overstating it makes the
        warning worthless: _drop_masked removes a detection when its FEET land
        in a mask, while a staff zone claims a person by box CENTRE. So an
        overlap does not empty the zone — it silently thins it, dropping
        whoever happens to stand with their feet inside the masked region and
        keeping everyone else. Desk coverage came back 73.2%, not 0%.

        That partial, position-dependent loss is harder to notice than a total
        one and still biases every number computed from the zone. The masks are
        drawn by hand against a still frame, so getting one wrong is ordinary.
        The geometry is knowable before a single frame is decoded, so it is
        checked here rather than discovered an hour later.
    """
    import numpy as np
    try:
        import cv2
    except ImportError:
        return []
    masks = [z for z, r in (zone_roles or {}).items() if "mask" in (r or [])]
    if not masks or not zones:
        return []
    pts = [np.asarray(p, float) for p in zones.values() if len(p)]
    if not pts:
        return []
    allp = np.vstack(pts)
    w = int(allp[:, 0].max()) + 2
    h = int(allp[:, 1].max()) + 2

    def _raster(name):
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.asarray(zones[name], np.int32)], 1)
        return m

    out = []
    dead = np.zeros((h, w), np.uint8)
    for mname in masks:
        if mname in zones:
            dead |= _raster(mname)
    for zname, roles_ in (zone_roles or {}).items():
        if zname not in zones or "mask" in (roles_ or []):
            continue
        if not (set(roles_ or []) & {"entry", "wait", "staff", "seating",
                                     "service"}):
            continue
        m = _raster(zname)
        area = int(m.sum())
        if area <= 0:
            continue
        frac = float((m & dead).sum()) / area
        if frac >= error_frac:
            out.append(("ERROR",
                        f"dead-area mask covers {frac:.0%} of '{zname}' "
                        f"({'/'.join(roles_)}) — any detection whose FEET "
                        f"land there is dropped, so this zone loses people "
                        f"by where they stand and every metric from it is "
                        f"biased low. Redraw the mask "
                        f"({', '.join(masks)}) so it does not overlap it."))
        elif frac >= warn_frac:
            out.append(("WARN",
                        f"dead-area mask covers {frac:.0%} of '{zname}' "
                        f"({'/'.join(roles_)}) — detections there are being "
                        f"suppressed; confirm that is intended."))
    return out


def _write_tracks(run, out_dir):
    """Persist what the tracker saw: MOT 1.1 predictions + the raw frame log.

    WHY THIS IS NOT OPTIONAL PLUMBING
        run["frame_log"] lived in memory and died with the process, so three
        capabilities that were fully built could never be reached from a
        codebase run:

          eval_harness / gt_kit   score HOTA against ground truth — but there
                                  was no predictions.txt to score
          learn_zones             propose the door from the tracks — but
                                  propose_zones.py reads a frame-log FILE
          any A/B at all          comparing two runs needs both runs' output
                                  to still exist afterwards

        An hour of GPU time produced a summary and threw the evidence away.
        Every "we should A/B this" in the last day was blocked here.

    Track ids are canonicalised, so these are FINAL identities after merging —
    the same thing the report counted, not the raw tracker fragments.
    """
    import gzip
    import json as _json
    fl = run.get("frame_log") or []
    if not fl:
        return []
    # the engine returns this as canon_map; "canon" silently yielded
    # {} and exported RAW tracker fragments instead of merged people
    canon = run.get("canon_map") or {}
    dbg = Path(out_dir) / DEBUG_SUBDIR
    dbg.mkdir(parents=True, exist_ok=True)
    cam = run.get("camera_id") or "run"
    mot, made = [], []
    try:
        for idx, _t, boxes in fl:
            for tid, x1, y1, x2, y2 in boxes:
                cid = canon.get(tid, tid)
                # MOT 1.1 frames are 1-indexed; conf 1.0 because these are
                # accepted tracks, not raw detections with scores attached.
                mot.append(f"{int(idx) + 1},{cid},{float(x1):.2f},"
                           f"{float(y1):.2f},{float(x2) - float(x1):.2f},"
                           f"{float(y2) - float(y1):.2f},1,-1,-1,-1")
        pred = dbg / f"{cam}_predictions.txt"
        pred.write_text("\n".join(mot) + "\n", encoding="utf-8")
        made.append(pred)

        flog = dbg / f"{cam}_frames.json.gz"
        with gzip.open(flog, "wt", encoding="utf-8") as fh:
            _json.dump([[int(i), float(t), [list(b) for b in bx]]
                        for i, t, bx in fl], fh)
        made.append(flog)
        _log.info(f"tracks persisted: {len(mot):,} rows -> {pred.name}, "
                  f"{len(fl):,} frames -> {flog.name}  "
                  f"(score with: python -m kevacv score gt.txt {pred.name})")
    except Exception as exc:                      # noqa: BLE001
        _log.warning(f"(track export skipped: {exc})")
    # E5 (2026-08-19): the crossings file check_closure has always wanted.
    #
    # tools/check_closure.py implements the one accuracy check that needs NO
    # labels at all -- over a closed period everyone who entered also left, so
    # |IN - OUT| is our own error, measured exactly, and occupancy can never
    # go negative. It reads output/*/debug/<cam>_crossings.json, and NOTHING
    # in this repo has ever written that file. The tool has been unusable
    # since it was written, and its --from-summary fallback scrapes a per-door
    # line that report_slim does not print either.
    try:
        _cr = run.get("crossings") or []
        if _cr:
            _cp = dbg / f"{cam}_crossings.json"
            _cp.write_text(_json.dumps(
                [{"t": float(c.get("t", 0.0)),
                  "direction": c.get("direction"),
                  "line": c.get("line"),
                  "track_id": str(c.get("track_id"))} for c in _cr],
                indent=1), encoding="utf-8")
            made.append(_cp)
            _ins = sum(1 for c in _cr if c.get("direction") == "in")
            _outs = sum(1 for c in _cr if c.get("direction") == "out")
            _log.info(f"   crossings -> {_cp.name}  (IN {_ins} / OUT {_outs}; "
                      f"score with tools/check_closure.py — over a CLOSED "
                      f"period these must balance, and a 10-min chunk is not "
                      f"a closed period)")
    except Exception as _ce:
        _log.info(f"(crossings not persisted: {_ce})")
    return made


def _propose_entry(run, frame_wh):
    """Where the tracks say the door is. -> dict, or None if it cannot tell.

    Never raises: a proposal is a nicety on top of a run that already produced
    its answers, so a failure here must not cost the report.
    """
    try:
        from .learn_zones import describe, learn_entry_zones, to_zone_config
        fl = run.get("frame_log") or []
        if not fl:
            return None
        props, stats = learn_entry_zones(fl, canon=run.get("canon_map"),
                                         frame_wh=frame_wh)
        _log.warning(describe(props, stats=stats))
        if not props:
            return None
        cfg = to_zone_config(props, frame_wh=frame_wh)
        _log.warning("   paste-ready polygons for the zones file:\n"
                     f"{__import__('json').dumps(cfg, indent=2)}")
        return {"entries": props, "stats": stats, "zone_config": cfg}
    except Exception as exc:                      # noqa: BLE001
        _log.warning(f"(entry-zone proposal skipped: {exc})")
        return None


def staff_gallery_findings(gallery_dir):
    """Is face-based staff identification actually available? -> [(level, msg)]

    Checked BEFORE the GPU work, because the failure is silent and total: every
    face mechanism (live matching, the C2 sweep, the face veto) is gated on a
    non-empty gallery, so with no photos staff identity falls back entirely to
    zone dwell — which cannot separate a receptionist from a guest standing at
    the counter. Symptom 2, and half of symptom 1.
    """
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    d = Path(gallery_dir)
    if not d.exists():
        return [("ERROR", f"no staff gallery at {d} — face-based staff "
                          f"identification is entirely disabled; staff will "
                          f"be guessed from zone dwell alone")]
    imgs = [f for f in d.iterdir() if f.suffix.lower() in exts]
    if not imgs:
        return [("ERROR", f"{d} contains no face images — face-based staff "
                          f"identification is entirely disabled. Add one photo "
                          f"per staff member; the filename becomes their id.")]
    return [("INFO", f"{len(imgs)} staff photo(s) present: "
                     f"{', '.join(sorted(f.stem for f in imgs)[:8])}")]


def preflight(zones, zone_roles, events=None, roles=None, strict=False,
              staff_gallery=None):
    """Can these zones answer the questions, BEFORE we spend the GPU?

    Returns (ok, findings). `strict` raises instead of returning False, for
    callers that would rather not produce a report at all than produce one
    with a silently absent metric.
    """
    findings = []
    entry = {z for z, r in (zone_roles or {}).items() if "entry" in (r or [])}
    interior = {z for z, r in (zone_roles or {}).items()
                if set(r or []) & {"wait", "staff", "seating", "service"}}
    if not entry:
        findings.append(("ERROR", "no zone has the ENTRY role — arrivals "
                                  "cannot be counted by any method"))
    if not interior:
        findings.append(("ERROR", "no INTERIOR zone — there is nowhere to "
                                  "arrive INTO"))
    if not zones:
        findings.append(("ERROR", "no zone polygons at all"))

    findings += _mask_swallows_zone(zones, zone_roles)
    if staff_gallery is not None:
        findings += staff_gallery_findings(staff_gallery)

    # If a previous run's events are available, the strongest check is whether
    # people were ever SEEN in the entry zone — a polygon in the wrong place
    # passes every structural test and still counts nobody.
    if events:
        cov = entry_zone_coverage(events, zone_roles, roles=roles)
        if cov and cov["non_staff"] >= 5 and cov["share_with_entry"] < 0.5:
            findings.append(("ERROR",
                             f"entry zone is misplaced: only {cov['with_entry']}"
                             f" of {cov['non_staff']} non-staff people were ever"
                             f" seen inside it"))
    ok = not any(lvl == "ERROR" for lvl, _ in findings)
    if not ok and strict:
        # PreflightValidationError takes (message, errors, warnings) and its
        # __str__ renders the full block — pass the lists, don't flatten them
        # into the message, or the formatted report comes out empty.
        from .preflight import PreflightValidationError
        raise PreflightValidationError(
            "pre-flight failed: zones cannot answer the questions",
            [m for lvl, m in findings if lvl == "ERROR"],
            [m for lvl, m in findings if lvl != "ERROR"])
    return ok, findings


def resolve_identities(track_windows, embeddings, merge_fn=None, *,
                       positions=None, zones=None, zone_roles=None,
                       frame_wh=None, use_topology=True, **merge_kw):
    """Merge fragments into people, with the topology veto applied FIRST.

    Order matters. The greedy union widens a group's window every time it
    absorbs a fragment, so an early wrong merge starves later right ones — 356
    candidates were blocked that way against 69 accepted. Removing physically
    impossible pairs before the union means fewer wrong merges polluting the
    windows, which is a second-order win on top of the obvious one.
    """
    if merge_fn is None:
        from .analytics import merge_fragmented_tracks as merge_fn

    doors = doors_from_zones(zones or {}, zone_roles or {}) if use_topology else []
    vetoed = []
    if doors and positions and frame_wh:
        pairs = [{"a": a, "b": b,
                  "death_pos": positions[a][1], "birth_pos": positions[b][0],
                  "gap_s": track_windows[b][0] - track_windows[a][1]}
                 for a in track_windows for b in track_windows
                 if a != b and a in positions and b in positions
                 and track_windows[b][0] >= track_windows[a][1]]
        _, vetoed = veto_pairs(pairs, doors, frame_wh)
        blocked = {(p["a"], p["b"]) for p in vetoed}
        _log.info(f"topology veto removed {len(blocked)} impossible pair(s) "
                  f"from {len(pairs)} candidates")
        merge_kw["blocked_pairs"] = blocked
        merge_kw["role_hint"] = merge_kw.get("role_hint")
    mapping, edges, diag = merge_fn(track_windows, embeddings,
                                    positions=positions, **merge_kw)
    diag = dict(diag or {})
    diag["topology_vetoed"] = len(vetoed)
    diag["doors_used"] = len(doors)
    return mapping, edges, diag


def run_camera(video_path, zones_path, out_dir, *, camera_id="CAM",
               analyse_fn=None, config=None, zones=None, zone_roles=None,
               frame_wh=None, strict_preflight=False,
               tz_name="America/Chicago", **analyse_kw):
    """One camera, one chunk, start to finish. -> the run dict plus paths."""
    cfg = config or TRACKING_DEFAULTS
    out = Path(out_dir)
    (out / DEBUG_SUBDIR).mkdir(parents=True, exist_ok=True)
    result = {"camera_id": camera_id, "written": [], "findings": []}
    _clock_start = None        # VERIFIED wall-clock start, or None
    _clock_why = "preflight did not reach the clock checks"   # why it is None

    with stage("run", module="pipeline") as run_st:
        run_st.count("camera", camera_id)

        with stage("preflight") as st:
            from . import engine as _Eg
            ok, findings = preflight(zones or {}, zone_roles or {},
                                     strict=strict_preflight,
                                     staff_gallery=_Eg.STAFF_GALLERY_DIR)
            # The clock is the product. Check it BEFORE the GPU, because a
            # three-hour offset makes every downstream number worthless while
            # each individual step still looks correct — which is exactly how
            # 19:30 got stamped onto 16:30 footage.
            # Will it fit? Knowable before a frame is decoded, and a full
            # disk previously destroyed 28 minutes of completed analysis.
            try:
                import cv2 as _cv2
                from .capabilities import disk_findings
                _cap = _cv2.VideoCapture(str(video_path))
                _dur = ((_cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
                        / (_cap.get(_cv2.CAP_PROP_FPS) or 30.0))
                _cap.release()
                findings += disk_findings(
                    out_dir, _dur, _Eg.FPS_TARGET,
                    want_render=getattr(_Eg, "RENDER_VIDEO", True),
                    want_eval=getattr(_Eg, "EVAL_EXPORT", False),
                    want_dataset=getattr(_Eg, "ENABLE_DATASET_EXPORT", False),
                    eval_max_frames=getattr(_Eg, "EVAL_MAX_FRAMES", None),
                    frame_w=getattr(_Eg, "ANALYSIS_MAX_W", 1280))
            except Exception as _dfe:
                findings.append(("INFO", f"disk check skipped: {_dfe}"))
            # Kept in their OWN list before being merged into `findings`, so
            # the observation layer's clock gate can be judged on the CLOCK
            # evidence alone. `findings` is a merged list -- zone roles, disk
            # space, staff gallery -- and a preflight ERROR does not abort the
            # run, so gating on it would blank `ts` for a venue whose only
            # complaint is a missing zone polygon. A polygon has nothing to say
            # about whether the file we decoded is the file the clock came from.
            _clock_findings = verify_provenance(
                analyse_kw.get("selected_name"), video_path,
                analyse_kw.get("clock_source_name"))
            start = parse_start(video_path)
            _clock_findings += check_dst_span(
                start, analyse_kw.get("expect_hours", 1), tz_name)
            findings += _clock_findings
            st.count("clock_start", str(start))
            result["clock"] = {"start": str(start), "tz": tz_name}
            # The observation layer stamps wall clock onto every row, and only
            # a VERIFIED clock may do that: a provenance ERROR means the file
            # we decoded is not the file the clock came from -- exactly how
            # 19:30 got stamped onto 16:30 footage. Unverified stays None, and
            # `_clock_why` records WHICH check said so, so a NULL ts column is
            # traceable to a line in the log instead of to a mystery.
            _clock_errs = [m for l, m in _clock_findings if l == "ERROR"]
            _clock_start = None if (start is None or _clock_errs) else start
            _clock_why = ("chunk filename carries no parsable start stamp"
                          if start is None
                          else ("; ".join(_clock_errs) if _clock_errs else None))
            st.count("findings", len(findings))
            result["findings"] = findings
            for lvl, msg in findings:
                _log.log(40 if lvl == "ERROR" else 30, msg)
            if not ok:
                banner("PREFLIGHT FAILED — zones cannot answer the questions",
                       [m for _, m in findings], level="ERROR")
            st.count("ok", ok)

        with stage("analyse") as st:
            engine_kw = {k: v for k, v in analyse_kw.items()
                         if k not in ('selected_name', 'clock_source_name',
                                      'expect_hours')}
            engine_kw.setdefault("camera_id", camera_id)
            # OBSERVATION LAYER (Task 5). One row per person per frame, out
            # through a bounded non-blocking queue to append-only JSONL. The
            # engine never opens the file itself; the queue is injected the
            # same way BASE/OUTPUT_DIR are. Off by default.
            from . import engine as _Eg_obs
            _obs_eq = None
            _obs_path = None
            if getattr(_Eg_obs, "ENABLE_OBSERVATIONS", False):
                from .event_queue import EventQueue as _OEQ, jsonl_sink as _osink
                _obs_path = out / "observations.jsonl"
                _obs_eq = _OEQ(sink=_osink(str(_obs_path)),
                               maxsize=int(getattr(_Eg_obs, "OBS_QUEUE_MAXSIZE",
                                                   20000)))
                _obs_eq.start()
                # run_id must match EXACTLY what engine.process_video computes
                # for this same chunk (see its _obs_run_id comment) -- the
                # ingest tool does delete-then-insert keyed on run_id, so a
                # mismatch here would leave this run row orphaned instead of
                # matching the obs/emb rows it describes. Never recompute the
                # formula differently; mirror it.
                _chunk_tag = analyse_kw.get("chunk_tag") or ""
                _start_seconds = analyse_kw.get("start_seconds") or 0.0
                _run_id = (f"{camera_id}_{_chunk_tag}" if _chunk_tag else
                           f"{camera_id}_{Path(video_path).stem}_{int(_start_seconds)}")
                from .build_id import compute as _build_compute
                _obs_eq.put({"kind": "run", "run_id": _run_id,
                             "camera_id": camera_id,
                             "video_sha": None,
                             "fps_analysed": float(getattr(_Eg_obs, "FPS_TARGET", 0) or 0),
                             "started_at": _clock_start,
                             "frames_analysed": None,
                             "zones_cfg_hash": None,
                             "git_sha": _build_compute()})
                _Eg_obs.OBS_QUEUE = _obs_eq
                # the datetime parse_start already returned, not a re-parse
                _Eg_obs.VIDEO_START_DT = _clock_start
                if _clock_start is None:
                    _log.warning(f"observations: clock not verified — every row "
                                 f"ships ts = NULL (t_s is still exact). "
                                 f"Reason: {_clock_why}")
            try:
                run = (analyse_fn or _default_analyse)(video_path, zones_path,
                                                       **engine_kw)
            finally:
                # in a finally, so a crashing run still flushes what it had
                if _obs_eq is not None:
                    _Eg_obs.OBS_QUEUE = None
                    _Eg_obs.VIDEO_START_DT = None   # no stale start next run
                    _st = _obs_eq.close()
                    result["observations"] = {**_st, "path": str(_obs_path)}
                    _log.info(f"\U0001f4dd observations -> {_obs_path.name}: "
                              f"{_st['written']} written, {_st['dropped']} dropped, "
                              f"{_st['sink_errors']} sink error(s)")
                    if _st["lost"]:
                        _log.error(f"!! observation queue LOST {_st['lost']} row(s) — "
                                   f"the JSONL is INCOMPLETE for this run")
            st.count("events", len(run.get("events") or []))
            st.count("crossings", len(run.get("crossings") or []))
            st.count("duration_s", round(run.get("duration_s") or 0))
            # per-stage timings into the ledger, so the NEXT run diffs the
            # PROFILE automatically: "did that make it faster" answered by
            # the run itself instead of a stopwatch and a memory.
            for _k, _v in (run.get("profile_ms") or {}).items():
                st.count(_k, _v)
            # the engine returns tracks; the answers need arrivals, contacts
            # and observed windows. Derive them ONCE, here, so what was
            # measured and what was inferred is visible in one place.
            enrich(run)
            if frame_wh is None and run.get("frame_size_analysed"):
                frame_wh = tuple(run["frame_size_analysed"])
            result["run"] = run

        with stage("phantoms") as st:
            # Runs on frame_log AFTER the engine, so no engine surgery is
            # needed. Three independent geometric channels, none of which the
            # detector can influence — which is the point, because the
            # detector is the thing that got the category wrong.
            fl = run.get("frame_log") or []
            keep = protected_ids(crossings=run.get("crossings") or (),
                                 # 'face_ids' is NOT a key the engine returns -- the key is
                                 # 'staff_matched_names'. So face protection here was
                                 # ALWAYS empty, and a face-recognised receptionist
                                 # standing at the desk past the flat 120s bar could be
                                 # deleted from the count by this stage.
                                 face_ids=run.get("staff_matched_names") or ())
            # C14 (2026-08-19): the ENGINE already runs static_track_ids, and
            # runs it BETTER -- engine.py:4515 passes canon=mapping (so a
            # stitched identity is judged as one thing, not as its fragments),
            # min_life_by_id (per-zone patience: 240s at a desk where staff
            # legitimately stand still, 30s at a doorway) and the full
            # protected set. This second pass had none of that: no canon, a
            # flat 120s bar for every zone.
            #
            # Re-judging tracks the better-informed pass deliberately KEPT is
            # how a face-recognised receptionist standing at the desk gets
            # deleted from the count. So: skip it when the engine already did
            # the work, and when it does run, give it the canon map.
            _canon = run.get("canon_map") or run.get("id_merges") or {}
            if run.get("static_dropped") is not None:
                still = {}
                _log.info(f"   static filter: already applied by the engine "
                          f"({run.get('static_dropped')} dropped, with per-zone "
                          f"patience and the canon map). Not re-judging here.")
            else:
                still = static_track_ids(fl, canon=_canon, protected=keep)
            rigid = rigid_track_ids(fl, protected=keep,       # never deforms
                                    frame_wh=frame_wh)  # and never travels
            mirror = mirrored_pair_ids(fl, protected=keep)    # never drifts
            st.count("static", len(still)).count("rigid", len(rigid))
            st.count("mirrored_pairs", len(mirror))
            result["phantoms"] = {"static": still, "rigid": rigid,
                                  "mirrored": mirror}
            for tid, ev in list(rigid.items())[:3]:
                _log.warning(f"track {tid} looks rigid — {ev['why']}")
            for (a, b), ev in list(mirror.items())[:3]:
                _log.warning(f"tracks {a}/{b} may be a reflection — {ev['why']}")
            if mirror:
                # Flagged, never deleted: which of the pair is the reflection
                # needs the zone map, and removing the wrong one is worse than
                # counting both.
                result["findings"].append(
                    ("WARN", f"{len(mirror)} track pair(s) moved in lockstep — "
                             f"possible reflections, review before trusting the "
                             f"headcount"))

            # DETECTING A PHANTOM AND STILL COUNTING IT IS WORSE THAN NOT
            # DETECTING IT. Until now this stage logged "static=3 rigid=4
            # mirrored=64" and changed nothing — the furniture stayed in the
            # guest count, wearing an id, sometimes labelled staff.
            #
            # static and rigid are removed: a thing that never moves and never
            # deforms is not a person, and there is no second interpretation.
            # mirrored pairs are NOT removed, because deciding which half of
            # the pair is the reflection needs geometry this stage does not
            # have, and deleting the real person is worse than counting both.
            # They widen the uncertainty instead of silently biasing it.
            drop = set(still) | set(rigid)
            if drop:
                before = len(run.get("guest_ids") or [])
                # C12 (2026-08-19): this hand-rolled loop filtered events,
                # guests, roles, arrivals and contacts -- but NOT `crossings`.
                # `line_n` is computed from `crossings` twenty lines below, so
                # a phantom removed from every other structure still counted as
                # an ARRIVAL. detect_filters.drop_tracks exists precisely so
                # "a phantom cannot survive in one place after being removed
                # from another", and it also carries the canon map so a merged
                # phantom stops being DRAWN as well as counted.
                run["events"], run["crossings"], run["frame_log"] = drop_tracks(
                    run.get("events") or [],
                    run.get("crossings") or [],
                    run.get("frame_log") or [],
                    drop,
                    canon=run.get("canon_map") or run.get("id_merges") or {})
                run["guest_ids"] = [g for g in (run.get("guest_ids") or [])
                                    if g not in drop]
                for tid in drop:
                    (run.get("roles") or {}).pop(tid, None)
                    (run.get("arrivals_by_id") or {}).pop(tid, None)
                    (run.get("contacts") or {}).pop(tid, None)
                _log.warning(
                    f"🪑 removed {len(drop)} phantom identity(ies) from the "
                    f"count — {len(still)} never moved, {len(rigid)} never "
                    f"changed shape. Guests {before} -> "
                    f"{len(run.get('guest_ids') or [])}")
            result["phantoms"]["removed"] = sorted(map(str, drop))
            if mirror:
                pair_ids = {i for pr in mirror for i in pr}
                result["phantoms"]["mirror_ids"] = sorted(map(str, pair_ids))
                _log.warning(
                    f"🪞 {len(mirror)} lockstep pair(s) touching "
                    f"{len(pair_ids)} identity(ies) — kept, but the guest "
                    f"count's lower bound must assume some are reflections")

        with stage("identity") as st:
            st.count("roles", len(run.get("roles") or {}))

        with stage("answers") as st:
            ev, zr = run.get("events") or [], run.get("zone_roles") or {}
            region_n, _, _ = arrivals_from_regions(ev, zr,
                                                   roles=run.get("roles"))
            cov = entry_zone_coverage(ev, zr, roles=run.get("roles"))
            # C3 (2026-08-18): this counted EVERY inbound crossing on EVERY
            # door with no staff filter, while region_n above filters both.
            # So the two halves of the cross-check were never counting the same
            # thing: line_n was inflated by interior transits and by every
            # staff member walking in, which is what drove the "LINE IS BROKEN"
            # / DISAGREE alarms on a healthy camera.
            # analytics.entered_count does it correctly and was called from
            # nothing but tests.
            _crossings = run.get("crossings") or []
            from .analytics import entered_count, venue_entry_lines
            _doors = venue_entry_lines({c.get("line") for c in _crossings
                                        if c.get("line")}) or None
            line_n = entered_count(_crossings, roles=run.get("roles"),
                                   lines=_doors)
            movers = len({e["track_id"] for e in ev})

            # E1: THIRD, INDEPENDENT ARRIVAL ESTIMATE — slit-scan.
            #
            # Both estimates above depend on identity holding together. On this
            # camera it does not: track ids are REUSED (one id swept the whole
            # frame width inside a 13s window), the camera flips colour<->IR 96
            # times in 20 minutes so appearance Re-ID cannot separate people,
            # and guests arrive in groups. Scored against the hand-read windows
            # in eval/gt_entries_*.json:
            #
            #     tracking counter      0% held-out recall, 1 false positive
            #     slit-scan            100% held-out recall, 1 false positive
            #
            # The slit has NO identity in it -- it samples a line of pixels per
            # frame and counts blobs in the time-stacked image -- so id reuse,
            # IR flips and fragmentation are structurally impossible there.
            # It has been CLI-only (tools/slit_count.py) while the 0% estimator
            # is the one wired in.
            #
            # Added as a THIRD OPINION, not a replacement: it is reported and
            # cross-checked, and it does not overwrite line/region until it has
            # been scored on more than three hand-read windows.
            slit_n = None
            try:
                # The venue door, in SOURCE pixel coordinates, straight from
                # the zones file -- not the built-in default, which is 1080p
                # coordinates and would put the slit across the reception desk
                # on a 4K source.
                import json as _json
                from .analytics import venue_entry_lines as _vel
                _zc = _json.loads(Path(zones_path).read_text(encoding="utf-8"))
                _lines = dict(_zc.get("entry_lines") or {})
                if not _lines and _zc.get("entry_line"):
                    _lines["entry"] = _zc["entry_line"]
                _door = _vel(set(_lines)) if _lines else set()
                _ln = next((_lines[k] for k in _lines if k in _door), None)
                # Scale to the SOURCE resolution, not the analysed one.
                # slit_count samples raw frames straight from the video file,
                # so the line must be in SOURCE pixels. frame_wh here is the
                # ANALYSED size (1920x1080 on a 3840x2160 source) -- using it
                # would halve the coordinates and put the slit across the
                # reception desk, which is a landmine that has already fired
                # once in this project.
                _ref = _zc.get("frame_size")
                _cap2 = _cv2.VideoCapture(str(video_path))
                _srcw = int(_cap2.get(_cv2.CAP_PROP_FRAME_WIDTH))
                _srch = int(_cap2.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                _cap2.release()
                if _ln and _ref and _srcw and tuple(_ref) != (_srcw, _srch):
                    _sx, _sy = _srcw / _ref[0], _srch / _ref[1]
                    _ln = [[_ln[0][0] * _sx, _ln[0][1] * _sy],
                           [_ln[1][0] * _sx, _ln[1][1] * _sy]]
                    _log.info(f"   slit line scaled {_ref[0]}x{_ref[1]} -> "
                              f"{_srcw}x{_srch} (source)")
                if _ln:
                    from tools.slit_count import count as _slit_count
                    _A, _B = _ln[0], _ln[1]
                    _t0, _t1 = (run.get("observed_windows")
                                or [(0.0, run.get("duration_s") or 0.0)])[0]
                    _n, _evs = _slit_count(str(video_path),
                                           float(_t0), float(_t1),
                                           A=tuple(_A), B=tuple(_B))
                    slit_n = sum(1 for e in _evs if e.get("dir") == "IN")
                    result["arrivals_slit"] = {
                        "in": slit_n,
                        "out": sum(1 for e in _evs if e.get("dir") == "OUT"),
                        "events": _evs}
            except Exception as _se:
                # Never let a third opinion take the run down.
                _log.info(f"(slit arrival estimate unavailable: {_se})")

            xc = cross_check(line_n, region_n, movers=movers, coverage=cov)
            st.count("line", line_n).count("region", region_n)
            if slit_n is not None:
                st.count("slit", slit_n)
                if line_n is not None and abs(slit_n - line_n) > max(2, 0.5 * max(slit_n, line_n)):
                    _log.info(
                        f"\u26a0\ufe0f  arrival estimates disagree: line={line_n} "
                        f"region={region_n} slit={slit_n}. The slit needs no "
                        f"identity, so a large gap points at id churn in the "
                        f"other two, not at the slit.")
            st.count("trust", xc["trust"])
            result["arrivals"] = {"line": line_n, "region": region_n,
                                  "slit": slit_n,
                                  "cross_check": xc, "coverage": cov}
            if xc["trust"] == "neither":
                banner("ENTRY ZONE MISPLACED — trust neither arrival count",
                       [xc["detail"]], level="ERROR")
                result["findings"].append(("ERROR", xc["detail"]))

            # Saying "your entry zone is wrong" and stopping there leaves the
            # only fix as a human squinting at a still frame — which is how it
            # got drawn wrong in the first place. learn_zones proposes the door
            # from where tracks are BORN and DIE (Makris & Ellis 2002), and the
            # tracks for this hour are already in memory. It was built, tested,
            # and reachable only through a script that reads a frame-log file
            # this pipeline never writes, so nothing could call it.
            #
            # Proposals only. A learned zone is evidence, not authority.
            # C5 (2026-08-18): this read `if xc["trust"] != "both"`, but
            # cross_check only ever returns line/region/neither -- "both" is
            # never produced -- so the condition was ALWAYS true and this full
            # zone-learning pass over the frame log ran on every single run,
            # including ones where the two sensors agreed.
            if xc["trust"] == "neither":
                result["learned_zones"] = _propose_entry(run, frame_wh)

            # The answers themselves. Denominator is OBSERVED footage, never
            # elapsed — a blind camera must not read as an uncovered desk.
            observed = run.get("observed_windows") or [(0.0, run.get("duration_s") or 0.0)]
            zr = run.get("zone_roles") or {}
            staff_zones = [z for z, r in zr.items() if "staff" in (r or [])]
            wait_zones = [z for z, r in zr.items() if "wait" in (r or [])]
            answers = answer_set(
                events=ev, staff_zones=staff_zones, waiting_zones=wait_zones,
                observed_windows=observed, roles=run.get("roles"),
                arrivals=run.get("arrivals_by_id"), contacts=run.get("contacts"),
                unique_ids=run.get("guest_ids"),
                confidence=run.get("id_confidence"),
                # The sensor that actually supplied the times, recorded by
                # derive.arrivals_by_id — not the cross-check's opinion about
                # which one to believe. Those are different statements, and
                # labelling the count with the second while it came from the
                # first is how a report says "line" over a region number.
                arrival_source=run.get("arrival_source", "unknown"),
                findings=result["findings"],
                # Whether the INDEPENDENT arrival estimators agree. Without
                # this the report resolves a 1-vs-4 disagreement through
                # `trust=` and prints one confident number.
                agreement=_arrival_agreement(result))
            result["answers"] = to_report_rows(answers)
            result["answer_objects"] = answers
            for a in answers:
                st.count(a.key, a.display)

        with stage("report") as st:
            # WHEN was this? The header renders meta["start"]/["end"] and
            # neither was ever set, so every SUMMARY.txt opened with
            # "? -> ?  ·  0.0 h footage" — the first line a GM reads. The clock
            # is parsed and provenance-checked two stages earlier and was then
            # simply not carried here.
            _dur = float(run.get("duration_s") or 0)
            _start_raw = (result.get("clock") or {}).get("start")
            _start = _end = None
            if _start_raw:
                try:
                    from datetime import datetime, timedelta
                    _s = datetime.fromisoformat(str(_start_raw))
                    _start = _s.strftime("%H:%M")
                    _end = (_s + timedelta(seconds=_dur)).strftime("%H:%M")
                except (ValueError, TypeError):
                    _start = _end = None      # unparseable -> "?", never a guess
            meta = {"camera": camera_id,
                    "source": str(video_path),
                    "provenance_ok": bool(run.get("provenance_ok", True)),
                    "start": _start, "end": _end, "date": _start_raw,
                    "footage_h": round(_dur / 3600, 2),
                    "t_end_s": run.get("duration_s"),
                    "hota": run.get("hota"),
                    "video": describe_video(run.get("annotated_video"),
                                            clips=run.get("clips"))}
            # people/staff/anomalies were read here and never assigned, so all
            # three rendered as confident negatives ("nobody was identified",
            # "nothing flagged") on a run with 45 people. Built now, from the
            # same enriched run the answers came from.
            ppl, stf, anom = report_rows(result.get("run") or run,
                                         clock=result.get("wall_clock"))
            result["people"], result["staff"], result["anomalies"] = \
                ppl, stf, anom
            st.count("people", len(ppl)).count("staff", len(stf))
            st.count("anomalies", len(anom))
            written = write_slim_outputs(
                out, meta,
                answers=result.get("answers") or [],
                staff=stf, anomalies=anom, people=ppl,
                notes=[m for _, m in result["findings"]])
            # No snaps= : the engine's "snapshots" is a LIST of (t, image)
            # timeline thumbnails, not the {person_id: crop} this wants.
            # Nothing banks a per-person crop yet, so passing anything here
            # would either crash or write paths to files that do not exist.
            # S5: _write_tracks appends AFTER result["written"] was assigned,
            # so the predictions.txt and frame log never appeared in the run's
            # own list of outputs -- which is plausibly why nobody was scoring
            # them. Write the debug artefacts FIRST, then publish the list.
            written += _write_tracks(run, out)
            # E4: persist per-frame modality so a day/night split is possible.
            try:
                _ir = run.get("frame_ir") or {}
                if _ir:
                    _irp = out / "debug" / f"{camera_id}_modality.json"
                    _irp.parent.mkdir(parents=True, exist_ok=True)
                    _irp.write_text(json.dumps(
                        {str(k): bool(v) for k, v in _ir.items()}),
                        encoding="utf-8")
                    written.append(_irp)
                    _nir = sum(1 for v in _ir.values() if v)
                    _log.info(f"   modality: {_nir}/{len(_ir)} analysed frames "
                              f"infrared -> {_irp.name} (lets score_conditions "
                              f"split day vs night)")
            except Exception as _ie:
                _log.info(f"(modality not persisted: {_ie})")
            result["written"] = [str(p) for p in written]
            st.count("files", len(written))

            # E3: SCORE THE RUN, if ground truth for this chunk exists.
            #
            # kevacv/eval_harness.py is complete and validated against planted
            # errors, gt.txt sits in the repo, and run["hota"] was never
            # assigned by anything -- so report_slim's quality line rendered
            # blank on every run and every accuracy claim in this project was
            # self-reported. _write_tracks even prints the command a human
            # should type. Nobody typed it.
            try:
                _gt = next((g for g in (Path("gt.txt"),
                                        out / "gt.txt",
                                        Path(zones_path).parent / "gt.txt")
                            if g.exists()), None)
                _pred = next((Path(w) for w in written
                              if str(w).endswith("_predictions.txt")), None)
                if _gt and _pred:
                    from .eval_harness import load_mot, score_sequence
                    _g, _p = load_mot(str(_gt)), load_mot(str(_pred))
                    if _g and _p:
                        _sc = score_sequence(_g, _p)
                        result["hota"] = {k: _sc.get(k) for k in
                                          ("HOTA", "DetA", "AssA", "IDF1",
                                           "MOTA", "precision", "recall")}
                        _log.info(
                            f"\U0001f4cf scored against {_gt.name}: "
                            f"HOTA {_sc.get('HOTA', 0):.4f} "
                            f"DetA {_sc.get('DetA', 0):.4f} "
                            f"AssA {_sc.get('AssA', 0):.4f} "
                            f"recall {_sc.get('recall', 0):.4f}")
                        _log.info(
                            f"   NOTE: gt.txt is tied to the frame sampling of "
                            f"the run it came from. If this run used a "
                            f"different fps the frame numbers do not align -- "
                            f"use tools/score_by_time.py instead.")
            except Exception as _ee:
                _log.info(f"(run not scored: {_ee})")

        # RUN SCORECARD — the run states its own case.
        #
        # Every number needed to accept or reject this run, in one block, plus
        # the same content as JSON so two runs can be diffed mechanically.
        # Before this, that evidence lived in three tools somebody ran BY HAND
        # afterwards (the funnel in the log, tools/track_health.py,
        # tools/score_line_entries.py) -- which made the person running them
        # the instrument, and their summary the thing you had to trust.
        #
        # Fails open and LOUDLY: a scorecard that cannot be built must never
        # take the run down, but it must not disappear quietly either, because
        # a missing scorecard looks exactly like a clean one.
        try:
            from . import scorecard as _SC
            _card = _build_scorecard(result, out, camera_id, video_path,
                                     zones_path)
            _log.info("\n" + _SC.render(_card))
            _SC.write(_card, str(out), camera_id)
        except Exception as _sce:
            import traceback as _tb
            _log.error(f"!! RUN SCORECARD FAILED: {_sce}. This run produced no "
                       f"self-assessment -- judge it from the funnel and the "
                       f"crossings file by hand, and do not read the absence "
                       f"of a scorecard as a pass.")
            _log.error(_tb.format_exc())

        run_st.count("outputs", len(result["written"]))
    return result


def _arrival_agreement(result):
    """Cross-estimator agreement for this run, or None if unavailable."""
    try:
        from .confidence import arrival_tier
        arr = result.get("arrivals") or {}
        if arr.get("line") is None and arr.get("region") is None:
            return None
        return arrival_tier(arr.get("line"), arr.get("region"),
                            (result.get("arrivals_slit") or {}).get("count"))
    except Exception:
        return None


def _build_scorecard(result, out, camera_id, video_path, zones_path):
    """Assemble the scorecard from what the run already produced."""
    import gzip as _gz
    import json as _json
    from . import scorecard as _SC
    from . import engine as _Eg

    dbg = Path(out) / "debug"
    tracks, t_first, t_last = {}, None, None
    fp = dbg / f"{camera_id}_frames.json.gz"
    if fp.exists():
        acc = {}
        for _fi, _t, _dets in _json.load(_gz.open(fp, "rt")):
            for _tid, _x1, _y1, _x2, _y2 in _dets:
                acc.setdefault(_tid, []).append(
                    (float(_t), float(_x1), float(_y1), float(_x2), float(_y2)))
        for k in acc:
            acc[k].sort()
        tracks = acc
        _ts = [p[0][0] for p in tracks.values()] + [p[-1][0] for p in tracks.values()]
        t_first, t_last = (min(_ts), max(_ts)) if _ts else (None, None)

    crossings = []
    cp = dbg / f"{camera_id}_crossings.json"
    if cp.exists():
        _d = _json.load(open(cp))
        crossings = _d if isinstance(_d, list) else _d.get("crossings", _d)

    run = result.get("run") or {}

    # The engine already returns both of these; nothing was reading them.
    funnel_pct = {}
    for st in (run.get("detection_funnel") or {}).get("stages", []):
        funnel_pct[st["stage"]] = 100.0 * float(st.get("share_of_raw") or 0.0)

    plane = {}
    _pd = run.get("ground_plane")
    if _pd:
        import re as _re
        _h = _re.search(r"camera height ([\d.]+) m", str(_pd))
        _r = _re.search(r"horizon at row (-?\d+)", str(_pd))
        plane = {"ok": True, "describe": str(_pd),
                 "mode": run.get("ground_mode"),
                 "camera_h_m": float(_h.group(1)) if _h else None,
                 "horizon_row": int(_r.group(1)) if _r else None}

    arr = result.get("arrivals") or {}
    counts = {k: arr[k] for k in ("line", "region", "trust") if k in arr}
    if result.get("people") is not None:
        try:
            counts["guests"] = len(result["people"])
        except TypeError:
            pass

    aw = float(getattr(_Eg, "ANALYSIS_MAX_W", 1920) or 1920)
    imgsz = float(getattr(_Eg, "YOLO_IMGSZ", 1280) or 1280)
    card = {"run": str(out).rstrip("/").split("/")[-1],
            "build": {"build_id": (run.get("build_id")
                                   or result.get("build_id")
                                   or getattr(_Eg, "BUILD_ID", None) or "?"),
                      "config": (result.get("config")
                                 or getattr(_Eg, "RUN_CONFIG_PATH", None) or "?"),
                      "video": str(video_path),
                      "zones": str(zones_path),
                      "seconds": run.get("duration_s", "?"),
                      "changed": result.get("config_changed") or {}},
            "funnel": funnel_pct,
            "ground_plane": plane,
            "counts": counts}
    # Zone polygons, scaled into the pixel space the TRACKS live in, so the
    # scorecard can tell a person leaving through the dining doorway from the
    # tracker dropping someone in open floor. Without this the fragmentation
    # verdict counts doorways and furniture as failures and reads ~5x high.
    zpolys = None
    try:
        _z = _json.load(open(zones_path))
        _fw, _fh = _z.get("frame_size", [3840, 2160])[:2]
        _ah = aw * float(_fh) / float(_fw)
        _sx, _sy = aw / float(_fw), _ah / float(_fh)
        zpolys = {nm: [(px * _sx, py * _sy) for px, py in pts]
                  for nm, pts in (_z.get("polygons") or {}).items()}
    except Exception:
        zpolys = None

    if tracks:
        card["tracks"] = _SC.track_stats(tracks, aw, aw * 9.0 / 16.0,
                                         zone_x=1500.0 * (aw / 1920.0),
                                         t_first=t_first, t_last=t_last,
                                         zones=zpolys)
        card["pixel_height"] = _SC.pixel_height(tracks, aw, 3840.0, imgsz)
        # P3: fragments-per-person, lost-buffer recoveries and swap pressure.
        # None of these need labels, and they are the numbers a tracking change
        # has to move -- until now "did that help the tracker?" had no answer.
        card["track_quality"] = _SC.track_quality(
            tracks, fps=float(getattr(_Eg, "FPS_TARGET", 8) or 8))
    # Tier the headline arrival count by how far the INDEPENDENT estimators
    # agree. Additive: the count itself is untouched, but a run where line and
    # region disagree 1-vs-4 can no longer be read as a clean answer.
    from .confidence import arrival_tier as _tier
    _plane_ok = bool(plane) and not plane.get("failed_scale")
    card["arrival_confidence"] = _tier(
        arr.get("line"), arr.get("region"),
        (result.get("arrivals_slit") or {}).get("count"),
        plane_ok=_plane_ok)

    # OCCUPANCY RECONCILIATION (audit.txt RED FLAG #3). The overlay published
    # "people in frame = 5, entered = 8, exited = 6" and never noticed that
    # 1 + 8 - 6 = 3. analytics.occupancy_timeline had existed the whole time
    # with zero callers -- the observed half of the check was already written
    # and simply never compared against the doors.
    try:
        from .analytics import (reconcile_occupancy as _rec,
                                describe_reconciliation as _drec)
        # `result` is the OUTER dict; the analyse payload lives under "run"
        # (bound at the top of this function). Reading events off `result`
        # silently returned [] and the whole check skipped without a word --
        # which is exactly the class of failure this check exists to catch,
        # so the skip is now logged instead of swallowed.
        _evts = run.get("events") or []
        _dur = float(run.get("duration_s") or 0.0)
        if _evts and _dur > 0:
            card["occupancy_reconciliation"] = _rec(_evts, crossings, _dur,
                                                    line_name="entry line")
            _log.info("\n" + _drec(card["occupancy_reconciliation"]))
        else:
            _log.warning(f"!! occupancy reconciliation SKIPPED: "
                         f"{len(_evts)} events, duration {_dur}s")
    except Exception as _roe:
        _log.error(f"!! occupancy reconciliation failed: {_roe}")

    # Person -> Visit -> Event. "unique people" and "visits" were the same
    # field, so a guest who stepped out for a cigarette and came back was
    # either two people or one visit, depending on which estimator won.
    try:
        from .visits import build_visits as _bv
        card["visits"] = _bv(crossings, line_name="entry line")
    except Exception as _ve:
        _log.error(f"!! visit model failed: {_ve}")

    card["entry_score"] = _SC.score_windows(crossings, video=str(video_path))
    # observation layer stats, recorded by run_camera when the flag is on
    if result.get("observations"):
        card["observations"] = result["observations"]
    # EVENT QUEUE (P7). Stream the run's events to append-only JSONL through a
    # bounded, non-blocking queue. A module nothing imports is how this
    # codebase accumulated four built-and-unwired features, so this is wired
    # even though it defaults off.
    if getattr(_Eg, "ENABLE_EVENT_QUEUE", False):
        try:
            from .event_queue import EventQueue as _EQ, jsonl_sink as _sink
            _qpath = Path(out) / f"{camera_id}_events.jsonl"
            _eq = _EQ(sink=_sink(str(_qpath)),
                      maxsize=int(getattr(_Eg, "EVENT_QUEUE_MAXSIZE", 10000)))
            _eq.start()
            for _c in crossings:
                _eq.put({"kind": "crossing", **_c})
            for _v in (card.get("visits") or {}).get("visits", []):
                _eq.put({"kind": "visit", **_v})
            _qs = _eq.close()
            card["event_queue"] = _qs
            _log.info(f"\U0001f4e4 event queue -> {_qpath.name}: "
                      f"{_qs['written']} written, {_qs['dropped']} dropped, "
                      f"{_qs['sink_errors']} sink error(s)")
            if _qs["lost"]:
                _log.error(f"!! event queue LOST {_qs['lost']} event(s) — the "
                           f"JSONL is incomplete for this run")
        except Exception as _qe:
            _log.error(f"!! event queue failed: {_qe}")

    # PROVENANCE. Stamp what produced these numbers onto the OUTPUT, not just
    # into a log on a box. Three failures today would have been a five-second
    # lookup with this: a byte-identical funnel after a half-applied detector
    # change, a GMC A/B where a duplicated key meant the flag never ran, and a
    # held-out score measured against ground truth from a different hour.
    try:
        from .provenance import build_stamp as _stamp
        card["provenance"] = _stamp(
            build_id=card["build"].get("build_id"),
            config_path=card["build"].get("config"),
            zones_path=str(zones_path), video=str(video_path),
            detector=str(getattr(_Eg, "DETECTOR_MODEL", "?")),
            tracker=str(getattr(_Eg, "TRACKER_MODE", "?")),
            reid_weights=str(getattr(_Eg, "CLIP_REID_WEIGHTS", "?")),
            changed=card["build"].get("changed"),
            ground_plane=plane,
            fps=float(getattr(_Eg, "FPS_TARGET", 8) or 8),
            analysis_w=aw, imgsz=imgsz)
        import json as _j
        with open(Path(out) / f"{camera_id}_provenance.json", "w") as _fh:
            _j.dump(card["provenance"], _fh, indent=1, default=str)
    except Exception as _pe:
        _log.error(f"!! provenance stamp failed: {_pe}")

    # Turn UNCERTAIN into an ACTION. Until now the tier was printed and the
    # ambiguous cases still went into the same total as the confirmed ones.
    try:
        from .review import build_queue as _bq
        card["review_queue"] = _bq(
            arrival_confidence=card.get("arrival_confidence"),
            visits=card.get("visits"),
            camera=camera_id)
        import json as _j
        with open(Path(out) / f"{camera_id}_review_queue.json", "w") as _fh:
            _j.dump(card["review_queue"], _fh, indent=1, default=str)
    except Exception as _rqe:
        _log.error(f"!! review queue failed: {_rqe}")

    # DECISION LEDGER. Built in an earlier session to answer "a number with no
    # provenance" and then imported by NOTHING -- the same orphan pattern as
    # fit_robust_ground_plane and validate_entry_line. provenance.py now covers
    # WHAT PRODUCED the run; this covers WHAT THE RUN DECIDED, per stage. Both
    # are needed and neither replaces the other.
    try:
        from .decision_log import Ledger as _Ledger
        _led = _Ledger(run_id=str(out).rstrip("/").split("/")[-1])
        with _led.stage("scorecard", module="kevacv.scorecard",
                        does="assemble the run's own evidence"):
            for _st, _pct in (card.get("funnel") or {}).items():
                _led.flow(_st, note=f"{_pct:.1f}% of raw dropped here")
            for _v in card.get("verdicts") or []:
                if _v.get("state") in ("FAIL", "NO-TRUTH"):
                    _led.warn(_v["check"], why_it_matters=_v["detail"])
        card["decision_log"] = _led.summary()
        _led.write(str(out))
    except Exception as _dle:
        _log.error(f"!! decision ledger failed: {_dle}")

    # POSE / ACTIVITY (P9). Off by default and correctly so -- it answers "what
    # is this body doing", not "who is this", so it cannot move a count this
    # pipeline is wrong about. Wired so it is a FLAG rather than an orphan.
    if getattr(_Eg, "ENABLE_POSE", False) and tracks:
        try:
            from .pose import select_tracks as _sel
            _chosen = _sel(tracks,
                           max_tracks=int(getattr(_Eg, "POSE_MAX_TRACKS", 8)),
                           min_seconds=float(getattr(_Eg, "POSE_MIN_TRACK_S", 2.0)))
            card["pose"] = {"selected_tracks": _chosen,
                            "model": str(getattr(_Eg, "POSE_MODEL", "?")),
                            "stride": int(getattr(_Eg, "POSE_STRIDE", 8)),
                            "note": "tracks selected; keypoint inference is a "
                                    "second pass and is not run inline"}
            _log.info(f"\U0001f9cd pose layer: {len(_chosen)} track(s) selected "
                      f"of {len(tracks)} (budget "
                      f"{getattr(_Eg, 'POSE_MAX_TRACKS', 8)})")
        except Exception as _pe:
            _log.error(f"!! pose layer failed: {_pe}")

    card["verdicts"] = _SC.verdicts(card)
    return card
