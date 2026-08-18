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

from pathlib import Path

from .answers import answer_set, to_report_rows
from .arrivals import arrivals_from_regions, cross_check, entry_zone_coverage
from .clock import (check_dst_span, check_frame_clock, parse_start,
                    verify_provenance)
from .config import DEFAULT as TRACKING_DEFAULTS
from .derive import enrich, report_rows
from .detect_filters import (mirrored_pair_ids, protected_ids, rigid_track_ids,
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
            findings += verify_provenance(analyse_kw.get("selected_name"),
                                          video_path,
                                          analyse_kw.get("clock_source_name"))
            start = parse_start(video_path)
            findings += check_dst_span(start, analyse_kw.get("expect_hours", 1),
                                       tz_name)
            st.count("clock_start", str(start))
            result["clock"] = {"start": str(start), "tz": tz_name}
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
            run = (analyse_fn or _default_analyse)(video_path, zones_path,
                                                   **engine_kw)
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
                                 face_ids=run.get("face_ids") or ())
            still = static_track_ids(fl, protected=keep)      # never moves
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
                run["events"] = [e for e in (run.get("events") or [])
                                 if e.get("track_id") not in drop]
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
            line_n = len({c["track_id"] for c in (run.get("crossings") or [])
                          if c.get("direction") == "in"})
            movers = len({e["track_id"] for e in ev})
            xc = cross_check(line_n, region_n, movers=movers, coverage=cov)
            st.count("line", line_n).count("region", region_n)
            st.count("trust", xc["trust"])
            result["arrivals"] = {"line": line_n, "region": region_n,
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
            if xc["trust"] != "both":
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
                findings=result["findings"])
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
            result["written"] = [str(p) for p in written]
            written += _write_tracks(run, out)
            st.count("files", len(written))

        run_st.count("outputs", len(result["written"]))
    return result
