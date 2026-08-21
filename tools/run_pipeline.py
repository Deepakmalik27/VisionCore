"""run_pipeline.py — run the CODEBASE. No notebook, no papermill.

    python tools/run_pipeline.py --video data/chunk.mp4 --zones zones/CAM.112_zone.json
    python tools/run_pipeline.py --video ... --zones ... --dry     # 20s smoke test
    python tools/run_pipeline.py --check                          # environment only

WHY A SEPARATE ENTRY POINT
    run.sh executes notebooks/pipeline.ipynb under papermill. That path works,
    but it runs the NOTEBOOK's copy of the logic, so none of the codebase guards
    fire — no phantom stage, no provenance check, no stage timeline, no
    SUMMARY.txt.

    This runs kevacv.pipeline.run_camera directly. Same engine, but wrapped in
    everything the package has learned:

        preflight   zones · PROVENANCE MISMATCH · DST span
        analyse     engine.process_video
        phantoms    static (never moves) · rigid (never deforms AND never
                    travels) · mirrored (never drifts apart)
        answers     line vs region cross-check · entry-zone coverage
        report      SUMMARY.txt · people.csv · snaps/ · debug/

    Every phase is timed and counted in one root-to-leaf log.

FIRST RUN
    Use --dry. It caps analysis at 20 seconds of footage, which proves the
    weights load, CUDA is real, the zones scale and the writer works — in about
    a minute instead of an hour. A full run that dies at minute 50 on a missing
    model file teaches nothing that --dry would not have taught in one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kevacv  # noqa: E402
from kevacv import log as klog  # noqa: E402
from kevacv.pipeline import bind_runtime, run_camera  # noqa: E402


def check_env():
    """What is installed, what the GPU is, and whether the two copies agree."""
    import importlib
    import importlib.util as u
    print("=" * 70)
    print("  ENVIRONMENT")
    print("=" * 70)
    ok = True
    for m in ("numpy", "scipy", "cv2", "torch", "supervision", "ultralytics",
              "boxmot", "insightface", "onnxruntime"):
        if u.find_spec(m) is None:
            print(f"  MISSING  {m}")
            ok = False
            continue
        try:
            mod = importlib.import_module(m)
            print(f"  ok       {m:<14} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"  BROKEN   {m:<14} {type(e).__name__}: {e}")
            ok = False
    try:
        import torch
        print(f"  cuda     available={torch.cuda.is_available()} "
              f"devices={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"           [{i}] {torch.cuda.get_device_name(i)}")
        if not torch.cuda.is_available():
            print("  !!       no GPU — a full night on CPU is not viable")
    except Exception as e:
        print(f"  cuda     unavailable ({e})")
        ok = False
    print(f"  kevacv   {kevacv.__version__}  ({len(kevacv.__all__)} public names)")
    return ok


def _choose_chunk(folder_id):
    """A numbered menu, so nobody has to paste a Drive id or guess a filter.

    Returns the chunk's own start-time text, which is what `select()` matches
    against — never the whole filename, because the range in the name means a
    bare substring also matches the PREVIOUS chunk's end time.
    """
    from kevacv.clock import parse_start
    from kevacv.drive import VIDEO_EXT, list_folder, start_part
    vids = [n for n in list_folder(folder_id) if n.lower().endswith(VIDEO_EXT)]
    vids.sort(key=lambda n: (parse_start(n) or "", n))
    if not vids:
        print("  no video chunks in that folder")
        return None
    print()
    print("=" * 70)
    print("  WHICH CHUNK?")
    print("=" * 70)
    for i, n in enumerate(vids, 1):
        st = parse_start(n)
        end = ""
        try:
            import re
            m = re.findall(r"(\d{1,2}\.\d{2}\.\d{2}[ap]m)", Path(n).name, re.I)
            if len(m) > 1:
                end = f" -> {m[-1]}"
        except Exception:
            pass
        print(f"  [{i}]  {st}{end}")
    print()
    raw = input("  number (or blank to cancel): ").strip()
    if not raw:
        print("  cancelled")
        return None
    try:
        chosen = vids[int(raw) - 1]
    except (ValueError, IndexError):
        print(f"  {raw!r} is not one of 1-{len(vids)}")
        return None
    import re
    m = re.findall(r"\d{1,2}\.\d{2}\.\d{2}[ap]m", start_part(chosen), re.I)
    token = m[-1] if m else Path(chosen).stem
    print(f"  -> {Path(chosen).name}")
    return token


def _public_host():
    """This instance's public DNS, so the printed scp line is copy-pasteable.

    EC2 metadata needs a token (IMDSv2); if anything about that fails we say
    so rather than printing a command with a placeholder someone will paste
    verbatim — which is exactly what happened with the Drive file ids.
    """
    try:
        import urllib.request as u
        req = u.Request("http://169.254.169.254/latest/api/token", method="PUT",
                        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        tok = u.urlopen(req, timeout=1).read().decode()
        req = u.Request("http://169.254.169.254/latest/meta-data/public-hostname",
                        headers={"X-aws-ec2-metadata-token": tok})
        return u.urlopen(req, timeout=1).read().decode()
    except Exception:
        return "<this-instance-address>"


def _package(out_dir, camera_id):
    """One zip of everything the run produced, ready to pull down."""
    import shutil
    import time
    out = Path(out_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(camera_id))
    # append, never with_suffix: camera ids carry dots (V73)
    base = out.parent / f"keva_{safe}_{stamp}"
    made = shutil.make_archive(str(base), "zip", root_dir=str(out))
    return Path(made)


def _run_all(a, p, argv):
    """Every chunk in the folder, oldest first. -> exit code.

    WHY EACH CHUNK GETS ITS OWN OUTPUT FOLDER
        SUMMARY.txt, people.csv and the annotated video are written to fixed
        names. Looping into one directory would leave only the last chunk's
        report standing, with nothing to say the earlier ones had been
        overwritten — a night of GPU time reduced to its final hour.

    A failing chunk does not stop the night. One corrupt file at 02:00 must not
    cost the six hours after it; the failure is recorded and the loop moves on,
    because an unattended run nobody is watching has to degrade, not halt.

    NOTE: chunks are independent here. An identity that leaves at 18:29 and
    returns at 18:31 is two people across the boundary — kevacv.seams exists to
    stitch that and has no caller yet, so cross-chunk totals are upper bounds.
    """
    import sys as _sys
    from pathlib import Path as _P

    from kevacv.clock import parse_start
    from kevacv.drive import VIDEO_EXT, list_folder, start_part

    if not a.drive_folder:
        p.error("--all needs --drive-folder")
    names = [n for n in list_folder(a.drive_folder)
             if str(n).lower().endswith(VIDEO_EXT)]
    if not names:
        print("  no video chunks in that folder")
        return 2
    names.sort(key=lambda n: (str(parse_start(n) or ""), str(n)))

    base = argv if argv is not None else _sys.argv[1:]
    keep, skip_next = [], False
    for tok in base:
        if skip_next:
            skip_next = False
            continue
        if tok == "--all":
            continue
        if tok in ("--chunk", "--out"):
            skip_next = True
            continue
        if tok.startswith("--chunk=") or tok.startswith("--out="):
            continue
        keep.append(tok)

    print("=" * 70)
    print(f"  ALL CHUNKS — {len(names)} to process, oldest first")
    print("=" * 70)
    for i, n in enumerate(names, 1):
        print(f"  [{i}] {parse_start(n)}  {_P(n).name}")
    print()

    results = []
    for i, n in enumerate(names, 1):
        stem = _P(n).stem
        sub = _P(a.out) / "".join(c if c.isalnum() or c in "-_." else "_"
                                  for c in stem)
        print("\n" + "=" * 70)
        print(f"  CHUNK {i}/{len(names)}  {parse_start(n)}  -> {sub}")
        print("=" * 70)
        argv_i = keep + ["--chunk", start_part(_P(n).name), "--out", str(sub)]
        try:
            rc = main(argv_i)
        except Exception as exc:                  # noqa: BLE001
            print(f"  CHUNK FAILED: {type(exc).__name__}: {exc}")
            rc = 1
        results.append((parse_start(n), _P(n).name, rc))

    print("\n" + "=" * 70)
    print("  NIGHT COMPLETE")
    print("=" * 70)
    for st, nm, rc in results:
        print(f"  {'ok  ' if rc == 0 else 'FAIL'}  {st}  {nm}")
    bad = sum(1 for _s, _n, rc in results if rc != 0)
    print(f"\n  {len(results) - bad} of {len(results)} chunks succeeded")
    if bad:
        print("  Failures are per-chunk; the rest of the night still ran.")
    return 1 if bad == len(results) else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", help="a local file (skips Drive entirely)")
    p.add_argument("--drive-folder", help="Drive folder id to pick a chunk from")
    p.add_argument("--chunk", help="e.g. 7.30.00pm — matched against each "
                                   "chunk's OWN start time")
    p.add_argument("--list", action="store_true",
                   help="show the chunks in the Drive folder and exit")
    p.add_argument("--zones")
    p.add_argument("--tiled", action="store_true",
                   help="SAHI tiled detection: run the detector on overlapping "
                        "tiles so far-away people get full effective "
                        "resolution. Costs more GPU per frame; see the cost "
                        "line printed at startup")
    p.add_argument("--dataset-export", action="store_true",
                   help="write frame+box pairs in YOLO format for retraining "
                        "best.pt on this venue (symptom 8). Pseudo-labels: "
                        "correct them before training")
    p.add_argument("--eval-export", action="store_true",
                   help="write the frames a human corrects into gt.txt — the "
                        "only thing that unblocks HOTA scoring, re-ID "
                        "threshold calibration and detector retraining")
    p.add_argument("--eval-window", nargs=2, type=float, metavar=("START", "END"),
                   help="seconds to export (default: the whole chunk). "
                        "Labelling is the expensive step; two representative "
                        "minutes beat an unlabelled hour")
    p.add_argument("--config", default=str(ROOT / "config" / "cam112.yaml"),
                   help="run settings as data; its analysis block overrides "
                        "the kevacv.config defaults")
    p.add_argument("--out", default=str(ROOT / "output"))
    p.add_argument("--camera-id", default="CAM.112")
    # DEFAULT CHANGED best.pt -> yolo11x.pt, 2026-08-19.
    # MEASURED 2026-08-19 against gt.txt (100 labelled frames, 600 boxes, 6 people),
    # single variable, same config/fps/zones, head recovery off in both:
    #
    #                        best.pt (CrowdHuman)   yolo11x (stock)   change
    #   HOTA                      0.2590               0.4762          +84%
    #   DetA                      0.1871               0.3976         +113%
    #   AssA                      0.3680               0.5807          +58%
    #   precision                 0.1954               0.6440         +230%
    #   recall                    0.0850               0.4883         +474%
    #   TP / FP                   51 / 210             293 / 162
    #
    # Strictly better on every metric -- recall up AND false positives down.
    # HOTA 0.4762 clears SUCCESS_CRITERIA.md hota_floor 0.40 for the first time.
    #
    # WHY the fine-tune lost: trained on CrowdHuman *fbox* (amodal -- full body
    # INCLUDING the occluded part), so it predicts box bottoms through the
    # reception desk; 10 epochs on ~4-6k outdoor-daylight images for an indoor
    # IR-flipping lobby; and it was a capacity downgrade too (yolo11x -> yolo11m).
    # Two confounded changes, never scored. Every notebook version used stock.
    p.add_argument("--detector", default="yolo11x.pt")
    p.add_argument("--staff-gallery", default=str(ROOT / "staff_gallery"))
    p.add_argument("--tz", default="America/Chicago")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--dry", action="store_true",
                   help="analyse only 20s — proves the whole path in a minute")
    p.add_argument("--strict", action="store_true",
                   help="refuse to produce a report if preflight fails")
    p.add_argument("--check", action="store_true", help="environment only")
    p.add_argument("--pick", action="store_true",
                   help="choose the chunk from a numbered menu")
    p.add_argument("--all", action="store_true",
                   help="every chunk in the Drive folder, oldest first, each "
                        "into its own output subfolder")
    p.add_argument("--no-render", action="store_true",
                   help="skip PASS 2. Every number is computed in PASS 1; "
                        "the annotated video is a human-audit artefact "
                        "worth about a fifth of the run only if someone watches it")
    p.add_argument("--render-window", nargs=2, type=float,
                   metavar=("START", "END"),
                   help="render ONLY these seconds. The middle ground between "
                        "a full render (11 min and ~8 GB on an hour) and "
                        "--no-render (nothing to watch, so 'is that box on a "
                        "real person?' stays unanswerable). Failures cluster, "
                        "so two chosen minutes beat sixty unwatched ones. "
                        "Ignored with --no-render.")
    p.add_argument("--no-zip", action="store_true",
                   help="skip packaging the results into a single zip")
    a = p.parse_args(argv)

    if a.check:
        return 0 if check_env() else 1
    if a.list:
        from kevacv.drive import describe, list_folder
        if not a.drive_folder:
            p.error("--list needs --drive-folder")
        print(describe(list_folder(a.drive_folder), a.chunk))
        return 0

    if a.all:
        return _run_all(a, p, argv)

    if a.drive_folder:
        # The zone map and staff photos live beside the footage. Fetch the
        # kilobytes automatically; the gigabytes only ever by explicit choice.
        from kevacv.drive import fetch_assets, fetch_chunk
        klog.setup(log_dir=str(ROOT / "logs"), name="pipeline")
        # An explicit --zones means "use THIS file", belt and braces. Since
        # 2026-08-13 the repo owns zones (drive.DRIVE_AUTHORITATIVE = ()), so
        # local wins either way — this just makes it impossible for a future
        # change to that default to silently overwrite a map passed on the
        # command line.
        _prefer = () if a.zones else None
        if a.zones:
            print(f"  --zones given explicitly: {a.zones} — NOT fetching zones "
                  f"from Drive for this run")
        # DRIVE SUPPLIES THE VIDEO. NOTHING ELSE.
        #
        # Operator instruction 2026-08-13: the repo is the source of truth for
        # every small asset. Zones are geometry the code is tuned against, and
        # the staff gallery is a curated set -- both are versioned WITH the
        # code, so a run is reproducible from a commit. Only the footage, which
        # is gigabytes and cannot live in git, is fetched.
        #
        # gallery_dir=None means fetch_assets will not write a single photo. A
        # Drive copy is reported as ignored rather than silently pulled, which
        # is what "fetching receptionist_sarah.jpg" was doing: quietly adding an
        # enrolled identity that exists in nobody's checkout.
        assets = fetch_assets(a.drive_folder, zones_dir=ROOT / "zones",
                              gallery_dir=None, prefer_drive=_prefer)
        _gal = ROOT / a.staff_gallery if not Path(a.staff_gallery).is_absolute() \
            else Path(a.staff_gallery)
        _imgs = sorted(p.name for p in _gal.glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        print(f"  staff gallery: LOCAL ONLY — {len(_imgs)} photo(s) in "
              f"{_gal}{' : ' + ', '.join(_imgs) if _imgs else ' (EMPTY — every '
              'staff member will be zone-inferred, never named)'}")
        if not a.zones and assets["zones"]:
            a.zones = assets["zones"][0]
            print(f"  using zone map from Drive: {a.zones}")
        if not a.video:
            if a.pick and not a.chunk:
                a.chunk = _choose_chunk(a.drive_folder)
                if a.chunk is None:
                    return 0
            # SELECT first, fetch second. The returned path is the only source
            # of truth for what was analysed and what time it is.
            a.video = str(fetch_chunk(a.drive_folder, ROOT / "data",
                                      chunk_filter=a.chunk))
    if not a.video or not a.zones:
        p.error("--video (or --drive-folder) and --zones are required")

    # analysis.detector, actually consumed.
    #
    # It was listed in RUN_CONFIG_CALLER_KEYS as "consumed by the caller" and
    # then read by nobody -- so a venue profile could name a detector, be
    # exempted from the unknown-key warning, and be silently ignored. That is
    # worse than an unrecognised key, because the yaml reads as authoritative.
    # An explicit --detector on the command line still wins.
    if a.config and "--detector" not in sys.argv:
        try:
            import yaml as _yaml
            _cfgd = (_yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
                     or {}).get("analysis") or {}
            if _cfgd.get("detector"):
                a.detector = str(_cfgd["detector"])
                print(f"  detector from {Path(a.config).name}: {a.detector}")
        except Exception as _de:
            print(f"  (could not read analysis.detector: {_de})")

    # The detector is the single largest measured lever in this pipeline
    # (best.pt vs yolo11x: GT coverage 66.2% -> 76.3%), and a 5-minute
    # annotated video was once produced with the WRONG one because a config
    # file still named it. Print it where nobody can miss it, and say out loud
    # when it is not the measured-best choice.
    print(f"  DETECTOR: {a.detector}")
    if "best.pt" in str(a.detector):
        print("  !! WARNING: models/best.pt is the CrowdHuman fine-tune, "
              "measured WORSE than stock yolo11x on this camera "
              "(66.2% vs 76.3% GT coverage). It was trained on amodal boxes "
              "that include the occluded part of a body, so it predicts box "
              "bottoms through the reception desk. Pass --detector yolo11x.pt "
              "unless you are deliberately A/B-ing it.")

    video, zones_path = Path(a.video), Path(a.zones)
    _checks = [("video", video), ("zones", zones_path)]
    # A BARE model name ("yolo11x.pt") is not a path -- ultralytics resolves and
    # downloads it. Only check the detector when it is given as a path, or the
    # stock-model default fails preflight on a clean box.
    if "/" in a.detector or "\\" in a.detector or Path(a.detector).exists():
        _checks.append(("detector", Path(a.detector)))
    for label, path in _checks:
        if not path.exists():
            print(f"  MISSING {label}: {path}")
            return 2

    klog.setup(log_dir=str(ROOT / "logs"), name="pipeline")
    check_env()

    zcfg = json.loads(zones_path.read_text(encoding="utf-8"))
    polygons = zcfg.get("polygons", {})
    # zone roles come from the engine's own classifier during the run; this is
    # only what preflight needs to judge the zone SET before the GPU starts.
    # ROLES COME FROM THE SAME PLACE THE ENGINE GETS THEM. This block used to
    # read zcfg["zone_roles"] -- a key nothing writes; the mapper and every
    # zone file use "roles" -- and then fall back to an inline keyword list of
    # its own that did NOT know "door" means entry.
    #
    # So a zone map with a polygon called "door" produced, in the same log,
    # 30 lines apart:
    #     preflight  no zone has the ENTRY role — arrivals cannot be counted
    #     analyse    entry_line + entry/wait/seating zones all present
    #
    # PREFLIGHT FAILED on a zone file the engine was perfectly happy with. Two
    # role systems that disagree are worse than one that is wrong, because the
    # contradiction trains you to ignore the louder of the two -- and this one
    # is the gate that is supposed to stop a bad run before the GPU starts.
    from kevacv.helpers import classify_zones
    roles = dict(classify_zones(list(polygons)))
    for _z, _r in (zcfg.get("roles") or zcfg.get("zone_roles") or {}).items():
        if _z in polygons:                 # explicit beats inferred, as in the engine
            roles[_z] = list(_r) if isinstance(_r, (list, tuple)) else [_r]

    bind_runtime(base=ROOT, output_dir=Path(a.out), device=a.device,
                 detector=a.detector, input_root=video.parent,
                 staff_gallery=a.staff_gallery)

    # The run config is applied AFTER bind_runtime, because bind_runtime
    # imports kevacv.engine and this must overwrite the module defaults that
    # import establishes. Printed in full either way: the whole point is that
    # a setting can no longer change the run without saying so, or appear in
    # the log without changing it.
    from kevacv.config import apply_run_config, describe_run_config
    _rc = apply_run_config(a.config)
    print(describe_run_config(_rc))

    # Put the SETTINGS in the ledger next to the counters they produced.
    # A ledger that says "crossings 0 -> 12" and cannot say WHICH setting
    # changed is half an answer; this makes the next run's auto-diff read
    # "dedup_nms_iou 0.7 -> 0.55  AND  crossings 0 -> 12" in one place.
    with klog.stage("run_config") as _st:
        for _k, _v in sorted((_rc.get("applied") or {}).items()):
            _st.count(_k, _v)
        if _rc.get("unknown"):
            _st.note(f"unknown config key(s) IGNORED: {_rc['unknown']}", "WARN")

    # WHICH CODE IS THIS? Set before anything expensive, because the answer
    # decides whether the rest of the log is worth reading. engine.py already
    # burns _BUILD_ID onto the annotated video's HUD band — it just never had
    # a value on this path, so every video ever rendered said "build ?" while
    # a stale copy on the GPU box produced five days of identical output.
    from kevacv import build_id as _bid
    from kevacv import engine as _E2
    _E2._BUILD_ID = _bid.compute()
    print(_bid.describe())

    if a.no_render:
        from kevacv import engine as _E
        _E.RENDER_VIDEO = False
    elif a.render_window:
        from kevacv import engine as _E
        _E.RENDER_WINDOW = (float(a.render_window[0]), float(a.render_window[1]))

    if a.tiled:
        _E2.ENABLE_TILED_DETECT = True

    if a.dataset_export:
        _E2.ENABLE_DATASET_EXPORT = True
        print(f"  dataset export ON -> {Path(a.out) / 'venue_dataset'}")

    if a.eval_export:
        _E2.EVAL_EXPORT = True
        _E2.EVAL_WINDOW = tuple(a.eval_window) if a.eval_window else None
        print(f"  eval export ON — frames for labelling will be written under "
              f"{Path(a.out) / 'eval_frames'}"
              + (f" for t={a.eval_window[0]:.0f}s..{a.eval_window[1]:.0f}s"
                 if a.eval_window else " (whole chunk)"))
    max_seconds = 20.0 if a.dry else a.max_seconds
    if a.dry:
        print("\n  --dry: analysing 20s only. Proves weights, CUDA, zones and "
              "the writer without spending an hour to find out.\n")

    try:
        result = run_camera(
            video, zones_path, a.out, camera_id=a.camera_id,
            zones=polygons, zone_roles=roles, tz_name=a.tz,
            strict_preflight=a.strict,
            selected_name=video.name, clock_source_name=video.name,
            max_seconds=max_seconds)
    except Exception as exc:
        print(f"\n  RUN FAILED: {type(exc).__name__}: {exc}")
        print(f"  the stage that failed is named in {ROOT / 'logs'}")
        raise
    finally:
        klog.close()

    print()
    print("=" * 70)
    print("  DONE")
    print("=" * 70)
    for path in result.get("written", []):
        print(f"  {path}")
    ph = result.get("phantoms") or {}
    if ph:
        print(f"  phantoms: static={len(ph.get('static', {}))} "
              f"rigid={len(ph.get('rigid', {}))} "
              f"mirrored={len(ph.get('mirrored', {}))}")
    arr = result.get("arrivals") or {}
    if arr:
        print(f"  arrivals: line={arr.get('line')} region={arr.get('region')} "
              f"trust={(arr.get('cross_check') or {}).get('trust')}")
    for lvl, msg in result.get("findings", []):
        print(f"  [{lvl}] {msg}")
    if not a.no_zip:
        try:
            z = _package(a.out, a.camera_id)
            mb = z.stat().st_size / 1_048_576
            print()
            print(f"  PACKAGED  {z}  ({mb:.1f} MB)")
            print()
            print("  Pull it to your Windows Downloads folder — in WSL:")
            print(f"    scp -i ~/ssh/yolo-surveillance-key.pem "
                  f"ubuntu@{_public_host()}:{z} /mnt/c/Users/prabh/Downloads/")
        except Exception as e:
            print(f"  (could not build the zip: {e})")
    print()
    print("  READ THESE FIRST:")
    print("    1. SOURCE crop height   <100px -> resolution is the ceiling")
    print("    2. ENTRY ZONE MISPLACED fires? -> zones, not counting, are wrong")
    print("    3. role_blocked_transitive     -> staff/customer fusions prevented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
