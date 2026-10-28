#!/usr/bin/env python3
"""cvat_seed_task.py — create a CVAT task with the pipeline's tracks already in it.

WHY THIS EXISTS
    Labelling from an empty canvas is the slowest possible way to get ground
    truth, and ground truth is the binding constraint on this project: the
    detector memorised its single 80-second training clip (DetA 0.85 on unseen
    footage -> 0.42), and the crowded score is unusable because ~half the
    people in frame were never boxed.

    The fix is more labelled footage. The fix for THAT is to stop drawing.
    Seeding a task with the pipeline's own tracks turns the job from "draw
    every person in 1200 frames" into "delete what is wrong, add what is
    missing, set the roles" -- typically a 5-10x reduction in clicks.

THE CIRCULARITY LINE, STATED PLAINLY
    These boxes come from the model. Corrected by a human they are ground
    truth. NOT corrected, they are the model's own opinion, and scoring the
    model against them is self-grading -- the exact failure this project spent
    weeks escaping. So the task is created with a name that says PRELABELLED,
    and the labels carry a `reviewed` flag that must be set. Nothing here makes
    an unreviewed task safe to score against; it only makes reviewing cheap.

ROLE IS IMMUTABLE HERE, DELIBERATELY
    Task 1/3 declared `role` mutable, so CVAT stored it per-SHAPE and set it
    only from the frame you clicked FORWARD -- every frame before that stayed
    unknown, and tools/gt_from_cvat.py had to majority-vote to recover it.
    A person does not change role mid-clip. Immutable = one click per track,
    applies to the whole track, no propagation trap.

Usage:
    python tools/cvat_seed_task.py --video /data/videos/X.mp4 \
        --run /data/outputs/pre_X --name "GT CAM.112 W1 QUIET (PRELABELLED)"
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LABELS = [
    {"name": "person", "attributes": [
        {"name": "role", "mutable": False, "input_type": "select",
         "values": ["unknown", "staff", "guest"], "default_value": "unknown"},
        {"name": "reviewed", "mutable": False, "input_type": "checkbox",
         "values": ["false"], "default_value": "false"},
    ]},
]


def api(base, path, token=None, data=None, method=None, raw=None, ctype=None):
    url = base.rstrip("/") + path
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    r = urllib.request.Request(url, data=body,
                               method=method or ("POST" if body else "GET"))
    r.add_header("Content-Type", ctype or "application/json")
    if token:
        r.add_header("Authorization", "Token " + token)
    try:
        with urllib.request.urlopen(r, timeout=600) as resp:
            txt = resp.read().decode()
            return json.loads(txt) if txt.strip()[:1] in "{[" else txt
    except urllib.error.HTTPError as e:
        raise SystemExit(f"CVAT {path} -> HTTP {e.code}: {e.read().decode()[:400]}")


def get_token(base, user, password):
    tf = Path.home() / ".cvat_token"
    if tf.exists():
        t = tf.read_text(encoding="utf-8").strip()
        try:
            api(base, "/users/self", token=t)
            return t
        except SystemExit:
            pass
    if not password:
        raise SystemExit("no valid ~/.cvat_token and no CVAT_PASSWORD")
    return api(base, "/auth/login", data={"username": user, "password": password})["key"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--run", required=True, help="a pipeline --out dir with debug/*_frames.json.gz")
    p.add_argument("--name", required=True)
    p.add_argument("--base", default="http://localhost:8080/api")
    p.add_argument("--user", default="prabh")
    p.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    p.add_argument("--min-frames", type=int, default=15,
                   help="drop seeded tracks shorter than this (flicker). A real "
                        "person cannot appear for 0.2 s, and every junk track "
                        "seeded is a deletion the reviewer has to make.")
    a = p.parse_args(argv)

    flog = sorted(glob.glob(str(Path(a.run) / "debug" / "*_frames.json.gz")))
    if not flog:
        raise SystemExit(f"no debug/*_frames.json.gz in {a.run}")
    raw = json.loads(gzip.open(flog[0], "rt", encoding="utf-8").read())

    by_track = {}
    for e in raw:
        for b in e[2]:
            by_track.setdefault(b[0], []).append(
                (int(e[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])))
    kept = {t: v for t, v in by_track.items() if len(v) >= a.min_frames}
    print(f"  {len(by_track)} tracks in the run -> {len(kept)} seeded "
          f"({len(by_track) - len(kept)} dropped as under {a.min_frames} frames)")
    if not kept:
        raise SystemExit("nothing to seed")

    tok = get_token(a.base, a.user, a.password)
    task = api(a.base, "/tasks", token=tok,
               data={"name": a.name, "labels": LABELS})
    tid = task["id"]
    print(f"  created task {tid}: {a.name}")

    # upload the video
    boundary = "----kevaseed"
    fname = Path(a.video).name
    parts = []
    for k, v in (("image_quality", "70"), ("use_zip_chunks", "true"),
                 ("use_cache", "true"), ("sorting_method", "lexicographical")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"client_files[0]\"; "
        f"filename=\"{fname}\"\r\nContent-Type: video/mp4\r\n\r\n".encode())
    parts.append(Path(a.video).read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    api(a.base, f"/tasks/{tid}/data", token=tok, raw=b"".join(parts),
        ctype=f"multipart/form-data; boundary={boundary}")
    print("  video uploaded, waiting for CVAT to unpack it...")
    for _ in range(180):
        st = api(a.base, f"/tasks/{tid}/status", token=tok)
        if st.get("state") in ("Finished", "Failed"):
            break
        time.sleep(5)
    else:
        raise SystemExit("timed out waiting for CVAT to process the video")
    if st.get("state") == "Failed":
        raise SystemExit(f"CVAT failed to process the video: {st}")

    label_id = api(a.base, f"/labels?task_id={tid}", token=tok)["results"][0]["id"]
    tracks = []
    for t, rows in kept.items():
        rows.sort()
        shapes = []
        prev = None
        for fi, x1, y1, x2, y2 in rows:
            # An `outside` shape closes the segment whenever the track skipped
            # frames. Without it CVAT interpolates straight through the gap and
            # draws a box on frames where the pipeline saw nothing -- inventing
            # ground truth that was never observed.
            if prev is not None and fi > prev + 1:
                shapes.append({"type": "rectangle", "frame": prev + 1,
                               "points": shapes[-1]["points"], "occluded": False,
                               "outside": True, "attributes": []})
            shapes.append({"type": "rectangle", "frame": fi,
                           "points": [x1, y1, x2, y2], "occluded": False,
                           "outside": False, "attributes": []})
            prev = fi
        shapes.append({"type": "rectangle", "frame": prev + 1,
                       "points": shapes[-1]["points"], "occluded": False,
                       "outside": True, "attributes": []})
        tracks.append({"label_id": label_id, "frame": rows[0][0],
                       "group": 0, "source": "auto", "shapes": shapes,
                       "attributes": []})
    api(a.base, f"/tasks/{tid}/annotations", token=tok, method="PUT",
        data={"version": 0, "tags": [], "shapes": [], "tracks": tracks})
    n_shapes = sum(len(t["shapes"]) for t in tracks)
    print(f"  seeded {len(tracks)} tracks / {n_shapes} shapes")
    print()
    print(f"  open: http://localhost:8080/tasks/{tid}")
    print("  the job is REVIEW, not drawing:")
    print("    1. delete tracks that are not people")
    print("    2. add anyone the pipeline missed  <- this is the part that matters")
    print("    3. set role on each track (immutable: one click covers it)")
    print()
    print("  UNTIL a human does step 2, this is the model's opinion, not ground")
    print("  truth, and scoring against it would be self-grading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
