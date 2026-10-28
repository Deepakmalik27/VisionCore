# Observation layer — per-frame rows into Supabase

Date: 2026-08-21
Status: approved in chat, not yet implemented
Source: "Computer Vision - Video Analytics Discussion" transcript (Deepak, 2026-08-20)

## The ask, in one line

Deepak owns an **intelligence layer** that reads rows out of a database and builds
per-person journeys. This repo owns the **data layer** and must stop at correct
detection. His words: *"Detection should simply be correct. Then the entire game
happens in the intelligence layer."*

Today keva_vision's only DB write is `output/<run>/people.csv` -> `tools/ingest_db.py`
-> SQLite `people`: **one row per person per run**. His layer needs one row per
person **per frame**. That gap is this spec.

## Target record

```json
{"timestamp":"20:15:32.4","camera_id":"cam01","track_id":103,
 "bbox":[412,180,530,620],"zone":"reception","zone_confidence":0.91,
 "position":{"x":471,"y":620},"motion":{"direction":"stationary","speed":0.03},
 "reid":{"embedding_id":"emb_8831"},"pose":{"available":true},
 "detection_confidence":0.94}
```

Already produced inside the frame loop (`engine.py` ~3540): timestamp, camera_id,
track ids (raw and canonical), bbox, detection confidence, foot anchor, IR flag.
Missing: zone + zone_confidence on the row, motion, persisted re-ID vector, pose.

## Decisions (settled 2026-08-21)

| # | Decision | Why |
|---|---|---|
| D1 | Row carries **both** `raw_track_id` (BotSort, pre-merge) and `canon_id` (our re-ID answer, nullable) | His layer re-decides identity: *"check whether a new ID was detected within a few frames around the same coordinates ... then apply vector embeddings and match."* Handing him only our canonical id makes that impossible and unauditable. |
| D2 | Re-ID **vectors are persisted**, not just an opaque `embedding_id` | Same reason. `emb_8831` alone cannot be matched against anything. |
| D3 | Sink is **Supabase Postgres**, credentials from env | Prabh's call. He pulls the DB remotely in the morning; a local SQLite file cannot be pulled. |
| D4 | `zone_confidence` = **edge distance / box width, halved when feet not visible** | It must be a measured geometric quantity. This repo has already been burned by numbers that looked measured and were not (see `docs` audit trail: fabricated gt.txt, withdrawn accuracy claims). |
| D5 | Pose runs on a **stride with carry-forward** and a `pose_age_s` staleness field | Pose is the dominant GPU cost. Sitting/standing does not change in 125 ms. Staleness is stated in the data rather than hidden. |
| D6 | Rows leave the CV thread through the **existing `EventQueue`**, JSONL first, DB after | `kevacv/event_queue.py` was built for exactly this and is wired in `pipeline.py` behind `ENABLE_EVENT_QUEUE` (off by default); observations become its first real producer. *"The video pipeline should never wait for PostgreSQL."* |

## Architecture

```
engine.py frame loop
    |
    +-- kevacv/observe.py  build_row(...)          pure function, no I/O
    |
    +-- EventQueue.submit(row)      bounded, non-blocking, drops are COUNTED
            |
            +-- jsonl_sink -> output/<run>/observations.jsonl
run ends
    |
    +-- tools/ingest_obs.py  ->  Supabase Postgres (batched execute_values)
```

JSONL is written even though Postgres is the target. A dropped connection at
minute 52 must not destroy an hour of GPU time, and the file makes ingestion
replayable and idempotent. This mirrors `tools/ingest_db.py`, which already
exists and already uses delete-then-insert per run.

### Modules

- **`kevacv/observe.py`** (new, ~150 lines). Pure: takes frame context + one
  detection + zone polygons + previous track position + last pose sample,
  returns a dict. No file, no network, no model. Testable without a video.
- **`kevacv/engine.py`** (edit, ~15 lines). Inside the existing
  `for (bx1,by1,bx2,by2), tid, cid_ in zip(...)` loop, behind
  `ENABLE_OBSERVATIONS`, build the row and submit it.
- **`kevacv/config.py`** (edit). `ENABLE_OBSERVATIONS = False`, `OBS_EMB_STRIDE`,
  reuse `POSE_STRIDE`.
- **`tools/ingest_obs.py`** (new, ~120 lines). JSONL -> Postgres, `CREATE TABLE
  IF NOT EXISTS`, batched insert, per-run delete-then-insert.

## Schema

```sql
CREATE TABLE IF NOT EXISTS vision_runs (
  run_id          text PRIMARY KEY,
  camera_id       text NOT NULL,
  video_sha       text,
  fps_analysed    real,
  started_at      timestamptz,
  frames_analysed integer,
  zones_cfg_hash  text,
  git_sha         text,          -- kevacv.build_id.manifest()
  ingested_at     timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS vision_observations (
  run_id        text NOT NULL REFERENCES vision_runs(run_id),
  frame_idx     integer NOT NULL,
  raw_track_id  text    NOT NULL,
  camera_id     text    NOT NULL,
  ts            timestamptz,      -- wall clock, via kevacv.clock; NULL if unverified
  t_s           real    NOT NULL, -- seconds into the analysed span
  x1 integer, y1 integer, x2 integer, y2 integer,
  det_conf      real,
  foot_x        real, foot_y real,
  feet_visible  boolean,
  is_ir         boolean,
  zone          text,
  zone_conf     real,
  speed_px_s    real,
  speed_mps     real,             -- NULL unless GroundPlane is calibrated
  heading_deg   real,             -- NULL when stationary
  stationary    boolean,
  canon_id      text,             -- NULL until re-ID resolves
  emb_id        text,
  pose_activity text,             -- standing|walking|bending|sitting|unknown|NULL
  pose_age_s    real,
  PRIMARY KEY (run_id, frame_idx, raw_track_id)
);
CREATE INDEX IF NOT EXISTS obs_track ON vision_observations (run_id, raw_track_id, t_s);
CREATE INDEX IF NOT EXISTS obs_zone  ON vision_observations (run_id, zone, t_s);

CREATE TABLE IF NOT EXISTS vision_embeddings (
  emb_id       text PRIMARY KEY,
  run_id       text NOT NULL,
  camera_id    text NOT NULL,   -- cross-camera match is the point; see D2
  raw_track_id text NOT NULL,
  frame_idx    integer NOT NULL,
  blur_score   real,
  vec          bytea NOT NULL     -- float32 little-endian; see note
);
```

**pgvector:** if the Supabase project has the `vector` extension enabled, `vec`
becomes an unconstrained `vector` column instead of `bytea` (MEASURED 2026-08-21: the
re-ID backend on CAM.112 emits **1280-dim** vectors, not the 512 this spec first
assumed — so no dimension is hardcoded anywhere), so Deepak's re-match is one SQL query
(`ORDER BY vec <=> $1`) rather than a full pull into Python. The ingest tool
detects the extension at table-creation time and picks the column type; the JSONL
format is identical either way. Dimension is read from the first vector the run
produces, not hardcoded — the OSNet backend in `engine.py` can be swapped.

## Field definitions

**position** — foot anchor: `((x1+x2)/2, y2)`, the same anchor `engine.py` already
uses for zones (`bc` in the frame loop), except in desk zones where
`uses_centre_anchor()` applies. The transcript is explicit: *"We should not count
from the person's head. We should count from the lower body part."*

**feet_visible** — false when the box bottom is clipped by the frame edge, or the
box is inside a desk-occlusion mask, or the observed height is far below what
`PerspectiveModel.expected_h(foot_y)` predicts for that footline (person cut off
by furniture). This is the *"only the head is visible, so the person is behind the
desk"* case from the transcript, made explicit instead of inferred downstream.

**zone_conf**
```
d    = signed_distance(foot_anchor, polygon_edge) / box_width   # + inside, - outside
conf = clamp(0.5 + d, 0, 1) * (1.0 if feet_visible else 0.5)
```
Dead centre of dining -> ~0.95. One pixel over the reception line -> ~0.5 and
falling. Head-only behind the desk -> capped at 0.5. Every number traces to
something visible in the frame. `zone` is NULL when the anchor is in no polygon.

**motion** — foot-anchor displacement between analysed frames, median-smoothed
over the last 3 samples (raw per-frame deltas at 8 fps are jitter, not movement).
`speed_px_s` always; `speed_mps` only when `GroundPlane.ok()` (it already exposes
`speed_mps(p, q, dt)`), otherwise NULL — never a pixel value labelled as metres.
`stationary` = `speed_px_s` below the same threshold the statue filter uses.
`heading_deg` is a compass bearing in image space. **IN/OUT semantics are NOT
computed here** — "outside to inside" depends on camera angle and belongs to the
intelligence layer, which owns per-camera business logic.

**pose** — `yolo11n-pose` on every present box every `POSE_STRIDE` analysed frames
(default 8 ~= 1x/second at 8 fps), classified by the existing rule-based
`kevacv/pose.py`. Between samples the row repeats the last activity and increments
`pose_age_s`. Never sampled -> `pose_activity` NULL, `pose_age_s` NULL. Pose is
evidence in a column, never an input to identity — the module docstring's rule.

**emb_id** — written every `OBS_EMB_STRIDE` (default: the pose stride) per track,
not every frame. One vector per track per second is ~18k vectors/hour (~36 MB at
512 float32); one per frame would be eight times that for no new information.

## Volume

8 fps x 3600 s x ~5 people = ~145k observation rows per camera-hour, ~40 MB in
Postgres with indexes. Embeddings ~36 MB/hour. Acceptable for the one-hour runs
this is for; revisit partitioning if 24h runs across multiple cameras land.

## Failure handling

- Queue full -> row dropped and counted (`EventQueue.dropped`), CV loop never
  blocks. The run report prints `lost` — a silent discard would repeat this
  codebase's worst habit.
- Sink raises -> counted in `sink_errors`, run continues, JSONL keeps the rest.
- Ingest is idempotent: `DELETE FROM vision_observations WHERE run_id = ?` then
  insert, so re-ingesting a shorter re-run leaves no stale rows (the bug
  `ingest_db.py`'s selftest already guards for `people`).
- Missing Supabase credentials -> ingest exits non-zero with a clear message. The
  pipeline run itself never touches the network and never fails for this reason.

## Testing

`tests/test_observe.py`, no fixtures, no video:

1. zone_conf: anchor at polygon centre > anchor near the edge > anchor outside;
   feet-hidden row is exactly half the feet-visible row.
2. motion: a track with identical successive anchors -> `stationary=True`,
   `speed_px_s == 0`, `heading_deg is None`; a moving track -> correct bearing.
3. pose carry-forward: sampled frame -> `pose_age_s == 0`; the 7 frames after ->
   same activity, increasing age; never sampled -> both NULL.
4. `speed_mps is None` when the GroundPlane is `GroundPlane.none()`.

`tools/ingest_obs.py --selftest` against SQLite in memory (same trick
`ingest_db.py` uses) for the delete-then-insert idempotency.

End-to-end acceptance: one 60-second clip with `ENABLE_OBSERVATIONS=True`, then
`SELECT count(*), count(distinct raw_track_id) FROM vision_observations` matching
the run summary's person count, and `EventQueue` reporting `lost == 0`.

## Explicitly out of scope

- Staff vs customer verdicts (his layer; the transcript is explicit that this
  moves out of live detection and into confidence scoring downstream).
- IN/OUT counting and journey stitching (his layer).
- Cross-camera matching (his layer; this layer just makes the vectors available).
- Any ORM or migration framework; `CREATE TABLE IF NOT EXISTS` in one tool.
- Changing any existing count, filter or merge behaviour. This spec adds an
  output. If it changes a number, that is a bug.

## Dependency note

`psycopg[binary]` is not currently in the Dockerfile's pinned install block and
must be added there for `tools/ingest_obs.py`. The pipeline image itself does not
need it if ingestion runs on the host; pin it in the same style as the rest
(exact version, not a range).
