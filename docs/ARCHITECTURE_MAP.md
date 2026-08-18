# Architecture Map

Generated from the actual import graph and call order, not from memory.
Regenerate the facts with the commands at the bottom.

39 modules in `kevacv/`. The point of this document is that fine-tuning needs
to know **where a number is decided**, and there is exactly one place for each.

---

## 1. Entry — how a run starts

```
tools/deploy.sh                   your laptop -> the GPU box (rsync + build-id proof)
        │
        v
tools/run_pipeline.py             THE entry point. Parses flags, binds runtime.
        │  ├─ kevacv.drive              fetch chunk + zones + gallery from Drive
        │  ├─ kevacv.pipeline.bind_runtime    paths, device, GPU math modes
        │  ├─ kevacv.config.apply_run_config  config/cam112.yaml -> engine globals
        │  └─ kevacv.build_id           content hash -> _BUILD_ID -> video HUD
        v
kevacv/pipeline.py :: run_camera   THE ORCHESTRATOR. Everything below is its tree.
```

`run.sh` is a thin wrapper over the same path. The notebook is legacy and is
**not** in this chain.

---

## 2. The run tree — `run_camera`'s seven stages

Each is a `stage()` in the log, so the timeline reads root-to-leaf and a
failure names its phase.

```
run
 ├─ preflight    pipeline.preflight
 │                 ├─ _mask_swallows_zone      does a mask cover its own zone?
 │                 ├─ capabilities.disk_findings   will this run FIT? (GB)
 │                 ├─ pipeline.staff_gallery_findings   are there face photos?
 │                 └─ clock.verify_provenance / check_dst_span
 │
 ├─ analyse      engine.process_video           <-- 90% of the work. Section 3.
 │
 ├─ phantoms     detect_filters.static_track_ids     never moves
 │               detect_filters.rigid_track_ids      never deforms
 │               detect_filters.mirrored_pair_ids    never drifts apart
 │               detect_filters.drop_tracks          remove from events+
 │                                                   crossings+frame_log TOGETHER
 │
 ├─ identity     pipeline.resolve_identities
 │                 └─ topology.veto_pairs        physics veto on merges
 │
 ├─ answers      derive.enrich                   <-- Section 5
 │                 ├─ derive.arrivals_by_id       region primary, line cross-check
 │                 ├─ derive.guest_ids
 │                 └─ derive.staff_contacts
 │               arrivals.arrivals_from_regions / cross_check / entry_zone_coverage
 │               answers.answer_set               desk coverage, greet latency, guests
 │
 └─ report       report_slim.write_slim_outputs   SUMMARY.txt · people.csv · snaps/
```

---

## 3. `engine.process_video` — the frame loop

The expensive half. Two passes over the video.

### PASS 1 — decode → detect → track → identity → events

```
frame_source (ffmpeg NVDEC)
   │
   v
_analysis_stream()                  generator: gate, batch, hand to YOLO
   ├─ frame_exposure()              #19  dark/bright measured EVERY frame
   ├─ apply_clahe(clip=scaled)      #19  strength scales with severity
   ├─ IR detect (chroma + debounce)      colour evidence on/off
   ├─ MOTION GATE                        skip the detector on empty static frames
   └─ YOLO batch (DET_BATCH=24)
   │
   v
_filter_chain(dets, t)              ONE definition, three tracker branches
   │   yolo raw                     <- the funnel's denominator
   │   person/head split            _split_person_head
   │   split merged (+)             #9   two heads in one box = two people
   │   head recovery (+)            #12  orphan head = a body behind someone
   │   carried-object suppress      _suppress_carried
   │   implausible size             _drop_implausible  (needs the ground plane)
   │   dead-area mask               _drop_masked       (window_mask polygon)
   │   live phantom suppress        #5/6/7  phantoms.OnlineStaticSuppressor
   v
_dedup_nms()                        A1: two boxes on one body -> one
   v
TRACKER   BoT-SORT + CLIP-ReID (boxmot)  |  or sv.ByteTrack
   v
_IdentityMemory  (analytics.py)
   ├─ split_duplicate_raws()        #10  one canonical id, one raw track
   ├─ resolve()                     appearance + metric gate -> canonical id
   └─ try_swap()                    undo a mid-occlusion id trade
   v
LiveSeparability.observe()          #3/4/11/14  measures re-id, no labels needed
   v
zones.trigger() -> OccupancyRecorder.add()      dwell
line_zones.trigger() -> crossings               in/out
DatasetCollector.save_frame_pseudo_labels()     #8 retrain set (sampled)
```

### After the loop — offline identity + reporting

```
OccupancyRecorder.events()
   v
apply_staff_zone_override()   #1  SPREAD rule (visits x span), not dwell alone
   v
Re-ID stitching   merge_fragmented_tracks · global tracklet pass ·
                  face veto · cross-validation      (analytics.py)
   v
static / phantom sweep  ->  drop_tracks
   v
DIAGNOSTIC BLOCK   capabilities · funnel · staff decision · separability
```

### PASS 2 — render

```
render_annotated()  proxy JPEGs -> ffmpeg libx264
   HUD band carries: counts, entered/exited, STAFFED/AWAY, and _BUILD_ID
```

---

## 4. Module layers — who depends on whom

Measured in-degree from the import graph.

```
LOAD-BEARING (many importers)
  log            12 importers   stage timeline, counters
  config          3             every tuning constant
  ground_plane    3             pixels -> metres

THE TWO BIG ONES
  engine.py     imports: analytics camera_health capabilities config
                         dataset_collector detect_filters funnel ground_plane
                         helpers log phantoms reid_calibration venue_profile
  pipeline.py   imports: analytics answers arrivals capabilities clock config
                         derive detect_filters engine learn_zones log preflight
                         report_slim topology

LEAVES — import nothing in-package, so they are safe to change in isolation
  and are where most tuning actually belongs:
  arrivals · build_id · camera_health · capabilities · config · dataset_collector
  detect_filters · funnel · ground_plane · learn_zones · log · phantoms
  preflight · reid_calibration · report_slim · threshold · tiled · topology
  triage · venue_profile · graph_fusion · reid_engine · tracker_wrapper
  anomaly_baseline
```

---

## 5. Where each of the 22 symptoms is decided

The fine-tuning index. One row, one file, one place to look.

| # | Symptom | Decided in | Knob |
|---|---|---|---|
| 1 | Customers labelled staff | `analytics.apply_staff_zone_override` | `STAFF_MIN_VISITS`, `STAFF_MIN_SPREAD`, `STAFF_SOLE_DWELL_S` |
| 2 | Staff not recognised | `engine.discover_and_load_staff_gallery` | `staff_gallery/`, `STAFF_MATCH_THRESHOLD` |
| 3,4 | Staff/person id not persistent | `analytics._IdentityMemory.resolve` | `LIVE_REID_SIM_THRESHOLD` |
| 5,6,7 | Phantoms, plants, static ids | `phantoms.OnlineStaticSuppressor` + `detect_filters.static_track_ids` | `STATIC_MIN_LIFE_BY_ROLE` |
| 8 | Bad boxes | `models/best.pt` | retrain (`--dataset-export`) |
| 9 | Merged people | `engine._split_merged_persons` | `ENABLE_MERGED_SPLIT` |
| 10 | Duplicate ids | `_IdentityMemory.split_duplicate_raws` | `ENABLE_COVISIBILITY_BLOCK` |
| 11 | Flicker -> new id | detector recall + `engine._recover_bodies_from_heads` | `DETECT_CONF_FLOOR`, `HEAD_RECOVERY_MIN_CONF` |
| 12 | Occlusion | `_recover_bodies_from_heads`, `_boxes_occluding` | `OCCLUSION_IOU`, `OCCLUSION_CONTAIN` |
| 13,16 | IN/OUT, line crossing | `derive.arrivals_by_id`, `arrivals.py` | `zones/*.json` `entry_line` |
| 14 | Unique count | downstream of 3/4/11 | — |
| 15 | Zone assignment | `helpers.uses_centre_anchor` | zone polygons |
| 17 | Impossible counts | `engine.render_annotated` HUD | — (fixed) |
| 18 | Stale status | `engine.render_annotated` `staffed` | — (fixed) |
| 19 | Lighting / IR | `engine.frame_exposure` | `EXPOSURE_*`, `IR_CHROMA_THRESHOLD` |
| 20 | Unstable detection | = 8 + 19 | — |
| 21 | Upstream -> downstream | `funnel.DetectionFunnel` (visibility) | — |
| 22 | Overall | all of the above | — |

---

## 6. The three diagnostic tables every run prints

Read in this order:

1. **CAPABILITIES** (`capabilities.py`) — is this run trustworthy at all?
   Anything DEGR/MISS means numbers were produced from less evidence than the
   design assumes.
2. **DETECTION FUNNEL** (`funnel.py`) — of every detection YOLO made, what
   survived, and which stage removed the rest. `emptied` counts frames a stage
   took from someone-present to nobody — each is a candidate id break.
3. **STAFF DECISION** (`analytics.describe_staff_decision`) — every track that
   entered a staff zone, and why it was or was not called staff.

Plus **RE-ID SEPARABILITY** (`reid_calibration.LiveSeparability`) — what your
similarity thresholds actually cost, measured without labels.

---

## Regenerate

```bash
# import graph
python - <<'PY'
import ast, pathlib, collections
mods = {p.stem: p for p in pathlib.Path('kevacv').glob('*.py') if p.stem != '__init__'}
edges = collections.defaultdict(set)
for name, p in mods.items():
    for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module in mods:
            edges[name].add(n.module)
deg = collections.Counter(b for bs in edges.values() for b in bs)
for m, c in deg.most_common(10): print(f"{c:>3} <- {m}")
PY

# run stages, in order
grep -n 'with stage("' kevacv/pipeline.py
```
