# Graph Report - /mnt/c/Users/prabh/Documents/Aurika/keva_vision  (2026-08-24)

## Corpus Check
- Large corpus: 379 files · ~2,402,244 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 2389 nodes · 4543 edges · 139 communities (120 shown, 19 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 208 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Confidence Tiers & Review Queue
- Phantom Detection Filters
- Static Phantom Suppression
- Run Scorecard & Occupancy Reconciliation
- Google Drive Chunk Fetch
- Staff vs Customer Zone Rules
- Engine Frame Preprocessing
- Per-Frame Observation Rows
- Compute Triage & Venue Profile
- Re-ID Separability Calibration
- Identity Memory & Duplicate Split
- A/B Merge Policy Evaluation
- Venue Dataset Collector
- Fisheye Dewarp
- Preflight Disk & Staff Gallery
- Detection Funnel Accounting
- Pose & Activity Layer
- Wiring Audit
- Analytics: Tracks to People
- Decision Log Ledger
- Zone & Plot Helpers
- Tiled Native-Res Detection
- Test Head Recovery
- Reid Engine
- Test Entry Label Kit
- Derive
- Learn Zones
- Camera Health
- Answers
- Slit Count
- Test Build Id
- Arrivals
- Test Robust Ground Plane
- Ground Plane
- Log
- Event Queue
- Test Provenance
- Report Slim
- Build Static Mask
- Seams
- Analytics
- Pipeline
- Profiling
- Calibrate Plane
- Test Run Config
- Gt Kit
- Capabilities
- Test Score By Time
- Test Ground Plane
- Threshold
- Anomaly Baseline
- Test Config Completeness
- Analytics
- Test Covisibility Dedupe
- Clock
- Graph Fusion
- Test Phase A
- Test Preflight Entry Line
- Test Mixed Id Sort
- Log
- Validity
- Test Exposure
- Test Gt Validate
- Ablate Slit Rules
- Threshold Expiry
- Analytics
- Validity
- Test Slit Wiring
-   Main  
- Config
- Engine
- Test Det Classes
- Eval Harness
- Eval Harness
- Log
- Test Calibration Sample
- Analytics
- Conftest
- Test Phase A
- Test Mask Motion Veto
- Test Stationary Gap Cap
- Doorcount
- Ground Plane
- Test Duplicate Keys
- Preflight
- Test Pipeline
- Test V56 Phase3
- Embed Kevacv
- Track Health
- Test Graph Fusion Plane Trust
- Test Label Kit Plan
- Test Scale Reference
- Test V56 Phase2
- Audit Notebook Edge Cases
- Export Tensorrt
- Id Reuse Audit
- Analytics
- Test Crossing Prior Stability
- Resilience
- Resilience
- Test No Missing Names
- Test P0 Config Reach
- Make Trainset
- Mask Prototype
- Test Absurd Size
- Test Check Closure
- Test Engine Compiles
- Test Eval Scorer
- Test Phantom Gates Measured
- Test Track Buffer Units
- Test V58 Phasec
- Audit Frames
- Extract Pipeline
- Run Night
- Config
- Test Make Trainset
- Test Render Window
- Depth Prototype
- Ingest Db
- Engine
- Run
- Test Absurd Cap Vs Geometry
- Test Audit Frames
- Test Guestlink
- Test V56 Phase11
- Test Venue Provenance
- Check Closure
- Make Label Package
- Score Points
- Test V69 Birth Arrivals
- Build Clean From Source
- Deploy
- Frame Ritual
- Score Line Entries
- Verify Notebook Health
- Make Entry Labels
- Stamp Build

## God Nodes (most connected - your core abstractions)
1. `process_video()` - 78 edges
2. `run_camera()` - 40 edges
3. `GroundPlane` - 38 edges
4. `build_visits()` - 30 edges
5. `merge_fragmented_tracks()` - 26 edges
6. `Ledger` - 23 edges
7. `fit_robust_ground_plane()` - 23 edges
8. `arrival_tier()` - 21 edges
9. `get_logger()` - 21 edges
10. `DetectionFunnel` - 20 edges

## Surprising Connections (you probably didn't know these)
- `make_analyse()` --indirect_call--> `_f()`  [INFERRED]
  tests/test_pipeline.py → kevacv/observe.py
- `Plane` --uses--> `FusionWeights`  [INFERRED]
  tests/test_graph_fusion_plane_trust.py → kevacv/graph_fusion.py
- `FakePersp` --uses--> `GroundPlane`  [INFERRED]
  tests/test_ground_plane.py → kevacv/ground_plane.py
- `anchor_point()` --indirect_call--> `_cfg()`  [INFERRED]
  kevacv/helpers.py → tests/test_preflight_entry_line.py
- `FakeOOM` --uses--> `Checkpoint`  [INFERRED]
  tests/test_edge_cases.py → kevacv/resilience.py

## Import Cycles
- 3-file cycle: `kevacv/config.py -> kevacv/engine.py -> kevacv/helpers.py -> kevacv/config.py`
- 3-file cycle: `kevacv/__init__.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/__init__.py`
- 3-file cycle: `kevacv/__init__.py -> kevacv/pipeline.py -> kevacv/engine.py -> kevacv/__init__.py`
- 3-file cycle: `kevacv/analytics.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/analytics.py`
- 3-file cycle: `kevacv/config.py -> kevacv/engine.py -> kevacv/ground_plane.py -> kevacv/config.py`
- 4-file cycle: `kevacv/__init__.py -> kevacv/helpers.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/__init__.py`
- 4-file cycle: `kevacv/config.py -> kevacv/engine.py -> kevacv/observe.py -> kevacv/helpers.py -> kevacv/config.py`
- 4-file cycle: `kevacv/__init__.py -> kevacv/analytics.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/__init__.py`
- 4-file cycle: `kevacv/__init__.py -> kevacv/ground_plane.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/__init__.py`
- 4-file cycle: `kevacv/__init__.py -> kevacv/pipeline.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/__init__.py`
- 4-file cycle: `kevacv/__init__.py -> kevacv/preflight.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/__init__.py`
- 4-file cycle: `kevacv/analytics.py -> kevacv/ground_plane.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/analytics.py`
- 5-file cycle: `kevacv/__init__.py -> kevacv/geometry_calibration.py -> kevacv/ground_plane.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/__init__.py`
- 5-file cycle: `kevacv/__init__.py -> kevacv/analytics.py -> kevacv/ground_plane.py -> kevacv/config.py -> kevacv/engine.py -> kevacv/__init__.py`

## Communities (139 total, 19 thin omitted)

### Community 0 - "Confidence Tiers & Review Queue"
Cohesion: 0.05
Nodes (72): arrival_tier(), event_tier(), Any, Event confidence tiers — CONFIRMED / PROBABLE / UNCERTAIN / REJECTED. WHY ---…, Counts per tier plus the headline sentence a report should print., Tier for a WINDOW's arrival count, from how far the estimators agree.…, Tier for ONE crossing event. Vetoes are absolute., summarise() (+64 more)

### Community 1 - "Phantom Detection Filters"
Cohesion: 0.05
Nodes (62): describe(), drop_tracks(), implausible_size_mask(), mirrored_pair_ids(), _path_length(), _point_in_poly(), detect_filters.py — PHASE 7. Two filters for two phantoms the real run…, How long each track must sit still before the static filter believes it is… (+54 more)

### Community 2 - "Static Phantom Suppression"
Cohesion: 0.06
Nodes (42): describe(), drop_phantom_dets(), in_phantom(), _iou(), _median(), OnlineStaticSuppressor, phantom_regions(), phantoms.py — PHASE 10. Kill static false positives that CHANGE ID. WHY THIS… (+34 more)

### Community 3 - "Run Scorecard & Occupancy Reconciliation"
Cohesion: 0.07
Nodes (57): describe_reconciliation(), Does the door arithmetic agree with the people actually on screen? audit.txt…, The drift, in words, because a JSON field nobody reads is not a check., reconcile_occupancy(), _build_scorecard(), Assemble the scorecard from what the run already produced., pixel_height(), Any (+49 more)

### Community 4 - "Google Drive Chunk Fetch"
Cohesion: 0.07
Nodes (43): parse_start(), -> naive datetime of the chunk's START, from its filename. Reads the FIRST…, describe(), DriveError, fetch_assets(), fetch_chunk(), _folder_entries(), list_folder() (+35 more)

### Community 5 - "Staff vs Customer Zone Rules"
Cohesion: 0.10
Nodes (47): apply_staff_zone_override(), describe_staff_decision(), Per-track evidence for the staff/customer decision, as data. Returned…, Re-label tracks as 'staff' from where and how they spent their time. THE RULE…, Every track that spent ANY time in a staff zone, and the call made on it.…, staff_evidence(), _ev(), A wrong Re-ID merge must not be able to manufacture a staff member. Replays the… (+39 more)

### Community 6 - "Engine Frame Preprocessing"
Cohesion: 0.05
Nodes (45): _blur_score(), _botsort_yaml(), _cross2d(), _dashed_poly(), _dim(), discover_and_load_staff_gallery(), embed_face_scored(), exposure_clip_limit() (+37 more)

### Community 7 - "Per-Frame Observation Rows"
Cohesion: 0.10
Nodes (46): anchor_point(), The (x, y) that uses_centre_anchor() selected. box is xyxy. THE FOOT IS NOT AT…, build_row(), emb_row(), _f(), _median(), motion(), pose_field() (+38 more)

### Community 8 - "Compute Triage & Venue Profile"
Cohesion: 0.06
Nodes (30): coverage_report(), describe(), miss_risk(), plan_segments(), triage.py — PHASE 5a. Spend the compute where the people are. THE PROBLEM THIS…, Account for every second of the chunk in one of three states. A pipeline that…, Choose the stretches worth the full pipeline. scan: [(t_seconds, n_people),…, How likely is the cheap scan to miss a person entirely? A visit of length V… (+22 more)

### Community 9 - "Re-ID Separability Calibration"
Cohesion: 0.09
Nodes (38): calibrate(), compare_to_legacy(), cosine(), describe(), describe_live(), LiveSeparability, _looks_duplicate(), _ordered() (+30 more)

### Community 10 - "Identity Memory & Duplicate Split"
Cohesion: 0.10
Nodes (38): _dynamic_accept_thresh(), _IdentityMemory, Rolling memory of recently-seen identities: one best snapshot (anchor…, crop may be None (detection too small/low-conf to embed) — still tracked…, A canonical identity may belong to exactly ONE raw track at a time — a person…, check(), main(), _mem() (+30 more)

### Community 11 - "A/B Merge Policy Evaluation"
Cohesion: 0.08
Nodes (33): kevacv — the parts of the video pipeline that are not the notebook. WHY THIS…, ab_topology(), apply_topology_veto(), assert_matches(), describe(), greedy_union(), merge_ab.py — measure a merge-policy change before shipping it. WHY THIS EXISTS…, Split candidate pairs into (possible, vetoed) using the topology gate. A pair… (+25 more)

### Community 12 - "Venue Dataset Collector"
Cohesion: 0.06
Nodes (30): _boxes_occluding(), v44: True if two xyxy boxes mutually occlude — either their IoU is high or one…, DatasetCollector, ndarray, Path, dataset_collector.py — Phase D P13 Venue-Specific Dataset Collector & Pseudo-…, Venue-specific training sample collector and pseudo-label exporter., Save a frame image and normalized YOLO format bounding box label file. boxes:… (+22 more)

### Community 13 - "Fisheye Dewarp"
Cohesion: 0.09
Nodes (33): box_to_source(), _centre_scale(), dewarped_predict(), fit_k(), horizon_r(), in_domain(), fisheye.py — undo the lens before asking the detector what it sees. WHY THIS…, -> (cx, cy, s). s is the half-diagonal, so r is ~1.0 at the corners and k means… (+25 more)

### Community 14 - "Preflight Disk & Staff Gallery"
Cohesion: 0.13
Nodes (34): disk_findings(), Will this run fit on the disk? -> [(level, message)] WHY THIS IS A PREFLIGHT…, Is face-based staff identification actually available? -> [(level, msg)]…, staff_gallery_findings(), check(), main(), test_staff_gallery.py — two photos of one person are ONE staff member. WHY THIS…, The actual files that shipped — this is the bug, not a fixture. (+26 more)

### Community 15 - "Detection Funnel Accounting"
Cohesion: 0.11
Nodes (25): DetectionFunnel, funnel.py — where detections are born, and where each one dies. WHY THIS EXISTS…, The lines worth raising to ERROR/WARN, as (level, message). Thresholds are…, Per-stage detection accounting for one video chunk. Stages are keyed by name…, One stage, one frame. Returns n_out so call sites can chain., The detector's raw output — the denominator everything else is a share of., The table, as a human reads it in the run log., check() (+17 more)

### Community 16 - "Pose & Activity Layer"
Cohesion: 0.11
Nodes (32): classify(), demo(), draw(), get_model(), _iou(), keypoints_for_frame(), match_to_boxes(), _mid() (+24 more)

### Community 17 - "Wiring Audit"
Cohesion: 0.11
Nodes (31): check_A2_uncalled_by_design(), check_A_dead_modules(), check_B_shadowed_constants(), check_C_signature_defaults(), check_D_stale_modules(), check_E_unsettable_flags(), check_F_broken_key_mapping(), check_G_dead_tools() (+23 more)

### Community 18 - "Analytics: Tracks to People"
Cohesion: 0.08
Nodes (30): clip_to(), complement_intervals(), covered_windows(), detect_groups(), detect_parties(), find_optimal_threshold(), _gta_hungarian(), _gta_solve() (+22 more)

### Community 19 - "Decision Log Ledger"
Cohesion: 0.10
Nodes (17): Ledger, A per-run ledger of WHAT happened, WHY, and WHAT IT COST. WHY THIS EXISTS Today…, authority: which rule/threshold decided. recoverable: how to get it back., owns: file:line that makes the decision. does: one plain sentence., Record a parameter AND where its value came from., _safe(), The ledger must make today's actual failures visible. Each test replays a real…, The log said "id churn at the line". It was a door filter deleting every… (+9 more)

### Community 20 - "Zone & Plot Helpers"
Cohesion: 0.10
Nodes (25): classify_zones(), load_zone_config(), mmss(), plot_reid_pair_audit(), helpers.py — the small shared functions engine.py needs. EXTRACTED FROM…, Which point on a body decides whether it is inside this zone? Feet everywhere…, Fixed sorted order -> fixed colors, shared by charts AND video overlay., (v36) The actual crops behind a calibration number, side by side -- turns "why… (+17 more)

### Community 21 - "Tiled Native-Res Detection"
Cohesion: 0.09
Nodes (20): cost_estimate(), height_roi(), _iou(), nms(), tiled.py — run the detector at native resolution on slices, not on a downscaled…, Greedy NMS over (x1,y1,x2,y2,conf,cls). Highest confidence wins. Suppression is…, True if the box hugs a tile edge that is NOT also the frame edge., Detect on overlapping native-resolution slices; return full-frame boxes. frame… (+12 more)

### Community 22 - "Test Head Recovery"
Cohesion: 0.17
Nodes (29): check(), dets(), FakePlane, main(), test_head_recovery.py — a head with no body is a person, not a statistic. WHY…, Otherwise every visible person is duplicated — one ID on two boxes, which is…, Inferred, not observed. It must lose to a real detection anywhere the two…, Everyone hidden behind one another is exactly when this matters most. (+21 more)

### Community 23 - "Reid Engine"
Cohesion: 0.08
Nodes (17): ndarray, reid_engine.py — Phase B P4/P5 SOTA Re-ID Engine & Visible-Infrared (VI) Dual-…, Extract Re-ID embeddings for a batch of crops in a single parallel GPU pass., Production SOTA Re-ID embedding extractor with RGB and Infrared (IR) cross-…, Extract L2-normalized 1D Re-ID embedding vector from an RGB/IR person crop…, ReIDEmbeddingExtractor, Any, ndarray (+9 more)

### Community 24 - "Test Entry Label Kit"
Cohesion: 0.12
Nodes (27): _plan(), The entry-label harness must refuse label sets that cannot detect the failures…, A held-out set of only small windows cannot see the group regression., test_accepts_a_sound_label_set(), test_refuses_when_no_group_is_held_out(), test_refuses_when_nothing_held_out(), test_refuses_when_nothing_labelled(), test_refuses_without_a_quiet_window() (+19 more)

### Community 25 - "Derive"
Cohesion: 0.11
Nodes (24): arrivals_by_id(), build_customer_registry(), enrich(), guest_ids(), id_confidence(), observed_windows(), derive.py — turn what the engine returns into what the answers need. WHY THIS…, First entry-zone -> interior-zone transition per non-staff track. Walking in IS… (+16 more)

### Community 26 - "Learn Zones"
Cohesion: 0.12
Nodes (22): _clusters(), describe(), _edge_dist_frac(), learn_dwell_zones(), learn_entry_zones(), learn_zones.py — PHASE 12. Find the door in the data instead of drawing it. WHY…, Propose entrance polygons from where tracks are BORN and DIE. -> (proposals,…, Propose zones where people STOP — waiting areas, counters, queues. Movement… (+14 more)

### Community 27 - "Camera Health"
Cohesion: 0.09
Nodes (15): CameraHealth, _prep(), camera_health.py — PHASE 4 / U1. "Is this still the same camera view?" THE BUG…, Persist so chunk 7 is compared against chunk 1's view, not its own. V73: append…, Affine transform from the reference view to this frame. ORB + RANSAC on…, Worst zone-vertex displacement in ORIGINAL-frame pixels. The number that…, The other four ways the video is not what you think it is., Full verdict. `valid` False means: do NOT publish zone-based numbers. (+7 more)

### Community 28 - "Answers"
Cohesion: 0.14
Nodes (23): Answer, answer_set(), _clip(), desk_coverage(), desk_gaps(), greet_latency(), guest_count(), _merge() (+15 more)

### Community 29 - "Slit Count"
Cohesion: 0.12
Nodes (24): count(), debounce(), direction(), _drop(), foreground(), new_funnel(), Sample the door line at several perpendicular OFFSETS. One slit tells you…, Split the slice at colour<->IR transitions. -> list of (lo, hi) rows. MEASURED… (+16 more)

### Community 30 - "Test Build Id"
Cohesion: 0.18
Nodes (24): compute(), describe(), diff(), manifest(), build_id.py — which code produced this output? WHY THIS EXISTS Five days of…, Every .py in the package, sorted, excluding caches. Sorted because the id must…, Short content hash of the package source. -> str Bytes, not text: a file that…, Per-file hashes, so a mismatch can be localised to the file that differs… (+16 more)

### Community 31 - "Arrivals"
Cohesion: 0.13
Nodes (17): arrivals_from_regions(), cross_check(), describe(), entry_zone_coverage(), arrivals.py — PHASE 9. Count arrivals without depending on the line. WHY THIS…, What share of non-staff people were EVER seen in an entry zone? WHY THIS EXISTS…, Compare the two independent arrival counts and say what to believe. The point…, Arrivals inferred from zone transitions instead of a line crossing. An arrival… (+9 more)

### Community 32 - "Test Robust Ground Plane"
Cohesion: 0.14
Nodes (18): fit_robust_ground_plane(), calibration.py — Phase B P6 Automated Camera Calibration & Geometry…, Robustly fit ground plane line h(y) = a*y + b using RANSAC to exclude…, The plane refusal must name WHICH filter emptied the sample set. A bare…, Boxes a real oblique camera produces: aspect ~1.5, below the 2.0 gate., Zero-width boxes must not divide-by-zero the diagnostic., test_refusal_reports_counts_and_median(), test_refusal_survives_degenerate_boxes() (+10 more)

### Community 33 - "Ground Plane"
Cohesion: 0.14
Nodes (10): GroundPlane, Pixel height of a standing person with feet at row y (auto mode)., Image point (x, y) — the FEET — to floor (X, Z) in metres. Returns None above…, Metres per pixel at image row y. The number that was implicitly assumed…, Floor distance in metres between two FOOT points. None if either is above the…, Convert a metric threshold back to pixels AT ROW y. Lets existing pixel-based…, Implied camera height. For a level camera h = (Hp/Hc)*(y - y_horizon), so a ==…, Compare the plane against ONE known real distance in the scene. A negative… (+2 more)

### Community 34 - "Log"
Cohesion: 0.13
Nodes (19): banner(), close(), compare(), flatten(), flatten_or_dict(), _fmt(), _install_record_factory(), _previous_ledger() (+11 more)

### Community 35 - "Event Queue"
Cohesion: 0.16
Nodes (13): EventQueue, jsonl_sink(), Any, Events leave the CV thread through a queue, not a database call. WHY ---…, Append-only JSONL. A crash at hour 23 keeps hours 1-22., Never blocks. Returns False if the event was dropped., The CV loop must never wait for a sink, and must never lose an event quietly., A dropped event that is COUNTED is recoverable. A stalled decoder is not. (+5 more)

### Community 36 - "Test Provenance"
Cohesion: 0.17
Nodes (20): build_stamp(), diff(), fingerprint(), Any, What produced this number. Stamped onto the events, not just the log. WHY ---…, One record identifying everything that could change a count., Short, stable hash of everything that could change a count. Two runs with the…, What differs between two stamps — the answer to 'why did it change? (+12 more)

### Community 37 - "Report Slim"
Cohesion: 0.14
Nodes (17): coverage_strip(), describe_video(), _fmt(), _hm(), people_csv(), people_rows(), report_slim.py — one file a manager reads, one file a analyst reads. WHY THIS…, Normalise person dicts into the fixed column order, filling gaps. A missing… (+9 more)

### Community 38 - "Build Static Mask"
Cohesion: 0.14
Nodes (18): build_mask(), edge_energy(), line_hits(), main(), modality_of(), Streaming per-pixel mean/variance, so we never hold the frame stack., Local texture. Sobel magnitude, blurred so single-pixel noise doesn't read as…, distractor = quiet over time AND textured in space. (+10 more)

### Community 39 - "Seams"
Cohesion: 0.12
Nodes (13): apply(), bridge(), heads(), seams.py — a person who walks across a chunk boundary is still one person. WHY…, Rewrite a chunk's ids so bridged bodies keep the earlier identity., Tracks still alive in the last `within_s` of a chunk. -> {tid: (t, x, y)}, Tracks first seen in the first `within_s` of a chunk. -> {tid: (t, x, y)}, Match bodies across a chunk boundary. -> (mapping, findings). mapping is… (+5 more)

### Community 40 - "Analytics"
Cohesion: 0.12
Nodes (13): _appearance_hsv_contradicts(), _handoff_hsv_contradicts(), _hsv_gallery_sim(), merge_fragmented_tracks(), Best pairwise HSV cosine across two crop lists — same gallery philosophy as…, v53: veto an APPEARANCE-tier merge when the independent HSV colour space…, v48: independent 2nd opinion on a pure-spatial merge. Within a hand-off-sized…, Merge track fragments that are the same person reappearing. track_windows:… (+5 more)

### Community 41 - "Pipeline"
Cohesion: 0.14
Nodes (19): entered_count(), Unique track_ids that crossed INWARD at a venue entrance. `lines`: the door…, -> the shape report_slim.summary_txt() expects., to_report_rows(), protected_ids(), Track ids whose SHAPE never changes — a plant, mannequin, poster, sign. WHY…, Ids the static filter may never touch: anyone who crossed the entry line, and…, rigid_track_ids() (+11 more)

### Community 42 - "Profiling"
Cohesion: 0.11
Nodes (8): Profile, profiling.py — where the time actually goes, per stage, every run. WHY THIS…, The Phase C decision, stated by the data rather than by argument., Accumulates wall time per named stage. Not thread-safe by design: the analysis…, For a stage timed elsewhere (a subprocess, a library callback)., {stage_ms: int} for the run ledger — ints so the diff is readable., The table. Ordered by cost, because the first line is the only one most people…, test_profiling.py — the profiler, and the gate it exists to decide. The…

### Community 43 - "Calibrate Plane"
Cohesion: 0.17
Nodes (16): test_calibrate_plane.py — the check that makes the calibration trustworthy.…, _dist(), enough_evidence(), main(), parse_point(), calibrate_plane.py — turn four floor tiles into a real ground plane. WHY THIS…, -> (ok, worst_error_or_None). A pair the plane could not map at all is a hard…, -> (ok, why). Does this measurement set CONTAIN a real test? THE TRAP, found by… (+8 more)

### Community 44 - "Test Run Config"
Cohesion: 0.26
Nodes (17): describe_run_config(), One block a human can read in the log, showing what MOVED., check(), main(), test_run_config.py — config/*.yaml must actually reach the run. WHY THIS EXISTS…, Stand-in for kevacv.engine — the real module needs torch, and none of this…, The real file, not a fixture — this is the regression that shipped., detector/reid_weights are real settings consumed by the caller. If they were… (+9 more)

### Community 45 - "Gt Kit"
Cohesion: 0.20
Nodes (17): dump_errors_csv(), explain(), Print the number AND where it came from. Reads the diagnosis out loud so the…, Every FP and FN with its box, so they can be drawn on the frames. This is what…, Freeze a score so the NEXT change has something to be compared against., pairs: {condition_name: (gt_path, pred_path)} One aggregate number hides the…, save_baseline(), score_conditions() (+9 more)

### Community 46 - "Capabilities"
Cohesion: 0.15
Nodes (7): CapabilityLedger, onnx_providers(), capabilities.py — what is actually working, what quietly fell back. WHY THIS…, Which execution providers onnxruntime actually has. -> (available, has_gpu)…, One row per subsystem: what it is, how it is running, what that costs., impact: what this run CANNOT tell you because of this status. Required for…, Nothing missing and nothing degraded.

### Community 47 - "Test Score By Time"
Cohesion: 0.18
Nodes (15): load_mot(), MOT 1.1 / MOT16: frame,id,x,y,w,h,conf,... -> {frame: [(id,x,y,w,h)]} Lenient…, The time-matched scorer must reproduce the frame-index scorer exactly. A…, At 15fps against 7.5fps gt, two predictions compete for each gt frame; exactly…, A bad --pred-t0 must drop frames, so it reads as a mapping error rather than as…, test_double_rate_collapses_to_one_per_gt_frame(), test_gt_mapping_matches_the_manifest(), test_identity_when_rates_match() (+7 more)

### Community 48 - "Test Ground Plane"
Cohesion: 0.13
Nodes (11): ground_plane.py — PHASE 3. Stop measuring the world in pixels. THE PROBLEM…, Return project(X, Z, Y) -> (x, y) for a camera at height cam_h looking down the…, synth_camera(), project(), A deliberately non-affine (perspective) mapping, so the test would fail for…, FakePersp, fit_line(), Validate ground_plane.py against a SYNTHETIC CAMERA with known truth. This is… (+3 more)

### Community 49 - "Threshold"
Cohesion: 0.18
Nodes (12): compare(), cost_weighted_threshold(), describe(), _rates(), threshold.py — choose a merge threshold by what mistakes actually COST. WHY…, Is this signal good enough to merge on at ALL? A threshold chosen on a signal…, Current threshold vs the cost-optimal one, as a decision aid. Never returns…, An ASCII cost curve. Seeing the shape stops a single number implying more… (+4 more)

### Community 50 - "Anomaly Baseline"
Cohesion: 0.14
Nodes (9): Any, anomaly_baseline.py — Phase C P10 Learned Anomaly Baselines (Avigilon UMD-Style…, Learned anomaly baseline engine for per-zone occupancy time-series., Fit history from records: [{zone_id, hour, is_weekend, count}, ...], Score current zone occupancy count against baseline model. Returns dict with…, ZoneAnomalyDetector, test_phase_c_d.py — Phase C & D Verification Test Suite. Verifies: 1. Zone…, TestAnomalyBaseline (+1 more)

### Community 51 - "Test Config Completeness"
Cohesion: 0.25
Nodes (15): check(), _engine_tree(), late_bound_names(), main(), module_level_names(), test_config_completeness.py — no setting may exist only as an inline default.…, Named individually so a regression on any of them fails with a message that…, It writes thousands of JPEGs. Defaulting it ON would surprise a production run;… (+7 more)

### Community 52 - "Analytics"
Cohesion: 0.13
Nodes (14): _as_gallery(), build_identity_dossiers(), cross_validate_faces(), cross_validate_reid(), _hsv_embed_single(), occupancy_timeline(), One crop -> one HSV color-histogram vector (torso region only)., Diagnostic (v30): where a face embedding exists for BOTH tracks in a pair, face… (+6 more)

### Community 53 - "Test Covisibility Dedupe"
Cohesion: 0.25
Nodes (14): TIER A: how many people came through the door, WITHOUT depending on identity…, tier_a_crossings(), _c(), Co-visibility stops a GROUP collapsing into one arrival. Ground truth…, The core guarantee: overlapping lifetimes => two people., The other side: no overlap => still treated as one person., The four abreast ids measured on CAM.112, with their real lifetimes. Both fixes…, Too tight and one person's re-minted id splits into two. (+6 more)

### Community 54 - "Clock"
Cohesion: 0.19
Nodes (11): check_dst_span(), check_frame_clock(), describe(), localize(), clock.py — what time is it, really, and is that clock trustworthy? WHY THIS…, Does this run cross a DST transition? -> findings. Checked on the SPAN, not…, Is frame_index/fps the real clock? -> (source, worst_drift_pct, findings).…, The file we picked, the file we decoded, and the file the clock came from must… (+3 more)

### Community 55 - "Graph Fusion"
Cohesion: 0.27
Nodes (12): compute_pairwise_cost(), FusionWeights, Any, graph_fusion.py — P2 Graph-Based Identity Fusion (Min-Cost Flow / Global…, Solve global tracklet identity association graph. NOTE: this is greedy…, Configurable weights for the multi-evidence identity graph edge costs., Compute total fusion edge cost between track_a and track_b (lower = more likely…, solve_graph_fusion() (+4 more)

### Community 56 - "Test Phase A"
Cohesion: 0.17
Nodes (11): Any, Validate camera ground plane calibration quality., Execute complete fail-loud pre-flight validation suite. Raises…, Validate polygon definitions in zones configuration., run_preflight_checks(), validate_ground_plane(), validate_polygons(), Valid configuration should pass preflight without errors. (+3 more)

### Community 57 - "Test Preflight Entry Line"
Cohesion: 0.24
Nodes (14): Validate every entry line: shape, length, open ends, mask collision., validate_entry_line(), _cfg(), The entry-line preflight must catch what actually broke, not pass by design.…, The measured CAM.112 failure: entry line inside 'plant area mask'., 20 absolute px is 0.5% of a 4K frame -- effectively no check., A line floating mid-frame has floor to walk around at both ends., The overlap is FATAL at MASK_REQUIRE_MOTION=0 and survivable above it. Without… (+6 more)

### Community 58 - "Test Mixed Id Sort"
Cohesion: 0.26
Nodes (13): minute_summaries(), One row per window: who was present, what happened. Deterministic (non-LLM)…, track_sort_key(), check(), main(), test_mixed_id_sort.py — an enrolled staff NAME must not crash the stitcher. WHY…, Ties previously resolved by whatever order the dict happened to give, so two…, Ties on sim, with mixed id types, sorted the way the fix does it. (+5 more)

### Community 59 - "Log"
Cohesion: 0.15
Nodes (9): get_logger(), human(), _PathFilter, Configure the run's logger. Safe to call twice; the second call is a no-op…, A logger for one module. setup() need not have run — an unconfigured logger…, Kept for handlers that want it explicitly; the record factory below is what…, setup(), Capture (+1 more)

### Community 60 - "Validity"
Cohesion: 0.15
Nodes (4): DetectorCanary, validity.py — "I did not observe" is not "I observed nothing". THE PRINCIPLE…, Motion happened and the detector saw nobody. How often, and in a row? A model…, test_validity.py — "I did not observe" must never read as "I observed nothing".…

### Community 61 - "Test Exposure"
Cohesion: 0.35
Nodes (13): check(), flat(), main(), test_exposure.py — badly-lit frames must be corrected, not just IR ones. WHY…, A frame can average 'fine' while half of it is pinned pure white — the blown-…, The cap is the whole reason this is safe to enable by default., test_clahe_accepts_a_clip_limit(), test_clip_limit_is_capped() (+5 more)

### Community 62 - "Test Gt Validate"
Cohesion: 0.26
Nodes (12): _moving(), The GT validator must refuse the exact file that fooled us twice. gt.txt: 600…, A believable sequence: everyone drifts, shapes are person-like., test_accepts_a_believable_sequence(), test_entry_events_are_recognised(), test_flags_no_entry_events_and_fails_when_required(), test_refuses_copy_forward(), test_refuses_frozen_tracks() (+4 more)

### Community 63 - "Ablate Slit Rules"
Cohesion: 0.18
Nodes (12): drop_reversals(), load_truth(), main(), One runnable check: the reversal filter must drop the trailing event of a close…, -> (false_timestamps, tolerance, [(name, (t0,t1), truth_count, is_tuned)])., -> dict of the four numbers a rule is judged on. `held_out` is the only recall…, Remove the SECOND of any two opposite-direction events closer than min_gap_s. A…, Keep events whose blob sits before the marble/LED end of the line. Raises when… (+4 more)

### Community 64 - "Threshold Expiry"
Cohesion: 0.26
Nodes (13): check(), current_value(), function_hash(), load_registry(), main(), -> one verdict per registered constant, worst first. EXPIRED a dependency's…, Record that `spec` was (re-)measured now, against today's dependencies. Refuses…, One runnable check: a changed dependency must expire its constant. (+5 more)

### Community 65 - "Analytics"
Cohesion: 0.17
Nodes (12): apply_face_veto(), _canon_priority_key(), _cosine(), _gallery_sim(), _handoff_appearance_contradicts(), match_track_to_dossiers(), v47: veto a PURE-SPATIAL hand-off/stationary merge when both tracks carry body…, ACTIVE veto (v32, redesigned) — reverses a merge when face evidence confidently… (+4 more)

### Community 66 - "Validity"
Cohesion: 0.19
Nodes (5): Time we actually looked at usable pixels. SKIPPED_IDLE counts as observed: the…, Contiguous stretches actually observed, for clipping any metric., -> [(level, message)]. What a human must be told about this run., Observed time vs elapsed time, and why the difference exists. Every metric that…, ValidityLedger

### Community 67 - "Test Slit Wiring"
Cohesion: 0.20
Nodes (7): Which door lines mean "arrived at the venue" -> set of names. A venue has more…, venue_entry_lines(), line_arrivals_by_id(), First inward door crossing per non-staff track. -> {track_id: t} Returns {} —…, The wired slit estimator must pick the VENUE door in SOURCE coordinates. Two…, test_picks_the_venue_door_not_the_first_key(), test_venue_doors.py — a dining door is not an arrival. Every crossing has…

### Community 68 - "  Main  "
Cohesion: 0.27
Nodes (11): One line for the run log. Green means the numbers may be published., verdict_line(), compare(), A/B delta AND the config diff. Without the config diff a delta is not…, _ab(), main(), _profile(), kevacv CLI — the tools, without opening Kaggle. python -m kevacv score <gt.txt>… (+3 more)

### Community 69 - "Config"
Cohesion: 0.20
Nodes (8): apply_run_config(), _flatten(), _implicit_key(), config.py — every tracking-critical constant, in one importable place. WHY THIS…, Resolve `analysis.foo` -> `FOO` when this module defines a scalar FOO. WHY THIS…, Read a run-config yaml and apply it to the engine module. `target` defaults to…, test_apply_run_config_refuses_rather_than_guessing(), test_foot_anchor.py — the foot anchor must sit where the foot is. WHY THIS…

### Community 70 - "Engine"
Cohesion: 0.20
Nodes (12): _boxmot_dev(), build_online_tracker(), get_reid_embedder(), Real online Re-ID — supports BotSort and BoostTrack. No silent fallback: if the…, Single source of truth for a ReID weights path, generalized (v28) to work for…, Back-compat alias — OSNet is now the v28 FALLBACK backbone, kept as its own…, boxmot wants '0'/'1'/'cpu' — NOT 'cuda' or 'cuda:1' (parse_device raises)., Appearance embedder (lazy, cached), v28 chain: CLIP-ReID (heavy, stronger) ->… (+4 more)

### Community 71 - "Test Det Classes"
Cohesion: 0.29
Nodes (10): _det_classes(), _detector_has_head_class(), TRUE only for a real 2-class person/head detector. Hard gate: on stock COCO…, Class ids to ASK the detector for. #23 (2026-08-20). Measured on the p0verify…, _M, #23: ask the detector only for the classes we use. The p0verify run showed…, test_coco_bicycle_is_never_mistaken_for_a_head(), test_real_person_head_model_is_recognised() (+2 more)

### Community 72 - "Eval Harness"
Cohesion: 0.18
Nodes (5): eval_harness.py — PHASE 2. Turn "we think it improved" into a number.…, rows: iterable of (frame, id, x, y, w, h)., write_mot(), test_gt_kit.py — the ground-truth kit on a synthetic package. The worst bug…, test_label_tool.py — the local label tool must be self-contained and its export…

### Community 73 - "Eval Harness"
Cohesion: 0.18
Nodes (7): _hungarian(), iou_matrix(), a, b: lists of (x, y, w, h) -> IoU matrix. Empty-safe., Maximise-free wrapper. scipy if present, else a small O(n^3) fallback so the…, HOTA / DetA / AssA / IDF1 / MOTA + everything needed to debug them. gt, pr:…, score_sequence(), Validate eval_harness.py against cases whose answers are known BY CONSTRUCTION.…

### Community 74 - "Log"
Cohesion: 0.18
Nodes (7): Append a finished stage to the run ledger. The stage is still on the stack…, A named span of work. Returned by stage(); counters attach to it., Record a number against this stage. Adds if numeric, else sets., Time a stage and log its start, end and counters. with stage("detect") as st:…, _record(), _stack(), Stage

### Community 75 - "Test Calibration Sample"
Cohesion: 0.27
Nodes (10): calibrate_appearance_threshold(), Diagnostic (v30): measures THIS run's actual same-person vs different-person…, _case(), The 'CLIP cannot separate at any threshold' verdict may be an artefact.…, Two tracks 14px apart, `gap_s` seconds apart, with DIFFERENT appearance -- the…, Documents the behaviour that made the verdict circular., _same_n(), test_a_genuine_short_reappearance_is_still_admitted() (+2 more)

### Community 76 - "Analytics"
Cohesion: 0.20
Nodes (4): OccupancyRecorder, Collects per-frame zone occupancy and consolidates it into events. add(t, zone,…, -> list of dicts {track_id, role, zone, t_in, t_out, duration}.…, test_mixed_id_events.py — a matched staff face must not kill the run. track_id…

### Community 77 - "Conftest"
Cohesion: 0.20
Nodes (9): fixture, hookimpl, pytest_configure(), pytest_runtest_call(), Make check() failures actually fail the test. 50 test files in this directory…, Run in the CALL phase, so a failure reports as a failure. A teardown fixture…, Snapshot kevacv's constants BEFORE collection imports any test module.…, Give every test the pristine constants, whatever ran before it. Restored on the… (+1 more)

### Community 78 - "Test Phase A"
Cohesion: 0.20
Nodes (6): PreflightValidationError, Exception, Raised when pre-flight validation fails on critical schema, line, or geometry…, test_phase_a.py — Phase A Verification Test Suite. Verifies: 1. Preflight…, Test global association solver on matching vs non-matching tracklets., TestGraphFusion

### Community 79 - "Test Mask Motion Veto"
Cohesion: 0.20
Nodes (5): test_mask_motion_veto.py — a mask must suppress the plant, not the guest. WHY…, The rule as implemented in engine._drop_masked, replayed here. A spot is…, veto(), test_v68 — certifies the IR debounce + staff-pin uniqueness patch. Runs the…, run_veto()

### Community 80 - "Test Stationary Gap Cap"
Cohesion: 0.29
Nodes (7): _case(), _merged(), The stationary merge tier had no time bound. Measured on output/p0classfix2…, Two fragments 14px apart, `gap_s` seconds apart — the measured shape of the…, test_cap_blocks_the_long_gap(), test_cap_still_allows_a_genuine_short_reappearance(), test_without_a_cap_a_two_minute_gap_still_merges()

### Community 81 - "Doorcount"
Cohesion: 0.33
Nodes (8): dedupe(), door_events(), Door-local counting: count the way an eye does. WHY THIS EXISTS The pipeline…, track_id -> (t_first, t_last), for the co-visibility test., Every time ANY track crosses INTO the door band moving inward. The first…, Fold ONE person's fragments into one count -- without folding a GROUP. CO-…, _spans(), main()

### Community 82 - "Ground Plane"
Cohesion: 0.31
Nodes (5): Rebuild the ground plane from a run dict. The GroundPlane object cannot survive…, _rebuild_plane(), Build from the fitted line h(y) = a*y + b (pixel height of a standing person…, Exact: >=4 image points and their real floor coordinates in metres., Prefer an exact homography if the zones JSON carries one: "ground_points": […

### Community 83 - "Test Duplicate Keys"
Cohesion: 0.36
Nodes (8): duplicate_yaml_keys(), Keys that appear TWICE under the same mapping. YAML keeps the LAST one. This is…, A duplicated yaml key must stop the run, not silently change it. On 2026-08-20…, test_clean_file_is_clean(), test_comments_and_blank_lines_are_not_keys(), test_every_shipped_config_is_clean(), test_finds_a_duplicate(), test_same_key_at_different_depths_is_not_a_duplicate()

### Community 84 - "Preflight"
Cohesion: 0.25
Nodes (8): _entry_lines_of(), _line_intersect(), _point_in_poly(), preflight.py — P1 Pre-Flight Validation Engine (Fail-Loud Architecture). WHY…, Every entry line in a zone config, whatever shape it is written in. #22…, Check if line segment p1-p2 intersects line segment q1-q2., Ray casting; poly is [(x, y), ...]., _validate_one_entry_line()

### Community 85 - "Test Pipeline"
Cohesion: 0.22
Nodes (3): Capture, make_analyse(), test_pipeline.py — the orchestrator, exercised without a GPU. The whole point…

### Community 86 - "Test V56 Phase3"
Cohesion: 0.25
Nodes (4): at(), box_at(), Tests for patch_v56_phase3.py — metres instead of pixels. The ground plane…, floor position -> image foot point

### Community 87 - "Embed Kevacv"
Cohesion: 0.31
Nodes (8): collect(), main(), embed_kevacv.py — re-embed the kevacv package into the notebook bootstrap. WHY…, -> {filename: source}. __init__ first so the dict reads sensibly., Emit each module as r'''...''' — the format test_v56_phase6 parses. json.dumps…, Replace the _KEVACV_SRC literal, leaving header and tail untouched., render(), splice()

### Community 88 - "Track Health"
Cohesion: 0.39
Nodes (8): load_tracks(), main(), network_scale(), How much of a source pixel survives to the network, and where it goes., Where tracks are BORN and DIE, and whether that is physically sensible. A…, report_lifecycle(), report_pixel_height(), _self_check()

### Community 89 - "Test Graph Fusion Plane Trust"
Cohesion: 0.25
Nodes (3): Plane, test_graph_fusion_plane_trust.py — the hard veto must not run on a guess.…, Stand-in for GroundPlane. `mode` is the only thing under test.

### Community 90 - "Test Label Kit Plan"
Cohesion: 0.39
Nodes (6): _plan(), entry_label_kit is the only path to the ground truth this project has never…, _run_dir(), test_plan_keeps_a_quiet_window(), test_plan_never_emits_the_same_window_twice(), test_plan_still_holds_out_the_busiest_window()

### Community 91 - "Test Scale Reference"
Cohesion: 0.36
Nodes (6): A plane is accepted or rejected against a KNOWN distance in the room. Bounding…, Boxes for upright people on a flat floor under a known camera., _synthetic_people(), test_a_plane_off_by_60_percent_is_rejected(), test_a_plane_that_matches_the_floor_is_accepted(), test_no_reference_means_no_opinion_not_a_pass()

### Community 92 - "Test V56 Phase2"
Cohesion: 0.25
Nodes (5): align(), mk_run(), Tests for patch_v56_phase2.py — the ground-truth harness wiring. The metric…, Same rule the cell uses, in isolation., Synthetic frame_log: `busy` is a list of (t0,t1,n_people).

### Community 94 - "Export Tensorrt"
Cohesion: 0.36
Nodes (7): _iou(), main(), export_tensorrt.py — compile best.pt into a TensorRT engine for THIS GPU. WHAT…, Do the two models agree? -> True/False, and say how they differ. SMOKE TEST…, Do the two models agree ON REAL FOOTAGE, and where do they disagree? WHY THIS…, verify(), verify_real()

### Community 95 - "Id Reuse Audit"
Cohesion: 0.36
Nodes (7): audit(), load(), main(), One runnable check: a coast-and-return is not reuse; a teleport is., -> {track_id: [(frame_index, centre_x, centre_y), ...]} sorted by frame., -> (summary dict, [worst offenders]). A "stale gap" is one long enough that…, _selftest()

### Community 96 - "Analytics"
Cohesion: 0.43
Nodes (4): _apply_global_tracklet_pass(), _GtaUF, link_tracklets_global(), Additive: run the global linker over the GREEDY groups. Builds one super-…

### Community 97 - "Test Crossing Prior Stability"
Cohesion: 0.29
Nodes (3): confirm_crossings(), Drop crossings that a person did not COMMIT to. -> (kept, dropped) THE PROBLEM,…, test_crossing_prior_stability.py — a crossing needs a track that was already…

### Community 98 - "Resilience"
Cohesion: 0.38
Nodes (6): _empty_cache(), _is_oom(), resilience.py — an 8-hour run must not die at hour 7. WHY THIS EXISTS DET_BATCH…, True for a CUDA out-of-memory, without importing torch to find out., Apply `fn(chunk_of_items)` in batches, surviving out-of-memory. Yields results…, run_batched()

### Community 100 - "Test No Missing Names"
Cohesion: 0.33
Nodes (4): test_no_missing_names.py — nothing may reference a name that does not exist.…, Names a `from .x import *` brings in, resolved by importing x., star_imported(), unresolved()

### Community 102 - "Make Trainset"
Cohesion: 0.43
Nodes (6): img_size(), load_mot(), main(), Frames whose boxes are byte-identical to the previous frame's. These are…, JPEG dimensions without pulling in PIL., static_frames()

### Community 103 - "Mask Prototype"
Cohesion: 0.43
Nodes (6): get_segmenter(), load_mot(), main(), mask_box(), A segmenter, or a clear message about what to install., Tight bounding box of a boolean mask, or None if empty.

### Community 104 - "Test Absurd Size"
Cohesion: 0.33
Nodes (3): kept(), test_absurd_size.py — the D0 cap, and why it must not be relaxable. The audit's…, The D0 predicate, exactly as engine._filter_chain applies it.

### Community 106 - "Test Engine Compiles"
Cohesion: 0.33
Nodes (3): _first_lines(), test_engine_compiles.py — engine.py must at least COMPILE. WHY THIS EXISTS On…, (first assignment line, first read line) per name, top-level stmts only.

### Community 107 - "Test Eval Scorer"
Cohesion: 0.33
Nodes (3): test_eval_scorer.py — the HOTA scorer must be trustworthy BEFORE anyone spends…, A clean synthetic sequence. Self-contained: this test must run on a laptop with…, synth()

### Community 108 - "Test Phantom Gates Measured"
Cohesion: 0.33
Nodes (3): passes(), test_phantom_gates_measured.py — the phantom gates must admit the real statue.…, All three gates, as phantom_regions applies them.

### Community 109 - "Test Track Buffer Units"
Cohesion: 0.33
Nodes (3): boxmot_seconds(), test_track_buffer_units.py — the lost-track memory must be the number of…, Exactly what boxmot does with the number we hand it.

### Community 111 - "Audit Frames"
Cohesion: 0.53
Nodes (5): _boxes(), iou(), load(), main(), Frame records have drifted in shape across builds. Accept what exists rather…

### Community 112 - "Extract Pipeline"
Cohesion: 0.47
Nodes (5): defs_in(), load_cells(), main(), extract_pipeline.py — lift the notebook's cells into an importable package. WHY…, Top-level def/class names, by regex — the source may not import here.

### Community 113 - "Run Night"
Cohesion: 0.47
Nodes (5): collect_overrides(), inject(), main(), run_night.py — headless production runner for the video-surv notebook. The…, Insert the override cell right after the CONFIG cell.

### Community 114 - "Config"
Cohesion: 0.40
Nodes (3): All of the above as one object, so a run can carry its settings around instead…, The audit, printed from the object that will actually be used — so it cannot…, TrackingConfig

### Community 117 - "Depth Prototype"
Cohesion: 0.60
Nodes (4): get_depth_model(), load_mot(), main(), Depth Anything V2, or a clear install line.

### Community 118 - "Ingest Db"
Cohesion: 0.90
Nodes (4): connect(), ingest(), main(), selftest()

### Community 119 - "Engine"
Cohesion: 0.50
Nodes (4): _heads_without_person(), Heads whose centre falls in no person box — each one is a missed person., Add a person box for every head the detector left orphaned. WHY THIS IS THE…, _recover_bodies_from_heads()

### Community 120 - "Run"
Cohesion: 0.67
Nodes (3): PYTHONIOENCODING, run.sh script, usage()

### Community 123 - "Test Guestlink"
Cohesion: 0.50
Nodes (3): Check the whole-night guest logic: a guest who leaves and comes back an hour…, cos=None -> a fresh identity. cos=x -> the same person seen again with cosine…, vec()

### Community 126 - "Check Closure"
Cohesion: 0.67
Nodes (3): load_crossings(), main(), [{t, direction, line, track_id}] from whatever the run wrote.

### Community 127 - "Make Label Package"
Cohesion: 0.83
Nodes (3): build(), find_inputs(), main()

### Community 128 - "Score Points"
Cohesion: 0.83
Nodes (3): load_boxes(), load_points(), main()

## Knowledge Gaps
- **3 isolated node(s):** `PYTHONIOENCODING`, `FakeFrame`, `make_entry_labels.sh script`
  These have ≤1 connection - possible missing edges or undocumented components.
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `process_video()` connect `Venue Dataset Collector` to `Phantom Detection Filters`, `Static Phantom Suppression`, `Staff vs Customer Zone Rules`, `Engine Frame Preprocessing`, `Per-Frame Observation Rows`, `Re-ID Separability Calibration`, `Identity Memory & Duplicate Split`, `A/B Merge Policy Evaluation`, `Fisheye Dewarp`, `Detection Funnel Accounting`, `Analytics: Tracks to People`, `Zone & Plot Helpers`, `Tiled Native-Res Detection`, `Camera Health`, `Test Robust Ground Plane`, `Analytics`, `Pipeline`, `Profiling`, `Capabilities`, `Analytics`, `Test Covisibility Dedupe`, `Analytics`, `Validity`, `Test Slit Wiring`, `  Main  `, `Engine`, `Test Det Classes`, `Test Calibration Sample`, `Analytics`, `Ground Plane`, `Analytics`, `Test Crossing Prior Stability`, `Resilience`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Why does `GroundPlane` connect `Ground Plane` to `Analytics`, `Test Robust Ground Plane`, `Engine Frame Preprocessing`, `Per-Frame Observation Rows`, `Identity Memory & Duplicate Split`, `A/B Merge Policy Evaluation`, `Analytics`, `Venue Dataset Collector`, `Calibrate Plane`, `Test Phase A`, `Test Ground Plane`, `Analytics: Tracks to People`, `Ground Plane`, `Test V56 Phase3`, `Reid Engine`, `Test Scale Reference`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `arrival_tier()` connect `Confidence Tiers & Review Queue` to `Pipeline`, `Run Scorecard & Occupancy Reconciliation`, `Answers`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `GroundPlane` (e.g. with `_GtaUF` and `_IdentityMemory`) actually correct?**
  _`GroundPlane` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PYTHONIOENCODING`, `FakeFrame`, `make_entry_labels.sh script` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Confidence Tiers & Review Queue` be split into smaller, more focused modules?**
  _Cohesion score 0.051189400782896716 - nodes in this community are weakly interconnected._
- **Should `Phantom Detection Filters` be split into smaller, more focused modules?**
  _Cohesion score 0.05405405405405406 - nodes in this community are weakly interconnected._