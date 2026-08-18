"""kevacv — the parts of the video pipeline that are not the notebook.

WHY THIS PACKAGE EXISTS
    The notebook is 34 cells and ~6000 lines, patched by matching literal
    strings. Anything that lives only inside it cannot be imported, cannot be
    tested outside Kaggle, and gets a second copy every time it is embedded as
    a cell. Everything in here is ordinary Python: importable, testable on a
    laptop, and with exactly one copy on disk.

CELL 5 AND CELL 7 ARE NOW HERE
    They used to be notebook-only, held back until a HOTA number existed so a
    later regression would be attributable. They were extracted before that
    number arrived, so the attribution problem was solved a different way:

        analytics.py  <- Cell 5. tests/test_analytics_extraction.py execs the
                         CELL and demands identical output from the module —
                         intervals, thresholds, the full merge (mapping, edges,
                         tier counts, blocked count) and both calibration
                         distributions.
        engine.py     <- Cell 7. tests/test_engine_extraction.py proves all 39
                         top-level definitions are CHARACTER-IDENTICAL to the
                         cell, and runs the pure helpers both ways.

    Those tests are the missing baseline. Both cells still exist in the
    notebook and are still the thing that runs; delete them, and delete the
    matching test, only once a scored run says the module path agrees.

    engine.py is NOT imported here: it needs torch/ultralytics/boxmot, which a
    laptop may not have. `import kevacv.engine` explicitly.

THE PUBLIC SURFACE
    log        stage / banner       one nested timeline for the whole run
    geometry   GroundPlane          pixels -> metres on the floor
    health     CameraHealth         is this still the same camera view?
    filters    implausible_size_mask / static_track_ids   phantom removal
    config     load_profile etc     per-camera / per-venue settings as DATA
    metrics    score_sequence etc   HOTA / DetA / AssA, A/B comparison
    triage     plan_segments        analyse where the people are, account for the rest
    arrivals   arrivals_from_regions  an arrival count that survives a badly drawn line
    phantoms   phantom_regions        static false positives whose track ids churn
    learn_zones learn_entry_zones     find the door in the data, don't draw it

    python -m kevacv --help          the same tools from a shell
"""
from .log import banner, get_logger, setup as setup_logging, stage
from .answers import Answer, answer_set, to_report_rows
from .derive import enrich, id_confidence, observed_windows, staff_contacts
from .drive import fetch_chunk, select as select_chunk
from .pipeline import bind_runtime, preflight, resolve_identities, run_camera
from .helpers import classify_zones, load_zone_config, mmss, wall
from .validity import (DetectorCanary, ValidityLedger, frame_validity)
from .clock import (check_dst_span, check_frame_clock, localize,
                    parse_start, verify_provenance)
from .resilience import Checkpoint, run_batched
from . import seams
from .arrivals import arrivals_from_regions, cross_check, entry_zone_coverage
from .report_slim import (coverage_strip, people_csv, summary_txt,
                          write_slim_outputs)
from .topology import (doors_from_endpoints, doors_from_zones,
                       reappearance_verdict, veto_pairs)
from .threshold import cost_weighted_threshold, verdict as threshold_verdict
from .merge_ab import ab_topology, greedy_union
from .reid_calibration import calibrate, compare_to_legacy
# ── extracted from the notebook (Cell 5). Behaviour is pinned against the
# cell itself by tests/test_analytics_extraction.py, so the refactor cannot
# quietly move a number. ────────────────────────────────────────────────────
from .config import DEFAULT as TRACKING_DEFAULTS
from .config import TrackingConfig
# engine.py imports torch/ultralytics/boxmot, which a laptop may not have.
# Import it lazily via kevacv.engine so the rest of the package stays usable.
from .analytics import (OccupancyRecorder, calibrate_appearance_threshold,
                        clip_to, complement_intervals, covered_windows,
                        entered_count, merge_fragmented_tracks,
                        merge_intervals, minute_summaries, occupancy_timeline,
                        reception_absence, remap_events, seated_count,
                        total_duration, waited_over)
from .tiled import cost_estimate, height_roi, slice_grid, tiled_predict
# ── merged in from the computer_vision line (two sessions worked in parallel;
# that fork carried the older notebook but the more advanced package) ────────
from .preflight import PreflightValidationError, run_preflight_checks
from .graph_fusion import FusionWeights, solve_graph_fusion
from .reid_engine import ReIDEmbeddingExtractor
from .geometry_calibration import fit_robust_ground_plane
from .tracker_wrapper import TrackerWrapper
from .anomaly_baseline import ZoneAnomalyDetector
from .dataset_collector import DatasetCollector
from .learn_zones import (learn_dwell_zones, learn_entry_zones,
                         to_zone_config, track_endpoints)
from .phantoms import (drop_phantom_dets, in_phantom, phantom_regions)
from .camera_health import CameraHealth, verdict_line
from .detect_filters import (BODY_ASPECT, drop_tracks, implausible_size_mask,
                             mirrored_pair_ids, protected_ids,
                             rigid_track_ids, static_min_life_by_id,
                             static_track_ids)
# NOT `from .build_id import compute as build_id` — that binds the NAME
# build_id to a function and shadows the MODULE kevacv.build_id, so
# `from kevacv import build_id; build_id.compute()` breaks. Exactly the kind
# of silent import-shadowing this package's __init__ warns about elsewhere.
from .build_id import compute as compute_build_id, describe as describe_build
from .funnel import DetectionFunnel
from .eval_harness import (compare, dump_errors_csv, explain, iou_matrix,
                           load_mot, save_baseline, score_conditions,
                           score_sequence, write_mot)
from .ground_plane import PERSON_H_M, GroundPlane, synth_camera
from .triage import coverage_report, miss_risk, plan_segments
from .venue_profile import (DEFAULTS, describe, infer_entry_direction,
                            load_profile, local_clock, validate, write_template)

__version__ = "0.6.0"          # tracks the pipeline phase, not semver

__all__ = [
    "setup_logging", "get_logger", "stage", "banner",
    "run_camera", "preflight", "resolve_identities", "bind_runtime",
    "Answer", "answer_set", "to_report_rows",
    "enrich", "observed_windows", "staff_contacts", "id_confidence",
    "fetch_chunk", "select_chunk",
    "load_zone_config", "classify_zones", "mmss", "wall",
    "ValidityLedger", "DetectorCanary", "frame_validity",
    "parse_start", "localize", "check_dst_span", "check_frame_clock",
    "verify_provenance", "Checkpoint", "run_batched", "seams",
    "mirrored_pair_ids",
    "GroundPlane", "PERSON_H_M", "synth_camera",
    "CameraHealth", "verdict_line",
    "implausible_size_mask", "static_track_ids", "static_min_life_by_id",
    "rigid_track_ids", "compute_build_id", "describe_build", "DetectionFunnel",
    "protected_ids", "drop_tracks",
    "BODY_ASPECT",
    "plan_segments", "coverage_report", "miss_risk",
    "arrivals_from_regions", "cross_check", "entry_zone_coverage",
    "summary_txt", "people_csv", "coverage_strip", "write_slim_outputs",
    "reappearance_verdict", "veto_pairs", "doors_from_zones",
    "doors_from_endpoints", "cost_weighted_threshold", "threshold_verdict",
    "ab_topology", "greedy_union", "calibrate", "compare_to_legacy",
    "tiled_predict", "slice_grid", "height_roi", "cost_estimate",
    "phantom_regions", "in_phantom", "drop_phantom_dets",
    "learn_entry_zones", "learn_dwell_zones", "to_zone_config",
    "track_endpoints",
    "load_profile", "validate", "describe", "local_clock",
    "infer_entry_direction", "write_template", "DEFAULTS",
    "score_sequence", "explain", "compare", "score_conditions",
    "load_mot", "write_mot", "dump_errors_csv", "save_baseline", "iou_matrix",
    "PreflightValidationError", "run_preflight_checks",
    "FusionWeights", "solve_graph_fusion",
    "ReIDEmbeddingExtractor", "fit_robust_ground_plane",
    "TrackerWrapper", "ZoneAnomalyDetector", "DatasetCollector",
    # analytics, extracted from Cell 5
    "TrackingConfig", "TRACKING_DEFAULTS",
    "merge_fragmented_tracks", "calibrate_appearance_threshold",
    "OccupancyRecorder", "occupancy_timeline", "remap_events",
    "entered_count", "seated_count", "waited_over", "reception_absence",
    "minute_summaries", "merge_intervals", "total_duration",
    "complement_intervals", "clip_to", "covered_windows",
]
