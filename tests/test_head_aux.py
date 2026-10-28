"""test_head_aux.py — the auxiliary head detector must not become the detector.

WHAT THIS GUARDS
    On 2026-08-26 models/best.pt (CrowdHuman person+head) was A/B'd as the
    PRIMARY detector against hand-labelled ground truth and lost badly:

        yolo11x.pt   HOTA 0.7884   DetA 0.8501   recall 0.8857
        best.pt      HOTA 0.2558   DetA 0.3339   recall 0.4436

    Its head class, measured separately, recalls 73.3% — useful. So the design
    is: bodies from yolo11x, heads from best.pt, heads consumed and discarded
    inside the filter chain.

    Every check here defends one way that could silently turn back into the
    regression above.

Run: python tests/test_head_aux.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


ENG = (ROOT / "kevacv" / "engine.py").read_text(encoding="utf-8")
CFG = (ROOT / "kevacv" / "config.py").read_text(encoding="utf-8")

print()
print("=" * 74)
print("  it is OFF unless asked for")
print("=" * 74)
import kevacv.config as C  # noqa: E402
check(getattr(C, "HEAD_DETECTOR", "missing") is None,
      "HEAD_DETECTOR defaults to None",
      "an opt-in costs one forward pass per frame; a default costs it always")
check('"analysis.head_detector": "HEAD_DETECTOR"' in CFG,
      "it is reachable from a yaml, so an A/B needs no code edit")

print()
print("=" * 74)
print("  THE BICYCLE TRAP — the reason _has_head and _head_aux are separate")
print("=" * 74)
# _det_classes(has_head) decides which class ids to ASK the primary model for.
# On stock COCO weights class 1 is 'bicycle'. If the aux flag ever reached it,
# every parked bike would be read as a head and invented into a person.
import re  # noqa: E402
det_calls = re.findall(r"_det_classes\(([^)]*)\)", ENG)
check(det_calls and all("_head_aux" not in c for c in det_calls),
      "_det_classes is NEVER passed the aux flag",
      f"call sites: {sorted(set(det_calls))}")
check("_has_head = _detector_has_head_class(model)" in ENG,
      "_has_head still describes the PRIMARY model only")
check("if not _detector_has_head_class(_head_model):" in ENG,
      "a non-head model handed to HEAD_DETECTOR is REFUSED, not trusted")

print()
print("=" * 74)
print("  heads must never reach the tracker")
print("=" * 74)
# _split_person_head returns (persons, heads) and only persons continue. The
# aux heads are substituted at that same point, so they live and die inside
# _filter_chain -- they cannot be tracked, counted, or ReID'd as people.
i = ENG.index("def _filter_chain")
j = ENG.index("def ", i + 10)
body = ENG[i:j]
check("dets, _heads = _split_person_head(dets, _has_head)" in body,
      "persons and heads are separated inside _filter_chain")
check("if aux_heads is not None and len(aux_heads):" in body,
      "aux heads REPLACE the primary's heads at that same point")
check("classes=[1]" in ENG,
      "the aux model is asked for class 1 ONLY",
      "its person class scored recall 0.44 — merging it back is the regression")

print()
print("=" * 74)
print("  the consumers actually fire for aux heads")
print("=" * 74)
check("if ENABLE_MERGED_SPLIT and (_has_head or _head_aux):" in ENG,
      "merged-box splitting runs on aux heads",
      "two heads in one box is two people — the overlap case")
check("if ENABLE_HEAD_RECOVERY and (_has_head or _head_aux):" in ENG,
      "occlusion recovery runs on aux heads")
check(ENG.count("_detect_heads(frame)") == 3,
      "all THREE _filter_chain call sites feed it",
      f"found {ENG.count('_detect_heads(frame)')}")

print()
print("=" * 74)
print("  it degrades to the old behaviour, never to a crash")
print("=" * 74)
check("_log.warning(\"aux head detection failed on one frame" in ENG,
      "a per-frame failure warns and returns None, not raises")
check("continuing with " in ENG and "bodies only" in ENG,
      "a model that will not load falls back to the previous behaviour")

print()
print("=" * 74)


def test_script_level_checks_passed():
    assert not FAILED, f"head-aux checks failed: {FAILED}"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for x in FAILED:
            print(f"    - {x}")
        sys.exit(1)
    print("  ALL PASS")
    print("=" * 74)
