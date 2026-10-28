"""kevacv CLI — the tools, without opening Kaggle.

    python -m kevacv score  <gt.txt> <predictions.txt>
    python -m kevacv ab     <before_score.json> <after_score.json>
    python -m kevacv profile <out.json> [camera-id]
    python -m kevacv view   <reference.jpg> <current.jpg>

`score` is the one that matters: it turns a labelled slice and a prediction
file into HOTA / DetA / AssA on any machine, so a change can be judged without
a GPU session.
"""
import json
import sys
from pathlib import Path


def _score(argv):
    from .eval_harness import explain, load_mot, score_sequence
    if len(argv) < 2:
        return _usage("score needs <gt.txt> <predictions.txt>")
    gt, pr = load_mot(argv[0]), load_mot(argv[1])
    if not gt:
        print(f"ground truth {argv[0]} is empty or unparseable")
        return 1
    res = explain(score_sequence(gt, pr),
                  label=f"{Path(argv[0]).name} vs {Path(argv[1]).name}")
    if len(argv) > 2:
        from .eval_harness import save_baseline
        save_baseline(res, argv[2], label=Path(argv[1]).stem)
    return 0


def _ab(argv):
    from .eval_harness import compare
    if len(argv) < 2:
        return _usage("ab needs <before.json> <after.json>")
    a = json.loads(Path(argv[0]).read_text())
    b = json.loads(Path(argv[1]).read_text())
    compare(a, b, Path(argv[0]).stem, Path(argv[1]).stem)
    return 0


def _profile(argv):
    from .venue_profile import write_template
    if not argv:
        return _usage("profile needs an output path")
    p = write_template(argv[0], argv[1] if len(argv) > 1 else "")
    print(f"wrote {p} — edit it and drop it beside the video as "
          f"profile_<video-stem>.json")
    return 0


def _view(argv):
    import cv2
    from .camera_health import CameraHealth, verdict_line
    if len(argv) < 2:
        return _usage("view needs <reference-image> <current-image>")
    ref, cur = cv2.imread(argv[0]), cv2.imread(argv[1])
    if ref is None or cur is None:
        print("could not read one of the images")
        return 1
    res = CameraHealth.from_frame(ref).check(cur)
    print(verdict_line(res))
    return 0 if res["valid"] else 2


CMDS = {"score": _score, "ab": _ab, "profile": _profile, "view": _view}


def _usage(msg=""):
    if msg:
        print(f"error: {msg}\n")
    print(__doc__)
    return 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        return _usage()
    fn = CMDS.get(argv[0])
    return fn(argv[1:]) if fn else _usage(f"unknown command {argv[0]!r}")


if __name__ == "__main__":
    sys.exit(main())
