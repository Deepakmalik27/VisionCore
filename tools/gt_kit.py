"""gt_kit.py — Phase B ground-truth kit: correct, don't draw.

Cell 22 already exports label packages (images + predictions.txt in MOT 1.1 +
manifest + instructions) and kevacv.eval_harness already scores HOTA/DetA/
AssA/IDF1/MOTA. The missing piece was labor: HOW_TO_LABEL.txt asks a human to
draw a box around every person in every frame (~960 frames for one 2-min
window at 8 fps). This kit converts our own predictions into a CVAT-importable
annotation archive, so labelling becomes CORRECTING — delete the phantoms, add
the missed, fix the switched ids. That is 5-10x less clicking for the same
gt.txt.

HONESTY NOTE (read once): pre-seeding anchors the labeller toward what the
pipeline already believes. The rules below counter it (pass 1 is add/delete
with prediction boxes treated as suspect), and a biased-but-existing gt.txt
beats the current state — which is nothing measured at all. If a window scores
suspiciously perfectly, relabel it cold.

Commands
    python gt_kit.py seed  <package_dir | package.zip>
        -> writes <pkg>/cvat_seed.zip (CVAT "MOT 1.1" import archive)
    python gt_kit.py score <package_dir> <gt.txt> [--freeze]
        -> HOTA/DetA/AssA/IDF1/MOTA + errors CSV; --freeze saves baseline
    python gt_kit.py all   <eval_dir>
        -> finds every package with a matching *_gt.txt, scores per condition
    python gt_kit.py compare <before_score.json> <after_score.json>

# ponytail: no SAM2 propagation — CVAT's own interpolation + this seed covers
# the 2-min windows; add SAM2 only if labelling ever exceeds an afternoon.
"""
import json
import sys
import zipfile
from pathlib import Path

# the package lives one level UP from tools/, so inserting HERE put
# tools/ on the path and `import kevacv` failed — this script has
# never been runnable from a clean checkout.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kevacv.eval_harness import (compare, dump_errors_csv, explain, load_mot,
                                 save_baseline, score_conditions,
                                 score_sequence)


def _pkg_dir(arg):
    p = Path(arg)
    if p.suffix == ".zip":
        dest = p.with_suffix("")
        if not dest.exists():
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
            # zip may contain the package dir itself one level down
            inner = [d for d in dest.iterdir() if d.is_dir()]
            if not (dest / "predictions.txt").exists() and len(inner) == 1:
                dest = inner[0]
        p = dest
    if not (p / "predictions.txt").exists():
        sys.exit(f"not a label package (no predictions.txt): {p}")
    return p


def seed(arg):
    """predictions.txt -> cvat_seed.zip in CVAT's MOT 1.1 import layout:
    gt/gt.txt (frame,id,x,y,w,h,not_ignored,class,visibility) + gt/labels.txt.
    """
    pkg = _pkg_dir(arg)
    pred = load_mot(pkg / "predictions.txt")
    n_imgs = len(list((pkg / "images").glob("*.jpg")))
    frames = sorted(pred)
    if frames and (frames[0] < 1 or frames[-1] > max(n_imgs, 1)):
        print(f"  !! prediction frames span {frames[0]}..{frames[-1]} but the "
              f"package has {n_imgs} images — seed would misalign. Aborting.")
        sys.exit(1)

    rows = [f"{fr},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1"
            for fr in frames for tid, x, y, w, h in pred[fr]]
    out = pkg / "cvat_seed.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("gt/gt.txt", "\n".join(rows))
        z.writestr("gt/labels.txt", "person")
    n_ids = len({tid for fr in pred.values() for tid, *_ in fr})
    print(f"cvat_seed.zip written: {len(rows)} boxes, {n_ids} track ids, "
          f"{len(frames)} annotated frames of {n_imgs} images")
    print(f"""
HOW TO USE THE SEED  ({pkg.name})
{'=' * 66}
 1. cvat.ai -> Create task -> name it {pkg.name}
    label: person   (exactly that, lowercase — must match the seed)
 2. Upload ALL {n_imgs} files from images/ (default filename ordering).
 3. Open the created task -> Actions -> Upload annotations
    -> format "MOT 1.1" -> pick cvat_seed.zip.
    Every frame now shows OUR pipeline's boxes and track ids.
 4. CORRECT, in two passes:
      pass 1 — presence: DELETE every box on a non-person (plant, mirror,
               reflection); ADD a box for every missed person. Treat every
               seeded box as a suspect, not a fact.
      pass 2 — identity: one track id per real human for the whole window.
               Someone who leaves and returns keeps their original id;
               a box that jumps to a different person gets split.
 5. Export task dataset -> "MOT 1.1" -> the gt.txt inside is your ground
    truth. Rename it {pkg.name}_gt.txt and drop it next to the package.
 6. python gt_kit.py score {pkg} {pkg.name}_gt.txt --freeze
{'=' * 66}
If the annotation upload errors, fall back to HOW_TO_LABEL.txt (label from
scratch) — the scoring path is identical either way.""")
    return out


def _sanity(gt, pr, pkg):
    """The classic silent killers: frame numbering offset and empty overlap."""
    if not gt or not pr:
        sys.exit("  !! empty gt or predictions — nothing to score")
    g0, g1, p0, p1 = min(gt), max(gt), min(pr), max(pr)
    if g0 == 0 and p0 == 1:
        print("  !! gt frames start at 0, predictions at 1 — shifting gt +1 "
              "(CVAT sometimes exports 0-based)")
        gt = {f + 1: v for f, v in gt.items()}
        g0, g1 = g0 + 1, g1 + 1
    overlap = len(set(gt) & set(pr))
    if overlap < 0.5 * len(gt):
        print(f"  !! only {overlap}/{len(gt)} gt frames overlap predictions "
              f"(gt {g0}..{g1}, pred {p0}..{p1}) — check you exported the "
              f"right task before trusting this score")
    man = pkg / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        print(f"  window {m.get('window_clock')} · "
              f"{'IR' if m.get('is_infrared') else 'colour'} · "
              f"pred ids {m.get('n_pred_ids')}")
    return gt


def score(pkg_arg, gt_path, freeze=False):
    pkg = _pkg_dir(pkg_arg)
    gt = load_mot(gt_path)
    pr = load_mot(pkg / "predictions.txt")
    gt = _sanity(gt, pr, pkg)
    res = explain(score_sequence(gt, pr), label=pkg.name)
    dump_errors_csv(gt, pr, res, pkg / "errors.csv")
    print(f"  per-frame error detail -> {pkg / 'errors.csv'}")
    if freeze:
        man = pkg / "manifest.json"
        cfg = (json.loads(man.read_text()).get("config")
               if man.exists() else None)
        bp = pkg / "baseline_score.json"
        save_baseline(res, bp, config=cfg, label=pkg.name)
        print(f"  BASELINE FROZEN -> {bp}")
        print(f"  every future change: score again, then "
              f"`python gt_kit.py compare {bp} <new_score.json>`")
    return res


def score_all(eval_dir):
    ed = Path(eval_dir)
    pairs = {}
    for pred in sorted(ed.glob("*/predictions.txt")):
        pkg = pred.parent
        gt = next((g for g in (pkg.parent / f"{pkg.name}_gt.txt",
                               pkg / f"{pkg.name}_gt.txt",
                               pkg / "gt.txt") if g.exists()), None)
        if gt:
            pairs[pkg.name] = (gt, pred)
        else:
            print(f"  (no gt yet for {pkg.name} — run `seed`, label, return)")
    if not pairs:
        sys.exit("no (gt, predictions) pairs found")
    return score_conditions(pairs, out_dir=ed / "scores")


def main():
    # eval_harness.explain() reports HOTA/DetA/AssA, the worst frames and every
    # id switch through _log.info. Run as a script, nothing configures logging,
    # so the whole scorecard went to a handler that discards it -- `gt_kit.py
    # score` appeared to work and printed two cosmetic lines. Anyone seeing
    # that would conclude scoring was broken and go back to having no numbers
    # at all, which is exactly the state this kit exists to end.
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)

    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == "seed" and len(a) == 2:
        seed(a[1])
    elif cmd == "score" and len(a) >= 3:
        score(a[1], a[2], freeze="--freeze" in a)
    elif cmd == "all" and len(a) == 2:
        score_all(a[1])
    elif cmd == "compare" and len(a) == 3:
        compare(json.loads(Path(a[1]).read_text()),
                json.loads(Path(a[2]).read_text()),
                label_a=Path(a[1]).stem, label_b=Path(a[2]).stem)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
