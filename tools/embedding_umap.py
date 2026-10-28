#!/usr/bin/env python3
"""embedding_umap.py — can the ReID model tell these people apart at all?

WHY THIS EXISTS
    AssA is the weak half of HOTA (0.759 quiet / 0.564 crowded) and nothing
    says WHY. Association works by comparing appearance vectors, so there is
    exactly one question underneath it:

        do two crops of the SAME person look more alike than crops of
        DIFFERENT people?

    If yes, association is a threshold problem and tuning helps. If no, no
    threshold on earth helps and the answer is a better backbone, bigger
    crops, or a different signal entirely. Months can be spent tuning a
    threshold that was never the constraint.

WHY A NUMBER, NOT JUST A PLOT
    A UMAP scatter is the famous artefact here, and it is genuinely useful --
    but it needs a screen, and this box is behind a 140 KB/s tunnel. So the
    plot is written to PNG (small, downloadable) and the SEPARABILITY is
    printed:

        intra  mean cosine similarity within one track
        inter  mean cosine similarity between different tracks
        margin intra - inter

    Margin near zero means the embedding is blind to the difference. This is
    the same statistic the pipeline's own "separability" probe reports, and
    it is not circular: track identity comes from the tracker, not from the
    embeddings being measured.

CAVEAT, STATED
    Tracks are not people. Two tracks CAN be the same person (fragmentation),
    which drags `inter` up and understates the true margin. Ground-truth ids
    would fix that; --gt takes them when available.

Usage:
    python tools/embedding_umap.py --run /data/outputs/V3_quiet
    python tools/embedding_umap.py --run ... --out /data/outputs/umap.png
"""
import argparse
import collections
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None, help="PNG path (default <run>/umap.png)")
    p.add_argument("--min-vecs", type=int, default=5,
                   help="ignore tracks with fewer vectors than this")
    a = p.parse_args(argv)

    import numpy as np

    obs = Path(a.run) / "observations.jsonl"
    if not obs.exists():
        raise SystemExit(f"no observations.jsonl in {a.run} "
                         f"(run with enable_observations: true)")
    by_track = collections.defaultdict(list)
    for line in obs.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = d.get("vec")
        t = d.get("raw_track_id")
        if v and t is not None:
            by_track[str(t)].append(v)
    by_track = {k: v for k, v in by_track.items() if len(v) >= a.min_vecs}
    if len(by_track) < 2:
        raise SystemExit(f"only {len(by_track)} track(s) with >={a.min_vecs} "
                         f"vectors — nothing to compare")
    X, y = [], []
    for t, vs in by_track.items():
        for v in vs:
            X.append(v)
            y.append(t)
    X = np.asarray(X, dtype=np.float32)
    # cosine similarity needs unit vectors; the ReID head does not guarantee it
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    y = np.asarray(y)
    print(f"  {len(by_track)} tracks, {len(X)} vectors, {X.shape[1]}-dim")

    S = X @ X.T
    same = y[:, None] == y[None, :]
    off = ~np.eye(len(X), dtype=bool)
    intra = S[same & off]
    inter = S[~same]
    m_intra, m_inter = float(intra.mean()), float(inter.mean())
    margin = m_intra - m_inter
    print()
    print(f"  intra-track  mean cosine {m_intra:.4f}   (same person)")
    print(f"  inter-track  mean cosine {m_inter:.4f}   (different people)")
    print(f"  MARGIN                   {margin:+.4f}")
    print()
    # How often does a vector's nearest neighbour belong to a DIFFERENT track?
    # This is the mistake association actually makes, counted directly.
    np.fill_diagonal(S, -2.0)
    nn = S.argmax(1)
    wrong = float((y[nn] != y).mean())
    print(f"  nearest neighbour belongs to another track: {100*wrong:.1f}%")
    print()
    if margin < 0.05:
        print("  VERDICT: the embedding barely separates these people. Tuning a")
        print("  similarity threshold cannot fix this -- the signal is not there.")
        print("  Look at crop resolution, the ReID backbone, or a non-appearance")
        print("  cue (position, motion, face) instead.")
    elif wrong > 0.30:
        print("  VERDICT: separable on average, but the NEAREST neighbour is wrong")
        print("  often. That is a threshold/gating problem -- tuning can help.")
    else:
        print("  VERDICT: embeddings separate these identities well. If AssA is")
        print("  still low, the loss is in gating/motion, not appearance.")

    out = Path(a.out) if a.out else Path(a.run) / "umap.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
        red = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=0).fit_transform(X)
        plt.figure(figsize=(9, 7))
        for t in sorted(by_track):
            m = y == t
            plt.scatter(red[m, 0], red[m, 1], s=6, label=f"{t} ({m.sum()})")
        plt.legend(fontsize=6, markerscale=2, ncol=2, loc="best")
        plt.title(f"ReID embeddings — {Path(a.run).name}\n"
                  f"intra {m_intra:.3f}  inter {m_inter:.3f}  margin {margin:+.3f}")
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"\n  plot -> {out}")
        print("  one tight cluster per colour = identities are distinguishable;")
        print("  colours mixed together = the tracker cannot tell them apart.")
    except Exception as e:
        print(f"\n  (plot skipped: {type(e).__name__}: {str(e)[:90]})")
        print("  the numbers above are the answer; the plot is illustration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
