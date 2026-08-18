"""capabilities.py — what is actually working, what quietly fell back.

WHY THIS EXISTS
    A pipeline this size degrades rather than fails. Every subsystem has a
    fallback, each fallback is individually reasonable, and none of them is
    an error — so the run completes, the report is produced, and nothing says
    that half the evidence was never gathered.

    The first real run made the point three times over, and none of the three
    appeared as a failure:

      * onnxruntime had no CUDA provider, so InsightFace silently ran on CPU.
        One warning, buried in a model-loading dump, 40 lines above the
        progress bar.
      * the perspective fit implied a camera height of 1.1 m, so every metric
        gate was quietly working in nonsense units.
      * one gallery photo had no detectable face, so a staff member was
        enrolled from a single shot instead of two.

    Each was visible SOMEWHERE. None was visible TOGETHER, and the question
    an operator actually asks — "is this run trustworthy?" — had no single
    place to be answered.

WHAT THIS IS NOT
    Not a health check that blocks the run. A degraded run is often still the
    right run to have; the point is that the degradation is stated, next to
    its consequence, before an hour of GPU time is spent on it.
"""
from __future__ import annotations

OK, DEGRADED, MISSING = "OK", "DEGRADED", "MISSING"
_ORDER = {MISSING: 0, DEGRADED: 1, OK: 2}


class CapabilityLedger:
    """One row per subsystem: what it is, how it is running, what that costs."""

    def __init__(self):
        self.rows = []

    def record(self, name, status, detail="", impact=""):
        """impact: what this run CANNOT tell you because of this status.

        Required for anything not OK. A degradation reported without its
        consequence is a line people learn to scroll past — the consequence is
        the only part that changes a decision.
        """
        self.rows.append({"name": name, "status": status,
                          "detail": str(detail), "impact": str(impact)})
        return self

    def ok(self, name, detail=""):
        return self.record(name, OK, detail)

    def degraded(self, name, detail, impact):
        return self.record(name, DEGRADED, detail, impact)

    def missing(self, name, detail, impact):
        return self.record(name, MISSING, detail, impact)

    @property
    def counts(self):
        c = {OK: 0, DEGRADED: 0, MISSING: 0}
        for r in self.rows:
            c[r["status"]] = c.get(r["status"], 0) + 1
        return c

    def trustworthy(self):
        """Nothing missing and nothing degraded."""
        c = self.counts
        return c[MISSING] == 0 and c[DEGRADED] == 0

    def describe(self, width=78):
        c = self.counts
        L = ["=" * width,
             f"  CAPABILITIES — {c[OK]} ok, {c[DEGRADED]} degraded, "
             f"{c[MISSING]} missing",
             "=" * width]
        # worst first: the reason to read this block is at the top of it
        for r in sorted(self.rows, key=lambda r: (_ORDER[r["status"]], r["name"])):
            mark = {OK: "  ok  ", DEGRADED: " DEGR ", MISSING: " MISS "}[r["status"]]
            L.append(f"  [{mark}] {r['name']:<22} {r['detail']}")
            if r["impact"]:
                L.append(f"            -> {r['impact']}")
        L.append("  " + "-" * (width - 4))
        if self.trustworthy():
            L.append("  Every subsystem is running as designed.")
        else:
            L.append("  Rows above marked DEGR or MISS still produce numbers.")
            L.append("  They produce them from less evidence than the design")
            L.append("  assumes, and nothing downstream will say so again.")
        L.append("=" * width)
        return "\n".join(L)

    def as_dict(self):
        return {"counts": self.counts, "trustworthy": self.trustworthy(),
                "rows": list(self.rows)}


def disk_findings(out_dir, duration_s, eff_fps, want_render=True,
                  want_eval=False, want_dataset=False, eval_max_frames=None,
                  frame_w=1280):
    """Will this run fit on the disk? -> [(level, message)]

    WHY THIS IS A PREFLIGHT AND NOT A try/except
        A full disk killed a run at 95% of a 28-minute analysis. Everything had
        been computed; nothing had been written. The exports are now non-fatal,
        so that exact failure cannot recur — but "the run survived and silently
        produced no video" is only a better outcome, not a good one.

        The sizes are knowable before a single frame is decoded: frames, fps
        and JPEG quality are all fixed at that point. So say it up front, in
        gigabytes, while changing your mind is still free.

    Estimates are deliberately rough and rounded UP. Being told you need 12 GB
    and using 9 wastes nothing; the reverse costs half an hour.
    """
    import shutil
    out = []
    try:
        free_gb = shutil.disk_usage(str(out_dir)).free / 1e9
    except Exception as e:
        return [("INFO", f"could not check free disk space: {e}")]

    n_frames = max(1, int(duration_s * max(eff_fps, 0.1)))
    # JPEG size scales with PIXELS, not width. Raising ANALYSIS_MAX_W from
    # 1280 to 1920 is 2.25x the area and therefore 2.25x the disk — a fixed
    # per-frame estimate would under-report by that factor and fill the volume
    # exactly as before, which is the failure this check exists to prevent.
    px_scale = (float(frame_w) / 1280.0) ** 2
    need = 0.0
    parts = []
    if want_render:
        # proxy JPEGs, quality 72 -> ~120 KB each at 1280x720
        gb = n_frames * 120e3 * px_scale / 1e9
        need += gb
        parts.append(f"render proxy {gb:.1f} GB ({n_frames} frames)")
    if want_eval:
        n = min(n_frames, eval_max_frames or n_frames)
        gb = n * 190e3 * px_scale / 1e9          # quality 92 -> ~190 KB
        need += gb
        parts.append(f"eval frames {gb:.1f} GB ({n} frames)")
    if want_dataset:
        gb = (n_frames / 40.0) * 190e3 * px_scale / 1e9
        need += gb
        parts.append(f"dataset {gb:.1f} GB")
    need += 1.0                        # the annotated video and report files

    msg = (f"disk: {free_gb:.1f} GB free, this run needs about "
           f"{need:.1f} GB at {int(frame_w)}px wide — " + "; ".join(parts))
    if free_gb < need:
        out.append(("ERROR", msg + ". NOT ENOUGH. Drop --eval-export, narrow "
                             "it with --eval-window, or free space first."))
    elif free_gb < need * 1.5:
        out.append(("WARN", msg + ". Tight — under 50% headroom."))
    else:
        out.append(("INFO", msg))
    return out


def onnx_providers():
    """Which execution providers onnxruntime actually has. -> (available, has_gpu)

    InsightFace asks for CUDAExecutionProvider and falls back to CPU with a
    UserWarning if it is absent — which is what `pip install onnxruntime`
    (rather than onnxruntime-gpu) gets you. Face detection then runs perhaps
    20x slower, which on a face-sparse camera means the face path contributes
    almost nothing while appearing to be enabled.
    """
    try:
        import onnxruntime as ort
        avail = list(ort.get_available_providers())
    except Exception as e:
        return [f"(onnxruntime unavailable: {e})"], False
    return avail, any("CUDA" in p or "Tensorrt" in p for p in avail)
