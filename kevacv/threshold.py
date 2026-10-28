"""threshold.py — choose a merge threshold by what mistakes actually COST.

WHY THIS EXISTS
    find_optimal_threshold() sweeps for the best balanced accuracy:

        score = 1.0 - (false_reject + false_accept) / 2.0

    That weights the two mistakes equally. For this pipeline they are not
    remotely equal:

        FALSE REJECT   one person becomes two fragments.
                       Guest count +1. Everything else still correct.

        FALSE ACCEPT   two people become one person.
                       Guest count -1, AND their dwell times merge, AND their
                       zone visits merge, AND a customer can inherit a staff
                       member's desk minutes, AND greet latency is computed
                       from the wrong arrival. One bad merge corrupts several
                       unrelated numbers at once.

    On CAM.112 the balanced sweep suggested 0.340 at balanced accuracy 0.658.
    Adopting it would have merged aggressively on a signal that is barely
    better than a coin flip. CALIBRATION_AUTO_APPLY=False was the correct
    instinct; this module is that instinct written down as arithmetic.

THE PRINCIPLE
    Pick the threshold that minimises EXPECTED COST, not error count:

        cost(t) = fa_cost * FA(t) + fr_cost * FR(t)

    With fa_cost > fr_cost the optimum moves UP — toward conservative merging,
    toward leaving fragments unmerged rather than fusing strangers. That is the
    correct direction for a report whose headline metric (desk coverage) does
    not need identity at all, and whose guest count is allowed a ±10% band.

HONEST LIMIT
    No threshold rescues a separable-at-0.658 signal. This module tells you the
    least-bad point and, just as importantly, prints how bad the least-bad
    point is — so "we tuned the threshold" can never be mistaken for "we fixed
    the problem".
"""
from __future__ import annotations

# One wrong merge corrupts roughly this many times more downstream numbers
# than one wrong split. Deliberately a round, arguable number: it is a policy
# choice, not a measurement, and it belongs somewhere a human can see it.
DEFAULT_FA_COST = 8.0
DEFAULT_FR_COST = 1.0


def _rates(same_sims, diff_sims, t):
    """(false_reject_rate, false_accept_rate) at threshold t."""
    fr = sum(1 for s in same_sims if s < t) / len(same_sims)
    fa = sum(1 for s in diff_sims if s >= t) / len(diff_sims)
    return fr, fa


def cost_weighted_threshold(same_sims, diff_sims, fa_cost=DEFAULT_FA_COST,
                            fr_cost=DEFAULT_FR_COST, lo=0.0, hi=1.0, step=0.01):
    """-> (threshold, report). The threshold minimising expected cost.

    Ties break toward the HIGHER threshold: when two points cost the same,
    the more conservative one is chosen, because the cost model already says
    which error we would rather make.
    """
    if not same_sims or not diff_sims:
        return None, {"note": "insufficient data — need both same and diff sims",
                      "n_same": len(same_sims or []), "n_diff": len(diff_sims or [])}

    n = int(round((hi - lo) / step))
    best_t, best_cost = None, float("inf")
    # How good the SIGNAL is, and where we choose to OPERATE on it, are two
    # different questions. A conservative cost policy deliberately accepts a
    # worse error count; reading signal quality off the chosen point would make
    # every cautious policy look like bad data.
    best_balanced = 0.0
    curve = []
    for i in range(n + 1):
        t = round(lo + i * step, 4)
        fr, fa = _rates(same_sims, diff_sims, t)
        cost = fa_cost * fa + fr_cost * fr
        curve.append((t, fr, fa, cost))
        best_balanced = max(best_balanced, 1.0 - (fr + fa) / 2.0)
        if cost <= best_cost:          # <= so later (higher) t wins ties
            best_t, best_cost = t, cost

    fr, fa = _rates(same_sims, diff_sims, best_t)
    same_sorted = sorted(same_sims)
    diff_sorted = sorted(diff_sims)

    def _p(arr, q):
        return arr[min(len(arr) - 1, max(0, int(q * len(arr))))]

    balanced = 1.0 - (fr + fa) / 2.0
    # When the signal cannot separate and a wrong merge is expensive, the
    # arithmetic optimum is "merge NOTHING" — the sweep walks the threshold up
    # until no pair clears it. That is a real answer, not a bug, but returning
    # 0.99 with no comment would look like a tuned threshold. Say it plainly.
    degenerate = fr >= 0.99
    return best_t, {
        "threshold": best_t,
        "degenerate_no_merge": degenerate,
        "expected_cost": round(best_cost, 4),
        "false_reject_rate": round(fr, 4),
        "false_accept_rate": round(fa, 4),
        "balanced_accuracy": round(balanced, 4),          # at the chosen point
        "best_balanced_accuracy": round(best_balanced, 4),  # of the signal itself
        "fa_cost": fa_cost, "fr_cost": fr_cost,
        "n_same": len(same_sims), "n_diff": len(diff_sims),
        "same_p10": round(_p(same_sorted, 0.10), 4),
        "same_p50": round(_p(same_sorted, 0.50), 4),
        "diff_p50": round(_p(diff_sorted, 0.50), 4),
        "diff_p90": round(_p(diff_sorted, 0.90), 4),
        "separable": bool(_p(same_sorted, 0.10) > _p(diff_sorted, 0.90)),
        "curve": curve,
    }


def verdict(report, usable_balanced=0.80):
    """Is this signal good enough to merge on at ALL?

    A threshold chosen on a signal that cannot separate is still a bad
    threshold. This says so out loud rather than letting a tuned number imply
    a solved problem.
    """
    if not report or report.get("threshold") is None:
        return "NO DATA — cannot choose a threshold"
    # judge the SIGNAL, not the operating point we chose on it
    ba = report.get("best_balanced_accuracy", report["balanced_accuracy"])
    if report.get("degenerate_no_merge"):
        return (f"MERGE NOTHING is cheapest — at a {report['fa_cost']}:"
                f"{report['fr_cost']} cost ratio no threshold on this signal "
                f"beats simply never merging on appearance (balanced accuracy "
                f"{ba}). This is the arithmetic agreeing with the physics: use "
                f"hand-off, stationary and topology evidence, and let unmatched "
                f"fragments stay separate.")
    if report["separable"]:
        return (f"SEPARABLE — same-person p10 ({report['same_p10']}) is above "
                f"different-person p90 ({report['diff_p90']}). A threshold "
                f"between them is a real decision boundary.")
    if ba >= usable_balanced:
        return (f"OVERLAPPING but usable — balanced accuracy {ba}. Merge, but "
                f"keep the hard constraints (co-visibility, topology, role) "
                f"doing the heavy lifting.")
    return (f"NOT SEPARABLE — balanced accuracy {ba} at the best possible "
            f"point. No threshold fixes this. Appearance must not be the "
            f"primary merge signal on this footage; prefer physical evidence "
            f"(hand-off, stationary, topology) and accept fragments.")


def compare(same_sims, diff_sims, current, fa_cost=DEFAULT_FA_COST,
            fr_cost=DEFAULT_FR_COST):
    """Current threshold vs the cost-optimal one, as a decision aid.

    Never returns 'apply this'. It returns what each choice costs, because
    changing a merge threshold without a scored A/B is how a pipeline quietly
    starts fusing people.
    """
    best_t, rep = cost_weighted_threshold(same_sims, diff_sims, fa_cost, fr_cost)
    if best_t is None:
        return rep
    cur_fr, cur_fa = _rates(same_sims, diff_sims, current)
    cur_cost = fa_cost * cur_fa + fr_cost * cur_fr
    return {
        "current": {"threshold": current, "false_reject_rate": round(cur_fr, 4),
                    "false_accept_rate": round(cur_fa, 4),
                    "expected_cost": round(cur_cost, 4)},
        "suggested": {"threshold": best_t,
                      "false_reject_rate": rep["false_reject_rate"],
                      "false_accept_rate": rep["false_accept_rate"],
                      "expected_cost": rep["expected_cost"]},
        "cost_delta": round(rep["expected_cost"] - cur_cost, 4),
        "direction": ("more conservative" if best_t > current
                      else "more aggressive" if best_t < current else "unchanged"),
        "verdict": verdict(rep),
    }


def describe(report, width=46):
    """An ASCII cost curve. Seeing the shape stops a single number implying
    more precision than the data supports — a flat basin means the exact
    threshold barely matters, which is itself the finding."""
    if not report or report.get("threshold") is None:
        return report.get("note", "no data")
    curve = report["curve"]
    lo = min(c[3] for c in curve)
    hi = max(c[3] for c in curve)
    span = max(hi - lo, 1e-9)
    L = [f"COST CURVE  (fa_cost={report['fa_cost']} fr_cost={report['fr_cost']})",
         f"  chosen t={report['threshold']}  cost={report['expected_cost']}  "
         f"FR={report['false_reject_rate']}  FA={report['false_accept_rate']}"]
    for t, fr, fa, cost in curve:
        if round(t * 100) % 10:
            continue
        fill = int(width * (cost - lo) / span)
        mark = " <- chosen" if abs(t - report["threshold"]) < 1e-9 else ""
        L.append(f"  {t:.2f} |{'#' * fill}{'.' * (width - fill)}| {cost:.3f}{mark}")
    L.append(f"  {verdict(report)}")
    return "\n".join(L)
