# Success Criteria — What "Good Enough" Means

> **This document exists because we've never written down which business decision this report drives.**
> Without it, optimisation is unbounded — we can always make a number slightly better, and never know when to stop.

---

## Primary Business Questions

These are the questions the system must answer. Each has a different accuracy requirement because the business cost of being wrong is different.

### Q1: "Was the desk covered?"
**Priority: CRITICAL**

> Was there a staff member at the reception desk during operating hours?

| Metric | Target | Current | How we verify |
|--------|--------|---------|---------------|
| Desk coverage detection | ≥ 90% accuracy | Untested (no ground truth) | Staff zone dwell + gallery match |
| Gap detection (desk empty > X min) | ≥ 85% accuracy | Untested | Zone occupancy tracking |
| False "desk empty" alerts | < 10% | Untested | Staff briefly stepping away ≠ empty |

**Why this is achievable:** Staff stay in one zone for long periods. Zone dwell is the strongest signal we have — identity doesn't even need to be perfect.

---

### Q2: "How fast were people greeted?"
**Priority: HIGH**

> From arrival at the door to first staff interaction — how long did each guest wait?

| Metric | Target | Current | How we verify |
|--------|--------|---------|---------------|
| Greet time (entry → staff proximity) | ± 30 seconds | PROXY tier | Ground truth labelling |
| Ungreeted guests detected | ≥ 75% recall | Untested | Entry crossing + no staff contact |

**Why this is hard:** "Greeted" is a semantic event — a staff member walked up and spoke. We approximate it with proximity (staff within 1.5m for ≥ 3s). That's a PROXY, not a measurement.

> The greet metric will always be PROXY tier with pure computer vision. The honest upgrade is VLM (Phase 7d): send a 10s clip → "staff approached and led them in." Different model, different cost.

---

### Q3: "How many guests tonight?"
**Priority: MEDIUM**

> Total unique visitors entering the venue during this chunk.

| Metric | Target | Current | How we verify |
|--------|--------|---------|---------------|
| Entry count | ± 5% of true count | Untested | HOTA after labelling |
| Double-counting rate | < 10% | Unknown | Ablation (Ph4.5) |

**±5% is hard.** What it requires, at minimum:
- Domain fine-tune on YOUR frames (kills the phantoms, closes 10-20pt domain gap)
- Ablation (Ph4.5) — remove identity mechanisms that fight each other
- Possibly a second camera or overhead sensor if occlusion is too high

**The physics ceiling:** From one oblique security camera, ±5% means counting 40 guests and reporting 38–42. Industry systems achieve this with purpose-built overhead counters, not general security cameras. We will try with software first — the domain fine-tune alone could get us there if detection is the main error source.


---

### Q4: "What happened during the night?" (Report)
**Priority: MEDIUM**

| Metric | Target | Current |
|--------|--------|---------|
| Report completeness | Covers Q1-Q3 + peak hours + anomalies | REPORT.md exists |
| Report honesty | Every number has a tier label (EXACT/PROXY/WEAK) | ✅ Built (Ph3) |
| Report reliability | Never publishes if camera moved or data invalid | ✅ Built (Ph4) |

---

## What This System CANNOT Do (from one oblique camera)

| Capability | Why not | What would fix it |
|-----------|---------|-------------------|
| Exact headcount (± 2%) | Occlusion, ReID on B&W, oblique angle | Overhead dedicated sensor |
| True interaction detection | Proximity ≠ conversation | VLM or audio |
| Customer identity across days | No persistent face DB (privacy) | Loyalty app check-in |
| Full venue coverage | Single camera, limited FOV | Multi-camera (Phase 8) |
| Real-time alerts | Batch architecture | Architecture change |

---

## Exit Criteria — When Is Each Phase "Done"?

| Phase | Exit Criteria |
|-------|--------------|
| Ph1-4, 6, 7 | ✅ 433 checks green, clean-room verified |
| Ph8 Privacy | Patch applied, face scope restricted, tests pass |
| Ph5a Triage | 80%+ saving on empty hours, verified on real data |
| Ground Truth | HOTA ≥ 0.40 on 2-min labelled slice |
| Domain Fine-tune | HOTA improves ≥ 5 points over best.pt |
| Ph4.5 Ablation | Each identity mechanism measured; bad ones removed |
| "Production Ready" | Q1 ≥ 90%, Q2 ≥ 75% recall, Q3 ± 15%, report never lies |

---

## RATIFIED — Prabh, 2026-08-06

1. **Priority: ALL THREE questions are first-class.** Desk coverage, greet
   latency and guest count must each pass their own bar — none is sacrificed
   to tune another. Tie-break rule when a change helps one and hurts another:
   the change is rejected unless the gt scores show the helped metric gains
   more than the hurt one loses, and desk coverage is never allowed to regress
   (it is the cheapest to keep and the one a GM acts on daily).
2. **Guest count: ±10% from the start.** This bar is deliberately aggressive —
   it will likely FAIL until the venue fine-tune and the Phase C tracker
   verdicts land. That is the intended behavior: the report shows the target
   as failing honestly; the goalposts do not move.
3. **Latency: overnight batch.** The current architecture is the correct one.
   Near-real-time alerts are a separately funded phase (dedicated GPU box),
   not a requirement on this pipeline.
4. **Horizon: per-night.** One report per service night, matching how the
   venue and the forecasting pipeline already operate.

### The pass/fail sheet (what every gt-scored run is judged against)

| Metric | Bar | Measured by |
|---|---|---|
| Desk coverage accuracy | ≥ 90%, never regresses | staff-zone ledger vs labelled window |
| Ungreeted-guest recall | ≥ 75% (PROXY tier, stated as such) | labelled window |
| Guest count | within ±10% of hand count | tier-A count vs labelled window |
| Tracking floor | HOTA ≥ 0.40 busy window (≥ 0.55 goal) | gt_kit score |
| IR allowance | night window may run ≤ 15% below daylight HOTA | score_conditions |
| Report honesty | every number tier-tagged; never publishes on a moved camera | built ✅ |

Verification ritual per milestone run: `gt_kit.py score` + `frame_ritual.py`
review + this sheet. A run without the sheet filled is an anecdote.
