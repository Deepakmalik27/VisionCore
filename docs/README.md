# keva_vision — start here

Orientation for anyone (human or agent) picking this up. Last verified
**2026-08-25** against a real 20-minute Delilah run.

---

## 1. What runs, and what does not

```
tools/run_pipeline.py  ->  kevacv.pipeline.run_camera  ->  7 stages
```

That is the ONLY entry point. `run.sh` is a thin wrapper over it.

**The notebook is gone.** `archive/notebooks/pipeline.ipynb` is kept for
provenance only. As of 2026-08-25 there are **zero executable references to any
`.ipynb`** anywhere on the run path (AST-verified across 44 reachable modules).
Docstrings still say "EXTRACTED FROM notebook Cell N" — that is deliberate
history, not debt: it records *why* each module is shaped as it is.

## 2. Where things live

| Path | What |
|---|---|
| `kevacv/` | the pipeline, 52 modules. `engine.py` and `pipeline.py` are the big two |
| `tools/` | entry point, scoring, calibration, labelling, audits |
| `config/` | run configs. **`cam112.yaml` is the measured baseline — do not edit for one run** |
| `zones/` | zone polygons + entry lines, in SOURCE pixels |
| `tests/` | 387 passing, 1 known failure (below) |
| `docs/analysis/` | the .txt/.md analysis history — audit.txt, idea.txt, audut.md etc. |
| `gt.txt` | **DATA, not a doc.** 19 code references. Stays at the repo root |
| `archive/` | the notebook era. Nothing here is on the run path |

## 3. Config: never edit for one run

Any scalar in `kevacv/config.py` is settable from a run-config yaml **by
convention** — `analysis.<lower_name>` resolves to `<UPPER_NAME>` via
`config._implicit_key`. It does not need an entry in `RUN_CONFIG_KEYS`.

```yaml
analysis:
  reid_max_gap_s: 1200
```

`apply_run_config` logs every value it moved and on whose authority. A duplicate
yaml key raises rather than silently taking the last one.

Configs that exist:
- `cam112.yaml` — the measured baseline. Every historical number came from this.
- `cam112_gpu.yaml` — L40S-sized: `imgsz` 1920, `yolo11x-pose`, `pose_stride` 4.
  `det_batch` deliberately stays 24 (measured: 64 was **slower**).
- `cam1_poc.yaml` — a generic corner-shop clip. **Its numbers say nothing about
  Delilah.** Smoke tests only.
- `m_*.yaml` — single-knob ablations.

## 4. What is TRUE as of 2026-08-25 (measured, not assumed)

From a 20-minute run on `7-28-2026 19:30-19:50`, 39,086 observation rows:

**Detection is healthy. The tracker and role assignment are not.**
```
yolo raw                 48,727
carried-object suppress     -207   0.4%
implausible size            -331   0.7%
dead-area mask            -6,731  13.8%   <- the plant mask, the only big filter
tracker (boxmot)          -2,366   4.9%   <- 149 EMPTIED FRAMES = 149 id breaks
SURVIVED                  39,086  19.8% lost
```
WARNING: `docs/analysis/audut.md` reports these as 67.9% and 42.5%. That is a
**100x misread** of the scorecard JSON, whose values are already percentages.
Detection over-suppression is NOT the top problem; identity is.

**Known-broken, with evidence:**

| What | Evidence |
|---|---|
| Ground plane | camera height 4.25 m, **horizon row -303** (geometrically odd). Reports `ok=True`. **CORRECTED 2026-08-25: the metre output is NOT garbage** — `speed_mps` median 0.16, mean 0.41, p99 3.41 m/s, only 1.4% above 3 m/s. Plausible for a lobby. An earlier note here claimed 37 m/s; that was `speed_px_s` misread as m/s. The horizon row is still worth explaining, but do not discard the metres on its account |
| Identity churn | 136 canonical ids for ~6-10 real people. 83% of tracks born AND die mid-scene |
| Cover count | line=2, region=6, slit=**21**. The pipeline says `UNCERTAIN` and is right to |
| Anti-fake-staff guard | reverts staff->customer below 2 visits; with 83% churn, real staff never qualify |
| No ground truth | `HOTA NOT MEASURED`. Every accuracy claim is self-graded |

**The slit-scan deserves attention:** the code's own note records *tracking
counter = 0% held-out recall, slit-scan = 100%*. The slit says 21 covers; the
report headlines 6.

**MEASURED 2026-08-25 — the tracker confirmation bar was the id-switch cause**

A/B on the same 20-min chunk, same zones, ONE knob (`new_track_conf`) plus the
ROLE LOCK. `config/cam112_gpu_nt25.yaml` is the winner.

```
                        BEFORE (0.45)   AFTER (0.25)
tracker dropped          2,365  4.9%      723  1.5%
EMPTIED FRAMES             149              14     <- 91% fewer id breaks
observations written     39,848          41,826
counts.line                  2               6
counts.region                6               6     <- they AGREE
arrival tier          UNCERTAIN       CONFIRMED    <- tier upgraded
desk_covered             30.5%           42.2%     <- ROLE LOCK
desk_gaps                    5               3
staff-guest contacts         2              14
```

WHY: `NEW_TRACK_CONF` gates track CONFIRMATION. Under IR the detector's
confidence falls, an unconfirmed track is never kept, and every reappearance
mints a NEW id -- `track_buffer` is irrelevant if the track was never confirmed.
Ladder on the 60 s clip: emptied frames 49 (0.45) -> 18 (0.30) -> 2 (0.25).

THE TRADE, stated honestly: internal churn counters got WORSE while the answers
got better.
```
track_ids       136 -> 165      recovery_gaps  440 -> 792
                                swap_pressure   35 ->  61
```
A lower bar keeps real people alive across blink-outs but also admits more
marginal short-lived tracks. Anyone reading only `track_ids` would conclude the
change was harmful. Read `emptied` and estimator agreement instead.

**6-vs-21 ANSWERED — all three estimators are wrong, in different directions**

`eval/gt_arrivals_CAM_112_20260728_1930.json` is the project's first
externally-checkable number. Method: every IN event the slit proposed was checked
by pulling the doorway crop at t-2,-1,0,+1,+2 s and looking at it. "Empty" means
no body in ANY of five frames, so it is not a timing artefact.

```
slit reported IN         21
  person clearly present  9
  ambiguous               2
  EMPTY doorway +-2s     10   <- false positives

              reported   truth 9-11   verdict
  line             6                  UNDER-counts
  region           6                  UNDER-counts
  slit            21                  OVER-counts ~2x, precision ~0.45
  customer reg.   10                  CLOSEST -- and it is not headlined
```

The slit fails its OWN stated standard. tools/slit_count.py, on the variant it
rejected: *"it manufactured FOUR blobs on a provably empty doorway. A false
positive on an empty scene is the one error a counter must never make."* The
shipped version does it ten times in twenty minutes.

DO NOT promote the slit. DO look at `counts.guests` / the customer registry --
it reported 10-11 while the report headlined 6.

CAVEATS on the ground truth, stated so it is not over-used: one 20-min window,
one camera, arrivals only (not HOTA, not tracking). Only slit-proposed times were
checked, so a crossing no estimator proposed is not in it -- the true count is a
LOWER bound. OUT events unverified.

**STAFF/GUEST RE-ARCHITECTED 2026-08-26 — evidence, not a verdict**

FOUR mechanisms were fighting over one boolean:
```
1 face-gallery sweep     renames the track to "staff3"  (runs AFTER the loop)
2 zone-dwell override    can MANUFACTURE staff from dwell alone
3 anti-manufacture guard reverts staff->customer below 2 visits
4 ROLE LOCK              forces named ids back to staff
```
Result on the 19:30 chunk: `staff3` -- on camera 19.8 of 20 min, plainly the
receptionist -- filed as a GUEST who waited 2m16s and was never greeted. ONE
wrong boolean produced THREE wrong headlines: desk coverage, guest count, and a
fabricated service anomaly.

The 2026-08-20 design call already prescribed the fix: the CV layer must not
decide staff, it must emit confidence. So observations now carry EVIDENCE:
```
staff_evidence: { face_sim, face_name, staff_zone_s, role_hint, role_source }
```
`role_source` distinguishes a gallery match from a zone guess, which was
previously indistinguishable. The intelligence layer applies its own rule and
can RETUNE IT WITHOUT RE-RUNNING 20 MINUTES OF 4K -- today changing
STAFF_MIN_VISITS costs a 35-minute GPU run.

STILL TO DO: `face_sim`/`face_name` are null in-loop because the gallery sweep
runs after it; they need a backfill pass. `staff_zone_s`, `role_hint` and
`role_source` are populated live from the next run onward.

**SLIT FALSE POSITIVES FIXED 2026-08-26 (measured)**

A shape gate in `split_blobs`, with thresholds found by scoring against an
INDEPENDENT signal (YOLO on the doorway crop alone at 2 fps -- shares no code
with the slit):
```
                 BEFORE   AFTER
events             41       19
true positives     20       18
false positives    21        1     <- 95% eliminated
precision        0.49    0.947
IN count           21       12     <- hand-verified truth 9-11
```
`MIN_WIDTH = 8` px (a body spans the line; a 3-4 px sliver does not) and
`MAX_ROWS = 300` (~10 s -- longer is loitering, not crossing). Both log every
drop to the funnel: `too_narrow` 23, `lingers_too_long` 9.

COST: 2 of 20 real crossings lost. Lower MIN_WIDTH first if a venue needs
recall over precision -- it does nearly all the work.

**RUN IDENTITY FIXED 2026-08-26** -- `build_id.make_run_id`. The old formula
gave three different runs over the same chunk the IDENTICAL run_id, and
ingest_obs does delete-then-insert on it, so ingesting the second would have
silently DELETED the first. Now: same chunk+config+zones+code -> same id
(idempotent re-ingest preserved); change any of them -> different id (A/B works).
One implementation, not two hand-mirrored copies.

**SCHEMA MOVED 2026-08-26** -- the vision tables were mistakenly created in
`public`, which in THIS database holds 546 restaurant tables. They now live in
the `vision` schema that tools/ingest_obs.py always targeted by design.
`vision.v_observation_json` is the view. `vision.vision_embeddings.vec` is
pgvector, so similarity search is available.

**Fixed 2026-08-25:**
- `boxmot` 19 moved its modules; `_import_tracker` now resolves any layout
- `sv.LineZone.trigger` does `int(tracker_id)` and died on named staff ids;
  `_numeric_tracker_ids` aliases names to negative ints for supervision only
- Ground truth is now **camera-scoped** — a bare `gt.txt` is refused, after it
  scored a UK corner shop against Delilah truth and printed `HOTA 0.0057`
- **ROLE LOCK** — a face-matched (non-numeric) id is staff and no zone vote may
  downgrade it. Three staff were being reported as ungreeted guests

## 5. Version drift — read this before debugging

```
local dev   supervision 0.26.1
EC2 box     supervision 0.30.0
```
The `LineZone` crash **cannot reproduce locally**. Always confirm on the box.

## 6. The GPU box

`g6e.2xlarge`, NVIDIA **L40S 45 GB**, CUDA 12.6. Env via `~/.keva_env`
(`.bashrc` returns early for non-interactive shells, so `BASH_ENV` is used).
Storage layout is documented on the host at `/data/STORAGE.md`.

**The pipeline is CPU-bound, not GPU-bound** — the L40S sat at 0-16% for a whole
run. Do not buy throughput with a bigger GPU; the limit is 4K decode, crop prep
and the observation queue.

Fast iteration: `/data/videos/CAM.112_TEST60_at0920.mp4` — 60 s from the region
`audut.md` flags as the clearest staff failure. Add `--no-render` for counts only.

## 7. Data out — for the intelligence layer

Storage is flat and indexed; the nested JSON the 2026-08-20 design discussion
asked for is a **view**, so the downstream developer never sees our column names.

```
Supabase (project mnraeesscqsaappkaldb)
  vision_runs / vision_observations / vision_embeddings
  v_observation_json   <- the agreed nested shape, one row per person per frame
```

```sql
select observation from v_observation_json limit 1;
```

WARNING: connect on **port 5432** (session pooler). The `SUPABASE_DB_URL` in
`.env` says 6543 (transaction pooler), which cannot run DDL.

`motion.speed` is `speed_px_s` (units are declared in `speed_units`).
`motion.speed_mps` is ALSO sane — median 0.16 m/s, p99 3.41 — despite the odd
horizon row; an earlier warning here wrongly called it garbage.

WARNING: `reid.embedding_id` is populated on ~2% of rows by design
(`OBS_EMB_STRIDE=8`) and currently embeds the whole source filename — it should
be a short hash.

## 8. The one known test failure

```
tests/test_run_config.py::test_shipped_config_sets_fps
  asserts FPS_TARGET == 8;  cam112.yaml says fps: 15
```
**Pre-existing** — reproduced against the original yaml at git HEAD. The yaml
comment claims 15 is authoritative, so the test is probably stale, but "make the
test match the code" is the wrong reflex without knowing which was intended.
Operator decision, deliberately not auto-fixed.

## 9. Honesty rules this codebase actually enforces

Worth knowing before changing anything, because the code will fight you:
- **"0" and "not measured" must never render the same.** `validity.py`
- **Every number carries a tier** — EXACT / PROXY / WEAK. `answers.py`
- **Percentages are of OBSERVED time, never elapsed.**
- **Fail loud.** A subsystem that degrades silently is treated as a bug.
- **One knob at a time, against the run ledger.** Otherwise a result is
  unattributable — see `cam112_gpu.yaml`'s header.
