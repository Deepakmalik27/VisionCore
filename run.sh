#!/usr/bin/env bash
# run.sh — execute the pipeline notebook headlessly, with real logs.
#
# WHY papermill AND NOT A REWRITE
#   kevacv/__init__.py: the analytics (Cell 5) and video engine (Cell 7) come
#   out of the notebook AFTER the first HOTA number, not before — moving 260 KB
#   of logic with no measured baseline makes any later regression
#   unattributable. papermill gives us headless execution, no session cap and
#   proper logs while running the exact notebook we already trust.
#
# USAGE
#   ./run.sh --video V --zones Z  run THE CODEBASE (default)
#   ./run.sh --dry --video V --zones Z    20s smoke test, proves the whole path
#   ./run.sh --check              verify the environment, run nothing
#   ./run.sh --tests              run the test suite only
#   ./run.sh --notebook           run the NOTEBOOK under papermill instead
#
# CODEBASE vs NOTEBOOK
#   The codebase path (default) runs kevacv.pipeline.run_camera, so every guard
#   the package has fires: preflight, provenance, DST, the phantom stage
#   (static/rigid/mirrored), the arrival cross-check, and SUMMARY.txt.
#   The notebook path runs the notebook's own copy of the logic and none of
#   those guards exist there. Keep it only for comparing the two.

set -Eeuo pipefail

# The suites print ✅/❌ and the engine logs emoji. On any console whose locale
# is not UTF-8 (a Windows cp1252 shell, a bare LANG=C container) that raises
# UnicodeEncodeError and the process exits non-zero — which reads exactly like
# a failing test. Six suites were counted as "known failures" for that reason
# alone; forcing the encoding took the suite from 35 pass/10 fail to 41/4.
export PYTHONIOENCODING=utf-8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${ROOT}/config/cam112.yaml"
NB="${ROOT}/notebooks/pipeline.ipynb"
LOGDIR="${ROOT}/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

usage() { sed -n '2,20p' "$0"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --video)  VIDEO="$2"; shift 2 ;;
    --zones)  ZONES="$2"; shift 2 ;;
    --camera) CAMERA="$2"; shift 2 ;;
    --dry)    DRY=--dry; shift ;;
    --check)  MODE=check; shift ;;
    --tests)  MODE=tests; shift ;;
    --notebook) MODE=notebook; shift ;;
    -h|--help|--usage) usage ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
MODE="${MODE:-codebase}"
DRY="${DRY:-}"
CAMERA="${CAMERA:-CAM.112}"

mkdir -p "$LOGDIR" "${ROOT}/data" "${ROOT}/output"
LOG="${LOGDIR}/${MODE}_${STAMP}.log"

# Everything below is tee'd, so the log is self-contained: what ran, on what,
# with which versions. A run you cannot reconstruct from its own log is an
# anecdote — SUCCESS_CRITERIA.md makes the same point about the score sheet.
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo " reception analytics · ${MODE} · ${STAMP}"
echo " config   ${CONFIG}"
echo " notebook ${NB}"
echo " log      ${LOG}"
echo "=============================================================="

echo "--- environment ---"
python3 - <<'PY'
import platform, importlib
print(f"  python   {platform.python_version()}  {platform.platform()}")
for m in ("numpy", "scipy", "torch", "ultralytics", "boxmot", "supervision",
          "cv2", "insightface", "onnxruntime"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<12} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<12} MISSING ({type(e).__name__})")
try:
    import torch
    print(f"  cuda         available={torch.cuda.is_available()} "
          f"devices={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    [{i}] {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"  cuda         unavailable ({e})")
PY

echo "--- config ---"
sed 's/^/  /' "$CONFIG"

echo "--- kevacv / notebook drift check ---"
# The package exists twice: kevacv/*.py and a frozen copy inside the notebook.
# They drifted once already and it cost a NameError an hour into a run.
#
# The check is FATAL only on the notebook path, because only that path loads
# the frozen copy. Under `set -Eeuo pipefail` an unconditional check made a
# stale notebook abort run.sh before the MODE dispatch — so --check, --tests
# and the default codebase run were all dead, none of which read the notebook
# at all. And it IS stale: the bootstrap embeds none of pipeline, config,
# engine, analytics, derive, answers, drive, clock, validity, seams or
# resilience. Fixing the pipeline in kevacv/ is supposed to be possible
# without re-freezing a notebook nobody is running.
if ! python3 "${ROOT}/tools/embed_kevacv.py" --check; then
  if [[ "${MODE}" == "notebook" ]]; then
    echo "  the notebook path RUNS the frozen copy — refusing a stale run." >&2
    echo "  run: python tools/embed_kevacv.py" >&2
    exit 1
  fi
  echo "  (stale, but this path runs kevacv/ directly — continuing)"
fi

if [[ "$MODE" == "check" ]]; then
  echo "--- check complete, nothing executed ---"
  exit 0
fi

if [[ "$MODE" == "tests" ]]; then
  echo "--- test suite ---"
  pass=0; fail=0
  for t in "${ROOT}"/tests/test_*.py; do
    if python3 "$t" >/dev/null 2>&1; then
      pass=$((pass+1))
    else
      fail=$((fail+1)); echo "  FAIL $(basename "$t")"
    fi
  done
  echo "  ${pass} passed, ${fail} failed"
  exit 0
fi

if [[ "$MODE" == "codebase" ]]; then
  echo "--- executing THE CODEBASE (kevacv.pipeline.run_camera) ---"
  if [[ -z "${VIDEO:-}" || -z "${ZONES:-}" ]]; then
    echo "  --video and --zones are required for the codebase path" >&2
    echo "  (or use --notebook to run the notebook instead)" >&2
    exit 2
  fi
  python3 "${ROOT}/tools/run_pipeline.py" --video "$VIDEO" --zones "$ZONES"       --camera-id "$CAMERA" --out "${ROOT}/output" ${DRY}
  echo "  log  : ${LOG}"
  exit $?
fi

OUT="${ROOT}/output/pipeline_${STAMP}.ipynb"
echo "--- executing notebook (LEGACY path — codebase guards do NOT run) ---"
# --log-output puts every cell's stdout into THIS log as it happens, instead of
# burying it in the output notebook where nobody reads it.
papermill "$NB" "$OUT" --log-output --no-progress-bar

echo "--- done ---"
echo "  executed notebook : ${OUT}"
echo "  artefacts         : ${ROOT}/output/"
echo "  log               : ${LOG}"
echo
echo "  READ THESE THREE FIRST:"
echo "    1. SOURCE crop height   (V76)  <100px -> resolution is the ceiling"
echo "    2. ENTRY ZONE MISPLACED (V74)  fires? -> zones, not counting, are wrong"
echo "    3. calibration same/diff        vs the circular 0.658 baseline"
