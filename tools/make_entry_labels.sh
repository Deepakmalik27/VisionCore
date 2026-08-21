#!/usr/bin/env bash
# One command from a finished run to a folder of sheets a human can label.
#
# The three steps were each reachable on their own and had never been chained,
# which is part of why entry ground truth was never collected: `sheets` did not
# exist until 2026-08-21, so the middle link was missing entirely.
#
#   usage:  tools/make_entry_labels.sh <run_dir> <source_video> [n_windows]
set -euo pipefail

RUN_DIR="${1:?usage: make_entry_labels.sh <run_dir> <source_video> [n]}"
VIDEO="${2:?need the SOURCE chunk that was analysed, not the annotated render}"
N="${3:-8}"
PLAN="eval/entry_plan.json"

[ -e "$VIDEO" ] || { echo "no such video: $VIDEO"; exit 1; }
case "$VIDEO" in
  *annotated*|*ALLWIRED*|*FIXED*)
    echo "!! '$VIDEO' looks like a RENDER, not a source chunk."
    echo "   Renders are time-lapsed and carry overlays; labelling one gives"
    echo "   timestamps that match nothing. Point this at the source."
    exit 1;;
esac

echo "== 1/3  choosing windows by detector activity (before anyone watches) =="
python3 tools/entry_label_kit.py plan "$RUN_DIR" --n "$N" --out "$PLAN"

echo
echo "== 2/3  rendering one contact sheet per window =="
python3 tools/entry_label_kit.py sheets "$PLAN" --video "$VIDEO" --out eval/sheets

echo
echo "== 3/3  state of the reference (REFUSED until labelled -- expected) =="
python3 tools/entry_label_kit.py check "$PLAN" || true

echo
echo "Now: read eval/sheets/HOW_TO_LABEL.txt, fill '$PLAN', re-run:"
echo "   python3 tools/entry_label_kit.py check $PLAN"
