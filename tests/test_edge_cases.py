"""test_edge_cases.py — OOM survival, chunk seams, and reflections.

The three stream-level failures that had no guard anywhere:

  OOM         batch size is chosen for the average frame; VRAM is consumed by
              the worst one. The run dies on the busiest minute of the night —
              the minute the report exists to describe.
  SEAMS       every chunk boundary splits anyone standing across it. Eleven
              seams in a 12-hour night, and the receptionist becomes eleven
              "different people working the desk".
  REFLECTIONS person-shaped, person-sized, and it moves. Every existing filter
              is blind to it.

Run: python tests/test_edge_cases.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.detect_filters import mirrored_pair_ids  # noqa: E402
from kevacv.resilience import Checkpoint, run_batched  # noqa: E402
from kevacv.seams import apply, bridge, heads, tails  # noqa: E402

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


print("=" * 74)
print("  an out-of-memory does not end an 8-hour run")
print("=" * 74)


class FakeOOM(Exception):
    pass


def make_fn(fail_above):
    seen = {"batches": [], "n": 0}

    def fn(chunk):
        seen["n"] += 1
        if len(chunk) > fail_above:
            raise FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB")
        seen["batches"].append(len(chunk))
        return [c * 2 for c in chunk]
    return fn, seen


fn, seen = make_fn(fail_above=3)
out = list(run_batched(list(range(20)), fn, batch=12))
check(out == [i * 2 for i in range(20)], "every item is still processed, in order",
      f"{len(out)} items")
check(max(seen["batches"]) <= 3, "the batch shrank until it fit",
      f"max batch {max(seen['batches'])}")

fn2, seen2 = make_fn(fail_above=3)
list(run_batched(list(range(40)), fn2, batch=12))
check(seen2["batches"][-1] <= 3,
      "and the smaller batch STICKS for the rest of the run",
      "a scene that OOMed once will OOM again a second later")


def always_oom(chunk):
    raise FakeOOM("CUDA out of memory")


try:
    list(run_batched([1, 2, 3], always_oom, batch=4, min_batch=1))
    check(False, "an unfixable OOM eventually raises")
except FakeOOM:
    check(True, "an unfixable OOM eventually raises rather than looping forever")


def other_error(chunk):
    raise ValueError("something else entirely")


try:
    list(run_batched([1, 2], other_error, batch=2))
    check(False, "a non-OOM error propagates")
except ValueError:
    check(True, "a non-OOM error propagates immediately, unretried",
          "retrying a logic bug just hides it")

print()
print("=" * 74)
print("  a crash costs one chunk, not the night")
print("=" * 74)
with tempfile.TemporaryDirectory() as td:
    cp = Checkpoint(Path(td) / "run.state.json")
    keys = ["CAM.112 4.30pm", "CAM.112 5.30pm", "CAM.112 6.30pm"]
    check(cp.pending(keys) == keys, "nothing done yet -> everything pending")
    cp.mark(keys[0], {"events": 300})
    check(cp.pending(keys) == keys[1:], "a completed chunk is skipped on resume")
    cp2 = Checkpoint(Path(td) / "run.state.json")
    check(cp2.is_done(keys[0]), "and the state survives a fresh process")
    check(cp2.done[keys[0]]["events"] == 300, "with its info intact")
    bad = Path(td) / "corrupt.json"
    bad.write_text("{not json", encoding="utf-8")
    check(Checkpoint(bad).pending(keys) == keys,
          "an unreadable checkpoint starts fresh rather than crashing")

print()
print("=" * 74)
print("  a person standing across a chunk boundary is ONE person")
print("=" * 74)


def log(t0, n, tid, x, step=0.5):
    return [(i, t0 + i * step, [(tid, x, 300, x + 40, 440)]) for i in range(n)]


prev = log(3400.0, 20, "A", 500)          # ends at 3409.5
nxt = log(3600.0, 20, "Z", 502)           # starts at 3600.0
t, h = tails(prev), heads(nxt)
check(set(t) == {"A"} and set(h) == {"Z"}, "tails and heads are found")

m, f = bridge(t, h, (1280, 720), prev_end_clock=3600.0, next_start_clock=3600.0)
check(m == {"Z": "A"}, "the new chunk adopts the earlier identity", str(m))
check(not f, "and nothing is flagged")

# a gap in the footage: bridging would invent continuity nobody observed
m, f = bridge(t, h, (1280, 720), prev_end_clock=3600.0, next_start_clock=3840.0)
check(m == {}, "a 4-minute footage gap is NOT bridged")
check(any("could have left and been replaced" in msg for _, msg in f),
      "and says why — unobserved time is not continuity")

m, f = bridge(t, h, (1280, 720), prev_end_clock=3600.0, next_start_clock=3550.0)
check(m == {} and any(l == "ERROR" for l, _ in f),
      "overlapping chunks are an ERROR — the same seconds would count twice")

far = heads(log(3600.0, 20, "Z", 1100))
m, f = bridge(t, far, (1280, 720), prev_end_clock=3600.0, next_start_clock=3600.0)
check(m == {}, "a body that reappears far away is not bridged")
check(any("too far apart to match" in msg for _, msg in f), "and is reported")

two_t = tails(log(3400.0, 20, "A", 500) + log(3400.0, 20, "B", 900))
two_h = heads(log(3600.0, 20, "Y", 902) + log(3600.0, 20, "Z", 498))
m, _ = bridge(two_t, two_h, (1280, 720), prev_end_clock=3600.0,
              next_start_clock=3600.0)
check(m == {"Z": "A", "Y": "B"}, "two bodies bridge to the RIGHT partners", str(m))

ev, cr, fl = apply({"Z": "A"}, events=[{"track_id": "Z", "zone": "reception"}],
                   crossings=[{"track_id": "Z", "direction": "in"}],
                   frame_log=log(3600.0, 2, "Z", 502))
check(ev[0]["track_id"] == "A" and cr[0]["track_id"] == "A",
      "events and crossings are rewritten")
check(fl[0][2][0][0] == "A", "and so is the frame log")

print()
print("=" * 74)
print("  an object and its reflection cannot drift apart")
print("=" * 74)


def pair_log(offsets, a_path, tid_a="P1", tid_b="P2"):
    rows = []
    for i, (x, off) in enumerate(zip(a_path, offsets)):
        rows.append((i, float(i), [(tid_a, x, 300, x + 40, 440),
                                   (tid_b, x + off, 300, x + off + 40, 440)]))
    return rows


path = [100 + i * 4 for i in range(60)]
refl = pair_log([120] * 60, path)                       # constant separation
real = pair_log([40 + i * 6 for i in range(60)], path)  # drifting apart

r = mirrored_pair_ids(refl)
check(("P1", "P2") in r, "a constant-offset pair is flagged",
      f"cv={r.get(('P1','P2'), {}).get('offset_cv')}")
check("cannot" in r[("P1", "P2")]["why"], "with the reasoning stated")
check(mirrored_pair_ids(real) == {},
      "two real people who drift apart are NOT flagged")
check(mirrored_pair_ids(pair_log([3] * 60, path)) == {},
      "a near-zero offset is a duplicate box, not a reflection",
      "min_offset_px keeps those out")
check(mirrored_pair_ids(refl, protected={"P1", "P2"}) == {},
      "a pair who both crossed the door is never called a reflection")
check(mirrored_pair_ids(pair_log([120] * 10, path[:10])) == {},
      "too few samples -> no verdict")

print()
print("=" * 74)
# PYTEST VISIBILITY. This file's checks run at IMPORT and reported only via
# an exit code. A module-level sys.exit aborts pytest COLLECTION, which hid
# 59 of 74 test files -- and simply guarding the exit would have hidden the
# FAILURES instead. So the same condition is also asserted as a real test.
def test_script_level_checks_passed():
    assert not (FAILED), "module-level checks in this file failed"


if __name__ == "__main__":
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"    - {f}")
        sys.exit(1)
print("  ALL PASS")
print("=" * 74)
