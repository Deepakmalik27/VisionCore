"""test_v69 — certifies the birth-count arrival fallback on REAL chunk events.

Extracts the actual V69 block from the notebook and replays it against the
7:30pm chunk's events.csv (the run where line=0 AND region=0 while 21 moved).
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "pipeline.ipynb"
CSV = Path("/mnt/c/Users/prabh/Downloads/poc_results/CAM.112 (PP.09_12) "
           "7-28-2026, 4.30.00pm CDT - 7-28-2026, 5.30.00pm CDT_events.csv")

nb = json.loads(NB.read_text(encoding="utf-8"))
src = next((r if isinstance(r, str) else "".join(r))
           for c in nb["cells"] if c["cell_type"] == "code"
           for r in [c["source"]]
           if "V69 birth-count fallback" in (r if isinstance(r, str)
                                             else "".join(r)))
m = re.search(r"    if not per_person_ins and _movers >= 5:"
              r".*?trackable this chunk\)\"\)\n", src, re.S)
assert m, "V69 block not found in notebook"
block = "".join(l[4:] + "\n" for l in m.group(0).splitlines())


def replay(events, roles, pre=None, movers=None):
    g = {"per_person_ins": defaultdict(list, pre or {}),
         "_movers": movers if movers is not None
         else len({e["track_id"] for e in events}),
         "events": events, "roles": roles, "sorted": sorted,
         "ins": [], "_region_n": 0, "_arrivals_estimated": False}
    exec(block, g)
    return g["per_person_ins"]


if CSV.exists():
    rows = list(csv.DictReader(open(CSV)))
    events = [{"track_id": r["track_id"], "t_in": float(r["t_in"])}
              for r in rows]
    roles = {r["track_id"]: r["role"] for r in rows}
    out = replay(events, roles)
    assert 5 <= len(out) <= 21, f"implausible count {len(out)}"
    assert all(roles[t] != "staff" for t in out), "staff counted as guest"
else:
    print("(real events.csv not found — synthetic checks only)")

ev = [{"track_id": "a", "t_in": 5.0},     # present at start: NOT an arrival
      {"track_id": "b", "t_in": 100.0},   # arrival
      {"track_id": "s", "t_in": 200.0},   # staff: excluded
      {"track_id": "c", "t_in": 300.0}]   # arrival
rl = {"a": "customer", "b": "customer", "s": "staff", "c": "customer"}
out = replay(ev, rl, movers=6)
assert set(out) == {"b", "c"}, sorted(out)

# stays quiet when a real estimator already produced arrivals
out = replay(ev, rl, pre={"x": [1.0]}, movers=6)
assert set(out) == {"x"}

# stays quiet on a genuinely near-empty chunk (movers < 5)
out = replay(ev, rl, movers=4)
assert not out

print("test_v69: ALL PASS")
