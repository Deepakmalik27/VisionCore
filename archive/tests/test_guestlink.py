"""Check the whole-night guest logic: a guest who leaves and comes back an hour
later must be ONE person, and two different people must not be merged."""
import json, numpy as np
from pathlib import Path
NB = Path(__file__).resolve().parent.parent / "notebooks" / "pipeline.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
cw = lambda n: next(s for s in code if n in s)
G = {"__name__": "__main__", "OUTPUT_DIR": Path("/tmp"), "QUEUE_REMOTE": [],
     "print": print}
exec("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt", G)
exec(cw("# Cell 5 — analytics logic"), G)          # _cosine, _gallery_sim
# take just the function defs out of the runner cell (skip its top-level run)
runner = cw("# Cell 9b — MULTI-CHUNK RUNNER")
exec(runner.split("if len(globals().get(\"QUEUE_REMOTE\"")[0], G)

def vec(seed, cos=None):
    """cos=None -> a fresh identity. cos=x -> the same person seen again with
    cosine similarity exactly x (what a real ReID model returns for one person
    across two sightings: typically 0.6-0.85)."""
    rg = np.random.default_rng(seed)
    v = rg.normal(size=128); v /= np.linalg.norm(v)
    if cos is None:
        return v.tolist()
    o = np.random.default_rng(seed + 999).normal(size=128)
    o -= o.dot(v) * v; o /= np.linalg.norm(o)
    w = cos * v + np.sqrt(1 - cos ** 2) * o
    return (w / np.linalg.norm(w)).tolist()

# alice leaves at t=600 and returns at t=4200 (an hour later, different chunk)
dossiers = {
    "c0_5":  {"person_id": "c0_5",  "t_first": 100, "t_last": 600,
              "face_embedding": vec(1), "anchor_embedding": vec(11)},
    "c1_9":  {"person_id": "c1_9",  "t_first": 4200, "t_last": 4600,
              "face_embedding": vec(1, 0.62), "anchor_embedding": vec(11, 0.55)},
    "c1_12": {"person_id": "c1_12", "t_first": 4300, "t_last": 4900,
              "face_embedding": vec(2), "anchor_embedding": vec(22)},
    "c2_3":  {"person_id": "c2_3",  "t_first": 30000, "t_last": 30500,   # way later
              "face_embedding": vec(1), "anchor_embedding": vec(11)},
}
roles = {k: "customer" for k in dossiers}
alias = G["link_returning_guests"](dossiers, list(dossiers), roles)
print("alias:", alias)
assert alias.get("c1_9") == "c0_5", "the returning guest must fold into the first sighting"
assert "c1_12" not in alias, "a different person must NOT be merged"
assert "c2_3" not in alias, "beyond GUEST_GLOBAL_GAP_S must not be merged"

ev = [{"track_id": "c1_9", "zone": "waiting", "role": "customer", "t_in": 4200,
       "t_out": 4400, "duration": 200}]
cr = [{"track_id": "c1_9", "direction": "in", "t": 4200},
      {"track_id": "c0_5", "direction": "in", "t": 100}]
fl = [(0, 4200.0, [("c1_9", 1, 2, 3, 4)])]
ev2, cr2, ro2, fl2 = G["_apply_alias"](ev, cr, dict(roles), fl, alias)
assert ev2[0]["track_id"] == "c0_5" and fl2[0][2][0][0] == "c0_5"
assert len({c["track_id"] for c in cr2}) == 1, "both crossings now belong to one guest"
print("guests counted:", len({c['track_id'] for c in cr2 if c['direction']=='in'}),
      "(2 door crossings, 1 person)")
print("PASS: leave-and-return counts once; different people stay separate")
