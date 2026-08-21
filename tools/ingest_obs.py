#!/usr/bin/env python3
"""ingest_obs.py — observations.jsonl into Postgres (or SQLite, for a local look).

WHY A FILE FIRST AND NOT A DIRECT INSERT
    "The video pipeline should never wait for PostgreSQL." A dropped
    connection at minute 52 must not cost an hour of GPU time, and a file
    makes ingestion replayable. Same reasoning, and the same delete-then-
    insert discipline, as tools/ingest_db.py.

USAGE
    export SUPABASE_DB_URL=postgresql://...        # never read from .env here
    python3 tools/ingest_obs.py --jsonl output/run1/observations.jsonl
    python3 tools/ingest_obs.py --jsonl ... --sqlite output/obs.db
    python3 tools/ingest_obs.py --selftest
"""
import argparse, json, os, pathlib, sys, struct

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kevacv.observe import OBS_COLUMNS, EMB_COLUMNS      # noqa: E402

RUN_COLUMNS = ("run_id", "camera_id", "video_sha", "fps_analysed",
               "started_at", "frames_analysed", "zones_cfg_hash", "git_sha")
# git_sha is build_id.compute() — a hash of package file contents, not a git commit SHA

TYPES_PG = {"frame_idx": "integer", "x1": "integer", "y1": "integer",
            "x2": "integer", "y2": "integer", "t_s": "real",
            "det_conf": "real", "foot_x": "real", "foot_y": "real",
            "zone_conf": "real", "speed_px_s": "real", "speed_mps": "real",
            "heading_deg": "real", "pose_age_s": "real", "blur_score": "real",
            "fps_analysed": "real", "frames_analysed": "integer",
            "feet_visible": "boolean", "is_ir": "boolean",
            "stationary": "boolean", "ts": "timestamptz"}


def _coltype(name, dialect):
    t = TYPES_PG.get(name, "text")
    if dialect == "sqlite":
        return {"timestamptz": "text", "boolean": "integer"}.get(t, t)
    return t


def create_tables(conn, dialect="pg", vec_type="bytea"):
    obs = ", ".join(f"{c} {_coltype(c, dialect)}" for c in OBS_COLUMNS)
    emb = ", ".join((f"{c} {vec_type}" if c == "vec"
                     else f"{c} {_coltype(c, dialect)}") for c in EMB_COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS vision_observations ({obs}, "
                 f"PRIMARY KEY (run_id, frame_idx, raw_track_id))")
    conn.execute(f"CREATE TABLE IF NOT EXISTS vision_embeddings ({emb}, "
                 f"PRIMARY KEY (emb_id))")
    conn.execute("CREATE INDEX IF NOT EXISTS obs_track ON vision_observations "
                 "(run_id, raw_track_id, t_s)")
    conn.execute("CREATE INDEX IF NOT EXISTS obs_zone ON vision_observations "
                 "(run_id, zone, t_s)")
    runs = ", ".join(
        f"{c} {'timestamptz' if c == 'started_at' and dialect == 'pg' else _coltype(c, dialect)}"
        for c in RUN_COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS vision_runs ({runs}, "
                 f"PRIMARY KEY (run_id))")


def _ph(dialect, n):
    return ", ".join(("?" if dialect == "sqlite" else "%s") for _ in range(n))


def _pack(vec, vec_type):
    if vec_type == "vector":
        return "[" + ",".join(f"{float(v):.6f}" for v in vec) + "]"
    return struct.pack(f"<{len(vec)}f", *[float(v) for v in vec])


def ingest(conn, jsonl, dialect="pg", vec_type="bytea"):
    obs, emb, run_rows, runs = [], [], [], set()
    with open(jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            runs.add(r.get("run_id"))
            if r.get("kind") == "emb":
                r["vec"] = _pack(r["vec"], vec_type)
                emb.append([r.get(c) for c in EMB_COLUMNS])
            elif r.get("kind") == "run":
                run_rows.append([r.get(c) for c in RUN_COLUMNS])
            else:
                obs.append([r.get(c) for c in OBS_COLUMNS])
    for run in runs:
        conn.execute(f"DELETE FROM vision_observations WHERE run_id = "
                     f"{_ph(dialect, 1)}", (run,))
        conn.execute(f"DELETE FROM vision_embeddings WHERE run_id = "
                     f"{_ph(dialect, 1)}", (run,))
        conn.execute(f"DELETE FROM vision_runs WHERE run_id = "
                     f"{_ph(dialect, 1)}", (run,))
    if obs:
        conn.executemany(
            f"INSERT INTO vision_observations ({', '.join(OBS_COLUMNS)}) "
            f"VALUES ({_ph(dialect, len(OBS_COLUMNS))})", obs)
    if emb:
        conn.executemany(
            f"INSERT INTO vision_embeddings ({', '.join(EMB_COLUMNS)}) "
            f"VALUES ({_ph(dialect, len(EMB_COLUMNS))})", emb)
    if run_rows:
        conn.executemany(
            f"INSERT INTO vision_runs ({', '.join(RUN_COLUMNS)}) "
            f"VALUES ({_ph(dialect, len(RUN_COLUMNS))})", run_rows)
    conn.commit()
    return {"obs": len(obs), "emb": len(emb), "run_rows": len(run_rows),
            "runs": sorted(r for r in runs if r)}


def _pg_connect(url):
    import psycopg
    conn = psycopg.connect(url)
    # pgvector turns Deepak's re-match into one SQL query instead of a full
    # pull into Python. Use it when the project has it; never CREATE EXTENSION
    # from here -- that is a database owner's decision, not an ingest tool's.
    has_vec = conn.execute(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    return conn, ("vector" if has_vec else "bytea")


def selftest():
    import sqlite3, tempfile, pathlib, json
    rows = [{"kind": "obs", "run_id": "r1", "camera_id": "cam01", "frame_idx": i,
             "raw_track_id": "103", "canon_id": None, "ts": None, "t_s": i * 0.125,
             "x1": 1, "y1": 2, "x2": 3, "y2": 4, "det_conf": 0.9,
             "foot_x": 2.0, "foot_y": 4.0, "feet_visible": True, "is_ir": False,
             "zone": "dining", "zone_conf": 0.8, "speed_px_s": 0.0,
             "speed_mps": None, "heading_deg": None, "stationary": True,
             "emb_id": None, "pose_activity": None, "pose_age_s": None}
            for i in range(3)]
    d = pathlib.Path(tempfile.mkdtemp())
    p = d / "observations.jsonl"
    run_row = {"kind": "run", "run_id": "r1", "camera_id": "cam01",
               "video_sha": "abc123", "fps_analysed": 8.0,
               "started_at": None, "frames_analysed": 3,
               "zones_cfg_hash": "z1", "git_sha": "deadbeef"}
    p.write_text("\n".join(json.dumps(r) for r in [run_row] + rows) + "\n")
    conn = sqlite3.connect(":memory:")
    create_tables(conn, dialect="sqlite", vec_type="blob")
    n = ingest(conn, str(p), dialect="sqlite")
    assert n["obs"] == 3, n
    assert conn.execute("SELECT count(*) FROM vision_runs").fetchone()[0] == 1
    # A re-ingest must UPDATE the run row, not duplicate it
    ingest(conn, str(p), dialect="sqlite")
    assert conn.execute("SELECT count(*) FROM vision_runs").fetchone()[0] == 1
    # Re-ingest a SHORTER run: stale rows must not survive. This is the exact
    # bug tools/ingest_db.py's selftest guards, and INSERT OR REPLACE misses.
    p.write_text(json.dumps(rows[0]) + "\n")
    ingest(conn, str(p), dialect="sqlite")
    got = conn.execute("SELECT count(*) FROM vision_observations").fetchone()[0]
    assert got == 1, got
    print("selftest ok")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl")
    p.add_argument("--sqlite")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.jsonl:
        p.error("--jsonl is required")
    if a.sqlite:
        import sqlite3
        conn, dialect, vec_type = sqlite3.connect(a.sqlite), "sqlite", "blob"
    else:
        url = os.environ.get("SUPABASE_DB_URL")
        if not url:
            print("SUPABASE_DB_URL is not set. Export it in this shell "
                  "(this tool never reads .env).", file=sys.stderr)
            return 2
        conn, vec_type = _pg_connect(url)
        dialect = "pg"
    create_tables(conn, dialect=dialect, vec_type=vec_type)
    n = ingest(conn, a.jsonl, dialect=dialect, vec_type=vec_type)
    print(f"  {n['obs']} observations, {n['emb']} embeddings, "
          f"{n['run_rows']} run row(s), runs: {', '.join(n['runs'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
