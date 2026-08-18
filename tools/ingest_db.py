#!/usr/bin/env python3
"""ingest_db.py — every run's people.csv into one queryable SQLite file.

WHY SQLITE AND NOT POSTGRES
  Total corpus today: 99 rows across 36 runs, written by one pipeline process
  at a time. A server process, a port, a password and a backup policy buy
  nothing at that size. SQLite is the same SQL against a single file on the
  EBS volume. Revisit when two machines write at once or something remote
  needs to connect -- not before.

WHY A DB AT ALL, GIVEN THE CSVs
  The CSVs answer "what happened in THIS run". They cannot answer "did the
  greet rate move after the phantom-gate change", because that is a question
  across runs, and comparing 36 directories by eye is how regressions hide.

  Re-ingesting a run REPLACES its rows, so running this after every pipeline
  execution is safe and the db never double-counts.

USAGE
  python3 tools/ingest_db.py                    # ingest output/*/people.csv
  python3 tools/ingest_db.py --db /path/keva.db --out /path/output
  python3 tools/ingest_db.py --selftest
  sqlite3 output/keva.db "select run, count(*) from people group by run"
"""
import argparse, csv, pathlib, sqlite3, sys, datetime

COLS = ["person", "snap", "role", "role_from", "first_seen", "last_seen",
        "minutes", "waited_s", "greeted", "greet_s", "confidence", "flags"]


def connect(db):
    c = sqlite3.connect(db)
    c.execute(f"""CREATE TABLE IF NOT EXISTS people (
        run TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        {', '.join(f'{k} TEXT' for k in COLS)},
        PRIMARY KEY (run, person))""")
    return c


def ingest(conn, run, rows):
    # Delete-then-insert, not INSERT OR REPLACE: a re-run with FEWER people
    # must not leave the extra rows from the previous run sitting there.
    conn.execute("DELETE FROM people WHERE run = ?", (run,))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    conn.executemany(
        f"INSERT INTO people (run, ingested_at, {', '.join(COLS)}) "
        f"VALUES (?, ?, {', '.join('?' * len(COLS))})",
        [(run, now, *[r.get(k, "") for k in COLS]) for r in rows])
    return len(rows)


def main(argv=None):
    root = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(root / "output"))
    p.add_argument("--db", default=None)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()

    out = pathlib.Path(a.out)
    conn = connect(a.db or out / "keva.db")
    total = 0
    for csv_path in sorted(out.glob("*/people.csv")):
        with csv_path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        n = ingest(conn, csv_path.parent.name, rows)
        total += n
        print(f"  {csv_path.parent.name:<20} {n:>4} rows")
    conn.commit()
    print(f"  {'TOTAL':<20} {total:>4} rows -> {a.db or out / 'keva.db'}")
    return 0


def selftest():
    conn = connect(":memory:")
    assert ingest(conn, "r1", [{"person": "1", "greeted": "no"},
                               {"person": "2", "greeted": "yes"}]) == 2
    # Re-ingest with one fewer person: the stale row must be gone, which is
    # exactly what INSERT OR REPLACE would have got wrong.
    ingest(conn, "r1", [{"person": "1", "greeted": "yes"}])
    got = conn.execute("SELECT person, greeted FROM people WHERE run='r1'").fetchall()
    assert got == [("1", "yes")], got
    ingest(conn, "r2", [{"person": "1", "greeted": "no"}])
    assert conn.execute("SELECT count(*) FROM people").fetchone()[0] == 2
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
