"""test_drive.py — select the chunk BEFORE fetching it.

The shipped failure: the notebook parsed the clock from vids[0], downloaded
vids[0], THEN applied CHUNK_FILTER which selected a different file, then
re-parsed the clock from the new pick. It analysed 16:30 footage and stamped
19:30 on it. Every step was individually correct; only their order was wrong.

Run: python tests/test_drive.py
"""
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kevacv.drive import DriveError, describe, fetch_chunk, select, start_part

FAILED = []


def check(cond, label, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILED.append(label)


def nm(h1, h2):
    return (f"CAM.112 (PP.09_12) 7-28-2026, {h1} CDT - "
            f"7-28-2026, {h2} CDT.mp4")


LISTING = [nm("4.30.00pm", "5.30.00pm"), nm("5.30.00pm", "6.30.00pm"),
           nm("6.30.00pm", "7.30.00pm"), nm("7.30.00pm", "8.30.00pm"),
           "notes.txt"]

print("=" * 74)
print("  the filter matches a chunk's OWN start, never the previous one's end")
print("=" * 74)
check(start_part(LISTING[0]).endswith("4.30.00pm CDT"),
      "start_part keeps only the text before the range separator")
got = select(LISTING, chunk_filter="7.30.00pm")
check("7.30.00pm CDT - " in got, "'7.30.00pm' selects the 7:30 chunk...", Path(got).name[:40])
check(got == LISTING[3], "...and NOT the 6:30 chunk whose range ENDS at 7:30",
      "a bare substring search matched both — that is the shipped bug")

print()
print("=" * 74)
print("  ambiguity is a question for a human, never a silent first-match")
print("=" * 74)
try:
    select(LISTING, chunk_filter="pm")
    check(False, "a filter matching several chunks raises")
except DriveError as e:
    check("matched 4 chunks" in str(e), "a filter matching several chunks raises",
          "analysing the wrong hour is worse than analysing none")
try:
    select(LISTING, chunk_filter="9.30.00pm")
    check(False, "a filter matching nothing raises")
except DriveError as e:
    check("matched none" in str(e), "a filter matching nothing raises")
    check("Start times available" in str(e), "and lists what IS there")
try:
    select(LISTING)
    check(False, "no selection at all raises")
except DriveError as e:
    check("no selection given" in str(e), "no selection at all raises")
try:
    select(["notes.txt", "readme.md"])
    check(False, "a folder with no videos raises")
except DriveError as e:
    check("no video files" in str(e), "a folder with no videos raises")
check(select(LISTING, index=0) == LISTING[0], "index selection works too")

print()
print("=" * 74)
print("  the file we DOWNLOAD is the file we SELECTED")
print("=" * 74)
with tempfile.TemporaryDirectory() as td:
    pulled = []

    def fake_dl(name, out):
        pulled.append(name)
        Path(out).write_bytes(b"video")

    p = fetch_chunk("folder", td, chunk_filter="7.30.00pm",
                    listing=LISTING, download=fake_dl)
    check(len(pulled) == 1, "exactly ONE file is downloaded", str(len(pulled)))
    check("7.30.00pm CDT - " in pulled[0],
          "and it is the SELECTED one, not listing[0]",
          "the notebook downloaded vids[0] then filtered afterwards")
    check(p.name == Path(pulled[0]).name,
          "the returned path IS the downloaded file — no second variable to drift")

    # a download that produces a different file must be refused, not analysed
    def wrong_dl(name, out):
        Path(out).write_bytes(b"video")

    with tempfile.TemporaryDirectory() as td2:
        try:
            fetch_chunk("folder", td2, chunk_filter="4.30.00pm",
                        listing=LISTING,
                        download=lambda n, o: Path(o).write_bytes(b"v"))
            check(True, "a consistent download is accepted")
        except DriveError as e:
            check(False, "a consistent download is accepted", str(e))

with tempfile.TemporaryDirectory() as td:
    target = Path(td) / Path(LISTING[3]).name
    target.write_bytes(b"already here")
    calls = []
    p = fetch_chunk("folder", td, chunk_filter="7.30.00pm", listing=LISTING,
                    download=lambda n, o: calls.append(n))
    check(not calls, "a chunk already on disk is not re-downloaded",
          "3.2 GB per run is the whole reason to leave Kaggle")
    check(p == target, "and the existing path is returned")

print()
print("=" * 74)
print("  describe() makes picking a chunk not-guesswork")
print("=" * 74)
d = describe(LISTING, chunk_filter="7.30.00pm")
check("4 chunk(s) available" in d, "counts only the videos", d.splitlines()[0])
check("matches your filter" in d, "and marks the one your filter picks")
check("notes.txt" not in d, "non-video files are not offered as chunks")


print()
print("=" * 74)
print("  the kilobytes come automatically, the gigabytes never do")
print("=" * 74)
from kevacv.drive import fetch_assets  # noqa: E402

ENTRIES = [("id_v1", nm("4.30.00pm", "5.30.00pm")),
           ("id_v2", nm("7.30.00pm", "8.30.00pm")),
           ("id_zone", "CAM.112_zone.json"),
           ("id_face", "receptionist_sarah.jpg"),
           ("id_zip", "cv.zip")]

with tempfile.TemporaryDirectory() as td:
    z, g = Path(td) / "zones", Path(td) / "gallery"
    pulled = []

    def dl(fid, out):
        pulled.append(fid)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"x")

    got = fetch_assets("folder", zones_dir=z, gallery_dir=g,
                       entries=ENTRIES, download=dl)
    check("id_v1" not in pulled and "id_v2" not in pulled,
          "NO video is ever pulled by the asset sync",
          "13 GB must only move on an explicit chunk choice")
    check("id_zone" in pulled, "the zone map is fetched")
    check("id_face" in pulled, "the staff photo is fetched")
    check("id_zip" not in pulled, "unrelated files are ignored", "cv.zip")
    check("cv.zip" in got["skipped"], "and reported as skipped, not silently dropped")
    check(len(got["zones"]) == 1 and len(got["gallery"]) == 1,
          "one of each ends up where the pipeline looks for it")

# WHO OWNS THE ZONE MAP — LOCAL. Reverted 2026-08-13 by operator instruction,
# after one day as ("zones",).
#
# The zone map is geometry the code is TUNED AGAINST. A line moved in the
# mapper and a threshold moved in config.py have to land in the same commit or
# the run is not reproducible, and Drive has no commits. So the repo owns it.
#
# The complaint behind the brief reversal was real: a corrected map was
# uploaded to Drive and silently ignored for an entire 32-minute run, behind a
# log line reading "already local — keeping yours, not Drive's", which sounds
# like the tool protecting your work rather than discarding someone else's.
# That is fixed by making the skip LOUD — asserted below — not by letting
# Drive overwrite a hand-corrected file, which is the more destructive of the
# two failures because it is unrecoverable.
#
# The staff gallery keeps local precedence for the reason it always did: those
# photos are curated in the repo and versioned with the code.
with tempfile.TemporaryDirectory() as td:
    z, g = Path(td) / "zones", Path(td) / "gallery"
    z.mkdir(parents=True)
    g.mkdir(parents=True)
    (z / "CAM.112_zone.json").write_text("MY EDITED ZONES", encoding="utf-8")
    (g / "receptionist_sarah.jpg").write_bytes(b"MY CURATED PHOTO")
    pulled = []
    got = fetch_assets("folder", zones_dir=z, gallery_dir=g, entries=ENTRIES,
                       download=lambda f, o: pulled.append(f))
    check("id_zone" not in pulled,
          "a hand-corrected local zone map is NOT overwritten by Drive",
          "the repo owns the geometry the code is tuned against")
    check((z / "CAM.112_zone.json").read_text(encoding="utf-8") == "MY EDITED ZONES",
          "and the local edit survives untouched")
    check("id_face" not in pulled,
          "a curated staff photo is NOT overwritten either",
          "the gallery is versioned with the code, not with the footage")
    check((g / "receptionist_sarah.jpg").read_bytes() == b"MY CURATED PHOTO",
          "and the local photo survives untouched")

# THE SKIP MUST BE LOUD. This is the whole reason the policy could be reverted
# safely: a Drive copy that exists and is being discarded gets a WARNING naming
# what to do about it, so "I re-drew it in the mapper and nothing changed" is
# visible in the log instead of being discovered 32 minutes later.
with tempfile.TemporaryDirectory() as td:
    z = Path(td) / "zones"
    z.mkdir(parents=True)
    (z / "CAM.112_zone.json").write_text("MY EDITED ZONES", encoding="utf-8")
    _records = []
    _h = logging.Handler()
    _h.emit = _records.append
    _dlog = logging.getLogger("kevacv.drive")
    _dlog.addHandler(_h)
    try:
        fetch_assets("folder", zones_dir=z, entries=ENTRIES,
                     download=lambda f, o: None)
    finally:
        _dlog.removeHandler(_h)
    _warns = [r.getMessage() for r in _records if r.levelno >= logging.WARNING]
    check(any("IGNORING" in m and "CAM.112_zone.json" in m for m in _warns),
          "skipping Drive's copy WARNS, naming the file",
          _warns or "no warning was emitted at all")

with tempfile.TemporaryDirectory() as td:
    # the precedence is a parameter, not a hardcoded policy — a venue that
    # really does draw zones in the mapper can flip it without editing the
    # module, and accept that Drive then overwrites local edits
    z = Path(td) / "zones"
    z.mkdir(parents=True)
    (z / "CAM.112_zone.json").write_text("LAST RUN'S ZONES", encoding="utf-8")
    pulled = []
    fetch_assets("folder", zones_dir=z, entries=ENTRIES, prefer_drive=("zones",),
                 download=lambda f, o: pulled.append(f))
    check("id_zone" in pulled,
          'prefer_drive=("zones",) restores Drive-wins for zones')


print()
print("=" * 74)
print("  a Drive download needs the file ID, never the filename")
print("=" * 74)
ENT = [(f"id{i}", n) for i, n in enumerate(LISTING)]
with tempfile.TemporaryDirectory() as td:
    got = []

    def dl_id(fid, out):
        got.append(fid)
        Path(out).write_bytes(b"v")

    p = fetch_chunk("folder", td, chunk_filter="7.30.00pm", entries=ENT,
                    download=dl_id)
    check(got == ["id3"], "the ID of the SELECTED chunk is what gets fetched",
          str(got))
    check("7.30.00pm CDT - " in p.name, "and the file lands under its real name")

with tempfile.TemporaryDirectory() as td:
    try:
        fetch_chunk("folder", td, chunk_filter="7.30.00pm", listing=LISTING)
        check(False, "a listing with no IDs refuses to guess")
    except DriveError as e:
        check("no Drive file id" in str(e),
              "a listing with no IDs refuses to guess",
              "passing the filename to gdown made it try to open a URL")

print()
print("=" * 74)
if FAILED:
    print(f"  {len(FAILED)} FAILED:")
    for f in FAILED:
        print(f"    - {f}")
    sys.exit(1)
print("  ALL PASS")
print("=" * 74)
