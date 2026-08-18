"""drive.py — pick ONE chunk from the Drive folder and fetch it.

THE BUG THIS IS SHAPED AROUND
    The notebook did this in an order that produced a wrong report:

        1. list the folder
        2. parse the clock from vids[0]              <- the 4:30pm file
        3. DOWNLOAD vids[0]                          <- the 4:30pm file
        4. apply CHUNK_FILTER="7.30.00pm"            <- selects a DIFFERENT file
        5. re-parse the clock from the new pick      <- 19:30
        6. analyse whatever is on disk               <- still the 4:30pm file

    Every step was individually correct. The run analysed 16:30 footage and
    stamped 19:30 on it, and nothing complained, because no single step was
    wrong — only their order.

    Two rules follow, and both are enforced here rather than documented:

        SELECT BEFORE FETCHING. The filter runs on the listing. Nothing is
        downloaded until exactly one file has been chosen.

        THE PATH IS THE ONLY OUTPUT. fetch_chunk returns the file it actually
        downloaded, and the caller derives the clock from THAT path. There is
        no second variable to drift.

MATCHING
    Filenames carry a RANGE:
        "CAM.112 (PP.09_12) 7-28-2026, 4.30.00pm CDT - 7-28-2026, 5.30.00pm CDT"
    A bare substring search for "5.30.00pm" matches this file's END as well as
    the next file's START. Matching is therefore done against the text BEFORE
    the range separator, i.e. the chunk's own start time.
"""
from __future__ import annotations

import re
from pathlib import Path

from .clock import parse_start
from .log import get_logger

_log = get_logger("drive")

SEP = " - "                    # separates start and end in the filename
VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov")


class DriveError(RuntimeError):
    pass


def start_part(name):
    """The text before the range separator — the chunk's OWN start time."""
    return str(name).split(SEP)[0]


def select(names, chunk_filter=None, index=None):
    """Choose exactly one chunk from a listing. -> the chosen name.

    Raises rather than guessing. A filter that matches several files, or none,
    is a question for a human — silently taking the first match is how the
    wrong hour gets analysed.
    """
    vids = [n for n in names if str(n).lower().endswith(VIDEO_EXT)]
    if not vids:
        raise DriveError(f"no video files in the listing ({len(names)} entries)")
    vids = sorted(vids, key=lambda n: (parse_start(n) or "", str(n)))

    if chunk_filter:
        hits = [n for n in vids if str(chunk_filter) in start_part(n)]
        if not hits:
            raise DriveError(
                f"CHUNK_FILTER {chunk_filter!r} matched none of {len(vids)} "
                f"chunks. Start times available: "
                f"{[str(parse_start(v)) for v in vids[:8]]}")
        if len(hits) > 1:
            raise DriveError(
                f"CHUNK_FILTER {chunk_filter!r} matched {len(hits)} chunks: "
                f"{[Path(h).name for h in hits]}. Narrow it — analysing the "
                f"wrong hour is worse than analysing none.")
        return hits[0]

    if index is not None:
        try:
            return vids[int(index)]
        except IndexError:
            raise DriveError(f"index {index} out of range; {len(vids)} chunks")

    raise DriveError(
        f"{len(vids)} chunks available and no selection given. Pass "
        f"chunk_filter or index. Start times: "
        f"{[str(parse_start(v)) for v in vids[:8]]}")


def list_folder(folder_id, cache=None):
    """-> [names] in a Drive folder, via gdown.

    gdown is imported here, not at module import: the rest of the package must
    stay usable on a machine that has never seen it.
    """
    if cache:
        return list(cache)
    try:
        import gdown
    except ImportError:
        raise DriveError("gdown is not installed — pip install gdown, or pass "
                         "an already-downloaded file with --video")
    try:
        files = gdown.download_folder(id=folder_id, quiet=True,
                                      skip_download=True) or []
    except Exception as e:
        raise DriveError(f"could not list Drive folder {folder_id}: {e}")
    return [getattr(f, "path", str(f)) for f in files]


def fetch_chunk(folder_id, dest_dir, chunk_filter=None, index=None,
                listing=None, entries=None, download=None):
    """Select ONE chunk, then fetch it. -> Path of the downloaded file.

    The returned path is the single source of truth for what was analysed and
    what time it is. Callers must not keep a separate "selected name".

    Drive downloads need the file ID, not the filename — the first version of
    this passed the name straight to gdown, which tried to treat
    "CAM.112 (PP.09_12) 7-28-2026, ....mp4" as a URL. The listing carries both,
    so we keep the pairing instead of throwing the ID away.
    """
    if entries is None and listing is None:
        entries = _folder_entries(folder_id)
    if entries is not None:
        id_of = {n: i for i, n in entries}
        names = [n for _i, n in entries]
    else:
        id_of, names = {}, list(listing)
    chosen = select(names, chunk_filter=chunk_filter, index=index)
    file_id = id_of.get(chosen)
    start = parse_start(chosen)
    _log.info(f"selected chunk: {Path(chosen).name}")
    _log.info(f"  its start time parses to {start}")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / Path(chosen).name

    if target.exists() and target.stat().st_size > 0:
        _log.info(f"  already on disk ({target.stat().st_size/1e9:.2f} GB) — "
                  f"not re-downloading")
        return target

    if download is None:
        if not file_id:
            raise DriveError(f"no Drive file id for {Path(chosen).name!r} — "
                             f"cannot download it by name alone")
        try:
            import gdown
        except ImportError:
            raise DriveError("gdown is not installed and the file is not on "
                             "disk")

        def download(fid, out):
            gdown.download(id=fid, output=str(out), quiet=False)
    _log.info(f"  downloading -> {target}")
    download(file_id or chosen, target)

    if not target.exists() or target.stat().st_size == 0:
        raise DriveError(f"download produced no file at {target}")
    # The path we return IS the file we analysed. Re-parsing the clock from it
    # is what makes a selected/decoded mismatch impossible rather than merely
    # unlikely.
    if parse_start(target.name) != start:
        raise DriveError(
            f"downloaded file {target.name!r} does not carry the start time of "
            f"the file that was selected ({start}) — refusing to continue")
    return target


ZONE_SUFFIX = ("_zone.json", "zones.json")
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def _folder_entries(folder_id):
    """-> [(file_id, name)] for everything in the folder, videos included."""
    try:
        import gdown
    except ImportError:
        raise DriveError("gdown is not installed")
    try:
        files = gdown.download_folder(id=folder_id, quiet=True,
                                      skip_download=True) or []
    except Exception as e:
        raise DriveError(f"could not list Drive folder {folder_id}: {e}")
    out = []
    for f in files:
        name = getattr(f, "path", str(f))
        out.append((getattr(f, "id", None), name))
    return out


# WHICH SIDE OWNS WHICH ASSET
#       zones    LOCAL wins. The zone map is versioned WITH the code, because
#                it is geometry the code is tuned against — a line moved in
#                the mapper and a threshold moved in config.py have to land in
#                the same commit or neither is reproducible.
#       video    DRIVE, always — it is only ever downloaded.
#       gallery  LOCAL wins. Staff photos are curated in staff_gallery/ and
#                versioned with the code, so a Drive copy must not overwrite
#                a deliberate local set.
#
#     Anything not listed defaults to local-wins, which is the conservative
#     choice: it can only ever leave you with a stale file, never destroy a
#     deliberate one.
#
#     HISTORY. This was ("zones",) for part of 2026-08-13, after a re-drawn map
#     was uploaded to Drive and ignored for a whole run. That made Drive able to
#     silently overwrite a hand-corrected local map — and it contradicted
#     fetch_assets' own docstring two lines down, which still promised "a zone
#     map you have edited beats whatever is sitting in Drive". Reverted by the
#     operator 2026-08-13: the repo is the source of truth for zones. The
#     original complaint is real, so fetch_assets LOGS the Drive copy it
#     skipped rather than staying silent about it.
DRIVE_AUTHORITATIVE = ()


def _wins_from_drive(key, prefer_drive=None):
    """Should Drive's copy overwrite an existing local file for this class?"""
    if prefer_drive is None:
        prefer_drive = DRIVE_AUTHORITATIVE
    return key in set(prefer_drive)


def fetch_assets(folder_id, zones_dir=None, gallery_dir=None, entries=None,
                 download=None, prefer_drive=None):
    """Pull the SMALL companions of the video — zone map and staff photos.

    WHY THIS BELONGS IN THE PIPELINE
        The zone JSON and the staff gallery live in the same Drive folder as
        the footage, and the notebook fetched them. Without them the codebase
        path silently degrades: no zone file means no analysis at all, and no
        staff photo means every staff member is an unnamed low-confidence
        'zone-inferred' row in the report.

        Videos are excluded on purpose. This is for the kilobytes, not the
        gigabytes — the chunk is selected deliberately by fetch_chunk, never
        swept up by an asset sync.

    Existing local files are NOT overwritten: a zone map you have edited beats
    whatever is sitting in Drive.
    """
    got = {"zones": [], "gallery": [], "skipped": []}
    _log.info(f"  asset sourcing: " + ", ".join(
        f"{k}={'DRIVE' if _wins_from_drive(k, prefer_drive) else 'local'}"
        for k in ("zones", "gallery")))
    ents = entries if entries is not None else _folder_entries(folder_id)

    if download is None:
        try:
            import gdown
        except ImportError:
            raise DriveError("gdown is not installed")

        def download(file_id, out):
            gdown.download(id=file_id, output=str(out), quiet=True)

    for fid, name in ents:
        base = Path(name).name
        low = base.lower()
        if low.endswith(VIDEO_EXT):
            continue
        if low.endswith(ZONE_SUFFIX) and zones_dir:
            target = Path(zones_dir) / base
            key = "zones"
        elif low.endswith(IMAGE_EXT) and gallery_dir:
            target = Path(gallery_dir) / base
            key = "gallery"
        else:
            got["skipped"].append(base)
            continue
        if target.exists() and not _wins_from_drive(key, prefer_drive):
            # Loud on purpose. The previous wording ("keeping yours") read as
            # reassurance, so a zone map re-drawn and re-uploaded to Drive was
            # ignored for a whole run and nobody noticed. Local still wins —
            # the repo owns the geometry — but a Drive copy EXISTS and is being
            # discarded, and that is a warning, not a status line.
            _log.warning(f"  {base}: Drive has a copy, IGNORING it — the local "
                         f"{key} file wins. If you re-drew this in the mapper, "
                         f"copy it into the repo; Drive uploads are not picked up.")
            got[key].append(str(target))
            continue
        if target.exists():
            # Drive is the source of truth for this asset class, so a local
            # copy is a STALE ARTEFACT of an earlier run, not an edit to
            # preserve. Saying so matters: a zone map re-drawn and re-uploaded
            # to Drive was silently ignored for a whole run because the old
            # local file won, and the log line that said so read like
            # reassurance ("keeping yours") rather than a warning.
            _log.warning(f"  {base} REPLACED from Drive (Drive is authoritative "
                         f"for {key}; the local copy was from an earlier run)")
        if fid is None:
            _log.warning(f"  {base}: no file id from the listing, cannot fetch")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _log.info(f"  fetching {base} -> {target}")
        download(fid, target)
        if target.exists():
            got[key].append(str(target))
    _log.info(f"assets: {len(got['zones'])} zone file(s), "
              f"{len(got['gallery'])} gallery image(s), "
              f"{len(got['skipped'])} ignored")
    return got


def describe(names, chunk_filter=None):
    """A human-readable listing, so picking a chunk is not guesswork."""
    vids = [n for n in names if str(n).lower().endswith(VIDEO_EXT)]
    L = [f"{len(vids)} chunk(s) available"]
    for n in sorted(vids, key=lambda x: (parse_start(x) or "", str(x))):
        st = parse_start(n)
        mark = ""
        if chunk_filter and str(chunk_filter) in start_part(n):
            mark = "   <- matches your filter"
        L.append(f"  {str(st):<20} {Path(n).name}{mark}")
    return "\n".join(L)
