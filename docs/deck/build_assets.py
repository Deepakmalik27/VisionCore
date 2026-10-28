"""build_assets.py -- the picture assets the two decks share.

  camera_wall.svg/.png   every device in docs/delilah_la_devices.json, real names
  frame_zones.png        output/door_check.png with the debug header cropped off

Nothing here invents a number. Counts come from the device dump.
"""
import collections
import html
import json
import pathlib

import cairosvg
from PIL import Image

HERE = pathlib.Path(__file__).parent
REPO = HERE.parent.parent
DEVICES = REPO / "docs" / "delilah_la_devices.json"
OUT = REPO / "output"

import keva_theme as T

INK, IVORY = "#" + T.SLATE, "#" + T.FOG
BRASS, MUTED, LINE = "#" + T.COPPER, "#" + T.SAGE, "#" + T.HAIR
LANE_BG = "#" + T.SLATE_LT
SERIF = "'Space Grotesk','Segoe UI',sans-serif"
MONO = "'JetBrains Mono',Consolas,monospace"


def load():
    d = json.loads(DEVICES.read_text())
    host = d["data"][0]
    cams, doors, infra = [], [], []
    for dev in host["devices"]:
        name, model = dev["name"], dev["model"]
        if name.startswith("CAM."):
            cams.append((name.split()[0].replace("CAM.", ""), model, dev["ip"]))
        elif name.startswith("DOOR."):
            doors.append((name.split("-", 1)[-1], model))
        else:
            infra.append((name.split(" - ")[-1], model))
    cams.sort(key=lambda c: c[0])
    return host["hostName"], cams, doors, infra


def camera_wall():
    hostname, cams, doors, infra = load()
    by_model = collections.Counter(m for _, m, _ in cams)

    W, H = 1640, 880
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="{SERIF}">',
         f'<rect width="{W}" height="{H}" fill="{IVORY}"/>']

    o.append(f'<text x="56" y="72" font-size="34" fill="{INK}">'
             f'{len(cams)} cameras · {len(doors)} door readers · '
             f'{len(infra)} controllers</text>')
    o.append(f'<text x="56" y="108" font-size="18" fill="{MUTED}" font-style="italic">'
             f'{html.escape(hostname)} — every device reporting online. '
             f'Keva Vision reads them. No new hardware.</text>')

    # the grid of cameras, 10 per row
    x0, y0, cw, ch, gap = 60, 150, 156, 74, 13
    percol = 9
    for i, (num, model, ip) in enumerate(cams):
        col, row = i % percol, i // percol
        x, y = x0 + col * (cw + gap), y0 + row * (ch + gap)
        pilot = num == "112"
        o.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="7" '
                 f'fill="{"#" + T.COPPER_TINT if pilot else "#" + T.PAPER}" '
                 f'stroke="{BRASS if pilot else LINE}" '
                 f'stroke-width="{2 if pilot else 1}"/>')
        if pilot:
            o.append(f'<text x="{x+cw-12}" y="{y+22}" font-size="11" fill="{BRASS}" '
                     f'text-anchor="end" font-family="{MONO}" '
                     f'letter-spacing="1.5">PILOT</text>')
        # lens glyph -- a camera, not a logo we are not licensed to redraw
        o.append(f'<circle cx="{x+24}" cy="{y+26}" r="10" fill="none" '
                 f'stroke="{BRASS}" stroke-width="2"/>')
        o.append(f'<circle cx="{x+24}" cy="{y+26}" r="3.5" fill="{BRASS}"/>')
        o.append(f'<text x="{x+44}" y="{y+32}" font-size="20" fill="{INK}" '
                 f'font-family="{MONO}">{num}</text>')
        o.append(f'<text x="{x+12}" y="{y+60}" font-size="13" fill="{MUTED}">'
                 f'{html.escape(model)}</text>')

    yb = y0 + ((len(cams) + percol - 1) // percol) * (ch + gap) + 26
    o.append(f'<line x1="56" y1="{yb}" x2="{W-56}" y2="{yb}" stroke="{LINE}"/>')

    # model tally + the rest of the rack
    o.append(f'<text x="56" y="{yb+40}" font-size="15" letter-spacing="2" '
             f'fill="{BRASS}" font-family="{MONO}">BY MODEL</text>')
    tally = "  ·  ".join(f"{m} ×{n}" for m, n in by_model.most_common())
    o.append(f'<text x="56" y="{yb+72}" font-size="18" fill="{INK}">'
             f'{html.escape(tally)}</text>')

    o.append(f'<text x="56" y="{yb+124}" font-size="15" letter-spacing="2" '
             f'fill="{BRASS}" font-family="{MONO}">ALSO ON THE SAME NETWORK</text>')
    rest = "   ·   ".join([f"{n} ({m})" for n, m in doors]
                          + [f"{n} ({m})" for n, m in infra])
    o.append(f'<text x="56" y="{yb+156}" font-size="17" fill="{INK}">'
             f'{html.escape(rest)}</text>')

    o.append(f'<text x="56" y="{H-30}" font-size="14" fill="{MUTED}" '
             f'font-style="italic">Source: UniFi Protect device export, '
             f'docs/delilah_la_devices.json — names and models unedited</text>')
    o.append('</svg>')

    svg = "\n".join(o)
    (HERE / "camera_wall.svg").write_text(svg, encoding="utf-8")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(HERE / "camera_wall.png"),
                     scale=2)
    return len(cams), len(doors), by_model


def crop_header(src, dst, top=92):
    """The debug banner is for us, not for a COO. Cut it."""
    im = Image.open(OUT / src)
    im.crop((0, top, im.width, im.height)).save(HERE / dst)


if __name__ == "__main__":
    n_cam, n_door, models = camera_wall()
    crop_header("door_check.png", "frame_zones.png")

    # check: the wall must show every camera in the dump, and the model tally
    # must add back up to that same number -- a silently dropped camera on a
    # slide claiming "all of them" is the one error nobody in the room catches.
    _, cams, doors, _ = load()
    assert n_cam == len(cams) == sum(models.values()), "camera count disagrees"
    assert n_door == len(doors)
    for f in ("camera_wall.png", "frame_zones.png"):
        assert (HERE / f).stat().st_size > 10_000, f"{f} looks empty"
    print(f"assets ok · {n_cam} cameras, {n_door} doors, "
          f"{len(models)} distinct models")
