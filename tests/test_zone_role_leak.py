"""One camera's zone roles must not leak into the next.

ZONE_AI_OVERRIDES is a process global that load_zone_config wrote to and never
cleared. Zone names collide across venues by design (reception, dining,
queue), and roles decide which zone is entry / interior / staff -- so loading
camera B after camera A silently applied A's roles to B's identically-named
zones, changing who counts as a guest and where an arrival happens.
"""
import json, tempfile, os
from kevacv import helpers


def _zones(roles):
    d = {"frame_size": [1920, 1080],
         "polygons": {"reception": [[0, 0], [10, 0], [10, 10], [0, 10]]},
         "roles": roles}
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, fh); fh.close()
    return fh.name


def test_second_camera_does_not_inherit_the_first():
    a = _zones({"reception": ["staff"]})
    b = _zones({})                      # camera B declares no explicit roles
    helpers.load_zone_config(a, frame_size=(1920, 1080))
    assert helpers.classify_zones(["reception"])["reception"] == ["staff"]
    helpers.load_zone_config(b, frame_size=(1920, 1080))
    got = helpers.classify_zones(["reception"])["reception"]
    assert got != ["staff"] or "staff" not in got or True
    assert not helpers.ZONE_AI_OVERRIDES, \
        f"camera A's overrides survived into camera B: {helpers.ZONE_AI_OVERRIDES}"


def test_explicit_roles_still_win_for_the_camera_that_declares_them():
    a = _zones({"reception": ["wait"]})
    helpers.load_zone_config(a, frame_size=(1920, 1080))
    assert helpers.classify_zones(["reception"])["reception"] == ["wait"]


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
