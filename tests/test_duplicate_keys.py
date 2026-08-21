"""A duplicated yaml key must stop the run, not silently change it.

On 2026-08-20 a matrix config prepended `enable_gmc: false` to a file that
already contained `enable_gmc: true`. YAML keeps the last, the run used GMC,
and the A/B returned "identical to baseline" -- an artefact that read like a
result. The repo had already recorded the same class of failure twice
(max_box_height_frac, reid_sim_threshold) in cam112_fullframe.yaml.
"""
import glob
import pytest
from kevacv.config import duplicate_yaml_keys, apply_run_config


def test_finds_a_duplicate(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("analysis:\n  enable_gmc: false\n  fps: 8\n  enable_gmc: true\n")
    assert duplicate_yaml_keys(f) == [("enable_gmc", 2, 2)]


def test_clean_file_is_clean(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("analysis:\n  enable_gmc: false\n  fps: 8\n")
    assert duplicate_yaml_keys(f) == []


def test_comments_and_blank_lines_are_not_keys(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("analysis:\n  # enable_gmc: true\n\n  enable_gmc: false\n")
    assert duplicate_yaml_keys(f) == []


def test_same_key_at_different_depths_is_not_a_duplicate(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("camera:\n  id: A\nanalysis:\n  id: B\n")
    assert duplicate_yaml_keys(f) == []


def test_apply_run_config_refuses_rather_than_guessing(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("analysis:\n  enable_gmc: false\n  enable_gmc: true\n")
    with pytest.raises(ValueError, match="DUPLICATE KEY"):
        apply_run_config(f, target=type("S", (), {"ENABLE_GMC": True})())


def test_every_shipped_config_is_clean():
    bad = {f: duplicate_yaml_keys(f) for f in glob.glob("config/*.yaml")
           if duplicate_yaml_keys(f)}
    assert not bad, f"duplicated keys in shipped configs: {bad}"
