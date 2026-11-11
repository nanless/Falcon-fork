import json
from pathlib import Path

import pytest

from tools.streaming.relocate_local_indexes import INDEX_NAMES, SPLITS, relocated_copies


def fixture_tree(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "relocated"
    source = new
    originals = {}
    for split in SPLITS:
        for name in ("audio.wav", "mouth.pt"):
            path = new / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"media")
        data = {
            "paths": [str(old / split / "audio.wav"), [str(old / split / "mouth.pt"), 123]],
            "unrelated": "/other/storage/file.wav",
        }
        for name in INDEX_NAMES:
            path = source / split / name
            path.write_text(json.dumps(data), encoding="utf-8")
            originals[path] = path.read_bytes()
    return old, new, source, originals


def test_copies_all_indexes_and_preserves_originals(tmp_path):
    old, new, source, originals = fixture_tree(tmp_path)
    output = tmp_path / "adapted"
    report = relocated_copies(source, old, new, output)
    assert len(report["indexes"]) == 6
    assert all(item["replaced_paths"] == 2 for item in report["indexes"])
    assert all(path.read_bytes() == data for path, data in originals.items())
    for split in SPLITS:
        for name in INDEX_NAMES:
            result = json.loads((output / split / name).read_text())
            assert result["paths"][0] == str((new / split / "audio.wav").resolve())
            assert result["paths"][1] == [str((new / split / "mouth.pt").resolve()), 123]
            assert result["unrelated"] == "/other/storage/file.wav"
    assert json.loads((output / "relocation-report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        relocated_copies(source, old, new, output)


def test_missing_media_fails_before_creating_output(tmp_path):
    old, new, source, originals = fixture_tree(tmp_path)
    (new / "cv" / "mouth.pt").unlink()
    output = tmp_path / "adapted"
    with pytest.raises(ValueError, match="missing"):
        relocated_copies(source, old, new, output)
    assert not output.exists()
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_path_traversal_and_root_prefix_collision_are_rejected(tmp_path):
    old, new, source, _ = fixture_tree(tmp_path)
    path = source / "tr" / "mix.json"
    path.write_text(json.dumps([str(old) + "/../escape"]), encoding="utf-8")
    with pytest.raises(ValueError, match="outside new root"):
        relocated_copies(source, old, new, tmp_path / "out")
    path.write_text(json.dumps([str(old) + "-other/file"]), encoding="utf-8")
    with pytest.raises(ValueError, match="no matching paths"):
        relocated_copies(source, old, new, tmp_path / "out")


def test_alias_in_legacy_paths_is_understood(tmp_path):
    old, new, source, _ = fixture_tree(tmp_path)
    old.mkdir()
    alias = tmp_path / "legacy_alias"
    alias.symlink_to(old, target_is_directory=True)
    report = relocated_copies(source, alias, new, tmp_path / "adapted")
    assert len(report["indexes"]) == 6
