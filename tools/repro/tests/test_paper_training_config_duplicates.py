import pytest

from tools.streaming.train_paper_lookahead import load_training_config


def test_current_config_loads_without_duplicate_keys():
    data = load_training_config("configs/RealSSAPaperLookahead.yml")
    assert data["batch_size"] == 1024
    assert data["lr"] == 0.01


@pytest.mark.parametrize("content", [
    "lr: 0.01\nlr: 0.1\n",
    "outer:\n  limit: 4\n  limit: 5\n",
])
def test_duplicate_config_keys_are_rejected(tmp_path, content):
    path = tmp_path / "duplicate.yml"
    path.write_text(content)
    with pytest.raises(ValueError, match="duplicate configuration key"):
        load_training_config(path)


def test_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "list.yml"
    path.write_text("- one\n- two\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_training_config(path)


@pytest.mark.parametrize("key, value", [
    ("batch_size", True),
    ("warmup_steps", 3.5),
    ("checkpoint_chunks", 1),
    ("lr", float("nan")),
    ("weight_decay", float("inf")),
    ("weight_decay", -0.1),
    ("min_delta", -0.1),
    ("lr_patience", -1),
    ("history_frames", -1),
    ("width", 193),
    ("heads", 0),
    ("seed", 2**32 - 1),
    ("visual_weights", ""),
])
def test_config_rejects_invalid_types_values_and_ranges(key, value):
    from tools.streaming.train_paper_lookahead import validate_training_config
    data = load_training_config("configs/RealSSAPaperLookahead.yml")
    data[key] = value
    with pytest.raises(ValueError):
        validate_training_config(data)


def test_config_accepts_actual_recipe_and_valid_zeroes():
    from tools.streaming.train_paper_lookahead import validate_training_config
    data = load_training_config("configs/RealSSAPaperLookahead.yml")
    validate_training_config(data)
    data.update(weight_decay=0, min_delta=0, warmup_steps=0, history_frames=0, lr_patience=0, workers=0)
    validate_training_config(data)
