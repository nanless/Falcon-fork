from collections import Counter
from random import Random

import pytest
import torch

from tools.streaming import rolling_init_ab as init
from tools.streaming.rolling_paired_supervision_ab import (
    ARMS, make_schedule, overlap_blocks, split_updates, validate_updates,
)


def test_matched_updates_use_same_labels_in_different_optimizer_batches():
    rows = list(range(8))
    blocks = [5, 6, 5, 7, 5, 9, 6, 5]
    item = split_updates(rows, blocks, Random(42))
    validate_updates(item)
    for arm in ARMS:
        assert Counter((r[0], r[1], r[-1]) for update in item[arm] for r in update) == {
            (i, direction, block): 1 for i, block in enumerate(blocks) for direction in (0, 1)
        }
    for update in item["paired_on"]:
        assert all(a[0] == b[0] and a[-1] == b[-1] and a[1] != b[1]
                   for a, b in zip(update[::2], update[1::2], strict=True))
    assert all(len({r[0] for r in update}) == 8 for update in item["paired_off"])
    item["paired_off"][0][0] = item["paired_off"][0][1]
    with pytest.raises(ValueError):
        validate_updates(item)


def test_schedule_only_samples_jointly_active_windows_and_covers_eligible_pairs():
    t = torch.zeros(2, init.TOTAL)
    t[:, 5 * init.BLOCK_SAMPLES: 5 * init.BLOCK_SAMPLES + 2 * init.TICK_SAMPLES] = .1
    t[0, 6 * init.BLOCK_SAMPLES: 7 * init.BLOCK_SAMPLES] = .1
    t[1, 7 * init.BLOCK_SAMPLES: 8 * init.BLOCK_SAMPLES] = .1
    assert overlap_blocks({"targets": t}) == [5]
    train = [{"targets": t}] * 13 + [{"targets": torch.zeros_like(t)}]
    schedule, eligible = make_schedule(train, 8)
    assert eligible == {i: [5] for i in range(13)}
    assert schedule == make_schedule(train, 8)[0]
    assert {row[0] for step in schedule for batch in step["paired_off"] for row in batch} == set(eligible)
