import copy
import hashlib
import json

import pytest
import torch

from look2hear.models.FalconStreaming import FalconStreaming
from tools.streaming.common import CHECKPOINT_FORMAT
from tools.streaming.train_ddp import TrainingWrapper
from tools.streaming.inspect_quality_embeddings import (
    candidate_groups, inspect_checkpoint, retrieval_summary, select_samples, validate_development_index,
)


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def fake_indices():
    rows = [{'id': f'item-{i}', 'source_group_sha256': h(f'group-{i % 64}'),
             'target_source_sha256': h(f'target-{i}'), 'shard': 0,
             'fields': {'target': [i, 20]}} for i in range(512)]
    parent = {'format': 'realssa-tar-index-v1', 'split': 'cv', 'source_split': 'tr',
              'partition_kind': 'derived_video_directory_group_holdout', 'sample_count': 512,
              'incomplete_count': 0, 'incomplete': [], 'shards': [{'path': '/example'}],
              'samples': copy.deepcopy(rows)}
    fast = copy.deepcopy(parent)
    fast['selection'] = {'role': 'fast_dev_from_source_tr', 'parent_index_sha256': h('parent'),
                         'parent_sample_count': 512}
    manifest = {'format': 'falcon-realssa-source-holdout-v1', 'source_split': 'tr',
                'derived_index_sha256': {'cv': h('parent')},
                'video_group_assignments': {r['source_group_sha256']: 'cv' for r in rows}}
    return fast, parent, manifest


def test_accepts_matching_development_index_and_rejects_tt(fake_indices):
    fast, parent, manifest = fake_indices
    assert len(validate_development_index(fast, parent, manifest, h('parent'))) == 512
    fast['split'] = 'tt'
    with pytest.raises(ValueError, match='source-tr-derived cv'):
        validate_development_index(fast, parent, manifest, h('parent'))


@pytest.mark.parametrize('corruption', ['row','group_role','shards','parent_hash','source'])
def test_rejects_forged_development_identity(fake_indices, corruption):
    fast, parent, manifest = fake_indices
    if corruption == 'row': fast['samples'][2]['fields']['target'][0] += 1
    if corruption == 'group_role': manifest['video_group_assignments'][fast['samples'][0]['source_group_sha256']] = 'tt'
    if corruption == 'shards': fast['shards'][0]['path'] = '/other'
    if corruption == 'parent_hash': fast['selection']['parent_index_sha256'] = h('wrong')
    if corruption == 'source': fast['source_split'] = 'tt'
    with pytest.raises(ValueError):
        validate_development_index(fast, parent, manifest, h('parent'))


def test_selection_is_deterministic_distinct_and_skips_silent(fake_indices):
    samples, _, _ = fake_indices
    ordered = candidate_groups(samples['samples'], 12)
    silent_index = ordered[0][1][0]
    mix = torch.ones(32000)
    target = torch.linspace(-.5, .5, 32000)
    video = torch.zeros((50, 88, 88))
    def loader(i):
        return mix, torch.zeros_like(target) if i == silent_index else target, video
    a, skipped = select_samples(samples['samples'], loader, seed=12, count=48)
    b, _ = select_samples(samples['samples'], loader, seed=12, count=48)
    assert [r['index'] for r in a] == [r['index'] for r in b]
    assert len({r['source_group_sha256'] for r in a}) == 48
    assert silent_index not in [r['index'] for r in a]
    assert {'index': silent_index, 'id': samples['samples'][silent_index]['id'], 'reason': 'degenerate_target'} in skipped


def test_retrieval_ranking_has_explicit_stable_ties_and_rejects_zero():
    result = retrieval_summary(torch.eye(6), torch.eye(6))
    assert result['top1_count'] == result['top5_count'] == 6
    assert result['mean_rank'] == pytest.approx(1.)
    tied = retrieval_summary(torch.ones(6, 4), torch.ones(6, 4))
    assert tied['top1_count'] == 1 and tied['top5_count'] == 5
    assert tied['mean_rank'] == pytest.approx(3.5) and tied['ties_to_correct_count'] == 6
    with pytest.raises(ValueError, match='zero-norm'):
        retrieval_summary(torch.zeros(2, 4), torch.ones(2, 4))


def test_actual_checkpoint_embeddings_and_missing_head(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(44)
    model = FalconStreaming(channels=8, dilations=(1,)).eval()
    wrapper = TrainingWrapper(model, True).eval()
    ckpt = {'format': CHECKPOINT_FORMAT, 'model_config': {'channels': 8, 'dilations': [1]},
            'state_dict': model.state_dict(), 'speaker_pool_state': wrapper.speaker_pool.state_dict(),
            'steps': 3, 'training_status': 'synthetic_smoke'}
    path = tmp_path/'model.pt'
    torch.save(ckpt, path)
    gen = torch.Generator().manual_seed(45)
    selected = [{'tensors': (torch.randn(32000, generator=gen), torch.randn(32000, generator=gen),
                              torch.randn(50, 88, 88, generator=gen))} for _ in range(2)]
    result = inspect_checkpoint(path, selected, 'cpu', 2)
    assert result['n'] == 2 and result['steps'] == 3
    assert len(result['ranks']) == 2 and result['sha256'] and result['model_config']['channels'] == 8
    ckpt.pop('speaker_pool_state')
    bad = tmp_path/'no-head.pt'
    torch.save(ckpt, bad)
    with pytest.raises(ValueError, match='speaker_pool_state'):
        inspect_checkpoint(bad, selected, 'cpu', 2)
