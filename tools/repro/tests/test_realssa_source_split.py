import hashlib
import json
from pathlib import Path

import pytest

from tools.streaming.split_realssa_sources import create, derive, id_digest


def fixtures():
    samples = [{'id': f'id-{i}', 'shard': 0, 'fields': {'target': [0, 1]}} for i in range(12)]
    index = {'format': 'realssa-tar-index-v1', 'split': 'tr', 'sample_count': len(samples),
             'incomplete_count': 0, 'incomplete': [], 'shards': [], 'samples': samples}
    # Two distinct target paths per group; six video groups.
    records = [{'split': 'tr', 'id_sha256': id_digest(s['id']),
                'target_sha256': hashlib.sha256(f'target-{i}'.encode()).hexdigest(),
                'video_sha256': hashlib.sha256(f'video-{i//2}'.encode()).hexdigest()}
               for i, s in enumerate(samples)]
    return index, records


def test_derivation_preserves_all_samples_and_disjoins_groups_and_targets():
    index, records = fixtures()
    before = json.dumps(index, sort_keys=True)
    partitions, manifest = derive(index, records, 17, .2, .2)
    assert before == json.dumps(index, sort_keys=True)
    assert sum(x['sample_count'] for x in partitions.values()) == 12
    assert {r['id'] for p in partitions.values() for r in p['samples']} == {r['id'] for r in index['samples']}
    sets = {role: {r['source_group_sha256'] for r in part['samples']} for role,part in partitions.items()}
    targets = {role: {r['target_source_sha256'] for r in part['samples']} for role,part in partitions.items()}
    for a,b in [('tr','cv'),('tr','tt'),('cv','tt')]:
        assert sets[a].isdisjoint(sets[b]) and targets[a].isdisjoint(targets[b])
    assert all(p['source_split'] == 'tr' and p['split'] == role for role,p in partitions.items())
    assert 'not verified speaker' in manifest['caveat']
    assert 'already seen' in manifest['warm_start']


def test_seed_is_reproducible_and_changes_assignment():
    idx, records = fixtures()
    a, ma = derive(idx, records, 17, .2, .2)
    b, mb = derive(idx, list(reversed(records)), 17, .2, .2)
    c, mc = derive(idx, records, 18, .2, .2)
    assert a == b and ma == mb
    assert ma['video_group_assignments'] != mc['video_group_assignments']


@pytest.mark.parametrize('alter, message', [
    ('missing', 'exactly match'),
    ('duplicate', 'duplicate'),
    ('target_video_conflict', 'multiple video'),
])
def test_unverifiable_audit_is_rejected(alter, message):
    idx, records = fixtures()
    if alter == 'missing': records.pop()
    if alter == 'duplicate': records.append(dict(records[0]))
    if alter == 'target_video_conflict': records[2]['target_sha256'] = records[0]['target_sha256']
    with pytest.raises(ValueError, match=message):
        derive(idx, records, 17, .2, .2)



def test_unknown_sources_are_explicitly_excluded_and_accounted():
    idx, records = fixtures()
    records[0]['video_sha256'] = None
    records[2]['target_sha256'] = None
    parts, manifest = derive(idx, records, 17, .2, .2)
    assert manifest['excluded_samples'] == 2
    assert {x['id'] for x in manifest['excluded']} == {'id-0', 'id-2'}
    assert sum(p['sample_count'] for p in parts.values()) + manifest['excluded_samples'] == len(idx['samples'])
    assert 'id-0' not in {s['id'] for part in parts.values() for s in part['samples']}



def test_output_is_non_overwriting_and_sha_bound(tmp_path):
    idx, records = fixtures()
    index_path=tmp_path/'index.json'; index_path.write_text(json.dumps(idx))
    records_path=tmp_path/'records.jsonl'; records_path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    audit_path=tmp_path/'audit.json'
    audit_path.write_text(json.dumps({'index_sha256': {'tr': hashlib.sha256(index_path.read_bytes()).hexdigest()}}))
    out=tmp_path/'derived'
    manifest=create(index_path,records_path,audit_path,out,seed=17,validation_fraction=.2,test_fraction=.2)
    for role in ('tr','cv','tt'):
        p=out/(role+'.json')
        assert manifest['derived_index_sha256'][role] == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        create(index_path,records_path,audit_path,out,seed=17,validation_fraction=.2,test_fraction=.2)


def test_audit_index_mismatch_is_rejected_before_creating_output(tmp_path):
    idx, records=fixtures()
    index_path=tmp_path/'index.json'; index_path.write_text(json.dumps(idx))
    records_path=tmp_path/'records.jsonl'; records_path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    audit_path=tmp_path/'audit.json';audit_path.write_text(json.dumps({'index_sha256':{'tr':'different'}}))
    out=tmp_path/'derived'
    with pytest.raises(ValueError,match='different tr index'):
        create(index_path,records_path,audit_path,out,seed=17,validation_fraction=.2,test_fraction=.2)
    assert not out.exists()


def test_missing_audit_record_refuses_derivation_not_silent_exclusion():
    idx, records = fixtures()
    records.pop(4)
    with pytest.raises(ValueError, match='exactly match'):
        derive(idx, records, 17, .2, .2)
