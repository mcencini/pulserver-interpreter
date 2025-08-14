"""Test pulserver-interpreter Sequence replacement for pypulseq Sequence."""

import numpy as np
from pulserver_interpreter.sequence import Sequence


# Helper to create and fill sequence
def _make_seq(mprage, Nz):
    seq = Sequence()
    seq = mprage(seq)
    seq.build_segments()
    expected = [1, 0, 0] + Nz * [-1, 0, 0, 0, 0] + [-1]
    return seq, expected


def test_trid_events_tracking(mprage, Nz):
    seq, expected = _make_seq(mprage, Nz)
    assert set(seq.trid_events.keys()) == {1}
    assert len(seq.trid_events) == 1
    assert seq.trid_events[1] == expected


def test_block_events_count(mprage, Nz):
    seq, expected = _make_seq(mprage, Nz)
    assert len(seq._seq.block_events) == len(expected)


def test_segment_library_definitions(mprage, Nz):
    seq, _ = _make_seq(mprage, Nz)
    assert set(seq.segment_library.keys()) == {1, 2, 3}
    assert tuple(seq.segment_library[1]) == (1, 2, 3)
    assert tuple(seq.segment_library[2]) == (4, 5, 6, 7, 3)
    assert tuple(seq.segment_library[3]) == (3,)


def test_mapping_arrays_shape(mprage, Nz):
    seq, expected = _make_seq(mprage, Nz)
    trid = 1
    seg_ids = seq.trid_to_segment_ids[trid]
    within_seg_idx = seq.trid_to_within_segment_idx[trid]
    assert len(seg_ids) == len(expected)
    assert len(within_seg_idx) == len(expected)


def test_segment_ids_in_library(mprage, Nz):
    seq, _ = _make_seq(mprage, Nz)
    trid = 1
    seg_ids = seq.trid_to_segment_ids[trid]
    for sid in seg_ids:
        assert sid in seq.segment_library


def test_block_library_consistency(mprage, Nz):
    seq, _ = _make_seq(mprage, Nz)
    all_blocks = set()
    for blocks in seq.segment_library.values():
        all_blocks.update(blocks)
    block_lib_blocks = set(range(1, len(seq.block_library.block_events) + 1))
    assert all_blocks == block_lib_blocks


def test_segment_definitions_vs_mapping(mprage, Nz):
    seq, _ = _make_seq(mprage, Nz)
    trid = 1
    seg_ids = seq.trid_to_segment_ids[trid]
    assert set(seq.trid_definitions[trid]) == set(seg_ids)


def test_create_segments_idempotency(mprage, Nz):
    seq, _ = _make_seq(mprage, Nz)
    prev_segment_library = dict(seq.segment_library)
    prev_trid_to_segment_ids = {k: v.copy() for k, v in seq.trid_to_segment_ids.items()}
    prev_trid_to_within_segment_idx = {
        k: v.copy() for k, v in seq.trid_to_within_segment_idx.items()
    }
    prev_trid_definitions = dict(seq.trid_definitions)
    prev_block_library_len = len(seq.block_library.block_events)
    seq.build_segments()
    assert seq.segment_library == prev_segment_library
    for k in prev_trid_to_segment_ids:
        assert np.array_equal(seq.trid_to_segment_ids[k], prev_trid_to_segment_ids[k])
        assert np.array_equal(
            seq.trid_to_within_segment_idx[k], prev_trid_to_within_segment_idx[k]
        )
    assert seq.trid_definitions == prev_trid_definitions
    assert len(seq.block_library.block_events) == prev_block_library_len
