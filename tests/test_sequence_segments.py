"""Test pulserver-interpreter Sequence replacement for pypulseq Sequence."""

import numpy as np
from pulserver_interpreter import Sequence


def _make_seq(mprage, Nz):
    """
    Create a Sequence filled with MPRAGE pattern and build segments.

    Parameters
    ----------
    mprage : callable
        Function that fills a Sequence with MPRAGE blocks.
    Nz : int
        Number of slices or repetitions in sequence.

    Returns
    -------
    seq : Sequence
        Sequence with segments built.
    expected_trid : list[int]
        Expected TRID array per block.
    """
    seq = Sequence()
    seq = mprage(seq)
    seq.get_seq_structure()
    expected_trid = [1, 0, 0] + Nz * [-1, 0, 0, 0, 0] + [-1]
    return seq, expected_trid


def test_unique_blocks(mprage, Nz):
    seq, _ = _make_seq(mprage, Nz)
    unique_blocks = seq.unique_blocks.block_events

    # There should be exactly 7 unique blocks
    assert (
        len(unique_blocks) == 7
    ), f"Expected 7 unique blocks, got {len(unique_blocks)}"


def test_block_id_array(mprage, Ny, Nz):
    seq, _ = _make_seq(mprage, Nz)

    # Expected repeated pattern per TR
    pattern = [1, 2, 3, *(Nz * [4, 5, 6, 7, 3]), 3]  # 7 blocks
    expected_block_ids = pattern * Ny

    assert np.array_equal(seq.block_id, expected_block_ids), "block_id array mismatch"


def test_segments(mprage, Nz):
    seq, _ = _make_seq(mprage, Nz)

    # Expected segments
    expected_segments = {
        1: (1, 2, 3),  # Adiabatic inversion
        2: (4, 5, 6, 7, 3),  # Flash segment
        3: (3,),  # Recovery period
    }

    for seg_id, blocks in expected_segments.items():
        assert tuple(seq.segments[seg_id]) == blocks, f"Segment {seg_id} mismatch"


def test_block_segment_id_array(mprage, Ny, Nz):
    seq, _ = _make_seq(mprage, Nz)

    # Expected repeated pattern of segment IDs
    pattern = [1, 1, 1, *(Nz * [2, 2, 2, 2, 2]), 3]
    expected_array = pattern * Ny

    assert np.array_equal(
        seq.block_segment_id, expected_array
    ), "block_segment_id array mismatch"


def test_trs_segments_unique_blocks_consistency(mprage, Nz):
    seq, expected_trid = _make_seq(mprage, Nz)

    # block_trid should be all ones
    assert np.all(seq.block_trid == 1), "block_trid array should be all ones"

    # TR definitions (seq.trs) should contain exactly the 3 TRs
    expected_trs = [1]
    assert set(seq.trs.keys()) == set(expected_trs), f"TRs mismatch: {seq.trs.keys()}"
