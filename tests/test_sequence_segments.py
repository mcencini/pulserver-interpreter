"""Test pulserver-interpreter Sequence replacement for pypulseq Sequence."""

from types import SimpleNamespace

import numpy as np

import pytest

from pulserver_interpreter.demo import MPRAGE, GRE


def make_mprage(Ny, Nz):
    """
    Create a Sequence filled with MPRAGE pattern and build segments.

    Parameters
    ----------
    Ny : int
        Number of phase encoding lines in sequence.
    Nz : int
        Number of slices or repetitions in sequence.

    Returns
    -------
    seq : Sequence
        Sequence with segments built.
    expected_trid : list[int]
        Expected TRID array per block.

    """
    mprage = MPRAGE()
    mprage.mode = "prep"
    seq = mprage(mtx=(256, Ny, Nz))

    expected = SimpleNamespace()

    # expected TR ID
    trid = [1, 0, 0] + Nz * [-1, 0, 0, 0, 0] + [-1]
    expected.block_trids = np.ones_like(trid * Ny)
    expected.trs = [1]

    # expected n blocks
    expected.num_blocks = 7

    # Expected repeated pattern of block IDs per TR
    pattern = [1, 2, 3, *(Nz * [4, 5, 6, 7, 3]), 3]  # 7 blocks
    expected.block_ids = pattern * Ny

    # Expected segments
    expected.segments = {
        1: (1, 2, 3),  # Adiabatic inversion
        2: (4, 5, 6, 7, 3),  # Flash segment
        3: (3,),  # Recovery period
    }

    # Expected repeated pattern of segment IDs per TR
    pattern = [1, 1, 1, *(Nz * [2, 2, 2, 2, 2]), 3]
    expected.segment_ids = pattern * Ny

    return seq, expected


def make_gre(Ny, Nz):
    """
    Create a Sequence filled with GRE pattern and build segments.

    Parameters
    ----------
    Ny : int
        Number of phase encoding lines in sequence.
    Nz : int
        Number of slices or repetitions in sequence.

    Returns
    -------
    seq : Sequence
        Sequence with segments built.
    expected_trid : list[int]
        Expected TRID array per block.

    """
    gre = GRE()
    gre.mode = "prep"
    seq = gre(mtx=(256, Ny, Nz))

    expected = SimpleNamespace()

    # expected TR ID
    trid = Nz * [1, 0, 0, 0, 0]
    expected.block_trids = np.ones_like(trid * Ny)
    expected.trs = [1]

    # expected n blocks
    expected.num_blocks = 5

    # Expected repeated pattern of block IDs per TR
    pattern = Nz * [1, 2, 3, 4, 5]  # 5 blocks
    expected.block_ids = pattern * Ny

    # Expected segments
    expected.segments = {
        1: (1, 2, 3, 4, 5),  # Flash segment
    }

    # Expected repeated pattern of segment IDs per TR
    pattern = Nz * [1, 1, 1, 1, 1]
    expected.segment_ids = pattern * Ny

    return seq, expected


@pytest.mark.parametrize("make_func", [make_mprage, make_gre])
def test_unique_blocks(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)
    unique_blocks = seq.unique_blocks.block_events

    # There should be exactly 7 unique blocks
    assert (
        len(unique_blocks) == expected.num_blocks
    ), f"Expected {expected.num_blocks} unique blocks, got {len(unique_blocks)}"


@pytest.mark.parametrize("make_func", [make_mprage, make_gre])
def test_block_id_array(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)

    assert np.array_equal(seq.block_id, expected.block_ids), "block_id array mismatch"


@pytest.mark.parametrize("make_func", [make_mprage, make_gre])
def test_segments(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)

    for seg_id, blocks in expected.segments.items():
        assert tuple(seq.segments[seg_id]) == blocks, f"Segment {seg_id} mismatch"


@pytest.mark.parametrize("make_func", [make_mprage, make_gre])
def test_block_segment_id_array(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)

    assert np.array_equal(
        seq.block_segment_id, expected.segment_ids
    ), "block_segment_id array mismatch"


@pytest.mark.parametrize("make_func", [make_mprage, make_gre])
def test_trs_segments_unique_blocks_consistency(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)

    # block_trid should be all ones
    assert np.array_equal(
        seq.block_trid, expected.block_trids
    ), "block_trid array mismatch"

    # TR definitions (seq.trs) should contain exactly the 3 TRs
    assert set(seq.trs.keys()) == set(expected.trs), f"TRs mismatch: {seq.trs.keys()}"
