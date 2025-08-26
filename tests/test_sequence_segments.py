"""Test pulserver-interpreter Sequence replacement for pypulseq Sequence."""

from types import SimpleNamespace

import numpy as np

import pytest

from pulserver_interpreter.demo import MPRAGE, GRE
from pulserver_interpreter.pulseq import concatenate


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


def make_composite(Ny, Nz):
    """
    Create a Sequence filled with GRE pattern (PI calibration) followed
    by MPRAGE (main sequence) and build segments.

    Assume that, for PI calibration, Ny == Ny // 2 and Nz == Nz // 2

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
    gre = GRE(mtx=(256, Ny // 2, Nz // 2))
    mprage = MPRAGE()
    composite_seq = concatenate(gre, mprage)
    composite_seq.mode = "prep"
    seq = composite_seq(mprage={"mtx": (256, Ny, Nz)})

    expected = SimpleNamespace()

    # expected TR ID
    trid_gre = (Nz // 2) * [1, 0, 0, 0, 0]
    trid_mprage = [2, 0, 0] + Nz * [-1, 0, 0, 0, 0] + [-1]
    expected.block_trids = np.concatenate(
        (np.ones_like(trid_gre * (Ny // 2)), 2 * np.ones_like(trid_mprage * Ny))
    )
    expected.trs = [1, 2]

    # expected n blocks
    expected.num_blocks = 9

    # Expected repeated pattern of block IDs per TR
    gre_pattern = (Nz // 2) * [1, 2, 3, 4, 5]  # 5 blocks
    mprage_pattern = [6, 7, 5, *(Nz * [1, 8, 3, 9, 5]), 5]  # 7 blocks
    expected.block_ids = gre_pattern * (Ny // 2) + mprage_pattern * Ny

    # Expected segments
    expected.segments = {
        1: (1, 2, 3, 4, 5),  # Flash segment (GRE)
        2: (6, 7, 5),  # Adiabatic inversion
        3: (1, 8, 3, 9, 5),  # Flash segment (MPRAGE)
        4: (5,),  # Recovery period
    }

    # Expected repeated pattern of segment IDs per TR
    gre_pattern = (Nz // 2) * [1, 1, 1, 1, 1]
    mprage_pattern = [2, 2, 2, *(Nz * [3, 3, 3, 3, 3]), 4]
    expected.segment_ids = gre_pattern * (Ny // 2) + mprage_pattern * Ny

    return seq, expected


@pytest.mark.parametrize("make_func", [make_gre, make_mprage, make_composite])
def test_unique_blocks(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)
    unique_blocks = seq.unique_blocks.block_events

    assert (
        len(unique_blocks) == expected.num_blocks
    ), f"Expected {expected.num_blocks} unique blocks, got {len(unique_blocks)}"


@pytest.mark.parametrize("make_func", [make_gre, make_mprage, make_composite])
def test_block_id_array(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)

    assert np.array_equal(seq.block_id, expected.block_ids), "block_id array mismatch"


@pytest.mark.parametrize("make_func", [make_gre, make_mprage, make_composite])
def test_segments(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)

    for seg_id, blocks in expected.segments.items():
        assert tuple(seq.segments[seg_id]) == blocks, f"Segment {seg_id} mismatch"


@pytest.mark.parametrize("make_func", [make_gre, make_mprage, make_composite])
def test_block_segment_id_array(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)

    assert np.array_equal(
        seq.block_segment_id, expected.segment_ids
    ), "block_segment_id array mismatch"


@pytest.mark.parametrize("make_func", [make_gre, make_mprage, make_composite])
def test_trs_segments_unique_blocks_consistency(make_func, Ny, Nz):
    seq, expected = make_func(Ny, Nz)

    # block_trid should be all ones
    assert np.array_equal(
        seq.block_trid, expected.block_trids
    ), "block_trid array mismatch"

    assert set(seq.trs.keys()) == set(expected.trs), f"TRs mismatch: {seq.trs.keys()}"
