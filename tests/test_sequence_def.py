"""Test pulserver-interpreter Sequence replacement for pypulseq Sequence."""

from pulserver_interpreter.sequence import Sequence


def test_sequence_definition(mprage, Nz):

    seq = Sequence()

    # Fill mprage
    seq = mprage(seq)
    seq.create_segments()

    # Assertions for TRID tracking and definitions (dictionary version)
    assert set(seq.trid_events.keys()) == {1}
    assert len(seq.trid_events) == 1
    expected = [1, 0, 0] + Nz * [-1, 0, 0, 0, 0] + [-1]
    assert seq.trid_events[1] == expected

    # Check that the internal pypulseq sequence only stores the first instance of the TR
    # The number of blocks should match the length of the TRID definition for the first TRID
    assert len(seq._seq.block_events) == len(expected)

    # Check that the number of segments and their definitions are as expected
    assert set(seq.segment_library.keys()) == {1, 2, 3}
    assert tuple(seq.segment_library[1]) == (1, 2, 3)
    assert tuple(seq.segment_library[2]) == (4, 5, 6, 7, 3)
    assert tuple(seq.segment_library[3]) == (3,)
