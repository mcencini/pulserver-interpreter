"""Test pulserver-interpreter Sequence replacement for pypulseq Sequence."""

from pulserver_interpreter.sequence import Sequence


def test_sequence_definition(mprage, Nz):

    seq = Sequence()

    # Fill mprage
    seq = mprage(seq)

    # Assertions for TRID tracking and definitions (dictionary version)
    assert set(seq.trid_definitions.keys()) == {1}
    assert len(seq.trid_definitions) == 1
    expected = [1, 0, 0] + Nz * [-1, 0, 0, 0, 0] + [-1]
    assert seq.trid_definitions[1] == expected

    # Check that the internal pypulseq sequence only stores the first instance of the TR
    # The number of blocks should match the length of the TRID definition for the first TRID
    assert len(seq._pypulseq_seq.block_events) == len(expected)

