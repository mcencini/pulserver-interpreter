"""Auxiliary test case function."""

import pytest

from pulserver_interpreter.demo import mprage as _mprage

ny = 4
nz = 4


@pytest.fixture
def Ny():
    return ny


@pytest.fixture
def Nz():
    return nz


@pytest.fixture
def mprage():
    def fill_mprage_seq(seq):
        """
        Fill the given pulserver Sequence object with an MPRAGE-like loop using base_blocks.
        This function is agnostic to the mode of the Sequence (prep/eval).
        """
        return _mprage(seq, ny, nz)

    return fill_mprage_seq
