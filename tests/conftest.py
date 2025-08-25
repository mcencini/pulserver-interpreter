"""Auxiliary test case function."""

import pytest

from pulserver_interpreter.demo import MPRAGE

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
    return MPRAGE()
