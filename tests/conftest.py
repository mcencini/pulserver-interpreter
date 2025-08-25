"""Auxiliary test case function."""

import pytest

ny = 4
nz = 4


@pytest.fixture
def Ny():
    return ny


@pytest.fixture
def Nz():
    return nz
