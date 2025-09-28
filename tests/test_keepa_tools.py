import pytest
from agents.keepa_tools import last_nonneg

def test_last_nonneg_basic():
    assert last_nonneg([None, -1, 5, None, 8]) == 8

def test_last_nonneg_none():
    assert last_nonneg(None) is None

def test_last_nonneg_empty():
    assert last_nonneg([]) is None