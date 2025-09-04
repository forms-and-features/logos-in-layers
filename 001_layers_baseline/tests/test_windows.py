#!/usr/bin/env python3

import _pathfix  # noqa: F401

from layers_core.windows import WindowManager


def test_append_and_trim_basic():
    w = WindowManager(2)
    out1 = w.append_and_trim("norm", "pos", "orig", 10)
    assert out1 == [10]
    out2 = w.append_and_trim("norm", "pos", "orig", 11)
    assert out2 == [10, 11]
    out3 = w.append_and_trim("norm", "pos", "orig", 12)
    # max length 2: oldest dropped
    assert out3 == [11, 12]


def test_independent_keys():
    w = WindowManager(3)
    w.append_and_trim("norm", "pos", "orig", 1)
    w.append_and_trim("norm", "pos", "orig", 2)
    a = w.append_and_trim("norm", "pos", "orig", 3)
    b = w.append_and_trim("prism", "pos", "orig", 99)
    # Different lens_type should maintain separate windows
    assert a == [1, 2, 3]
    assert b == [99]


def test_reset_variant():
    w = WindowManager(2)
    w.append_and_trim("norm", "ctl", "orig", 5)
    w.append_and_trim("prism", "ctl", "orig", 6)
    # Reset for (ctl, orig) should clear both lenses
    w.reset_variant("ctl", "orig")
    a = w.append_and_trim("norm", "ctl", "orig", 7)
    b = w.append_and_trim("prism", "ctl", "orig", 8)
    assert a == [7]
    assert b == [8]

