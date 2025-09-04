#!/usr/bin/env python3
"""Unit tests for head transform detection helper."""

import _pathfix  # noqa: F401

from layers_core.head_transforms import detect_head_transforms


def test_detect_from_cfg_first():
    class Cfg:
        final_logit_scale = 2.0
        final_logit_softcap = 4.0

    class Model:
        cfg = Cfg()

    s, c = detect_head_transforms(Model())
    assert s == 2.0
    assert c == 4.0


def test_detect_from_model_when_cfg_missing():
    class Model:
        final_logit_scale = 3.0
        final_logit_softcap = 5.0

    s, c = detect_head_transforms(Model())
    assert s == 3.0
    assert c == 5.0


def test_ignore_non_numeric_and_unknown():
    class Cfg:
        final_logit_scale = "x"
        # softcap absent

    class Model:
        cfg = Cfg()
        # model-level scale absent; softcap provided via alternate alias
        logit_softcap = 7.5

    s, c = detect_head_transforms(Model())
    # numeric scale not found → None; softcap found via alias
    assert s is None
    assert c == 7.5


def test_aliases_on_cfg_are_checked():
    class Cfg:
        final_logits_scale = 1.25
        final_logits_softcap = 6.0

    class Model:
        cfg = Cfg()

    s, c = detect_head_transforms(Model())
    assert s == 1.25
    assert c == 6.0

