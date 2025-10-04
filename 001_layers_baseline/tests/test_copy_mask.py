#!/usr/bin/env python3
"""Tests for copy mask transparency utilities."""

import _pathfix  # noqa: F401

from layers_core.collapse_rules import build_copy_ignore_mask


def test_build_copy_ignore_mask_basic():
    tokens = {
        0: " ",
        1: "!",
        2: "A",
        3: "Berlin",
        4: "\t",
        5: "?",
    }

    def decode(idx: int) -> str:
        return tokens[int(idx)]

    mask = build_copy_ignore_mask(decode, len(tokens))
    assert mask["size"] == 4
    ignored_ids = set(mask["ignored_token_ids"])
    assert {0, 1, 4, 5} == ignored_ids
    assert "A" not in mask["ignored_token_str_sample"]
    assert any(sample in {" ", "!", "\t", "?"} for sample in mask["ignored_token_str_sample"])


def test_build_copy_ignore_mask_sentencepiece_tokens():
    tokens = {
        0: "▁",
        1: "▁Berlin",
        2: "▁ ",
        3: "Berlin",
        4: "。",
    }

    def decode(idx: int) -> str:
        return tokens[int(idx)]

    mask = build_copy_ignore_mask(decode, len(tokens))
    ignored_ids = set(mask["ignored_token_ids"])
    assert 0 in ignored_ids  # stand-alone sentencepiece space
    assert 2 in ignored_ids  # space followed by literal space
    assert 4 in ignored_ids  # punctuation
    assert 1 not in ignored_ids and 3 not in ignored_ids


def test_build_copy_ignore_mask_bpe_space_tokens():
    tokens = {
        0: "Ġ",
        1: "ĠBerlin",
        2: "Berlin",
        3: "Ċ",
        4: ",",
    }

    def decode(idx: int) -> str:
        return tokens[int(idx)]

    mask = build_copy_ignore_mask(decode, len(tokens))
    ignored_ids = set(mask["ignored_token_ids"])
    assert ignored_ids.issuperset({0, 3, 4})
    assert 1 not in ignored_ids and 2 not in ignored_ids
