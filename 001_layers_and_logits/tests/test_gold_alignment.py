#!/usr/bin/env python3
"""Unit tests for gold-token alignment helper (§1.7).

CPU-only; uses a mock tokenizer to simulate space-sensitive tokenization.
"""

import _pathfix  # noqa: F401

from typing import List

from layers_core.gold import compute_gold_answer_info, compute_gold_answer_info_from_sequences


class MockTokenizerWS:
    """Mock tokenizer where ' with space' yields a different first-piece id.

    - encode('P') -> [1, 2]
    - encode('P Berlin') -> [1, 2, 101]
    - encode('PBerlin') -> [1, 2, 100]
    - convert_ids_to_tokens maps ids to strings with/without a leading-space marker
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if text == "P":
            return [1, 2]
        if text == "P Berlin":
            return [1, 2, 101]
        if text == "PBerlin":
            return [1, 2, 100]
        return [1, 2]

    def convert_ids_to_tokens(self, ids: List[int]):
        mapping = {100: "Berlin", 101: "ĠBerlin"}
        return [mapping.get(i, f"<id:{i}>") for i in ids]

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        i = ids[0]
        return {100: "Berlin", 101: " Berlin"}.get(i, f"<id:{i}>")


class MockTokenizerNoWS:
    """Mock tokenizer where with-space does not extend the context; fallback to no-space.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if text == "P":
            return [1, 2]
        if text == "P Berlin":
            return [1, 2]  # does not extend
        if text == "PBerlin":
            return [1, 2, 100]
        return [1, 2]

    def convert_ids_to_tokens(self, ids: List[int]):
        mapping = {100: "Berlin"}
        return [mapping.get(i, f"<id:{i}>") for i in ids]

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        i = ids[0]
        return {100: "Berlin"}.get(i, f"<id:{i}>")


def test_compute_gold_answer_info_with_space_pref():
    tok = MockTokenizerWS()
    info = compute_gold_answer_info(tok, "P", "Berlin", pieces_k=3)
    assert info["status"] == "ok"
    assert info["variant"] == "with_space"
    assert info["first_id"] == 101
    assert info["pieces"][0] == "ĠBerlin"
    assert info["answer_ids"] == [101]
    assert info["ctx_ids"] == [1, 2]


def test_compute_gold_answer_info_fallback_no_space():
    tok = MockTokenizerNoWS()
    info = compute_gold_answer_info(tok, "P", "Berlin", pieces_k=2)
    # with_space fails to extend; should pick no_space
    assert info["status"] == "ok"
    assert info["variant"] == "no_space"
    assert info["first_id"] == 100
    assert info["pieces"][0] == "Berlin"


def test_compute_gold_answer_info_unresolved_when_no_tokenizer():
    info = compute_gold_answer_info(None, "P", "Berlin", pieces_k=3)
    assert info["status"] == "unresolved"
    assert info["first_id"] is None


def test_compute_from_sequences_helper():
    ctx = [7, 8]
    ws = [7, 8, 42]
    ns = [7, 8, 99]
    info = compute_gold_answer_info_from_sequences(
        ctx, ws, ns, pieces_k=2, convert_ids_to_tokens=lambda ids: [f"tok:{i}" for i in ids], answer_str="X"
    )
    assert info["status"] == "ok"
    assert info["variant"] == "with_space"
    assert info["first_id"] == 42
    assert info["pieces"][0] == "tok:42"
