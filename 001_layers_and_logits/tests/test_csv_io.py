#!/usr/bin/env python3
"""CPU-only tests for CSV writers: header shape, row length, rest_mass."""

import _pathfix  # noqa: F401

import os
import csv
import tempfile

from layers_core.csv_io import write_csv_files


def test_csv_writers_headers_and_rows():
    json_data = {
        "records": [
            {"layer": 0, "pos": 0, "token": "Give", "entropy": 3.0, "topk": [["A", 0.4], ["B", 0.3]]},
            {"layer": 1, "pos": 1, "token": "the", "entropy": 2.0, "topk": [["C", 0.6]]},
        ],
        "pure_next_token_records": [
            {"layer": 0, "pos": 5, "token": "⟨NEXT⟩", "entropy": 1.0,
             "topk": [["Berlin", 0.9]], "copy_collapse": True, "entropy_collapse": False, "is_answer": True},
        ],
    }

    top_k = 3

    with tempfile.TemporaryDirectory() as td:
        records_path = os.path.join(td, "records.csv")
        pure_path = os.path.join(td, "pure.csv")
        write_csv_files(json_data, records_path, pure_path, top_k)

        with open(records_path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        header = rows[0]
        expected_len = 4 + 2 * top_k + 1
        assert len(header) == expected_len
        assert header[:4] == ["layer", "pos", "token", "entropy"]
        assert header[-1] == "rest_mass"
        for r in rows[1:]:
            assert len(r) == expected_len
            rest = float(r[-1])
            assert 0.0 <= rest <= 1.0

        with open(pure_path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        header = rows[0]
        expected_len = 4 + 2 * top_k + 1 + 3
        assert len(header) == expected_len
        assert header[-4:] == ["rest_mass", "copy_collapse", "entropy_collapse", "is_answer"]
        for r in rows[1:]:
            assert len(r) == expected_len
            rest = float(r[-4])
            assert 0.0 <= rest <= 1.0

