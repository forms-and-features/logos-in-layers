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
        expected_len = 1 + 4 + 2 * top_k + 1  # prompt_id + (layer,pos,token,entropy) + topk + rest
        assert len(header) == expected_len
        assert header[0] == "prompt_id"
        assert header[1:5] == ["layer", "pos", "token", "entropy"]
        assert header[-1] == "rest_mass"
        for r in rows[1:]:
            assert len(r) == expected_len
            rest = float(r[-1])
            assert 0.0 <= rest <= 1.0

        with open(pure_path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        header = rows[0]
        # Pure next-token CSV now includes §1.3 metrics plus §1.5 cosine column and §1.8 control_margin
        expected_len = (1 + 4) + 2 * top_k + 1 + 3 + 5 + 1 + 1  # prompt_id + base + topk + rest + flags + metrics + cos + control_margin
        assert len(header) == expected_len
        assert header[-11:] == [
            "rest_mass",
            "copy_collapse",
            "entropy_collapse",
            "is_answer",
            "p_top1",
            "p_top5",
            "p_answer",
            "kl_to_final_bits",
            "answer_rank",
            "cos_to_final",
            "control_margin",
        ]
        # Validate row shapes and rest_mass range
        rest_idx = header.index("rest_mass")
        for r in rows[1:]:
            assert len(r) == expected_len
            rest = float(r[rest_idx])
            assert 0.0 <= rest <= 1.0
