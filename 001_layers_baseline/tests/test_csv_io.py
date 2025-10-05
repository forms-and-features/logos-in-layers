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
             "topk": [["Berlin", 0.9]],
             "copy_collapse": True,
             "entropy_collapse": False,
             "is_answer": True,
             "copy_strict@0.95": True,
             "copy_soft_k1@0.5": True,
             "copy_soft_k2@0.5": False,
             "copy_soft_k3@0.5": False,
             },
        ],
        "copy_flag_columns": ["copy_strict@0.95", "copy_soft_k1@0.5", "copy_soft_k2@0.5", "copy_soft_k3@0.5"],
    }

    top_k = 3

    with tempfile.TemporaryDirectory() as td:
        records_path = os.path.join(td, "records.csv")
        pure_path = os.path.join(td, "pure.csv")
        write_csv_files(json_data, records_path, pure_path, top_k)

        with open(records_path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        header = rows[0]
        expected_len = 2 + 5 + 2 * top_k + 1  # prompt_id + prompt_variant + (layer,pos,token,entropy,entropy_bits) + topk + rest
        assert len(header) == expected_len
        assert header[0] == "prompt_id"
        assert header[1] == "prompt_variant"
        assert header[2:7] == ["layer", "pos", "token", "entropy", "entropy_bits"]
        assert header[-1] == "rest_mass"
        for r in rows[1:]:
            assert len(r) == expected_len
            float(r[6])
            rest = float(r[-1])
            assert 0.0 <= rest <= 1.0

        with open(pure_path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
        header = rows[0]
        copy_cols = json_data["copy_flag_columns"]
        expected_len = 7 + 2*top_k + 2 + len(copy_cols) + 25
        assert len(header) == expected_len
        assert header[5] == "entropy"
        assert header[6] == "entropy_bits"
        rest_idx = header.index("rest_mass")
        assert header[rest_idx:rest_idx + 2 + len(copy_cols)] == ["rest_mass", "copy_collapse", *copy_cols]
        tail = header[-25:]
        assert tail == [
            "entropy_collapse",
            "is_answer",
            "p_top1",
            "p_top5",
            "p_answer",
            "teacher_entropy_bits",
            "kl_to_final_bits",
            "kl_to_final_bits_norm_temp",
            "answer_rank",
            "topk_jaccard_raw_norm@50",
            "topk_jaccard_consecutive@50",
            "cos_to_final",
            "cos_to_answer",
            "cos_to_prompt_max",
            "geom_crossover",
            "echo_mass_prompt",
            "answer_mass",
            "answer_minus_echo_mass",
            "mass_ratio_ans_over_prompt",
            "topk_prompt_mass@50",
            "control_margin",
            "resid_norm_ratio",
            "delta_resid_cos",
            "answer_logit_gap",
            "answer_vs_top1_gap",
        ]
        # Validate row shapes and rest_mass range
        for r in rows[1:]:
            assert len(r) == expected_len
            rest = float(r[rest_idx])
            assert 0.0 <= rest <= 1.0
            float(r[6])  # entropy_bits should be numeric
