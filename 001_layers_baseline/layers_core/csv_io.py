import csv
from typing import Dict, Any, List


def write_csv_files(json_data: Dict[str, Any], csv_filepath: str, pure_csv_filepath: str, top_k_verbose: int) -> None:
    """Write CSV files from collected JSON data.

    - Records CSV: per-layer, per-position rows with padded top-k and rest_mass.
    - Pure next-token CSV: last-position only per layer with collapse flags.
    """
    records: List[Dict[str, Any]] = json_data["records"]
    pure_next_token_records: List[Dict[str, Any]] = json_data["pure_next_token_records"]
    copy_flag_columns: List[str] = list(dict.fromkeys(json_data.get("copy_flag_columns", [])))

    # Save records to CSV
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(
            f_csv,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            lineterminator="\n",
        )
        header = ["prompt_id", "prompt_variant", "layer", "pos", "token", "entropy"]
        for i in range(1, top_k_verbose + 1):
            header.extend([f"top{i}", f"prob{i}"])
        header.append("rest_mass")
        writer.writerow(header)

        for rec in records:
            row = [rec.get("prompt_id", ""), rec.get("prompt_variant", ""), rec.get("layer"), rec.get("pos"), rec.get("token"), rec.get("entropy")]
            topk_list = rec.get("topk", [])
            topk_prob_sum = 0.0
            for j in range(top_k_verbose):
                if j < len(topk_list):
                    tok, prob = topk_list[j]
                    topk_prob_sum += prob
                else:
                    tok, prob = "", ""
                row.extend([tok, prob])
            rest_mass = max(0.0, 1.0 - topk_prob_sum)
            row.append(rest_mass)
            writer.writerow(row)

    # Save pure next-token records to separate CSV
    with open(pure_csv_filepath, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(
            f_csv,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            lineterminator="\n",
        )
        header = ["prompt_id", "prompt_variant", "layer", "pos", "token", "entropy"]
        for i in range(1, top_k_verbose + 1):
            header.extend([f"top{i}", f"prob{i}"])
        # Extended schema per 001_LAYERS_BASELINE_PLAN §1.3
        header.extend([
            "rest_mass",
            "copy_collapse",
        ])
        header.extend(copy_flag_columns)
        header.extend([
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
            # Representation-drift cosine (001_LAYERS_BASELINE_PLAN §1.5)
            "cos_to_final",
            # Geometric surface crossover (001_LAYERS_BASELINE_PLAN §1.14)
            "cos_to_answer",
            "cos_to_prompt_max",
            "geom_crossover",
            # Surface mass (001_LAYERS_BASELINE_PLAN §1.13)
            "echo_mass_prompt",
            "answer_mass",
            "answer_minus_echo_mass",
            "mass_ratio_ans_over_prompt",
            # Top-K prompt coverage (001_LAYERS_BASELINE_PLAN §1.15)
            "topk_prompt_mass@50",
            # Negative control margin (001_LAYERS_BASELINE_PLAN §1.8)
            "control_margin",
            # Normalization diagnostics (001_LAYERS_BASELINE_PLAN §1.30)
            "resid_norm_ratio",
            "delta_resid_cos",
            # Answer-confidence margins (001_LAYERS_BASELINE_PLAN §1.35)
            "answer_logit_gap",
            "answer_vs_top1_gap",
        ])
        writer.writerow(header)

        for rec in pure_next_token_records:
            row = [rec.get("prompt_id", ""), rec.get("prompt_variant", ""), rec.get("layer"), rec.get("pos"), rec.get("token"), rec.get("entropy")]
            topk_list = rec.get("topk", [])
            topk_prob_sum = 0.0
            for j in range(top_k_verbose):
                if j < len(topk_list):
                    tok, prob = topk_list[j]
                    topk_prob_sum += prob
                else:
                    tok, prob = "", ""
                row.extend([tok, prob])
            rest_mass = max(0.0, 1.0 - topk_prob_sum)
            def _nz(val):
                # Normalize None to empty string for CSV cleanliness
                return "" if val is None else val
            row.extend([
                rest_mass,
                rec.get("copy_collapse", ""),
            ])
            for flag_col in copy_flag_columns:
                row.append(_nz(rec.get(flag_col)))
            row.extend([
                rec.get("entropy_collapse", ""),
                rec.get("is_answer", ""),
                _nz(rec.get("p_top1")),
                _nz(rec.get("p_top5")),
                _nz(rec.get("p_answer")),
                _nz(rec.get("teacher_entropy_bits")),
                _nz(rec.get("kl_to_final_bits")),
                _nz(rec.get("kl_to_final_bits_norm_temp")),
                _nz(rec.get("answer_rank")),
                _nz(rec.get("topk_jaccard_raw_norm@50")),
                _nz(rec.get("topk_jaccard_consecutive@50")),
                _nz(rec.get("cos_to_final")),
                _nz(rec.get("cos_to_answer")),
                _nz(rec.get("cos_to_prompt_max")),
                _nz(rec.get("geom_crossover")),
                _nz(rec.get("echo_mass_prompt")),
                _nz(rec.get("answer_mass")),
                _nz(rec.get("answer_minus_echo_mass")),
                _nz(rec.get("mass_ratio_ans_over_prompt")),
                _nz(rec.get("topk_prompt_mass@50")),
                _nz(rec.get("control_margin")),
                _nz(rec.get("resid_norm_ratio")),
                _nz(rec.get("delta_resid_cos")),
                _nz(rec.get("answer_logit_gap")),
                _nz(rec.get("answer_vs_top1_gap")),
            ])
            writer.writerow(row)


def write_raw_lens_window_csv(records: List[Dict[str, Any]], csv_filepath: str) -> None:
    """Write windowed raw-vs-norm comparison rows to CSV."""
    if not records:
        return

    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(
            f_csv,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            lineterminator="\n",
        )
        header = [
            "prompt_id",
            "prompt_variant",
            "layer",
            "lens",
            "p_top1",
            "top1_token_id",
            "top1_token_str",
            "p_answer",
            "answer_rank",
            "kl_norm_vs_raw_bits",
        ]
        writer.writerow(header)

        def _nz(val):
            return "" if val is None else val

        for rec in records:
            row = [
                _nz(rec.get("prompt_id")),
                _nz(rec.get("prompt_variant")),
                rec.get("layer"),
                rec.get("lens"),
                _nz(rec.get("p_top1")),
                rec.get("top1_token_id"),
                _nz(rec.get("top1_token_str")),
                _nz(rec.get("p_answer")),
                rec.get("answer_rank"),
                _nz(rec.get("kl_norm_vs_raw_bits")),
            ]
            writer.writerow(row)


def write_raw_lens_full_csv(records: List[Dict[str, Any]], csv_filepath: str) -> None:
    """Write full-depth raw-vs-norm per-layer rows to CSV (one row per layer)."""
    if not records:
        return

    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(
            f_csv,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            lineterminator="\n",
        )
        header = [
            "prompt_id",
            "prompt_variant",
            "layer",
            # raw lens
            "p_top1_raw",
            "top1_token_id_raw",
            "top1_token_str_raw",
            "p_answer_raw",
            "answer_rank_raw",
            # norm lens
            "p_top1_norm",
            "top1_token_id_norm",
            "top1_token_str_norm",
            "p_answer_norm",
            "answer_rank_norm",
            # cross
            "kl_norm_vs_raw_bits",
            "kl_raw_to_norm_bits",
            "js_divergence",
            "entropy_bits_norm",
            "entropy_bits_raw",
            "entropy_gap_bits",
            "l1_prob_diff",
            "topk_jaccard_raw_norm@50",
            "topk_jaccard_consecutive@50",
            "norm_only_semantics",
        ]
        writer.writerow(header)

        def _nz(val):
            return "" if val is None else val

        for rec in records:
            row = [
                rec.get("prompt_id", ""),
                rec.get("prompt_variant", ""),
                rec.get("layer"),
                _nz(rec.get("p_top1_raw")),
                rec.get("top1_token_id_raw"),
                _nz(rec.get("top1_token_str_raw")),
                _nz(rec.get("p_answer_raw")),
                _nz(rec.get("answer_rank_raw")),
                _nz(rec.get("p_top1_norm")),
                rec.get("top1_token_id_norm"),
                _nz(rec.get("top1_token_str_norm")),
                _nz(rec.get("p_answer_norm")),
                _nz(rec.get("answer_rank_norm")),
                _nz(rec.get("kl_norm_vs_raw_bits")),
                _nz(rec.get("kl_raw_to_norm_bits")),
                _nz(rec.get("js_divergence")),
                _nz(rec.get("entropy_bits_norm")),
                _nz(rec.get("entropy_bits_raw")),
                _nz(rec.get("entropy_gap_bits")),
                _nz(rec.get("l1_prob_diff")),
                _nz(rec.get("topk_jaccard_raw_norm@50")),
                _nz(rec.get("topk_jaccard_consecutive@50")),
                _nz(rec.get("norm_only_semantics")),
            ]
            writer.writerow(row)
