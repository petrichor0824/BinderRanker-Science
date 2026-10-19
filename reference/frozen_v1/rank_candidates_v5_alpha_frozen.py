#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rank_candidates_v5_alpha_frozen.py

BinderRanker v5-alpha frozen candidate.

Frozen architecture/parameters
------------------------------
beta_local      = 0.0
extent_variant  = coverage_cb_micro
extent_weight   = 0.90
micro_weight    = 0.0
safety_weight   = 0.0
roughness_weight= 0.0

Therefore:
    morphology_weight = 0.10

Final score:
    final_score_v5
      = 0.10 * global_morphology
      + 0.90 * coherent_extent

where coherent_extent is the geometric mean of target-normalized:
    - target_effective_coverage
    - cb_supported_weight_ratio
    - micro_fit_score_weighted

Safety/roughness remain available through the frozen v4-derived filters and
diagnostic columns, but they are not positive outer rewards in final_score_v5.

This wrapper deliberately exposes NO tuning flags for the frozen model.
It delegates calculation/output to the SHA-guarded v5 prototype and frozen v4.
"""

from pathlib import Path
import argparse
import hashlib
import subprocess
import sys

DEFAULT_V4 = Path("/home/u2023010109/binderranker_work/BinderRanker/rank_candidates_v4_layered.py")
DEFAULT_PROTO = Path("/home/u2023010109/binderranker_work/BinderRanker/rank_candidates_v5_proto.py")

EXPECTED_V4_SHA256 = "9871434c132ceb3fb0f5e54ff8ada17db241525e7f03c2124b6c6e746558ddfe"
EXPECTED_PROTO_SHA256 = "2d6ec85a02f2cf7186900bb535d5769196f14f2c1794d2c346312200fc939f18"

FROZEN_CONFIG = {
    "morph_alpha": "1.25",
    "min_mode_weight": "0.05",
    "q_low": "0.05",
    "q_high": "0.95",
    "beta_local": "0.0",
    "local_lambda": "1.0",
    "gain_plane": "1.0",
    "gain_compact": "1.0",
    "extent_variant": "coverage_cb_micro",
    "micro_weight": "0.0",
    "extent_weight": "0.90",
    "safety_weight": "0.0",
    "roughness_weight": "0.0",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise SystemExit(f"[FATAL] missing {label}: {path}")
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(
            f"[FATAL] {label} SHA256 mismatch\n"
            f"expected={expected}\n"
            f"got={got}\n"
            f"file={path}"
        )
    return got


def main() -> None:
    p = argparse.ArgumentParser(description="BinderRanker v5-alpha frozen candidate.")
    p.add_argument("--input_csv", required=True, help="contact_field_v3 CSV for ONE target")
    p.add_argument("--output_xlsx", required=True)
    p.add_argument("--output_tsv", default=None)
    p.add_argument("--output_txt", default=None)
    p.add_argument("--v4_script", default=str(DEFAULT_V4))
    p.add_argument("--v5_proto", default=str(DEFAULT_PROTO))
    p.add_argument("--top_n_sheets", type=int, default=500)
    p.add_argument("--top_k_report", type=int, default=30)
    args = p.parse_args()

    v4 = Path(args.v4_script)
    proto = Path(args.v5_proto)

    v4_sha = require_sha(v4, EXPECTED_V4_SHA256, "frozen v4")
    proto_sha = require_sha(proto, EXPECTED_PROTO_SHA256, "frozen v5 prototype")

    cmd = [
        sys.executable,
        str(proto),
        "--input_csv", args.input_csv,
        "--v4_script", str(v4),
        "--output_xlsx", args.output_xlsx,
        "--top_n_sheets", str(args.top_n_sheets),
        "--top_k_report", str(args.top_k_report),
        "--morph_alpha", FROZEN_CONFIG["morph_alpha"],
        "--min_mode_weight", FROZEN_CONFIG["min_mode_weight"],
        "--q_low", FROZEN_CONFIG["q_low"],
        "--q_high", FROZEN_CONFIG["q_high"],
        "--beta_local", FROZEN_CONFIG["beta_local"],
        "--local_lambda", FROZEN_CONFIG["local_lambda"],
        "--gain_plane", FROZEN_CONFIG["gain_plane"],
        "--gain_compact", FROZEN_CONFIG["gain_compact"],
        "--extent_variant", FROZEN_CONFIG["extent_variant"],
        "--micro_weight", FROZEN_CONFIG["micro_weight"],
        "--extent_weight", FROZEN_CONFIG["extent_weight"],
        "--safety_weight", FROZEN_CONFIG["safety_weight"],
        "--roughness_weight", FROZEN_CONFIG["roughness_weight"],
    ]

    if args.output_tsv:
        cmd.extend(["--output_tsv", args.output_tsv])
    if args.output_txt:
        cmd.extend(["--output_txt", args.output_txt])

    print("=" * 80)
    print("BinderRanker v5-alpha FROZEN CANDIDATE")
    print(f"Frozen v4 SHA256       : {v4_sha}")
    print(f"Frozen v5 proto SHA256 : {proto_sha}")
    print("Frozen score           : 0.10 * global_morphology + 0.90 * coherent_extent")
    print("coherent_extent        : coverage × CB-support × microfit (geometric mean)")
    print("Safety/Roughness outer : 0 / 0 (retained in filters/diagnostics)")
    print("=" * 80)

    subprocess.run(cmd, check=True)

    print("=" * 80)
    print("[PASS] V5-ALPHA FROZEN CANDIDATE RUN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
