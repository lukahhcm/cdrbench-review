from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CDR-Bench v3 data from Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Dataset repository containing the benchmark JSONL files.")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", default="data/benchmark_v3")
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=[
            "manifest.json",
            "benchmark_v3_all.jsonl",
            "benchmark_v3_all_prompts.jsonl",
            "tracks/*.jsonl",
            "tracks_all_prompts/*.jsonl",
        ],
        help="Hugging Face snapshot allow patterns.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=args.allow_patterns,
    )
    print(f"Downloaded {args.repo_id} -> {output_dir}")


if __name__ == "__main__":
    main()
