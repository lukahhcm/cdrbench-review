from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from cdrbench_v3.io import read_jsonl, write_json
from cdrbench_v3.schema import validate_v3_row


def validate_path(path: Path) -> dict:
    errors: dict[str, list[str]] = defaultdict(list)
    counts = Counter()
    rows = read_jsonl(path)
    for index, row in enumerate(rows, start=1):
        counts["rows"] += 1
        counts[f"track_family:{row.get('track_family')}"] += 1
        counts[f"benchmark_track:{row.get('benchmark_track')}"] += 1
        counts[f"source_track:{row.get('source_track')}"] += 1
        counts[f"scoring_profile:{row.get('scoring_profile')}"] += 1
        for error in validate_v3_row(row):
            errors[str(path)].append(f"row {index}: {error}")
    return {"path": str(path), "counts": dict(sorted(counts.items())), "errors": dict(errors)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CDR-Bench v3 JSONL files.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    reports = [validate_path(Path(path).resolve()) for path in args.paths]
    total_errors = sum(len(items) for report in reports for items in report["errors"].values())
    payload = {"num_files": len(reports), "num_errors": total_errors, "files": reports}
    if args.output_json:
        write_json(Path(args.output_json).resolve(), payload)
    if total_errors:
        for report in reports:
            for path, errors in report["errors"].items():
                for error in errors[:20]:
                    print(f"{path}: {error}")
        raise SystemExit(f"validation failed with {total_errors} errors")
    print(f"validation passed for {len(reports)} file(s)")


if __name__ == "__main__":
    main()
