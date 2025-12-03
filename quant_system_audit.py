"""Static audit tool for Precise Predictor project.

This script scans project Python files and surfaces potential risk patterns
without modifying any files. Outputs a CSV report under logs/performance.
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import re
import csv

import quant_paths


SLOT_NAMES = {"FRBD", "GZBD", "GALI", "DSWR"}
SKIP_DIRS = {"__pycache__", "logs", "output", "predictions"}


def should_skip(path: Path) -> bool:
    """Check whether a path is in a skipped directory."""
    return any(part in SKIP_DIRS for part in path.parts)


def levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr_row = [i]
        for j, cb in enumerate(b, 1):
            insertions = prev_row[j] + 1
            deletions = curr_row[j - 1] + 1
            substitutions = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def scan_file(file_path: Path) -> list:
    """Scan a single file and return warning rows."""
    warnings = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return warnings

    hard_path_pattern = re.compile(r"[A-Za-z]:\\\\Users\\\\|/home/|/Users/")

    for idx, line in enumerate(content, start=1):
        lower_line = line.lower()
        upper_line = line.upper()

        if hard_path_pattern.search(line):
            warnings.append({
                "file_path": str(file_path),
                "warning_type": "HARD_CODED_PATH",
                "line_number": idx,
                "snippet": line.strip()
            })

        if (".xlsx" in lower_line or ".xls" in lower_line) and "quant_paths" not in lower_line:
            warnings.append({
                "file_path": str(file_path),
                "warning_type": "RAW_EXCEL_PATH",
                "line_number": idx,
                "snippet": line.strip()
            })

        tokens = re.findall(r"\b[A-Z]{3,5}\b", line)
        for token in tokens:
            if token in SLOT_NAMES:
                continue
            if any(levenshtein_distance(token, slot) <= 1 for slot in SLOT_NAMES):
                warnings.append({
                    "file_path": str(file_path),
                    "warning_type": "SLOT_NAME_SUSPECT",
                    "line_number": idx,
                    "snippet": token
                })
                break

        if ("DATE" in upper_line or "to_datetime" in lower_line or "datetime.strptime" in lower_line) and \
                ("quant_data_core" not in lower_line and "quant_paths" not in lower_line):
            warnings.append({
                "file_path": str(file_path),
                "warning_type": "DATE_PARSING_DUPLICATE",
                "line_number": idx,
                "snippet": line.strip()
            })

        if "result" in lower_line and "excel" in lower_line and "quant_paths" not in lower_line and "quant_data_core" not in lower_line:
            warnings.append({
                "file_path": str(file_path),
                "warning_type": "RAW_RESULTS_FILE_REFERENCE",
                "line_number": idx,
                "snippet": line.strip()
            })

    return warnings


def run_audit():
    base_dir = quant_paths.get_base_dir()
    project_root = Path(base_dir)

    py_files = [p for p in project_root.rglob("*.py") if not should_skip(p)]

    all_warnings = []
    for file_path in py_files:
        all_warnings.extend(scan_file(file_path))

    logs_dir = project_root / "logs" / "performance"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / f"quant_system_audit_report_{datetime.now().strftime('%Y%m%d')}.csv"

    with report_path.open("w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file_path", "warning_type", "line_number", "snippet"])
        writer.writeheader()
        for row in all_warnings:
            writer.writerow(row)

    print("=== QUANT SYSTEM STATIC AUDIT ===")
    print(f"Scanned {len(py_files)} .py files")
    print(f"Found {len(all_warnings)} warnings")
    if all_warnings:
        type_counts = Counter(w["warning_type"] for w in all_warnings)
        file_counts = Counter(w["file_path"] for w in all_warnings)

        print("Warning types summary:")
        for wtype, count in type_counts.most_common():
            print(f"  {wtype}: {count}")

        print("Top 10 files by warnings:")
        for fpath, count in file_counts.most_common(10):
            print(f"  {Path(fpath).name}: {count}")
    if not all_warnings:
        print("No warnings found")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    run_audit()
