"""Build a system roadmap by combining filesystem metadata and audit warnings."""
from __future__ import annotations

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

import quant_paths


def get_project_root() -> Path:
    """Return the project root using quant_paths for consistency."""
    return quant_paths.get_base_dir()


def find_python_files(root_dir: Path) -> List[Path]:
    """Recursively collect all Python files under the root directory."""
    return [Path(dirpath) / fname
            for dirpath, _, files in os.walk(root_dir)
            for fname in files
            if fname.endswith(".py")]


def format_mtime(path: Path) -> str:
    """Convert mtime to a readable string."""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def file_has_main_guard(path: Path) -> bool:
    """Check whether the file contains a __main__ guard."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "if __name__ == \"__main__\"" in content or "if __name__ == '__main__'" in content


def categorize_file(file_name: str) -> str:
    """Assign a rough category based on filename heuristics."""
    lower_name = file_name.lower()
    if lower_name.startswith("deepseek_scr"):
        return "prediction_script"
    if "bet_pnl" in lower_name or "pnl" in lower_name:
        return "pnl_analysis"
    if "audit" in lower_name:
        return "tooling_audit"
    if "pattern" in lower_name or "pack" in lower_name:
        return "pattern_packs"
    if "core" in lower_name:
        return "core_engine"
    return "misc"


def normalize_path(value: str, base_dir: Path) -> str:
    """Normalize file paths from audit data to project-relative strings."""
    path = Path(value)
    if path.is_absolute():
        try:
            return path.relative_to(base_dir).as_posix()
        except ValueError:
            return path.name
    return path.as_posix()


def load_latest_audit(log_dir: Path, base_dir: Path) -> Dict[str, Dict[str, object]]:
    """Load the latest audit report and summarize warnings per file."""
    pattern = str(log_dir / "quant_system_audit_report_*.csv")
    audit_files = glob.glob(pattern)
    if not audit_files:
        return {}

    latest_path = max(audit_files, key=os.path.getmtime)
    try:
        df = pd.read_csv(latest_path)
    except Exception:
        return {}

    path_col = None
    for candidate in ["file_path", "file", "path", "filename"]:
        if candidate in df.columns:
            path_col = candidate
            break
    if path_col is None:
        return {}

    warning_col = "warning_type" if "warning_type" in df.columns else None

    summary: Dict[str, Dict[str, object]] = {}
    for _, row in df.iterrows():
        raw_path = str(row[path_col])
        normalized = normalize_path(raw_path, base_dir)
        entry = summary.setdefault(normalized, {"warnings_total": 0, "warning_types": set()})
        entry["warnings_total"] += 1
        if warning_col and pd.notna(row.get(warning_col)):
            entry["warning_types"].add(str(row[warning_col]))

    for rel_path, data in summary.items():
        data["warning_types"] = ",".join(sorted(data["warning_types"]))

    return summary


def build_roadmap_rows(py_files: Iterable[Path], audit_summary: Dict[str, Dict[str, object]], base_dir: Path) -> List[Dict[str, object]]:
    """Combine filesystem scan results with audit summary."""
    rows: List[Dict[str, object]] = []
    for path in py_files:
        rel_path = path.relative_to(base_dir).as_posix()
        file_name = path.name
        warnings_info = audit_summary.get(rel_path, {"warnings_total": 0, "warning_types": ""})

        rows.append({
            "file_name": file_name,
            "rel_path": rel_path,
            "category": categorize_file(file_name),
            "warnings_total": warnings_info.get("warnings_total", 0),
            "warning_types": warnings_info.get("warning_types", ""),
            "has_main_guard": file_has_main_guard(path),
            "size_bytes": path.stat().st_size,
            "last_modified": format_mtime(path),
            "notes": "",
        })
    return rows


def write_excel(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Write the roadmap Excel file."""
    df = pd.DataFrame(rows)
    if not df.empty:
        column_order = [
            "file_name",
            "rel_path",
            "category",
            "warnings_total",
            "warning_types",
            "has_main_guard",
            "size_bytes",
            "last_modified",
            "notes",
        ]
        df = df[column_order]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="roadmap", index=False)


def write_summary(rows: List[Dict[str, object]], summary_path: Path) -> None:
    """Write a human-readable text summary for quick reference."""
    total_files = len(rows)
    sorted_by_warnings = sorted(rows, key=lambda r: r.get("warnings_total", 0), reverse=True)
    top_10 = sorted_by_warnings[:10]

    category_counts: Dict[str, int] = {}
    for row in rows:
        category = row.get("category", "misc")
        category_counts[category] = category_counts.get(category, 0) + 1

    lines = [
        "=== Quant System Roadmap Summary ===",
        f"Total Python files: {total_files}",
        "",
        "Top files by warnings:",
    ]

    if top_10:
        for entry in top_10:
            fname = entry.get("file_name", "")
            count = entry.get("warnings_total", 0)
            lines.append(f"- {fname}: {count}")
    else:
        lines.append("- No warnings data available")

    lines.extend(["", "Files per category:"])
    for category, count in sorted(category_counts.items(), key=lambda item: item[0]):
        lines.append(f"- {category}: {count}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_builder() -> None:
    base_dir = get_project_root()
    performance_dir = base_dir / "logs" / "performance"

    py_files = find_python_files(base_dir)
    audit_summary = load_latest_audit(performance_dir, base_dir)

    rows = build_roadmap_rows(py_files, audit_summary, base_dir)
    roadmap_path = performance_dir / "quant_system_roadmap.xlsx"
    write_excel(rows, roadmap_path)

    summary_path = performance_dir / "quant_system_roadmap_summary.txt"
    write_summary(rows, summary_path)

    print("=== QUANT SYSTEM ROADMAP BUILDER ===")
    print(f"Scanned {len(rows)} Python files")
    if audit_summary:
        print(f"Loaded audit warnings for {len(audit_summary)} files")
    else:
        print("No audit warnings found; proceeding with filesystem data only")
    print(f"Roadmap Excel saved to: {roadmap_path}")
    print(f"Summary text saved to: {summary_path}")


if __name__ == "__main__":
    run_builder()
