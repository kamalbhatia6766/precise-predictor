"""Quant system roadmap builder with merge planning metadata."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import quant_paths

# Classification rules documentation:
# merge_group: string category for Master Predictor merge planning.
# keep_for_master: boolean flag indicating initial keep/delete stance for merge.

SKIP_DIRS = {"__pycache__", "logs", "output", ".git"}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def has_main_guard(text: str) -> bool:
    return "if __name__ == \"__main__\"" in text or "if __name__ == '__main__'" in text


def classify_for_merge(path: Path) -> Tuple[str, bool]:
    """Classify a file for merge planning and decide keep flag."""
    name = path.name
    lower_name = name.lower()

    predictor_core_names = {f"deepseek_scr{i}.py" for i in range(1, 10)}
    if name in predictor_core_names:
        merge_group = "PREDICTOR_CORE"
    elif name == "deepseek_scr11.py" or ("ultimate" in lower_name and "predict" in lower_name):
        merge_group = "AGGREGATOR_LEGACY"
    elif name in {
        "precise_bet_engine.py",
        "bet_pnl_tracker.py",
        "quant_pnl_signals.py",
        "quant_slot_health.py",
    } or ("bet_engine" in lower_name or "pnl" in lower_name):
        merge_group = "BET_ENGINE"
    elif name in {"quant_system_audit.py", "quant_roadmap_builder.py"} or (
        "audit" in lower_name or "roadmap" in lower_name
    ):
        merge_group = "INFRA"
    elif any(keyword in lower_name for keyword in ["tmp", "temp", "old", "backup", "bk", "scratch", "test", "sandbox"]):
        merge_group = "LEGACY_OR_SCRATCH"
    else:
        merge_group = "UNCLASSIFIED"

    keep_for_master = merge_group in {"PREDICTOR_CORE", "AGGREGATOR_LEGACY", "BET_ENGINE", "INFRA"}
    return merge_group, keep_for_master


def load_latest_audit_report(project_root: Path) -> Tuple[Dict[str, Dict[str, object]], Path | None]:
    performance_dir = project_root / "logs" / "performance"
    reports = list(performance_dir.glob("quant_system_audit_report_*.csv"))
    if not reports:
        return {}, None

    latest_report = max(reports, key=lambda p: p.stat().st_mtime)
    warnings_by_file: Dict[str, Dict[str, object]] = {}

    with latest_report.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_path_value = row.get("file_path", "").strip()
            warning_type = row.get("warning_type", "").strip()
            if not file_path_value:
                continue
            path_obj = Path(file_path_value)
            try:
                rel_path = path_obj.relative_to(project_root)
            except ValueError:
                rel_path = Path(file_path_value)
            rel_key = str(rel_path)

            if rel_key not in warnings_by_file:
                warnings_by_file[rel_key] = {"warnings_total": 0, "warning_types": set()}
            warnings_by_file[rel_key]["warnings_total"] += 1
            if warning_type:
                warnings_by_file[rel_key]["warning_types"].add(warning_type)

    for rel_key, info in warnings_by_file.items():
        info["warning_types"] = ", ".join(sorted(info["warning_types"]))

    return warnings_by_file, latest_report


def build_rows(project_root: Path, audit_data: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    py_files = [p for p in project_root.rglob("*.py") if not should_skip(p)]

    for path in py_files:
        rel_path = path.relative_to(project_root)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""

        merge_group, keep_for_master = classify_for_merge(path)
        warning_info = audit_data.get(str(rel_path), {"warnings_total": 0, "warning_types": ""})

        stat_info = path.stat()
        row = {
            "file_name": path.name,
            "relative_path": str(rel_path),
            "merge_group": merge_group,
            "keep_for_master": keep_for_master,
            "is_in_root": len(rel_path.parents) == 1,
            "folder_depth": len(rel_path.parts) - 1,
            "warnings_total": warning_info.get("warnings_total", 0),
            "warning_types": warning_info.get("warning_types", ""),
            "has_main_guard": has_main_guard(text),
            "size_bytes": stat_info.st_size,
            "last_modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(timespec="seconds"),
            "notes": "",
        }
        rows.append(row)

    return rows


def write_excel(rows: List[Dict[str, object]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    column_order = [
        "file_name",
        "relative_path",
        "merge_group",
        "keep_for_master",
        "is_in_root",
        "folder_depth",
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


def write_summary(rows: List[Dict[str, object]], audit_report: Path | None, summary_path: Path) -> None:
    summary_lines: List[str] = []
    summary_lines.append("=== QUANT SYSTEM ROADMAP SUMMARY ===")
    summary_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    summary_lines.append(f"Total Python files: {len(rows)}")
    if audit_report:
        audit_files = {row["relative_path"] for row in rows if row.get("warnings_total", 0)}
        summary_lines.append(f"Loaded audit warnings from: {audit_report.name} (files with warnings: {len(audit_files)})")
    else:
        summary_lines.append("Audit report not found; warnings not loaded.")

    merge_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        group = row.get("merge_group", "UNCLASSIFIED")
        keep_flag = bool(row.get("keep_for_master", False))
        merge_counts[group]["total"] += 1
        if keep_flag:
            merge_counts[group]["keep_true"] += 1

    summary_lines.append("")
    summary_lines.append("=== MERGE GROUP SUMMARY ===")
    summary_lines.append(f"total_files = {len(rows)}")
    for group in [
        "PREDICTOR_CORE",
        "AGGREGATOR_LEGACY",
        "BET_ENGINE",
        "INFRA",
        "LEGACY_OR_SCRATCH",
        "UNCLASSIFIED",
    ]:
        total = merge_counts[group]["total"]
        keep_true = merge_counts[group]["keep_true"]
        keep_note = "keep_for_master=True" if group != "LEGACY_OR_SCRATCH" and group != "UNCLASSIFIED" else "keep_for_master=False"
        summary_lines.append(f"{group}:  {total:2d}  ({keep_note}: {keep_true:2d})")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def run_builder() -> None:
    base_dir = quant_paths.get_base_dir()
    project_root = Path(base_dir)

    print("=== QUANT SYSTEM ROADMAP BUILDER ===")

    audit_data, audit_report = load_latest_audit_report(project_root)
    if audit_report:
        loaded_files = len(audit_data)
        print(f"Loaded audit warnings for {loaded_files} files from {audit_report.name}")
    else:
        print("Audit report not found; proceeding without warnings data")

    rows = build_rows(project_root, audit_data)
    print(f"Scanned {len(rows)} Python files")

    output_dir = project_root / "logs" / "performance"
    excel_path = output_dir / "quant_system_roadmap.xlsx"
    write_excel(rows, excel_path)
    print(f"Roadmap Excel saved to: {excel_path}")

    summary_path = output_dir / "quant_system_roadmap_summary.txt"
    write_summary(rows, audit_report, summary_path)
    print(f"Summary text saved to: {summary_path}")

    merge_counts = Counter(row.get("merge_group", "UNCLASSIFIED") for row in rows)
    if merge_counts:
        merge_parts = [f"{group}={count}" for group, count in merge_counts.items()]
        print("Merge groups: " + ", ".join(merge_parts))


if __name__ == "__main__":
    run_builder()
