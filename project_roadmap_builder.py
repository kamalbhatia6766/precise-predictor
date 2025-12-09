# project_roadmap_builder.py - Auto-maps project scripts/data into a roadmap sheet
# Generates output/precise_predictor_roadmap_YYYYMMDD.xlsx
# Examples:
#   py -3.12 project_roadmap_builder.py
#   py -3.12 project_roadmap_builder.py --root .

import argparse
import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def get_first_docstring_or_comment(path: Path) -> str:
    """Return the first module docstring or top-level comment snippet"""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    snippet: List[str] = []
    try:
        tree = ast.parse(text)
        doc = ast.get_docstring(tree)
        if doc:
            snippet.append(doc)
    except Exception:
        pass

    if not snippet:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                snippet.append(stripped.lstrip("# "))
            elif stripped:
                break
            if len(snippet) >= 8:
                break

    joined = " ".join(snippet).strip()
    return joined[:400]


def infer_category(filename: str, hint_text: str) -> str:
    name = filename.lower()
    hint_lower = hint_text.lower()
    if "deepseek" in name or "predict" in name or "scr" in name:
        return "prediction_logic"
    if any(word in name for word in ["bet", "pnl", "stake", "money", "bankroll"]):
        return "betting_pnl"
    if any(word in name for word in ["learn", "backtest", "train", "model", "lstm", "xgboost"]):
        return "learning_ml"
    if "util" in name or "helper" in name or "tool" in name:
        return "utilities"
    if "pack" in name or "pattern" in name:
        return "pattern_packs"
    if any(word in hint_lower for word in ["ensemble", "prediction", "slot"]):
        return "prediction_logic"
    return "unknown"


def detect_internal_imports(text: str, module_names: set) -> str:
    imports: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            parts = re.split(r"[, ]+", stripped.replace("import", "").strip())
            for part in parts:
                base = part.split(".")[0]
                if base in module_names and base not in imports:
                    imports.append(base)
        elif stripped.startswith("from "):
            match = re.match(r"from\s+([\w\.]+)\s+import", stripped)
            if match:
                base = match.group(1).split(".")[0]
                if base in module_names and base not in imports:
                    imports.append(base)
    return ", ".join(imports)


def is_entry_point(text: str) -> bool:
    return "if __name__ == \"__main__\":" in text or "if __name__ == '__main__':" in text


def scan_scripts(root: Path) -> List[Dict]:
    py_files = list(root.rglob("*.py"))
    module_names = {p.stem for p in py_files}
    records: List[Dict] = []

    for path in py_files:
        rel_path = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""

        snippet = get_first_docstring_or_comment(path)
        category = infer_category(path.name, snippet)
        entry_point = is_entry_point(text)
        imports_internal = detect_internal_imports(text, module_names)

        records.append({
            "type": "script",
            "filename": path.name,
            "relative_path": str(rel_path),
            "inferred_category": category,
            "first_docstring_or_comment": snippet,
            "is_entry_point": entry_point,
            "imports_internal": imports_internal,
            "guessed_role": ""
        })

    return records


def scan_data_files(root: Path) -> List[Dict]:
    records: List[Dict] = []
    data_files = list(root.glob("*.xlsx")) + list(root.glob("*.csv"))
    data_files += list((root / "predictions").glob("*.xlsx")) if (root / "predictions").exists() else []

    for path in data_files:
        rel_path = path.relative_to(root)
        name = path.name.lower()
        role = "data_file"
        if "number prediction learn" in name:
            role = "results_history"
        elif "bet_plan_master" in name:
            role = "daily_bet_plans"

        records.append({
            "type": "data",
            "filename": path.name,
            "relative_path": str(rel_path),
            "inferred_category": "",
            "first_docstring_or_comment": "",
            "is_entry_point": "",
            "imports_internal": "",
            "guessed_role": role
        })

    return records


def build_roadmap(root: Path, output_dir: Path) -> Path:
    script_records = scan_scripts(root)
    data_records = scan_data_files(root)
    all_records = script_records + data_records

    df = pd.DataFrame(all_records)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"precise_predictor_roadmap_{datetime.now().strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="roadmap", index=False)

    print(f"Found {len(script_records)} scripts, {len(data_records)} data files")
    print(f"Roadmap written to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Builds a project roadmap of scripts and data")
    parser.add_argument("--root", type=str, default=".", help="Root folder to scan (default: current directory)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "output"

    build_roadmap(root, output_dir)


if __name__ == "__main__":
    main()
