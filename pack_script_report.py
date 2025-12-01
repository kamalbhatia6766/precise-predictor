import json
from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def load_pattern_stats(path: Path):
    if not path.exists():
        print(f"Missing file: {path}")
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Error reading {path}: {exc}")
        return None

    pattern_stats = data.get("pattern_stats", {})
    rows = []
    for name, stats in pattern_stats.items():
        hits = stats.get("hits", 0)
        opps = stats.get("opportunities", 0)
        weight = stats.get("weight", stats.get("score", 0))
        hit_rate = (hits / opps * 100) if opps else 0
        rows.append({
            "name": name,
            "hits": hits,
            "hit_rate": hit_rate,
            "weight": weight,
        })

    rows.sort(key=lambda x: x.get("weight", 0), reverse=True)
    return rows[:10]


def load_script_hits(path: Path):
    if not path.exists():
        print(f"Missing file: {path}")
        return None

    try:
        df = pd.read_excel(path)
    except Exception as exc:
        print(f"Error reading {path}: {exc}")
        return None

    if df.empty:
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def pick_number_column(df: pd.DataFrame):
    for col in ["hit_number", "number", "result", "winning_number", "value"]:
        if col in df.columns:
            return col
    return None


def summarize_script_hits(df: pd.DataFrame):
    if df is None:
        return None
    if df.empty:
        return []

    total_hits = len(df)
    script_col = "script" if "script" in df.columns else None
    hit_family_col = "hit_family" if "hit_family" in df.columns else None
    number_col = pick_number_column(df)

    summaries = []
    if script_col:
        for script, group in df.groupby(script_col):
            hits = len(group)
            share = (hits / total_hits * 100) if total_hits else 0
            cross_hits = group[hit_family_col].str.contains("cross", case=False, na=False).sum() if hit_family_col else 0
            direct_hits = group[hit_family_col].str.contains("direct", case=False, na=False).sum() if hit_family_col else 0

            s40_hits = non_s40_hits = None
            if number_col:
                from pattern_packs import is_s40

                nums = group[number_col].dropna()
                s40_hits = sum(1 for n in nums if is_s40(n))
                non_s40_hits = hits - s40_hits

            summaries.append({
                "script": script,
                "hits": hits,
                "share": share,
                "cross": cross_hits,
                "direct": direct_hits,
                "s40": s40_hits,
                "non_s40": non_s40_hits,
            })

    return summaries


def print_pattern_section(top_patterns):
    if top_patterns is None:
        return False

    print("TOP PATTERN FAMILIES:")
    if not top_patterns:
        print("  (no pattern data)")
        return True

    for row in top_patterns:
        name = row["name"]
        marker = "⭐ " if name in {"S40", "164950"} else ""
        print(f"  {marker}{name}: hit_rate={row['hit_rate']:.1f}% | hits={row['hits']} | weight={row['weight']}")
    return True


def print_script_section(script_summaries):
    if script_summaries is None:
        return False

    print("\nSCRIPT PERFORMANCE SUMMARY:")
    if not script_summaries:
        print("  (no script hit data)")
        return True

    for row in sorted(script_summaries, key=lambda r: r.get("script")):
        cross = row.get("cross", 0)
        direct = row.get("direct", 0)
        share = row.get("share", 0)
        print(f"  {row['script']}: hits={row['hits']} | CROSS={cross} | DIRECT={direct} | share={share:.1f}%")
    return True


def print_pack_overlap(script_summaries):
    if not script_summaries:
        return

    print("\nSCRIPT PACK CONTRIBUTION (approx):")
    for row in sorted(script_summaries, key=lambda r: r.get("script")):
        s40 = row.get("s40")
        non_s40 = row.get("non_s40")
        if s40 is None or non_s40 is None:
            print(f"  {row['script']}: hits={row['hits']} (pack split unavailable)")
        else:
            print(f"  {row['script']}: hits={row['hits']}, S40≈{s40}, NON≈{non_s40}")


def main():
    pattern_file = BASE_DIR / "logs" / "performance" / "pattern_intelligence.json"
    script_hit_file = BASE_DIR / "logs" / "performance" / "script_hit_memory.xlsx"

    top_patterns = load_pattern_stats(pattern_file)
    if top_patterns is None:
        return 0

    script_df = load_script_hits(script_hit_file)
    if script_df is None:
        return 0

    script_summaries = summarize_script_hits(script_df)

    print_pattern_section(top_patterns)
    print_script_section(script_summaries)
    print_pack_overlap(script_summaries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
