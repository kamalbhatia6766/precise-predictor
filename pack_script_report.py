import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

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
    pattern_weights = data.get("pattern_weights", {})
    rows = []
    for name, stats in pattern_stats.items():
        hits = stats.get("hits", 0)
        opps = stats.get("opportunities", 0)
        weight = stats.get("weight")
        if weight is None:
            weight = pattern_weights.get(name, stats.get("score", 1.0))
        if weight == 0:
            weight = 1.0  # Avoid zero weights unless explicitly set
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
    script_stats = {}
    if script_col:
        for script, group in df.groupby(script_col):
            hits = len(group)
            share_ratio = (hits / total_hits) if total_hits else 0
            cross_hits = group[hit_family_col].str.contains("cross", case=False, na=False).sum() if hit_family_col else 0
            direct_hits = group[hit_family_col].str.contains("direct", case=False, na=False).sum() if hit_family_col else 0

            s40_hits = non_s40_hits = None
            if number_col:
                from pattern_packs import is_s40

                nums = group[number_col].dropna()
                s40_hits = sum(1 for n in nums if is_s40(n))
                non_s40_hits = hits - s40_hits

            script_stats[str(script).upper()] = {
                "script": str(script).upper(),
                "hits": hits,
                "share_ratio": share_ratio,
                "cross": cross_hits,
                "direct": direct_hits,
                "s40": s40_hits,
                "non_s40": non_s40_hits,
            }

    for idx in range(1, 10):
        key = f"SCR{idx}"
        if key not in script_stats:
            script_stats[key] = {
                "script": key,
                "hits": 0,
                "share_ratio": 0,
                "cross": 0,
                "direct": 0,
                "s40": None,
                "non_s40": None,
            }

    scores = {name: stats.get("direct", 0) * 2.0 + stats.get("cross", 0) for name, stats in script_stats.items()}
    max_score = max(scores.values()) if scores else 0
    for name, stats in script_stats.items():
        score_val = scores.get(name, 0)
        if max_score == 0:
            weight = 1.0
        else:
            base = score_val / max_score
            weight = 0.5 + 1.5 * base
        stats["score"] = score_val
        stats["weight"] = weight

    summaries = list(script_stats.values())

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
        share_pct = (row.get("share_ratio", 0) or 0) * 100
        weight = row.get("weight", 1.0)
        hits = row.get("hits", 0)
        if hits == 0 or share_pct == 0:
            status = "DORMANT"
        elif weight < 0.8:
            status = "LOW-CONFIDENCE"
        elif weight > 1.4:
            status = "HIGH-CONFIDENCE"
        else:
            status = "MID-CONFIDENCE"

        print(
            f"  {row['script']}: hits={hits} | CROSS={cross} | DIRECT={direct} | share={share_pct:.1f}% | weight={weight:.2f} | status={status}"
        )
    return True


def print_pack_overlap(script_summaries):
    if not script_summaries:
        return

    print("\nSCRIPT PACK CONTRIBUTION (approx):")
    for row in sorted(script_summaries, key=lambda r: r.get("script")):
        s40 = row.get("s40")
        non_s40 = row.get("non_s40")
        if s40 is None or non_s40 is None:
            print(f"  {row['script']}: hits={row['hits']} (pack split unavailable – TODO for per-script pack split)")
        else:
            print(f"  {row['script']}: hits={row['hits']}, S40≈{s40}, NON≈{non_s40}")


def _to_json_safe(obj):
    """
    Recursively convert numpy / pandas scalar types into
    plain Python int/float/bool so json.dump doesn't choke.
    """
    import numpy as np

    # Scalars
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if isinstance(obj, (np.bool_, )):
        return bool(obj)

    # ndarrays -> lists
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]

    # Containers
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]

    # Everything else unchanged
    return obj


def write_script_weights_json(script_summaries, output_path: Path):
    if not script_summaries:
        return

    payload = {
        "scripts": [
            {
                "name": row.get("script"),
                "hits": row.get("hits", 0),
                "cross": row.get("cross", 0),
                "direct": row.get("direct", 0),
                "share": row.get("share_ratio", 0),
                "score": row.get("score", 0),
                "weight": row.get("weight", 1.0),
            }
            for row in sorted(script_summaries, key=lambda r: r.get("script"))
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _to_json_safe(payload)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe_payload, f, indent=2, ensure_ascii=False)


def main():
    pattern_file = BASE_DIR / "logs" / "performance" / "pattern_intelligence.json"
    script_hit_file = BASE_DIR / "logs" / "performance" / "script_hit_memory.xlsx"
    script_weight_file = BASE_DIR / "logs" / "performance" / "script_performance_summary.json"
    script_weight_file_alias = BASE_DIR / "logs" / "performance" / "script_weights.json"

    top_patterns = load_pattern_stats(pattern_file)
    if top_patterns is None:
        return 0

    script_df = load_script_hits(script_hit_file)
    if script_df is None:
        return 0

    script_summaries = summarize_script_hits(script_df)

    write_script_weights_json(script_summaries, script_weight_file)
    # Also write a friendly alias expected by downstream ensemble engines
    write_script_weights_json(script_summaries, script_weight_file_alias)

    print_pattern_section(top_patterns)
    print_script_section(script_summaries)
    print_pack_overlap(script_summaries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
