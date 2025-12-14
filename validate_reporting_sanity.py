from __future__ import annotations

# PR57: sanity check for pattern membership numerators

import sys
from typing import Dict, Optional

import quant_daily_brief
import quant_learning_core


def _extract_hits(block: Optional[Dict]) -> int:
    if not isinstance(block, dict):
        return 0
    return int(block.get("hits_total") or block.get("exact_hits") or block.get("hits") or 0)


def main() -> int:
    summary = quant_learning_core.load_pattern_summary_json()
    if not summary:
        print("Pattern summary not found; run pattern_intelligence_enhanced.py first.")
        return 1

    patterns = summary.get("patterns", {}) if isinstance(summary, dict) else {}
    s40_raw = patterns.get("S40", {}) or {}
    fam_raw = patterns.get("PACK_164950") or patterns.get("FAMILY_164950") or {}

    brief_summary = quant_daily_brief.load_pattern_summary_from_intel()
    s40_hits_brief = int((brief_summary.s40 or {}).get("hits_total") or (brief_summary.s40 or {}).get("hits") or 0)
    fam_hits_brief = int((brief_summary.fam_164950 or {}).get("hits_total") or (brief_summary.fam_164950 or {}).get("hits") or 0)

    s40_hits_expected = _extract_hits(s40_raw)
    fam_hits_expected = _extract_hits(fam_raw)

    ok = True
    if s40_hits_brief != s40_hits_expected:
        print(
            f"S40 membership mismatch: brief={s40_hits_brief} vs PatternIntel={s40_hits_expected}")
        ok = False
    if fam_hits_brief != fam_hits_expected:
        print(
            f"164950 membership mismatch: brief={fam_hits_brief} vs PatternIntel={fam_hits_expected}")
        ok = False

    if not ok:
        return 2

    print("Reporting sanity check passed: membership numerators match PatternIntel summaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
