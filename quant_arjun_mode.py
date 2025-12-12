from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import quant_paths
from quant_slot_health import load_slot_health, SlotHealth
from quant_stats_core import get_quant_stats

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]
quant_stats = get_quant_stats()
TOPN_POLICY_PATH = Path("data") / "topn_policy.json"
SLOT_HEALTH_PATH = Path("data") / "slot_health.json"


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[quant_arjun_mode] Warning: unable to parse {path}: {exc}")
        return None


def _regime_score(label: Optional[str]) -> int:
    mapping = {"BOOST": 2, "NORMAL": 1, "OFF": 0}
    return mapping.get(str(label).upper(), 0) if label is not None else 0


def _load_slot_health(base_dir: Path) -> Dict[str, Dict[str, object]]:
    snapshot_path = base_dir / SLOT_HEALTH_PATH
    if snapshot_path.exists():
        data = _load_json(snapshot_path) or {}
        slot_health: Dict[str, Dict[str, object]] = {}
        for slot in SLOTS:
            record = data.get(slot, {}) if isinstance(data, dict) else {}
            slot_health[slot] = {
                "roi": float(record.get("roi_30", record.get("roi_percent", 0.0)) or 0.0),
                "hit_rate": float(record.get("hit_rate", 0.0) or 0.0),
                "slump": bool(record.get("slump", False)),
                "slot_level": str(record.get("slot_level", "MID")).upper(),
            }
        return slot_health

    fallback_path = base_dir / "logs" / "performance" / "quant_reality_pnl.json"
    slot_map: Dict[str, Dict[str, object]] = {}
    try:
        raw = load_slot_health(str(fallback_path))
    except Exception as exc:
        print(f"[quant_arjun_mode] Warning: unable to load slot health: {exc}")
        raw = {}
    if raw:
        for slot in SLOTS:
            health: SlotHealth = raw.get(slot) if isinstance(raw, dict) else None
            if health:
                slot_map[slot] = {
                    "roi": getattr(health, "roi_percent", 0.0),
                    "hit_rate": getattr(health, "hit_rate", 0.0),
                    "slump": bool(getattr(health, "slump", False)),
                    "slot_level": "MID",
                }
    return slot_map


def _load_topn_summary(base_dir: Path) -> Dict:
    policy_path = base_dir / TOPN_POLICY_PATH
    if policy_path.exists():
        data = _load_json(policy_path) or {}
        per_slot: Dict[str, Dict[str, object]] = {}
        if isinstance(data, dict):
            for slot in SLOTS:
                record = data.get(slot, {}) if isinstance(data, dict) else {}
                per_slot[slot] = {
                    "roi": float(record.get("roi_final_best", record.get("roi_best_exact", 0.0)) or 0.0),
                    "roi_final_best": float(record.get("roi_final_best", record.get("roi_best_exact", 0.0)) or 0.0),
                    "final_best_n": record.get("final_best_n", record.get("best_n_exact")),
                }
        return {"per_slot": per_slot}

    path = base_dir / "logs" / "performance" / "topn_roi_summary.json"
    return _load_json(path) or {}


def _load_pattern_regimes(base_dir: Path) -> Optional[Dict]:
    candidates = [
        base_dir / "logs" / "performance" / "pattern_regimes_summary.json",
        base_dir / "logs" / "performance" / "pattern_regime_summary.json",
    ]
    data: Optional[Dict] = None
    for path in candidates:
        data = _load_json(path)
        if data:
            break
    if not data:
        return None

    regimes = {slot: {"S40": "NORMAL", "FAMILY_164950": "NORMAL"} for slot in SLOTS}
    extra_best = {
        slot: {"family": None, "regime": "NORMAL", "score": (-float("inf"), -float("inf"), -float("inf"))}
        for slot in SLOTS
    }

    def _score(regime_label: str, hits: int, cover: int):
        mapping = {"BOOST": 2, "NORMAL": 1, "OFF": 0}
        return (int(hits), int(cover), mapping.get(str(regime_label).upper(), 0))

    families = data.get("families", {}) if isinstance(data, dict) else {}

    for fam_key in ["S40", "FAMILY_164950"]:
        fam_block = families.get(fam_key, {}) if isinstance(families, dict) else {}
        per_slot_block = fam_block.get("per_slot", {}) if isinstance(fam_block, dict) else {}
        for slot in SLOTS:
            slot_stats = per_slot_block.get(slot, {}) if isinstance(per_slot_block, dict) else {}
            regime = slot_stats.get("regime", "NORMAL")
            regimes[slot][fam_key] = str(regime).upper()

    for fam_name, fam_block in families.items():
        if fam_name in ("S40", "FAMILY_164950"):
            continue
        per_slot_block = fam_block.get("per_slot", {}) if isinstance(fam_block, dict) else {}
        for slot in SLOTS:
            slot_stats = per_slot_block.get(slot, {}) if isinstance(per_slot_block, dict) else {}
            regime = slot_stats.get("regime", "NORMAL")
            hits = slot_stats.get("hits", 0) or 0
            cover = slot_stats.get("days_covered", 0) or 0
            score = _score(regime, hits, cover)
            if score > tuple(extra_best[slot]["score"]):
                extra_best[slot] = {"family": fam_name, "regime": str(regime).upper(), "score": score}

    return {
        "per_slot": regimes,
        "extra_family": {slot: {"family": info["family"], "regime": info["regime"]} for slot, info in extra_best.items()},
    }


def _load_hero_weak(base_dir: Path) -> Dict:
    path = base_dir / "logs" / "performance" / "script_hero_weak.json"
    return _load_json(path) or {}


def _slot_pattern_regime(patterns: Optional[Dict], slot: str, family: str) -> Optional[str]:
    if not patterns:
        return None
    per_slot_summary = patterns.get("per_slot", {}) if isinstance(patterns, dict) else {}
    if per_slot_summary:
        slot_entry = per_slot_summary.get(slot, {}) if isinstance(per_slot_summary, dict) else {}
        return slot_entry.get(family)

    families = patterns.get("families", {}) if isinstance(patterns, dict) else {}
    fam_block = families.get(family, {}) if isinstance(families, dict) else {}
    per_slot = fam_block.get("per_slot", {}) if isinstance(fam_block, dict) else {}
    slot_entry = per_slot.get(slot, {}) if isinstance(per_slot, dict) else {}
    return slot_entry.get("regime")


def _slot_extra_family(patterns: Optional[Dict], slot: str) -> Optional[Dict[str, Optional[str]]]:
    if not patterns:
        return None
    extra_family = patterns.get("extra_family", {}) if isinstance(patterns, dict) else {}
    slot_entry = extra_family.get(slot) if isinstance(extra_family, dict) else None
    if slot_entry:
        return {"family": slot_entry.get("family"), "regime": slot_entry.get("regime")}
    return None


def _pattern_score_for_slot(patterns: Optional[Dict], slot: str) -> Dict[str, object]:
    if not patterns:
        return {"score": 0.0, "s40": None, "fam": None, "extra": None}

    s40_regime = _slot_pattern_regime(patterns, slot, "S40") or "NORMAL"
    fam_regime = _slot_pattern_regime(patterns, slot, "FAMILY_164950") or "NORMAL"
    extra = _slot_extra_family(patterns, slot)

    def _single(label: Optional[str]) -> int:
        if label is None:
            return 0
        label = str(label).upper()
        if label == "BOOST":
            return 1
        if label == "OFF":
            return -1
        return 0

    extra_score = _single(extra.get("regime")) if extra else 0
    score = _single(s40_regime) + _single(fam_regime) + extra_score
    return {"score": float(score), "s40": s40_regime, "fam": fam_regime, "extra": extra}


def _quant_slot_score(slot: str) -> Dict[str, object]:
    slot_key = str(slot).upper()
    pnl_block = quant_stats.get("pnl", {}) if isinstance(quant_stats, dict) else {}
    slot_pnl = (pnl_block.get("slots", {}) if isinstance(pnl_block, dict) else {}).get(slot_key, {})
    slot_roi = float(slot_pnl.get("roi", 0.0) or slot_pnl.get("roi_percent", 0.0) or 0.0)

    slot_health_block = quant_stats.get("slot_health", {}) if isinstance(quant_stats, dict) else {}
    health_entry = slot_health_block.get(slot_key, {}) if isinstance(slot_health_block, dict) else {}
    slump_flag = bool(health_entry.get("slump", False))
    roi_bucket = str(health_entry.get("roi_bucket", "")).upper()

    topn_root = quant_stats.get("topn", {}) if isinstance(quant_stats, dict) else {}
    topn_slots = topn_root.get("slots", {}) if isinstance(topn_root, dict) else {}
    topn_slot = topn_slots.get(slot_key, {}) if isinstance(topn_slots, dict) else {}
    best_n = topn_slot.get("best_N")
    best_roi = topn_slot.get("best_roi")

    pattern_root = quant_stats.get("patterns", {}) if isinstance(quant_stats, dict) else {}
    fams = (pattern_root.get("slots", {}) if isinstance(pattern_root, dict) else {}).get(slot_key, {})
    fam_block = fams.get("families", {}) if isinstance(fams, dict) else {}
    s40_regime = (fam_block.get("S40", {}) or {}).get("regime_30d") or (fam_block.get("S40", {}) or {}).get("regime")
    fam_regime = (fam_block.get("FAMILY_164950", {}) or {}).get("regime_30d") or (fam_block.get("FAMILY_164950", {}) or {}).get("regime")

    score = slot_roi / 50.0
    if slump_flag:
        score -= 2.0
    if roi_bucket == "LOW":
        score -= 0.5
    if best_roi is not None and best_roi > 0:
        score += 0.5
    if str(s40_regime or "").upper() == "BOOST":
        score += 0.4
    if str(fam_regime or "").upper() == "BOOST":
        score += 0.2

    return {
        "slot": slot_key,
        "slot_roi": slot_roi,
        "slump": slump_flag,
        "roi_bucket": roi_bucket,
        "best_n": best_n,
        "best_roi": best_roi,
        "s40_regime": s40_regime,
        "fam_regime": fam_regime,
        "score": score,
    }


def _topn_score_for_slot(topn: Dict, slot: str) -> Dict[str, object]:
    per_slot_topn = topn.get("per_slot", topn.get("slots", {})) if isinstance(topn, dict) else {}
    best_n_per_slot = topn.get("best_n_per_slot", {}) if isinstance(topn, dict) else {}
    topn_slot = per_slot_topn.get(slot, {}) if isinstance(per_slot_topn, dict) else {}
    best_n = topn_slot.get("final_best_n") or topn_slot.get("best_N", best_n_per_slot.get(slot)) if isinstance(topn_slot, dict) else None
    roi = float(topn_slot.get("roi_final_best", topn_slot.get("roi", 0.0)) or 0.0) if isinstance(topn_slot, dict) else 0.0

    score = 0.0
    if roi > 0:
        score += min(roi / 200.0, 1.0)
    if best_n:
        score += 0.3 if 2 <= int(best_n) <= 6 else 0.1

    return {"score": score, "roi": roi, "best_n": best_n, "roi_final_best": roi, "final_best_n": best_n}


def _select_slot(slot_health: Dict[str, Dict[str, object]], topn: Dict, patterns: Optional[Dict]) -> Optional[str]:
    quant_candidates = [_quant_slot_score(slot) for slot in SLOTS]
    quant_candidates = [c for c in quant_candidates if c]
    if quant_candidates:
        quant_candidates.sort(
            key=lambda c: (
                -float(c.get("score", 0.0) or 0.0),
                -float(c.get("slot_roi", 0.0) or 0.0),
            ),
        )
        top_score = quant_candidates[0].get("score", 0.0)
        tied = [c for c in quant_candidates if c.get("score", 0.0) == top_score]
        tie_order = {"GZBD": 0, "GALI": 1, "DSWR": 2, "FRBD": 3}
        winner = sorted(tied, key=lambda c: tie_order.get(c.get("slot", ""), 99))[0]
        return winner.get("slot")

    if not slot_health:
        return None

    candidates = []
    for slot, stats in slot_health.items():
        level = str(stats.get("slot_level", "MID")).upper()
        if level == "OFF":
            continue
        roi = float(stats.get("roi", 0.0) or 0.0)
        slump = bool(stats.get("slump", False))
        pattern_info = _pattern_score_for_slot(patterns, slot)
        topn_info = _topn_score_for_slot(topn, slot)
        base_score = roi / 50.0
        if slump:
            base_score -= 2.0
        level_bias = {"HIGH": 0.6, "MID": 0.3, "LOW": -0.2}.get(level, 0.0)
        roi_bias = 0.4 if (topn_info.get("roi_final_best") or 0.0) > 0 else 0.0
        slot_score = base_score + 0.8 * pattern_info["score"] + 0.5 * topn_info["score"] + level_bias + roi_bias
        candidates.append(
            {
                "slot": slot,
                "roi": roi,
                "slump": slump,
                "slot_level": level,
                "slot_score": slot_score,
                "pattern": pattern_info,
                "topn": topn_info,
            }
        )

    viable = [c for c in candidates if not c.get("slump") and c.get("slot_level") in {"HIGH", "MID"}]
    ranked_pool = viable if viable else candidates
    if not ranked_pool:
        return None

    ranked_pool.sort(
        key=lambda c: (
            -float(c.get("slot_score", 0.0) or 0.0),
            -{"HIGH": 2, "MID": 1, "LOW": 0}.get(str(c.get("slot_level", "MID")), 0),
            -float(c.get("roi", 0.0) or 0.0),
            -_regime_score((c.get("pattern") or {}).get("s40")),
            -_regime_score((c.get("pattern") or {}).get("fam")),
            -float((c.get("topn") or {}).get("roi", 0.0) or 0.0),
        )
    )
    return ranked_pool[0].get("slot")


def _pick_number_from_plan(slot: str, base_dir: Path) -> Optional[Dict[str, str]]:
    debug_path = base_dir / "data" / "bet_engine_debug.json"
    if debug_path.exists():
        try:
            debug_data = json.loads(debug_path.read_text())
            slot_block = (debug_data.get("slots", {}) or {}).get(slot, {}) if isinstance(debug_data, dict) else {}
            numbers = slot_block.get("numbers", []) if isinstance(slot_block, dict) else []
            if numbers:
                numbers_sorted = sorted(numbers, key=lambda x: (-float(x.get("final_score", 0.0)), x.get("tier", "Z")))
                chosen = numbers_sorted[0]
                num = str(chosen.get("number", "")).zfill(2)
                return {"number": num, "andar": num[0], "bahar": num[1], "tier": chosen.get("tier"), "final_score": chosen.get("final_score")}
        except Exception:
            pass

    latest_plan = quant_paths.find_latest_bet_plan_master()
    if not latest_plan or not Path(latest_plan).exists():
        return None

    try:
        sheets = pd.read_excel(latest_plan, sheet_name=None)
    except Exception as exc:
        print(f"[quant_arjun_mode] Warning: unable to read bet plan {latest_plan}: {exc}")
        return None

    def _extract(df: pd.DataFrame) -> Optional[Dict[str, str]]:
        if df is None or df.empty:
            return None
        col_map = {str(c).strip().lower(): c for c in df.columns}
        slot_col = col_map.get("slot")
        if not slot_col:
            return None
        slot_mask = df[slot_col].astype(str).str.upper() == slot
        slot_df = df[slot_mask]
        if slot_df.empty:
            return None

        tier_col = col_map.get("tier")
        if tier_col and not slot_df[tier_col].empty:
            tier_mask = slot_df[tier_col].astype(str).str.upper().str.startswith("A")
            tier_df = slot_df[tier_mask]
            if not tier_df.empty:
                slot_df = tier_df

        stake_col = None
        for cand in ["stake", "total_stake"]:
            if cand in col_map:
                stake_col = col_map[cand]
                break
        if stake_col is None:
            stake_candidates = [c for key, c in col_map.items() if "stake" in key]
            stake_col = stake_candidates[0] if stake_candidates else None

        number_col = None
        for cand in ["number_or_digit", "number", "num", "prediction"]:
            if cand in col_map:
                number_col = col_map[cand]
                break
        if number_col is None:
            number_candidates = [c for key, c in col_map.items() if "number" in key]
            number_col = number_candidates[0] if number_candidates else None

        if number_col is None:
            return None

        if stake_col is not None:
            stakes = pd.to_numeric(slot_df[stake_col], errors="coerce").fillna(0)
            slot_df = slot_df.assign(_stake_sort=stakes)
            slot_df = slot_df.sort_values("_stake_sort", ascending=False)
        else:
            slot_df = slot_df.copy()
            slot_df["_stake_sort"] = 0

        top_row = slot_df.iloc[0]
        raw_number = str(top_row.get(number_col)).strip()
        digits = "".join([ch for ch in raw_number if ch.isdigit()])
        if len(digits) >= 2:
            number = digits[-2:]
        elif digits:
            number = digits.zfill(2)
        else:
            return None

        return {
            "number": number,
            "andar": number[0],
            "bahar": number[1],
        }

    # Prefer "bets" sheet if present
    if isinstance(sheets, dict):
        if "bets" in sheets:
            pick = _extract(sheets["bets"])
            if pick:
                return pick
        for _, df in sheets.items():
            pick = _extract(df)
            if pick:
                return pick
    return None


def _build_reason(
    slot: str, slot_health: Dict[str, object], patterns: Optional[Dict], topn: Dict, hero: Optional[str]
) -> str:
    notes = []
    if slot_health:
        level = str(slot_health.get("slot_level", "MID")).upper()
        notes.append(f"level={level}")
        if not slot_health.get("slump"):
            notes.append("non-slump")
        roi_val = slot_health.get("roi")
        if roi_val is not None:
            notes.append("strong ROI" if roi_val > 0 else "weak ROI")
    s40 = _slot_pattern_regime(patterns, slot, "S40")
    fam = _slot_pattern_regime(patterns, slot, "FAMILY_164950")
    extra = _slot_extra_family(patterns, slot)
    pattern_bits = []
    if s40:
        pattern_bits.append(f"S40{'+' if str(s40).upper() == 'BOOST' else '-' if str(s40).upper() == 'OFF' else '='}")
    if fam:
        pattern_bits.append(
            f"164950{'+' if str(fam).upper() == 'BOOST' else '-' if str(fam).upper() == 'OFF' else '='}"
        )
    if extra and extra.get("family"):
        regime = str(extra.get("regime") or "").upper()
        tag = f"{extra['family']}" + (
            "+" if regime == "BOOST" else "-" if regime == "OFF" else "="
        )
        pattern_bits.append(tag)
    if pattern_bits:
        notes.append("patterns=" + ",".join(pattern_bits))

    topn_slot = (topn.get("per_slot", {}) or {}).get(slot, {}) if isinstance(topn, dict) else {}
    topn_best = (topn.get("best_n_per_slot", {}) or {}).get(slot) if isinstance(topn, dict) else None
    if isinstance(topn_slot, dict) and (topn_slot.get("roi") is not None or topn_slot.get("roi_final_best") is not None):
        roi_val = float(topn_slot.get("roi_final_best", topn_slot.get("roi", 0.0)) or 0.0)
        best_n = topn_slot.get("final_best_n") or topn_slot.get("best_N", topn_best)
        if best_n:
            notes.append(f"TopN final_best_n={best_n} ROI {roi_val:+.1f}%")
        else:
            notes.append(f"TopN ROI {roi_val:+.1f}%")
    if hero:
        notes.append(f"hero script={hero}")
    return ", ".join(notes) if notes else "signals unavailable"


def main() -> int:
    base_dir = quant_paths.get_base_dir()
    slot_health = _load_slot_health(base_dir)
    patterns = _load_pattern_regimes(base_dir)
    topn = _load_topn_summary(base_dir)
    hero_weak = _load_hero_weak(base_dir)

    if not slot_health:
        fallback: Dict[str, Dict[str, object]] = {}
        per_slot_topn = topn.get("per_slot", {}) if isinstance(topn, dict) else {}
        for slot in SLOTS:
            topn_entry = per_slot_topn.get(slot, {}) if isinstance(per_slot_topn, dict) else {}
            fallback[slot] = {
                "roi": float(topn_entry.get("roi", 0.0) or 0.0),
                "hit_rate": 0.0,
                "slump": False,
            }
        slot_health = fallback

    chosen_slot = _select_slot(slot_health, topn, patterns)
    if not chosen_slot:
        print("No Arjun pick available (slot signals missing).")
        return 0

    pick = _pick_number_from_plan(chosen_slot, base_dir)
    if not pick:
        print("No Arjun pick available (bet plan missing).")
        return 0

    slot_quant = _quant_slot_score(chosen_slot)
    reasons = []
    slot_health_entry = slot_health.get(chosen_slot, {}) if isinstance(slot_health, dict) else {}
    level_label = str(slot_health_entry.get("slot_level", "")).upper()
    if level_label:
        reasons.append(f"slot level {level_label}")
    roi_val = slot_quant.get("slot_roi") if slot_quant else None
    if roi_val is not None:
        reasons.append(f"slot ROI {float(roi_val):+.1f}%")
    if slot_quant and slot_quant.get("slump"):
        reasons.append("slot in slump")
    topn_slot_info = (topn.get("per_slot", {}) or {}).get(chosen_slot, {}) if isinstance(topn, dict) else {}
    best_n = topn_slot_info.get("final_best_n") or slot_quant.get("best_n") if slot_quant else topn_slot_info.get("final_best_n")
    best_roi = topn_slot_info.get("roi_final_best") if isinstance(topn_slot_info, dict) else None
    if best_roi is None:
        best_roi = slot_quant.get("best_roi") if slot_quant else None
    if best_n:
        reasons.append(f"best_N={best_n} ROI {float(best_roi or 0.0):+.1f}%")
    s40_regime = slot_quant.get("s40_regime") if slot_quant else None
    fam_regime = slot_quant.get("fam_regime") if slot_quant else None
    if s40_regime:
        reasons.append(f"S40 {s40_regime}")
    if fam_regime:
        reasons.append(f"164950 {fam_regime}")
    reasons.append("chosen from bet plan shortlist")

    hero_map = hero_weak.get("per_slot", {}) if isinstance(hero_weak, dict) else {}
    hero_script = None
    slot_entry = hero_map.get(chosen_slot, {}) if isinstance(hero_map, dict) else {}
    if slot_entry:
        hero_script = slot_entry.get("hero_script")
        if hero_script:
            reasons.append(f"hero script={hero_script}")

    result = {
        "date": date.today().isoformat(),
        "slot": chosen_slot,
        "number": pick["number"],
        "andar": pick["andar"],
        "bahar": pick["bahar"],
        "reasons": reasons,
        "sources": {
            "slot_health": slot_health.get(chosen_slot, {}),
            "topn": (topn.get("per_slot", {}) or {}).get(chosen_slot, {}) if isinstance(topn, dict) else {},
            "patterns": {
                "S40": _slot_pattern_regime(patterns, chosen_slot, "S40"),
                "FAMILY_164950": _slot_pattern_regime(patterns, chosen_slot, "FAMILY_164950"),
                "extra_family": _slot_extra_family(patterns, chosen_slot),
            },
            "hero_script": hero_script,
        },
    }

    output_path = Path(base_dir) / "data" / "arjun_pick.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(json.dumps(result, indent=2))
    except Exception as exc:
        print(f"[quant_arjun_mode] Warning: unable to write arjun_pick.json: {exc}")

    reason = _build_reason(
        chosen_slot,
        slot_health.get(chosen_slot, {}),
        patterns,
        topn,
        hero_script,
    )
    if reasons:
        reason = reason + "; " + "; ".join(reasons) if reason else "; ".join(reasons)

    print("=== ARJUN MODE – FOCUSED SHOT ===")
    print(f"Slot      : {chosen_slot}")
    print(f"Number    : {pick['number']}  (ANDAR={pick['andar']}, BAHAR={pick['bahar']})")
    print(f"Reason    : {reason}")
    print(f"JSON saved: {output_path.relative_to(base_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
