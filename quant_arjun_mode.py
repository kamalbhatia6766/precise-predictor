from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import quant_paths
from quant_slot_health import load_slot_health, SlotHealth

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


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
    snapshot_path = base_dir / "data" / "slot_health.json"
    if snapshot_path.exists():
        data = _load_json(snapshot_path) or {}
        slot_health: Dict[str, Dict[str, object]] = {}
        for slot in SLOTS:
            record = data.get(slot, {}) if isinstance(data, dict) else {}
            slot_health[slot] = {
                "roi": float(record.get("roi_30", record.get("roi_percent", 0.0)) or 0.0),
                "hit_rate": float(record.get("hit_rate", 0.0) or 0.0),
                "slump": bool(record.get("slump", False)),
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
                }
    return slot_map


def _load_topn_summary(base_dir: Path) -> Dict:
    path = base_dir / "logs" / "performance" / "topn_roi_summary.json"
    return _load_json(path) or {}


def _load_pattern_regimes(base_dir: Path) -> Dict:
    path = base_dir / "logs" / "performance" / "pattern_regimes_summary.json"
    return _load_json(path) or {}


def _load_hero_weak(base_dir: Path) -> Dict:
    path = base_dir / "logs" / "performance" / "script_hero_weak.json"
    return _load_json(path) or {}


def _slot_pattern_regime(patterns: Dict, slot: str, family: str) -> Optional[str]:
    families = patterns.get("families", {}) if isinstance(patterns, dict) else {}
    fam_block = families.get(family, {}) if isinstance(families, dict) else {}
    per_slot = fam_block.get("per_slot", {}) if isinstance(fam_block, dict) else {}
    slot_entry = per_slot.get(slot, {}) if isinstance(per_slot, dict) else {}
    return slot_entry.get("regime")


def _select_slot(slot_health: Dict[str, Dict[str, object]], topn: Dict, patterns: Dict) -> Optional[str]:
    if not slot_health:
        return None

    per_slot_topn = topn.get("per_slot", {}) if isinstance(topn, dict) else {}

    candidates = []
    for slot, stats in slot_health.items():
        roi = float(stats.get("roi", 0.0) or 0.0)
        slump = bool(stats.get("slump", False))
        s40_regime = _slot_pattern_regime(patterns, slot, "S40") or "NORMAL"
        fam_regime = _slot_pattern_regime(patterns, slot, "FAMILY_164950") or "NORMAL"
        topn_slot = per_slot_topn.get(slot, {}) if isinstance(per_slot_topn, dict) else {}
        candidates.append(
            {
                "slot": slot,
                "roi": roi,
                "slump": slump,
                "s40_regime": s40_regime,
                "fam_regime": fam_regime,
                "topn_roi": float(topn_slot.get("roi", 0.0) or 0.0),
                "topn_best_n": topn_slot.get("best_N"),
            }
        )

    viable = [c for c in candidates if not c.get("slump")]
    ranked_pool = viable if viable else candidates
    if not ranked_pool:
        return None

    ranked_pool.sort(
        key=lambda c: (
            -float(c.get("roi", 0.0) or 0.0),
            -_regime_score(c.get("s40_regime")),
            -_regime_score(c.get("fam_regime")),
            -float(c.get("topn_roi", 0.0) or 0.0),
        )
    )
    return ranked_pool[0].get("slot")


def _pick_number_from_plan(slot: str, base_dir: Path) -> Optional[Dict[str, str]]:
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


def _build_reason(slot: str, slot_health: Dict[str, object], patterns: Dict, topn: Dict, hero: Optional[str]) -> str:
    notes = []
    if slot_health:
        if not slot_health.get("slump"):
            notes.append("non-slump")
        roi_val = slot_health.get("roi")
        if roi_val is not None:
            notes.append("strong ROI" if roi_val > 0 else "weak ROI")
    s40 = _slot_pattern_regime(patterns, slot, "S40")
    fam = _slot_pattern_regime(patterns, slot, "FAMILY_164950")
    if s40:
        notes.append(f"pattern {s40}")
    elif fam:
        notes.append(f"164950 {fam}")
    topn_slot = (topn.get("per_slot", {}) or {}).get(slot, {}) if isinstance(topn, dict) else {}
    if topn_slot.get("roi") is not None:
        notes.append(f"TopN ROI {float(topn_slot.get('roi', 0.0)):+.1f}%")
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

    hero_map = hero_weak.get("per_slot", {}) if isinstance(hero_weak, dict) else {}
    hero_script = None
    slot_entry = hero_map.get(chosen_slot, {}) if isinstance(hero_map, dict) else {}
    if slot_entry:
        hero_script = slot_entry.get("hero_script")

    result = {
        "date": date.today().isoformat(),
        "slot": chosen_slot,
        "number": pick["number"],
        "andar": pick["andar"],
        "bahar": pick["bahar"],
        "sources": {
            "slot_health": slot_health.get(chosen_slot, {}),
            "topn": (topn.get("per_slot", {}) or {}).get(chosen_slot, {}) if isinstance(topn, dict) else {},
            "patterns": {
                "S40": _slot_pattern_regime(patterns, chosen_slot, "S40"),
                "FAMILY_164950": _slot_pattern_regime(patterns, chosen_slot, "FAMILY_164950"),
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

    print("=== ARJUN MODE – FOCUSED SHOT ===")
    print(f"Slot      : {chosen_slot}")
    print(f"Number    : {pick['number']}  (ANDAR={pick['andar']}, BAHAR={pick['bahar']})")
    print(f"Reason    : {reason}")
    print(f"JSON saved: {output_path.relative_to(base_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
