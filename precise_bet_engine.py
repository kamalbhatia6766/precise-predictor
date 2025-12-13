# precise_bet_engine.py - ULTRA v5 ROCKET MODE - CLEAR DATES + BREAKDOWN
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import warnings
import argparse
import json
import quant_data_core
from quant_slot_health import get_slot_health, SlotHealth
from quant_stats_core import compute_topn_roi, get_quant_stats
from pattern_intelligence_engine import load_near_miss_boosts
from utils_2digit import is_valid_2d_number, to_2d_str
from pattern_helpers import (
    get_families_for_number,
    is_164950_number,
    is_s40_number,
)
warnings.filterwarnings('ignore')

quant_stats = get_quant_stats()

# Pre-compute family memberships for speed and consistency across the chain.
PATTERN_FAMILY_CACHE: Dict[str, List[str]] = {}

# Tunable horizons (kept as constants for easy future adjustments)
ROI_WINDOW_DAYS_DEFAULT = 30
NEAR_MISS_WINDOW_DAYS_DEFAULT = 30

# Policy files
TOPN_POLICY_PATH = Path("data") / "topn_policy.json"
SLOT_HEALTH_PATH = Path("data") / "slot_health.json"

# Defensive stake controls
BAHAR_EXTRA_MULTIPLIER = 0.5
SLOT_LEVEL_MULTIPLIERS = {
    "OFF": 0.0,
    "LOW": 0.5,
    "MID": 1.0,
    "HIGH": 1.25,
}

# Baseline main-stake total used for smooth distribution when expanding shortlist
BASE_MAIN_TOTAL_UNITS = 3.5  # Equivalent to the legacy A/B/C pattern (2 + 1 + 0.5)


def _precompute_family_cache() -> None:
    if PATTERN_FAMILY_CACHE:
        return
    for i in range(100):
        num_str = to_2d_str(i)
        tags = get_families_for_number(num_str)
        PATTERN_FAMILY_CACHE[num_str] = tags


_precompute_family_cache()

PATTERN_BOOST_S40 = 1.20
PATTERN_PENALTY_S40 = 0.85
PATTERN_BOOST_164950 = 1.15
PATTERN_PENALTY_164950 = 0.90
PATTERN_BOOST_EXTRA = 1.10
PATTERN_PENALTY_EXTRA = 0.95
PATTERN_MULT_MIN = 0.5
PATTERN_MULT_MAX = 1.8
PATTERN_REGIME_MULTIPLIERS = {"BOOST": 1.2, "NORMAL": 1.0, "OFF": 0.7}
PATTERN_REGIME_CLAMP = (0.8, 1.5)
DEFAULT_BEST_N = 3
MIN_SHORTLIST_K = 3
MAX_SHORTLIST_K = 12

DEFAULT_SLOT_LEVEL_POLICY = {
    "very_strong": {"roi_min": 400.0, "hit_rate_min": 0.20, "multiplier": 1.4, "label": "HIGH"},
    "strong": {"roi_min": 150.0, "hit_rate_min": 0.18, "multiplier": 1.1, "label": "MID"},
    "normal": {"roi_min": 0.0, "hit_rate_min": 0.0, "multiplier": 1.0, "label": "MID"},
    "slump": {"roi_max": -10.0, "multiplier": 0.25, "label": "SLUMP"},
    "off": {"roi_max": -30.0, "multiplier": 0.0, "label": "OFF"},
}

DEFAULT_LAYER_POLICY = {
    "MAIN": {
        "steps": [
            {"roi_min": 0.0, "multiplier": 1.0},
            {"roi_min": 50.0, "multiplier": 1.05},
        ]
    },
    "ANDAR": {
        "steps": [
            {"roi_min": 0.0, "multiplier": 1.0},
            {"roi_min": 25.0, "multiplier": 1.05},
            {"roi_min": 50.0, "multiplier": 1.1},
        ]
    },
    "BAHAR": {
        "steps": [
            {"roi_max": -10.0, "multiplier": 0.3},
            {"roi_min": 10.0, "multiplier": 0.5},
            {"roi_min": 50.0, "multiplier": 1.0},
        ],
        "fallback": 0.5,
    },
}

OVERLAY_POLICY = {
    # Filtered S36 overlay per slot
    "S36": {
        "enabled": True,
        # Minimum slot ROI (percent) required to even consider S36
        "min_slot_roi": 0.0,
        # If slot is in slump, force OFF regardless of ROI
        "allow_in_slump": False,
        # Stake units per slot (before slot_multiplier), in multiples of base_unit (₹10)
        "stake_units": 0.0,
    },
    # Core 4/4 packs
    "PackCore": {
        "enabled": True,
        "min_slot_roi": 150.0,   # needs reasonably positive ROI
        "allow_in_slump": False,
        "stake_units": 1.0,
    },
    # Booster 2/2 packs
    "PackBooster": {
        "enabled": True,
        "min_slot_roi": 250.0,   # only for very strong slots
        "allow_in_slump": False,
        "stake_units": 0.5,
    },
}


def apply_overlay_policy_to_bets(bets_df, slot_health_map, base_unit):
    """
    Apply ROI/slump-gated overlay stakes for S36 / PackCore / PackBooster.
    Returns a modified copy of bets_df.
    """
    if bets_df.empty:
        return bets_df

    df = bets_df.copy()

    for slot_name, health in slot_health_map.items():
        # Skip if we don't have health info
        if health is None:
            continue

        slot_mask = df['slot'] == slot_name

        for layer_type, policy in OVERLAY_POLICY.items():
            layer_mask = slot_mask & (df['layer_type'] == layer_type)
            if not layer_mask.any():
                continue

            # Policy controls
            if not policy.get("enabled", True):
                df.loc[layer_mask, ['stake', 'potential_return']] = 0.0
                continue

            roi_percent = getattr(health, "roi_percent", 0.0) or 0.0
            in_slump = bool(getattr(health, "slump", False))
            min_slot_roi = float(policy.get("min_slot_roi", 0.0))
            allow_in_slump = bool(policy.get("allow_in_slump", False))
            stake_units = float(policy.get("stake_units", 0.0))

            # Gating logic
            if roi_percent < min_slot_roi:
                # Slot ROI not strong enough
                df.loc[layer_mask, ['stake', 'potential_return']] = 0.0
                continue

            if in_slump and not allow_in_slump:
                # Slot in slump, overlay disabled
                df.loc[layer_mask, ['stake', 'potential_return']] = 0.0
                continue

            if stake_units <= 0:
                # Infra only; keep overlay effectively OFF
                df.loc[layer_mask, ['stake', 'potential_return']] = 0.0
                continue

            # Compute stake AFTER slot multiplier has already been baked into base layer stakes.
            # We just treat overlays as additional stake blocks per slot.
            overlay_stake = stake_units * float(base_unit)

            # Assign the same overlay stake to each row of that layer for that slot.
            df.loc[layer_mask, 'stake'] = overlay_stake

            # Potential return: treat overlay as a MAIN-style layer that pays at 90x on a hit.
            # (If another convention already exists in code, follow that same factor.)
            df.loc[layer_mask, 'potential_return'] = overlay_stake * 90.0

    return df


def fmt_rupees(value: float) -> str:
    """Cosmetic helper to keep rupee values tidy in logs."""
    try:
        amt = float(value)
    except (TypeError, ValueError):
        return "₹0"

    if abs(amt - round(amt)) < 0.01:
        return f"₹{int(round(amt))}"
    return f"₹{amt:.2f}"

try:
    import pattern_packs
    PATTERN_PACKS_AVAILABLE = True
except ImportError:
    print("⚠️  pattern_packs.py not found - pattern bonuses disabled")
    PATTERN_PACKS_AVAILABLE = False

# Compatibility alias to maintain older naming used in this file
is_164950_family = is_164950_number


def belongs_to_family(num: object, family_name: str) -> bool:
    if not family_name:
        return False
    family_upper = str(family_name).upper()
    try:
        tags = [str(t).upper() for t in get_families_for_number(num)]
        return family_upper in tags
    except Exception:
        return False


def _safe_slot_block(key: str) -> Dict:
    block = quant_stats.get(key, {}) if isinstance(quant_stats, dict) else {}
    return block if isinstance(block, dict) else {}


def _extract_script_numbers(entry: Dict) -> set:
    numbers = set()
    if not isinstance(entry, dict):
        return numbers
    for key in ["numbers", "top_numbers", "best_numbers"]:
        vals = entry.get(key)
        if isinstance(vals, list):
            numbers.update({to_2d_str(v) for v in vals})
    hits_block = entry.get("hits_by_number") if isinstance(entry, dict) else None
    if isinstance(hits_block, dict):
        numbers.update({to_2d_str(k) for k in hits_block.keys()})
    return numbers

class PreciseBetEngine:
    def __init__(self):
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.base_unit = 10
        self.risk_context = {
            "zone": "UNKNOWN",
            "risk_mode": "BASE",
            "multiplier": 1.0,
            "pre_risk_total": 0,
            "final_total": 0,
        }
        
        # ULTRA v5 constants
        self.N_DAYS = 30
        self.W_DIRECT = 0.15
        self.W_CROSS = 0.05
        self.W_S40_HIT = 0.05
        self.S40_BONUS = 0.20
        self.DIGIT_PACK_BONUS = 0.05
        self.MAX_PATTERN_BONUS = 0.5
        
        self.GOLDEN_DIGIT_BOOST = 0.08
        self.HERO_NUMBER_BOOST = 0.10
        self.TIME_AWARENESS_BOOST = 0.05
        self.MAX_QUANTUM_BOOST = 0.25
        
        self.NEAR_MISS_BOOST = 0.06
        self.MIRROR_BOOST = 0.04
        self.DIGITAL_ROOT_BOOST = 0.03
        
        self.EV_GAP = 0.03
        self.HARD_CAP = 12
        self.SOFT_CAP = 15
        self.MIN_BINS = 3
        self.MAX_PER_BIN = 6

        self.pattern_config = self.load_enhanced_pattern_intelligence()
        self.adaptive_packs = self.load_adaptive_pattern_packs()
        self.golden_insights = self.load_golden_insights()
        self.quant_stats = quant_stats or {}
        self.bet_engine_debug: Dict[str, Dict] = {}

        self.topn_roi_profile = {}
        self.topn_shortlist_profile = {}
        self.pattern_regime_summary = {}
        self.topn_policy = self._load_topn_policy()
        self.slot_health_snapshot = self._load_slot_health_snapshot()
        self.near_miss_boosts = load_near_miss_boosts()
        self.script_metrics = self._load_script_metrics()
        self.slot_level_policy = self._load_slot_level_policy()
        self.layer_risk_policy = self._load_layer_risk_policy()

    def _load_topn_roi_profile(self, window_days: int = ROI_WINDOW_DAYS_DEFAULT) -> Dict:
        try:
            return compute_topn_roi(window_days=window_days) or {}
        except Exception:
            return {}

    def _load_topn_policy(self) -> Dict[str, Dict[str, object]]:
        try:
            path = TOPN_POLICY_PATH
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            print(f"⚠️  Unable to load Top-N policy: {exc}")
        return {}

    def _load_slot_level_policy(self) -> Dict[str, Dict[str, object]]:
        config_path = Path("config") / "slot_level_policy.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                print(f"⚠️  Unable to read slot_level_policy.json: {exc}")
        return DEFAULT_SLOT_LEVEL_POLICY

    def _load_layer_risk_policy(self) -> Dict[str, Dict[str, object]]:
        config_path = Path("config") / "layer_risk_policy.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                print(f"⚠️  Unable to read layer_risk_policy.json: {exc}")
        return DEFAULT_LAYER_POLICY

    def _load_script_metrics(self) -> Dict[str, Dict[str, object]]:
        metrics_path = Path("logs") / "performance" / "script_hit_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                print(f"⚠️  Unable to read script_hit_metrics.json: {exc}")
        print("[WARN] Script hit metrics unavailable; using equal weights.")
        return {}

    def _load_slot_health_snapshot(self) -> Dict[str, Dict[str, object]]:
        try:
            path = SLOT_HEALTH_PATH
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {str(k).upper(): v for k, v in data.items() if isinstance(v, dict)}
        except Exception as exc:
            print(f"⚠️  Unable to load slot_health.json: {exc}")
        return {}

    def _get_slot_quant(self, key: str, slot: str) -> Dict:
        block = _safe_slot_block(key)
        slot_block = block.get(slot) if isinstance(block, dict) else None
        return slot_block if isinstance(slot_block, dict) else {}

    def _get_number_families(self, number: object) -> List[str]:
        num_str = to_2d_str(number)
        return PATTERN_FAMILY_CACHE.get(num_str, [])

    def _get_family_regime(self, slot: str, family: str) -> Dict:
        try:
            slot_block = (self.pattern_regime_summary or {}).get("slots", {}).get(slot, {})
            fam_block = slot_block.get("families", {}).get(family, {})
            return fam_block if isinstance(fam_block, dict) else {}
        except Exception:
            return {}

    def _family_drift_label(self, slot: str, family: str) -> str:
        fam_block = self._get_family_regime(slot, family)
        return fam_block.get("drift_label") or "NORMAL"

    def _slot_health_multiplier(self, slot: str) -> float:
        health_block = (self.quant_stats.get("slot_health") or {}).get(slot, {}) if isinstance(self.quant_stats, dict) else {}
        slump = bool(health_block.get("slump")) if isinstance(health_block, dict) else False
        roi_pct = float(health_block.get("roi_percent", 0.0) or 0.0) if isinstance(health_block, dict) else 0.0
        multiplier = 1.0
        if slump:
            multiplier *= 0.92
        elif roi_pct > 150:
            multiplier *= 1.05
        return multiplier

    def _load_topn_shortlist_profile(self) -> Dict:
        base_dir = Path(__file__).resolve().parent
        path = base_dir / "logs" / "performance" / "topn_roi_summary.json"
        profile = {"overall_best_N": DEFAULT_BEST_N, "best_n_per_slot": {}}
        if not path.exists():
            return profile
        try:
            with open(path, "r") as f:
                data = json.load(f)
            overall_best = data.get("overall_best_N") or data.get("overall", {}).get("best_N")
            best_per_slot = data.get("best_n_per_slot") or data.get("per_slot") or {}
            clean_per_slot = {}
            if isinstance(best_per_slot, dict):
                for k, v in best_per_slot.items():
                    try:
                        clean_per_slot[str(k).upper()] = int(v.get("best_N", v))
                    except Exception:
                        continue
            profile["overall_best_N"] = int(overall_best) if overall_best else DEFAULT_BEST_N
            profile["best_n_per_slot"] = clean_per_slot
        except Exception:
            return profile
        return profile

    def _load_pattern_regime_summary(self) -> Dict:
        base_dir = Path(__file__).resolve().parent
        candidates = [
            base_dir / "logs" / "performance" / "pattern_regimes_summary.json",
            base_dir / "logs" / "performance" / "pattern_regime_summary.json",
        ]
        regimes = {slot: {"S40": "NORMAL", "FAMILY_164950": "NORMAL"} for slot in self.slots}
        extra_best = {slot: {"family": None, "regime": "NORMAL", "score": (-float("inf"), -float("inf"), -float("inf"))} for slot in self.slots}

        def _score(regime_label: str, hits: int, cover: int):
            mapping = {"BOOST": 2, "NORMAL": 1, "OFF": 0}
            return (int(hits), int(cover), mapping.get(str(regime_label).upper(), 0))

        data = None
        for path in candidates:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    break
                except Exception:
                    data = None
        if not data:
            return {"per_slot": regimes, "extra_family": {slot: None for slot in self.slots}}

        families = data.get("families", {}) if isinstance(data, dict) else {}

        for fam_key in ["S40", "FAMILY_164950"]:
            fam_block = families.get(fam_key, {}) if isinstance(families, dict) else {}
            per_slot_block = fam_block.get("per_slot", {}) if isinstance(fam_block, dict) else {}
            for slot in self.slots:
                slot_stats = per_slot_block.get(slot, {}) if isinstance(per_slot_block, dict) else {}
                regime = slot_stats.get("regime", "NORMAL")
                regimes[slot][fam_key] = str(regime).upper()

        for fam_name, fam_block in families.items():
            if fam_name in ("S40", "FAMILY_164950"):
                continue
            per_slot_block = fam_block.get("per_slot", {}) if isinstance(fam_block, dict) else {}
            for slot in self.slots:
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

    def _compute_pattern_multiplier(self, slot: str, number: object) -> float:
        slot_key = str(slot).upper()
        summary = self.pattern_regime_summary.get("per_slot", {}) if isinstance(self.pattern_regime_summary, dict) else {}
        slot_regimes = summary.get(slot_key, {}) if isinstance(summary, dict) else {}
        mult = 1.0

        s40_regime = str(slot_regimes.get("S40", "NORMAL")).upper()
        if is_s40_number(number):
            if s40_regime == "BOOST":
                mult *= PATTERN_BOOST_S40
            elif s40_regime == "OFF":
                mult *= PATTERN_PENALTY_S40

        fam_regime = str(slot_regimes.get("FAMILY_164950", "NORMAL")).upper()
        if is_164950_family(number):
            if fam_regime == "BOOST":
                mult *= PATTERN_BOOST_164950
            elif fam_regime == "OFF":
                mult *= PATTERN_PENALTY_164950

        extra_map = self.pattern_regime_summary.get("extra_family", {}) if isinstance(self.pattern_regime_summary, dict) else {}
        extra_info = extra_map.get(slot_key, {}) if isinstance(extra_map, dict) else {}
        extra_family = extra_info.get("family")
        extra_regime = str(extra_info.get("regime", "NORMAL")).upper()
        if extra_family and belongs_to_family(number, extra_family):
            if extra_regime == "BOOST":
                mult *= PATTERN_BOOST_EXTRA
            elif extra_regime == "OFF":
                mult *= PATTERN_PENALTY_EXTRA

        mult = max(PATTERN_MULT_MIN, min(mult, PATTERN_MULT_MAX))
        return mult

    def get_pattern_multiplier(self, slot: str, num: object) -> float:
        """
        Combine S40 and FAMILY_164950 regime multipliers for a given slot/number.
        Returns a factor clamped to PATTERN_REGIME_CLAMP.
        """
        try:
            num_str = to_2d_str(num)
        except Exception:
            return 1.0

        if not is_valid_2d_number(num_str):
            return 1.0

        slot_key = str(slot).upper()
        summary = self.pattern_regime_summary.get("per_slot", {}) if isinstance(self.pattern_regime_summary, dict) else {}
        slot_regimes = summary.get(slot_key, {}) if isinstance(summary, dict) else {}

        s40_regime = str(slot_regimes.get("S40", "NORMAL")).upper()
        fam_regime = str(slot_regimes.get("FAMILY_164950", "NORMAL")).upper()

        m_s40 = PATTERN_REGIME_MULTIPLIERS.get(s40_regime, 1.0) if is_s40_number(num_str) else 1.0
        m_164950 = PATTERN_REGIME_MULTIPLIERS.get(fam_regime, 1.0) if is_164950_family(num_str) else 1.0

        combined = m_s40 * m_164950
        low, high = PATTERN_REGIME_CLAMP
        return max(low, min(combined, high))

    def _get_shortlist_width(self, slot: str, candidate_count: int) -> int:
        profile = self.topn_shortlist_profile or {}
        overall_best = profile.get("overall_best_N") or DEFAULT_BEST_N
        per_slot = profile.get("best_n_per_slot", {}) if isinstance(profile, dict) else {}
        slot_best = per_slot.get(str(slot).upper(), overall_best) or overall_best
        try:
            k_val = int(slot_best)
        except Exception:
            k_val = DEFAULT_BEST_N
        k_val = max(MIN_SHORTLIST_K, min(k_val, MAX_SHORTLIST_K))

        slot_cap_rupees = self.base_unit * self.SOFT_CAP
        min_unit = 0.5 * self.base_unit
        if min_unit > 0:
            max_numbers_cap = int(slot_cap_rupees // min_unit)
            if max_numbers_cap > 0:
                k_val = min(k_val, max_numbers_cap)

        if candidate_count:
            k_val = min(k_val, candidate_count)
        return max(1, k_val)

    def _slot_roi_dampener(self, slot: str, health: SlotHealth, topn_per_slot: Dict[str, object]) -> float:
        slot_key = str(slot).upper()

        roi_block = topn_per_slot.get(slot_key, {}) if isinstance(topn_per_slot, dict) else {}
        roi_map = roi_block.get("roi_by_N", {}) if isinstance(roi_block, dict) else {}
        roi1 = roi_map.get(1)
        roi5 = roi_map.get(5)
        candidates = [roi_map.get(n) for n in range(1, 6) if roi_map.get(n) is not None]

        if (
            getattr(health, "slump", False)
            and roi1 is not None
            and roi5 is not None
            and candidates
            and roi1 <= 0.0
            and roi5 <= 0.0
            and max(candidates) <= 0.0
        ):
            return 0.5
        return 1.0

    def _slot_pnl_snapshot(self, slot: str) -> Dict[str, object]:
        pnl_block = self.quant_stats.get("pnl") if isinstance(self.quant_stats, dict) else {}
        if not pnl_block:
            return {}
        slots_block = pnl_block.get("slots") if isinstance(pnl_block, dict) else {}
        slot_block = slots_block.get(slot) if isinstance(slots_block, dict) else {}
        return slot_block if isinstance(slot_block, dict) else {}

    def _compute_slot_multiplier(self, slot: str, health: SlotHealth) -> Tuple[float, str]:
        """Decide per-slot stake multiplier based on slot P&L and health."""
        policy = self.slot_level_policy or DEFAULT_SLOT_LEVEL_POLICY
        slot_key = str(slot).upper()
        pnl_snapshot = self._slot_pnl_snapshot(slot_key)
        roi_pct = float(
            pnl_snapshot.get("roi_30d")
            or pnl_snapshot.get("roi_percent")
            or getattr(health, "roi_percent", 0.0)
            or 0.0
        )
        hit_rate = float(pnl_snapshot.get("hit_rate_30d") or pnl_snapshot.get("hit_rate") or 0.0)

        # Off/slump guards first
        if getattr(health, "slump", False):
            slump_cfg = policy.get("slump", {}) if isinstance(policy, dict) else {}
            return float(slump_cfg.get("multiplier", 0.25)), str(slump_cfg.get("label", "SLUMP")).upper()
        off_cfg = policy.get("off", {}) if isinstance(policy, dict) else {}
        if off_cfg and roi_pct <= float(off_cfg.get("roi_max", -30)):
            return float(off_cfg.get("multiplier", 0.0)), str(off_cfg.get("label", "OFF")).upper()

        vs_cfg = policy.get("very_strong", {}) if isinstance(policy, dict) else {}
        if roi_pct >= float(vs_cfg.get("roi_min", 400)) and hit_rate >= float(vs_cfg.get("hit_rate_min", 0.2)):
            return float(vs_cfg.get("multiplier", 1.4)), str(vs_cfg.get("label", "HIGH")).upper()

        strong_cfg = policy.get("strong", {}) if isinstance(policy, dict) else {}
        if roi_pct >= float(strong_cfg.get("roi_min", 150)) and hit_rate >= float(strong_cfg.get("hit_rate_min", 0.18)):
            return float(strong_cfg.get("multiplier", 1.1)), str(strong_cfg.get("label", "MID")).upper()

        normal_cfg = policy.get("normal", {}) if isinstance(policy, dict) else {}
        if normal_cfg:
            return float(normal_cfg.get("multiplier", 1.0)), str(normal_cfg.get("label", "MID")).upper()

        return 1.0, "MID"

    def _compute_layer_multipliers(self) -> Dict[str, float]:
        pnl_block = self.quant_stats.get("pnl") if isinstance(self.quant_stats, dict) else {}
        layers_block = pnl_block.get("layers") if isinstance(pnl_block, dict) else pnl_block if isinstance(pnl_block, dict) else {}
        policy = self.layer_risk_policy or DEFAULT_LAYER_POLICY
        multipliers: Dict[str, float] = {}

        for layer in ["MAIN", "ANDAR", "BAHAR"]:
            stats = layers_block.get(layer) if isinstance(layers_block, dict) else {}
            roi_val = float(stats.get("roi_30d") or stats.get("roi") or 0.0) if isinstance(stats, dict) else 0.0
            steps = policy.get(layer, {}).get("steps") if isinstance(policy.get(layer, {}), dict) else policy.get(layer, {}).get("steps")
            if steps is None and isinstance(policy.get(layer, {}), dict):
                steps = policy.get(layer, {}).get("steps")
            if steps is None:
                steps = []
            best_mult = None
            for step in steps:
                roi_min = step.get("roi_min")
                roi_max = step.get("roi_max")
                meets_min = roi_min is None or roi_val >= float(roi_min)
                meets_max = roi_max is None or roi_val <= float(roi_max)
                if meets_min and meets_max:
                    best_mult = float(step.get("multiplier", best_mult or 1.0))
            if best_mult is None:
                best_mult = float(policy.get(layer, {}).get("fallback", 1.0)) if isinstance(policy.get(layer, {}), dict) else 1.0
            multipliers[layer] = best_mult
        return multipliers

    def load_dynamic_stake_plan(self, target_date):
        plan_path = Path(__file__).resolve().parent / "logs" / "performance" / "dynamic_stake_plan.json"
        if not plan_path.exists():
            return {}
        try:
            with open(plan_path, "r") as f:
                data = json.load(f)
        except Exception:
            return {}

        target_str = target_date.strftime("%Y-%m-%d") if target_date else None

        def extract_slots(plan_obj):
            slots = plan_obj.get("slot_stakes") or plan_obj.get("final_slot_stakes") or plan_obj.get("slot_allocations") or {}
            return {str(k).upper(): float(v) for k, v in slots.items() if v is not None}

        if isinstance(data, list):
            for plan_obj in data:
                if plan_obj.get("target_date") == target_str:
                    return extract_slots(plan_obj)
        if isinstance(data, dict):
            if data.get("target_date") and target_str and data.get("target_date") != target_str:
                return {}
            return extract_slots(data)
        return {}

    def load_loss_recovery_context(self):
        plan_path = Path(__file__).resolve().parent / "logs" / "performance" / "loss_recovery_plan.json"
        context = {
            "zone": "UNKNOWN",
            "risk_mode": "BASE",
            "multiplier": 1.0,
        }

        if not plan_path.exists():
            return context

        try:
            with open(plan_path, "r") as f:
                data = json.load(f)
        except Exception:
            return context

        zone_val = (data.get("zone") or data.get("current_zone") or "UNKNOWN").upper()
        risk_mode = (data.get("risk_mode") or data.get("current_mode") or "BASE").upper()
        multiplier = data.get("stake_multiplier") or data.get("multiplier")

        if multiplier is None:
            zone_map = {"GREEN": 1.0, "YELLOW": 0.8, "RED": 0.6}
            multiplier = zone_map.get(zone_val, 1.0)

        try:
            multiplier = float(multiplier)
        except Exception:
            multiplier = 1.0

        context.update({"zone": zone_val, "risk_mode": risk_mode, "multiplier": multiplier})
        return context

    def apply_stake_overlays(self, bets_df, summary_df, target_date):
        dynamic_targets = self.load_dynamic_stake_plan(target_date)
        risk_context = self.load_loss_recovery_context()

        self.risk_context.update(
            {
                "zone": risk_context.get("zone", "UNKNOWN"),
                "risk_mode": risk_context.get("risk_mode", "BASE"),
                "multiplier": float(risk_context.get("multiplier", 1.0)),
                "pre_risk_total": sum(float(v) for v in dynamic_targets.values()) if dynamic_targets else summary_df['total_stake'].sum(),
            }
        )

        loss_recovery_mult = self.risk_context["multiplier"]

        if not dynamic_targets and loss_recovery_mult == 1.0:
            self.risk_context["final_total"] = summary_df['total_stake'].sum()
            return bets_df, summary_df

        adjusted_bets = bets_df.copy()
        adjusted_summary = summary_df.copy()

        for slot in self.slots:
            slot_mask = adjusted_bets['slot'] == slot
            slot_summary_mask = adjusted_summary['slot'] == slot
            base_total = adjusted_summary.loc[slot_summary_mask, 'total_stake'].sum()
            factor = loss_recovery_mult
            if dynamic_targets and slot in dynamic_targets and base_total > 0:
                factor *= float(dynamic_targets[slot]) / base_total

            if factor == 1.0:
                continue

            stake_numeric = pd.to_numeric(adjusted_bets['stake'], errors='coerce')
            numeric_mask = slot_mask & stake_numeric.notna()
            adjusted_bets.loc[numeric_mask, 'stake'] = (stake_numeric.loc[numeric_mask] * factor).round(1)

            try:
                numeric_potential = pd.to_numeric(adjusted_bets.loc[numeric_mask, 'potential_return'], errors='coerce')
                updated_returns = adjusted_bets.loc[numeric_mask, 'stake'] * 90
                adjusted_bets.loc[numeric_mask, 'potential_return'] = updated_returns.round(1).where(~numeric_potential.isna(), adjusted_bets.loc[numeric_mask, 'potential_return'])
            except Exception:
                pass

            slot_rows = adjusted_bets[slot_mask]
            main_total = slot_rows[slot_rows['layer_type'] == 'Main']['stake'].sum()
            andar_total = slot_rows[slot_rows['layer_type'] == 'ANDAR']['stake'].sum()
            bahar_total = slot_rows[slot_rows['layer_type'] == 'BAHAR']['stake'].sum()
            max_total_return = pd.to_numeric(slot_rows['potential_return'], errors='coerce').sum()

            adjusted_summary.loc[slot_summary_mask, 'main_stake'] = main_total
            adjusted_summary.loc[slot_summary_mask, 'andar_stake'] = andar_total
            adjusted_summary.loc[slot_summary_mask, 'bahar_stake'] = bahar_total
            adjusted_summary.loc[slot_summary_mask, 'total_stake'] = main_total + andar_total + bahar_total
            adjusted_summary.loc[slot_summary_mask, 'max_total_return'] = max_total_return

        self.risk_context["final_total"] = adjusted_summary['total_stake'].sum()
        return adjusted_bets, adjusted_summary

    # ✅ ROCKET MODE: CRYSTAL CLEAR DATE MAPPING
    def print_rocket_summary(self, bets_df, summary_df, target_date, source_file, target_mode):
        """🚀 ULTRA CLEAR SUMMARY - NO CONFUSION"""
        system_today = datetime.now().date()
        
        print("\n" + "="*80)
        print("🎯 PRECISE BET ENGINE - ULTRA v5 ROCKET MODE")
        print("="*80)
        
        # ✅ CRYSTAL CLEAR DATE MAPPING
        print("📅 DATE MAPPING (CRYSTAL CLEAR):")
        print(f"   🖥️  System Date: {system_today}")
        print(f"   🎯 Target Date: {target_date}") 
        print(f"   📊 Mode: {target_mode.upper()}")
        print(f"   📁 Source: {source_file.name}")
        
        print(f"💰 Base Unit: {fmt_rupees(self.base_unit)}")
        
        print("\n📊 SLOT BREAKDOWN (MAIN + ANDAR/BAHAR):")
        print("-" * 80)
        
        grand_total = 0
        for slot in self.slots:
            slot_bets = bets_df[(bets_df['slot'] == slot) & (bets_df['layer_type'] == 'Main')]
            slot_andars = bets_df[(bets_df['slot'] == slot) & (bets_df['layer_type'] == 'ANDAR')]
            slot_bahars = bets_df[(bets_df['slot'] == slot) & (bets_df['layer_type'] == 'BAHAR')]
            
            if not slot_bets.empty:
                # Main numbers with tiers
                main_numbers = []
                main_total = 0
                for _, bet in slot_bets.iterrows():
                    number = bet['number_or_digit']
                    tier = bet['tier']
                    stake = bet['stake']
                    main_numbers.append(f"{number}({tier} {fmt_rupees(stake)})")
                    main_total += stake
                
                # ANDAR/BAHAR digits
                andar_digit = slot_andars['number_or_digit'].iloc[0] if not slot_andars.empty else "None"
                bahar_digit = slot_bahars['number_or_digit'].iloc[0] if not slot_bahars.empty else "None"
                andar_stake = slot_andars['stake'].iloc[0] if not slot_andars.empty else 0
                bahar_stake = slot_bahars['stake'].iloc[0] if not slot_bahars.empty else 0
                
                slot_total = main_total + andar_stake + bahar_stake
                grand_total += slot_total
                
                print(f"   {slot}:")
                print(f"     🔢 Main: {', '.join(main_numbers)}")
                print(
                    f"     📊 ANDAR: {andar_digit}({fmt_rupees(andar_stake)}), "
                    f"BAHAR: {bahar_digit}({fmt_rupees(bahar_stake)})"
                )
                print(f"     💰 Total: {fmt_rupees(slot_total)}")
                print()

        print(f"💵 GRAND TOTAL: {fmt_rupees(grand_total)}")
        print(f"🚀 ULTRA v5 QUANTUM SELF-LEARNING: ACTIVE")

        risk_zone = self.risk_context.get("zone", "UNKNOWN")
        risk_mode = self.risk_context.get("risk_mode", "BASE")
        risk_mult = self.risk_context.get("multiplier", 1.0)
        pre_risk = self.risk_context.get("pre_risk_total", grand_total)
        final_total = self.risk_context.get("final_total", grand_total)

        print("\nRISK LINK:")
        print(f"   Zone: {risk_zone}")
        print(f"   Risk Mode: {risk_mode}")
        print(f"   Loss-Recovery Multiplier: {risk_mult:.2f}x")
        print(f"   Dynamic Stake Total (pre-risk): {fmt_rupees(pre_risk)}")
        print(f"   Final Applied Stake Total: {fmt_rupees(final_total)}")

    # ✅ ALL WORKING METHODS FROM YOUR CURRENT VERSION
    def find_latest_predictions_file(self, source='scr9'):
        if source == 'fusion':
            predictions_dir = Path(__file__).resolve().parent / "predictions" / "fusion"
            if not predictions_dir.exists():
                raise FileNotFoundError(f"Fusion directory not found: {predictions_dir}")
            fusion_files = list(predictions_dir.glob("fusion_predictions_*.xlsx"))
            if not fusion_files:
                raise FileNotFoundError("No fusion_predictions_*.xlsx files found")
            return max(fusion_files, key=lambda x: x.stat().st_mtime)
        else:
            scr9_dir = Path(__file__).resolve().parent / "predictions" / "deepseek_scr9"
            if not scr9_dir.exists():
                raise FileNotFoundError(f"SCR9 directory not found: {scr9_dir}")
            ultimate_files = list(scr9_dir.glob("ultimate_predictions_*.xlsx"))
            if not ultimate_files:
                raise FileNotFoundError("No ultimate_predictions_*.xlsx files found")
            return max(ultimate_files, key=lambda x: x.stat().st_mtime)

    def load_ultimate_predictions(self, file_path):
        try:
            df = pd.read_excel(file_path)
            df.columns = [str(col).strip().lower() for col in df.columns]
            print(f"📁 Loaded: {file_path.name}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading predictions: {e}")

    def select_target_data(self, df, target_mode):
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        if 'type' not in df.columns:
            print("⚠️  No 'type' column found - using first row")
            return df.iloc[[0]]
        
        print(f"🎯 Target mode: {target_mode}")
        
        if target_mode == 'today':
            target_rows = df[df['type'].str.upper() == 'TODAY_EMPTY']
            if not target_rows.empty:
                print("   Using TODAY_EMPTY rows")
                return target_rows
            else:
                print("⚠️  No TODAY_EMPTY rows found - falling back to TOMORROW")
                target_mode = 'tomorrow'
        
        if target_mode == 'tomorrow':
            target_rows = df[df['type'].str.upper() == 'TOMORROW']
            if not target_rows.empty:
                print("   Using TOMORROW rows")
                return target_rows
            else:
                print("⚠️  No TOMORROW rows found - using first available row")
                return df.iloc[[0]]
        
        if target_mode == 'auto':
            today_rows = df[df['type'].str.upper() == 'TODAY_EMPTY']
            if not today_rows.empty:
                print("   Auto-selected TODAY_EMPTY rows")
                return today_rows
            else:
                tomorrow_rows = df[df['type'].str.upper() == 'TOMORROW']
                if not tomorrow_rows.empty:
                    print("   Auto-selected TOMORROW rows (TODAY_EMPTY not available)")
                    return tomorrow_rows
                else:
                    print("⚠️  No TODAY_EMPTY or TOMORROW rows found - using first available row")
                    return df.iloc[[0]]
        
        return df.iloc[[0]]

    def convert_wide_to_long_format(self, df, target_rows):
        slot_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if (any(slot.lower() in col_lower for slot in self.slots) and 
                '_opp' not in col_lower and
                col_lower not in ['date', 'type']):
                slot_columns.append(col)
        
        if not slot_columns:
            raise ValueError("No slot columns found in the data")
        
        print(f"🔍 Found slot columns: {slot_columns}")
        
        long_data = []
        for _, row in target_rows.iterrows():
            date_val = row.get('date', '')
            for slot_col in slot_columns:
                numbers_str = row[slot_col]
                if pd.notna(numbers_str):
                    numbers = self.parse_numbers(str(numbers_str))
                    if numbers:
                        slot_name = None
                        for slot in self.slots:
                            if slot.lower() in slot_col.lower():
                                slot_name = slot
                                break
                        if slot_name:
                            long_data.append({
                                'date': date_val,
                                'slot': slot_name,
                                'numbers': numbers
                            })
        
        long_df = pd.DataFrame(long_data)
        print(f"🔄 Converted to long format: {len(long_df)} rows")
        return long_df

    def parse_numbers(self, numbers_str):
        if pd.isna(numbers_str):
            return []
        numbers = []
        for num_str in str(numbers_str).split(','):
            num_str = num_str.strip()
            if num_str.isdigit():
                try:
                    numbers.append(int(num_str))
                except ValueError:
                    continue
        return numbers

    def get_target_date(self, long_df):
        today = datetime.now().date()
        dates = []
        for date_val in long_df['date'].unique():
            try:
                if isinstance(date_val, datetime):
                    parsed_date = date_val.date()
                elif isinstance(date_val, str):
                    parsed_date = pd.to_datetime(date_val).date()
                else:
                    continue
                dates.append(parsed_date)
            except:
                continue
        
        if not dates:
            raise ValueError("No valid dates found in predictions")
        
        future_dates = [d for d in dates if d >= today]
        if not future_dates:
            print("⚠️  No future dates found - using latest date")
            target_date = max(dates)
        else:
            target_date = min(future_dates)
        
        return target_date

    def load_script_hit_memory(self, target_date):
        memory_file = Path(__file__).resolve().parent / "logs" / "performance" / "script_hit_memory.xlsx"
        if not memory_file.exists():
            print("⚠️  No script_hit_memory.xlsx found - using pure SCR9 ranks")
            return None
        try:
            df = pd.read_excel(memory_file)
            df.columns = [str(c).strip() for c in df.columns]

            possible_hit_type_cols = ['hit_type', 'HIT_TYPE', 'HitType', 'hitType', 'HIT TYPE']
            normalized_map = {re.sub(r"[\s_]+", "", str(col)).lower(): col for col in df.columns}
            resolved_col = None
            for col in possible_hit_type_cols:
                normalized = re.sub(r"[\s_]+", "", col).lower()
                if normalized in normalized_map:
                    resolved_col = normalized_map[normalized]
                    break

            if resolved_col is None:
                print("⚠️  Warning: No hit_type/HIT_TYPE column found in script_hit_memory; skipping hit-type-based weighting.")
                df['hit_type'] = 'UNKNOWN'
                resolved_col = 'hit_type'
            elif resolved_col != 'hit_type':
                df['hit_type'] = df[resolved_col]

            print(f"✅ Using hit type column: {resolved_col} → exposed as 'hit_type'")
            df['date'] = pd.to_datetime(df['date']).dt.date
            cutoff_date = target_date - timedelta(days=self.N_DAYS)
            filtered_df = df[df['date'] >= cutoff_date]
            print(f"📊 Loaded script hit memory: {len(filtered_df)} records")
            return filtered_df
        except Exception as e:
            print(f"⚠️  Error loading script_hit_memory: {e}")
            return None

    def build_history_table(self, memory_df):
        history = defaultdict(lambda: {'direct_hits': 0, 'cross_hits': 0, 's40_hits': 0, 'digit_tags': set()})
        if memory_df is None:
            return history
        for _, row in memory_df.iterrows():
            slot = row['real_slot']
            number = row['real_number']
            hit_type = row['hit_type']
            is_s40 = row.get('is_s40', False)
            digit_tags = str(row.get('digit_pack_tags', '')).split(',')
            key = (slot, number)
            if hit_type == 'DIRECT':
                history[key]['direct_hits'] += 1
            elif hit_type == 'CROSS_SLOT':
                history[key]['cross_hits'] += 1
            if is_s40:
                history[key]['s40_hits'] += 1
            for tag in digit_tags:
                if tag.strip():
                    history[key]['digit_tags'].add(tag.strip())
        return history

    def assign_tiers(self, shortlist):
        if not shortlist:
            return {}
        tiers = {}
        n = len(shortlist)
        tier_sizes = {'A': n // 3, 'B': n // 3, 'C': n - 2 * (n // 3)}
        idx = 0
        for tier, size in tier_sizes.items():
            for i in range(size):
                if idx < len(shortlist):
                    tiers[shortlist[idx]['number']] = tier
                    idx += 1
        return tiers

    def get_andar_bahar(self, shortlist):
        if not shortlist:
            return None, None
        tens_digits = [int(to_2d_str(num['number'])[0]) for num in shortlist]
        ones_digits = [int(to_2d_str(num['number'])[1]) for num in shortlist]
        tens_counter = Counter(tens_digits)
        ones_counter = Counter(ones_digits)
        def break_tie(counter, digit_list):
            max_count = max(counter.values())
            candidates = [d for d, count in counter.items() if count == max_count]
            if len(candidates) == 1:
                return candidates[0]
            positions = {}
            for candidate in candidates:
                for pos, item in enumerate(shortlist):
                    num_str = to_2d_str(item['number'])
                    if (digit_list is tens_digits and int(num_str[0]) == candidate) or (digit_list is ones_digits and int(num_str[1]) == candidate):
                        positions[candidate] = pos
                        break
            return min(positions.keys(), key=lambda x: positions[x])
        andar_digit = break_tie(tens_counter, tens_digits)
        bahar_digit = break_tie(ones_counter, ones_digits)
        return andar_digit, bahar_digit

    def load_enhanced_pattern_intelligence(self):
        config = {
            "s40_enabled": True,
            "digit_packs_enabled": True,
            "memory_bonus_enabled": True,
            "pattern_weights": {
                "s40_bonus": self.S40_BONUS,
                "digit_pack_bonus": self.DIGIT_PACK_BONUS,
                "max_pattern_bonus": self.MAX_PATTERN_BONUS,
            },
        }
        try:
            pattern_file = Path(__file__).resolve().parent / "logs" / "performance" / "pattern_intelligence_config.json"
            if pattern_file.exists():
                with open(pattern_file, 'r') as f:
                    external_config = json.load(f)
                config.update(external_config)
                print("✅ Loaded enhanced pattern intelligence configuration")
        except Exception as e:
            print(f"⚠️  Error loading enhanced pattern config: {e}")
        return config

    def load_adaptive_pattern_packs(self):
        try:
            adaptive_file = Path(__file__).resolve().parent / "logs" / "performance" / "adaptive_pattern_packs.json"
            if adaptive_file.exists():
                with open(adaptive_file, 'r') as f:
                    adaptive_data = json.load(f)
                print("✅ Loaded adaptive pattern packs for quantum boost")
                return adaptive_data
        except Exception as e:
            print(f"⚠️  Error loading adaptive pattern packs: {e}")
        return {}

    def load_golden_insights(self):
        try:
            golden_file = Path(__file__).resolve().parent / "logs" / "performance" / "golden_block_insights.json"
            if golden_file.exists():
                with open(golden_file, 'r') as f:
                    insights = json.load(f)
                print("✅ Loaded golden block insights")
                return insights
        except Exception as e:
            print(f"⚠️  Error loading golden insights: {e}")
        return {}

    def _script_hero_boost(self, slot: str, number_str: str) -> float:
        scripts_block = _safe_slot_block("scripts")
        hero_block = scripts_block.get("hero", {}) if isinstance(scripts_block, dict) else {}
        weak_block = scripts_block.get("weak", {}) if isinstance(scripts_block, dict) else {}
        hero_entry = hero_block.get(slot, {}) if isinstance(hero_block, dict) else {}
        weak_entry = weak_block.get(slot, {}) if isinstance(weak_block, dict) else {}
        hero_numbers = _extract_script_numbers(hero_entry)
        weak_numbers = _extract_script_numbers(weak_entry)
        boost = 0.0
        if number_str in hero_numbers:
            boost += 0.12
        if number_str in weak_numbers:
            boost -= 0.08
        return boost

    def _topn_alignment_boost(self, slot: str, rank: int) -> float:
        topn_root = _safe_slot_block("topn")
        slots_block = topn_root.get("slots", {}) if isinstance(topn_root, dict) else {}
        slot_topn = slots_block.get(slot, {}) if isinstance(slots_block, dict) else {}
        roi_map = slot_topn.get("roi_by_N", {}) if isinstance(slot_topn, dict) else {}
        best_n = slot_topn.get("best_N") if isinstance(slot_topn, dict) else None
        best_roi = slot_topn.get("best_roi") if isinstance(slot_topn, dict) else None
        try:
            roi_map = {int(k): float(v) for k, v in roi_map.items()}
        except Exception:
            roi_map = {}
        if best_n is None and roi_map:
            try:
                best_roi = max(roi_map.values()) if roi_map else None
                best_candidates = [n for n, rv in roi_map.items() if rv == best_roi]
                best_n = min(best_candidates) if best_candidates else None
            except Exception:
                best_n = None
        align = 0.0
        if best_n and best_roi is not None and best_roi > 0:
            if rank <= int(best_n):
                align += 0.25 * (int(best_n) - rank + 1) / max(1, int(best_n))
        if roi_map:
            top_band = [roi_map.get(n) for n in range(1, 6) if roi_map.get(n) is not None]
            deep_positive = [(n, v) for n, v in roi_map.items() if n > 5 and v is not None and v > 0]
            if top_band and all((r is not None and r <= 0) for r in top_band) and deep_positive:
                if rank <= 5:
                    align -= 0.08
                elif best_n and rank <= int(best_n):
                    align += 0.18
                else:
                    align += 0.05
        return align

    def _pattern_regime_bonus(self, slot: str, families: List[str]) -> (float, Dict[str, float], Dict[str, str]):
        patterns_root = _safe_slot_block("patterns")
        slot_block = patterns_root.get("slots", {}) if isinstance(patterns_root, dict) else {}
        slot_patterns = slot_block.get(slot, {}) if isinstance(slot_block, dict) else {}
        fam_block = slot_patterns.get("families", {}) if isinstance(slot_patterns, dict) else {}
        total = 0.0
        boosts: Dict[str, float] = {}
        regimes: Dict[str, str] = {}
        for fam in families:
            fam_key = str(fam).upper()
            fam_info = fam_block.get(fam_key, {}) if isinstance(fam_block, dict) else {}
            regime30 = str(fam_info.get("regime_30d", fam_info.get("regime" ,"NORMAL"))).upper()
            regime90 = str(fam_info.get("regime_90d", "" )).upper()
            bonus = 0.0
            if regime30 == "BOOST" or regime90 == "BOOST":
                bonus = 0.12
            elif regime30 == "OFF":
                bonus = -0.10
            if bonus != 0:
                boosts[fam_key] = bonus
                total += bonus
            if regime30:
                regimes[fam_key] = regime30
        return total, boosts, regimes

    def calculate_quant_scores(self, numbers_list, slot):
        if not numbers_list:
            return [], {}
        scored_numbers = []
        debug_reasons = {}
        slot_key = str(slot).upper()
        for rank, number in enumerate(numbers_list, 1):
            num_str = to_2d_str(number)
            base_score = 1.0 / rank
            hero_boost = self._script_hero_boost(slot_key, num_str)
            topn_boost = self._topn_alignment_boost(slot_key, rank)
            families = self._get_number_families(number)
            pattern_bonus, pattern_boosts, regimes = self._pattern_regime_bonus(slot_key, families)
            final_score = base_score + hero_boost + topn_boost + pattern_bonus
            scored_numbers.append({
                'number': number,
                'rank': rank,
                'base_score': base_score,
                'pattern_bonus': pattern_bonus,
                'memory_bonus': 0.0,
                'final_score': final_score,
                'is_s40': 'S40' in families,
                'digit_pack_tags': ','.join(families),
                'direct_hits_30d': 0,
                'cross_hits_30d': 0,
                's40_hits_30d': 0,
                'quantum_boosted_score': final_score,
                'quantum_boost_components': {},
                'pattern_multiplier': 1.0,
            })
            debug_reasons[num_str] = {
                "base_rank": rank,
                "base_score": base_score,
                "script_hero_boost": hero_boost,
                "topn_align": topn_boost,
                "pattern_boosts": pattern_boosts,
                "final_score": final_score,
                "pattern_tags": families,
                "regimes": regimes,
            }
        scored_numbers = sorted(scored_numbers, key=lambda x: (-x['final_score'], x['rank']))
        return scored_numbers, debug_reasons

    def calculate_enhanced_scores(self, numbers_list, slot, history, target_date):
        if not numbers_list:
            return []
        scored_numbers = []
        for rank, number in enumerate(numbers_list, 1):
            base_score = 1.0 / rank
            digit_tags = []
            history_key = (slot, number)
            
            quantum_boosted_score, quantum_debug = self.apply_pattern_boost(slot, number, base_score, target_date)
            pattern_bonus = 0.0
            
            if PATTERN_PACKS_AVAILABLE and self.pattern_config.get('s40_enabled', True):
                if pattern_packs.is_s40(number):
                    s40_bonus = self.pattern_config.get('pattern_weights', {}).get('s40_bonus', self.S40_BONUS)
                    pattern_bonus += s40_bonus
                    
                if self.pattern_config.get('digit_packs_enabled', True):
                    digit_tags = pattern_packs.get_digit_pack_tags(number)
                    # ✅ UPDATED: Family-level bonuses instead of per-pack
                    families = self._get_family_categories(digit_tags)
                    for fam in families:
                        fam_bonus = self.DIGIT_PACK_BONUS
                        pattern_bonus += fam_bonus
            
            memory_bonus = 0.0
            if self.pattern_config.get('memory_bonus_enabled', True) and history_key in history:
                h = history[history_key]
                memory_bonus = (
                    self.W_DIRECT * h['direct_hits'] +
                    self.W_CROSS * h['cross_hits'] +
                    self.W_S40_HIT * h['s40_hits']
                )
                memory_bonus = memory_bonus / max(1, self.N_DAYS / 10)
            
            for other_slot in self.slots:
                if other_slot != slot:
                    cross_boost = self.get_cross_slot_boost(other_slot, slot, number)
                    pattern_bonus += cross_boost
            
            time_boost = self.get_time_awareness_boost(slot, target_date)
            if time_boost > 1.0:
                pattern_bonus += (time_boost - 1.0) * 0.5
            
            digit_boost = self.get_digit_preference_boost(slot, number)
            pattern_bonus += digit_boost
            
            max_pattern = self.pattern_config.get('pattern_weights', {}).get('max_pattern_bonus', self.MAX_PATTERN_BONUS)
            pattern_bonus = min(pattern_bonus, max_pattern)
            
            pattern_multiplier = self._compute_pattern_multiplier(slot, number)
            pattern_tags = self._get_number_families(number)

            drift_bonus = 0.0
            drift_penalty = 0.0
            for fam in pattern_tags:
                drift_label = self._family_drift_label(slot, fam)
                if drift_label == "BOOST_DRIFT":
                    drift_bonus += 0.05
                elif drift_label == "COOL_OFF":
                    drift_penalty += 0.05

            pattern_score = pattern_multiplier + drift_bonus - drift_penalty
            slot_health_mult = self._slot_health_multiplier(slot)
            final_score = (quantum_boosted_score + pattern_bonus + memory_bonus) * max(pattern_score, 0.5) * slot_health_mult
            
            direct_hits = history[history_key]['direct_hits'] if history_key in history else 0
            cross_hits = history[history_key]['cross_hits'] if history_key in history else 0
            s40_hits = history[history_key]['s40_hits'] if history_key in history else 0
            
            scored_numbers.append({
                'number': number,
                'rank': rank,
                'base_score': base_score,
                'pattern_bonus': pattern_bonus,
                'memory_bonus': memory_bonus,
                'final_score': final_score,
                'is_s40': PATTERN_PACKS_AVAILABLE and pattern_packs.is_s40(number),
                'digit_pack_tags': ','.join(digit_tags),
                'direct_hits_30d': direct_hits,
                'cross_hits_30d': cross_hits,
                's40_hits_30d': s40_hits,
                'quantum_boosted_score': quantum_boosted_score,
                'quantum_boost_components': quantum_debug,
                'pattern_multiplier': pattern_multiplier,
                'pattern_tags': ','.join(pattern_tags),
                'pattern_score': max(pattern_score, 0.5),
                'slot_health_multiplier': slot_health_mult,
                'drift_families': ','.join([f for f in pattern_tags if self._family_drift_label(slot, f) in {"BOOST_DRIFT", "COOL_OFF"}]),
            })
        return scored_numbers

    def _get_family_categories(self, digit_tags):
        """Convert fine-grained pack tags to family categories"""
        families = set()
        for tag in digit_tags:
            if tag.startswith("pack2_"):
                families.add("PACK_2DIGIT")
            elif tag.startswith("pack3_"):
                families.add("PACK_3DIGIT")
            elif tag.startswith("pack4_"):
                families.add("PACK_4DIGIT")
            elif tag.startswith("pack5_"):
                families.add("PACK_5DIGIT")
            elif tag.startswith("pack6_"):
                families.add("PACK_6DIGIT")
            elif tag == "PACK_164950":
                families.add("PACK_164950")
            elif tag == "S40":
                families.add("S40")
        return families

    def apply_pattern_boost(self, slot, number, base_score, target_date):
        debug_components = {
            'base_score': base_score,
            'digit_match_boost': 0.0,
            'hero_number_boost': 0.0,
            'time_boost': 0.0,
            'cross_slot_boost': 0.0,
            'near_miss_boost': 0.0,
            'mirror_boost': 0.0,
            'digital_root_boost': 0.0,
            'total_quantum_boost': 0.0
        }
        quantum_boost = 0.0
        tens_digit = number // 10
        ones_digit = number % 10
        
        if self.adaptive_packs.get('tens_core_base') and tens_digit in self.adaptive_packs['tens_core_base']:
            quantum_boost += self.GOLDEN_DIGIT_BOOST * 0.6
            debug_components['digit_match_boost'] += self.GOLDEN_DIGIT_BOOST * 0.6
        if self.adaptive_packs.get('ones_core_base') and ones_digit in self.adaptive_packs['ones_core_base']:
            quantum_boost += self.GOLDEN_DIGIT_BOOST * 0.6
            debug_components['digit_match_boost'] += self.GOLDEN_DIGIT_BOOST * 0.6
        if self.adaptive_packs.get('tens_core_golden') and tens_digit in self.adaptive_packs['tens_core_golden']:
            quantum_boost += self.GOLDEN_DIGIT_BOOST
            debug_components['digit_match_boost'] += self.GOLDEN_DIGIT_BOOST
        if self.adaptive_packs.get('ones_core_golden') and ones_digit in self.adaptive_packs['ones_core_golden']:
            quantum_boost += self.GOLDEN_DIGIT_BOOST
            debug_components['digit_match_boost'] += self.GOLDEN_DIGIT_BOOST
        
        if self.adaptive_packs.get('hero_numbers') and number in self.adaptive_packs['hero_numbers']:
            quantum_boost += self.HERO_NUMBER_BOOST
            debug_components['hero_number_boost'] += self.HERO_NUMBER_BOOST
        
        current_day = target_date.strftime('%A')
        time_boost_slots = self.adaptive_packs.get('time_boost_slots', {})
        if slot in time_boost_slots and time_boost_slots[slot].get('best_day') == current_day:
            quantum_boost += self.TIME_AWARENESS_BOOST
            debug_components['time_boost'] += self.TIME_AWARENESS_BOOST
        
        cross_pairs = self.adaptive_packs.get('cross_slot_pairs_top', [])
        for pair in cross_pairs[:2]:
            if f"→{slot}" in pair:
                quantum_boost += 0.03
                debug_components['cross_slot_boost'] += 0.03
        
        near_miss_multiplier, near_miss_score = self.get_near_miss_multiplier(slot, number)
        if near_miss_multiplier > 1.0:
            debug_components['near_miss_boost'] = near_miss_multiplier - 1.0

        mirror_boost = self.get_mirror_boost(number)
        quantum_boost += mirror_boost
        debug_components['mirror_boost'] += mirror_boost
        
        digital_root_boost = self.get_digital_root_boost(number)
        quantum_boost += digital_root_boost
        debug_components['digital_root_boost'] += digital_root_boost
        
        quantum_boost = min(quantum_boost, self.MAX_QUANTUM_BOOST)
        debug_components['total_quantum_boost'] = quantum_boost
        boosted_score = base_score * (1.0 + quantum_boost)
        boosted_score *= near_miss_multiplier
        try:
            if near_miss_multiplier > 1.0:
                print(
                    f"[NEARMISS BOOST] {slot}: {to_2d_str(number)} base={base_score:.2f}, near_miss_boost={near_miss_score:.2f} → eff={boosted_score:.2f}"
                )
        except Exception:
            pass
        return boosted_score, debug_components

    def get_near_miss_multiplier(self, slot, number):
        slot_key = str(slot).upper()
        num_str = to_2d_str(number)
        base_score = 0.0
        try:
            base_score = float((self.near_miss_boosts or {}).get(slot_key, {}).get(num_str, 0.0) or 0.0)
        except Exception:
            base_score = 0.0

        if base_score <= 0 and hasattr(self, 'real_numbers_history'):
            fallback_hits = sum(1 for real_num in self.real_numbers_history if abs(number - real_num) == 1)
            base_score = fallback_hits * 0.2

        k = 0.15
        multiplier = 1.0 + min(base_score * k, 1.0)
        return multiplier, base_score

    def get_mirror_boost(self, number):
        if number < 10:
            return 0.0
        tens = number // 10
        ones = number % 10
        mirror_num = ones * 10 + tens
        if mirror_num == number:
            return 0.0
        return self.MIRROR_BOOST

    def get_digital_root_boost(self, number):
        digital_root = number
        while digital_root > 9:
            digital_root = sum(int(d) for d in str(digital_root))
        common_roots = [3, 6, 9]
        if digital_root in common_roots:
            return self.DIGITAL_ROOT_BOOST
        return 0.0

    def get_cross_slot_boost(self, real_slot, predicted_slot, number):
        boost_key = f"{real_slot}_{predicted_slot}"
        cross_boost = self.pattern_config.get('cross_slot_boost', {}).get(boost_key, 0.0)
        if hasattr(self, 'history'):
            history_key = (real_slot, number)
            if history_key in self.history:
                cross_hits = self.history[history_key]['cross_hits']
                cross_boost += min(0.2, cross_hits * 0.05)
        return cross_boost

    def get_time_awareness_boost(self, slot, target_date):
        time_awareness = self.pattern_config.get('time_awareness', {})
        slot_config = time_awareness.get(slot, {})
        current_day = target_date.strftime('%A')
        preferred_day = slot_config.get('preferred_day')
        if current_day == preferred_day:
            return slot_config.get('boost_factor', 1.0)
        return 1.0

    def get_digit_preference_boost(self, slot, number):
        digit_preferences = self.pattern_config.get('digit_preferences', {})
        slot_digits = digit_preferences.get(slot, {})
        tens_digit = number // 10
        ones_digit = number % 10
        boost = 0.0
        if tens_digit in slot_digits.get('common_tens', []):
            boost += 0.05
        if ones_digit in slot_digits.get('common_ones', []):
            boost += 0.05
        return boost

    def build_dynamic_shortlist(self, scored_numbers, desired_k=3, ev_gap=0.03):
        if not scored_numbers:
            return [], []
        sorted_numbers = sorted(scored_numbers, key=lambda x: (-x['final_score'], x['rank']))
        if not sorted_numbers:
            return [], []
        top_score = sorted_numbers[0]['final_score']
        shortlist = sorted_numbers[:max(1, desired_k)]
        shortlist = self._apply_diversity_guards(shortlist, sorted_numbers)
        shortlist = self._apply_mirror_hedge(shortlist, sorted_numbers, top_score, ev_gap)
        shortlist = sorted(shortlist, key=lambda x: (-x['final_score'], x['rank']))[:max(1, desired_k)]
        shortlisted_numbers = [item['number'] for item in shortlist]
        for item in scored_numbers:
            item['shortlisted'] = item['number'] in shortlisted_numbers
        return shortlist, scored_numbers

    def _apply_diversity_guards(self, shortlist, all_sorted_numbers):
        if not shortlist:
            return shortlist
        bins = [item['number'] // 10 for item in shortlist]
        bin_counts = Counter(bins)
        unique_bins = set(bins)
        if len(unique_bins) < self.MIN_BINS:
            missing_bins = set(range(10)) - unique_bins
            for bin_num in missing_bins:
                if len(unique_bins) >= self.MIN_BINS:
                    break
                for candidate in all_sorted_numbers:
                    if candidate['number'] // 10 == bin_num and candidate not in shortlist:
                        most_common_bin = bin_counts.most_common(1)[0][0]
                        worst_in_bin = None
                        for item in shortlist:
                            if item['number'] // 10 == most_common_bin:
                                if worst_in_bin is None or item['final_score'] < worst_in_bin['final_score']:
                                    worst_in_bin = item
                        if worst_in_bin and worst_in_bin['final_score'] < candidate['final_score']:
                            shortlist.remove(worst_in_bin)
                            shortlist.append(candidate)
                            bin_counts[most_common_bin] -= 1
                            bin_counts[bin_num] = bin_counts.get(bin_num, 0) + 1
                            unique_bins.add(bin_num)
                        break
        for bin_num, count in list(bin_counts.items()):
            if count > self.MAX_PER_BIN:
                bin_items = [item for item in shortlist if item['number'] // 10 == bin_num]
                bin_items_sorted = sorted(bin_items, key=lambda x: x['final_score'])
                while len(bin_items) > self.MAX_PER_BIN and bin_items_sorted:
                    worst = bin_items_sorted.pop(0)
                    shortlist.remove(worst)
                    bin_items.remove(worst)
        return shortlist

    def _apply_mirror_hedge(self, shortlist, all_sorted_numbers, top_score, ev_gap):
        if len(shortlist) < 2:
            return shortlist
        top_numbers = shortlist[:2]
        for top_item in top_numbers:
            number = top_item['number']
            mirror_num = int(to_2d_str(number)[::-1])
            if mirror_num == number or any(item['number'] == mirror_num for item in shortlist):
                continue
            mirror_item = None
            for candidate in all_sorted_numbers:
                if candidate['number'] == mirror_num:
                    mirror_item = candidate
                    break
            if mirror_item and mirror_item['final_score'] >= top_score * (1 - 2 * ev_gap):
                if len(shortlist) < self.HARD_CAP:
                    shortlist.append(mirror_item)
                else:
                    worst_item = min(shortlist, key=lambda x: x['final_score'])
                    if mirror_item['final_score'] > worst_item['final_score']:
                        shortlist.remove(worst_item)
                        shortlist.append(mirror_item)
        return shortlist

    # ✅ FIXED: Near-miss bug resolved - proper date comparison
    def load_real_numbers_history(self, days=30, target_date=None):
        try:
            results_df = quant_data_core.load_results_dataframe()
            if results_df.empty:
                print("   No real results data found")
                return []

            results_df['DATE'] = pd.to_datetime(results_df['DATE'], errors='coerce')
            results_df = results_df.dropna(subset=['DATE'])
            results_df['DATE_ONLY'] = results_df['DATE'].dt.date

            latest_real_date = quant_data_core.get_latest_result_date(results_df)
            if not latest_real_date:
                print("   No real results data found")
                return []

            start_date = latest_real_date - timedelta(days=days - 1)
            filtered_df = results_df[(results_df['DATE_ONLY'] >= start_date) & (results_df['DATE_ONLY'] <= latest_real_date)]

            if filtered_df.empty:
                print("   No real results data found")
                return []

            real_numbers = []
            for slot in self.slots:
                if slot not in filtered_df.columns:
                    continue
                slot_numbers = pd.to_numeric(filtered_df[slot], errors='coerce').dropna().astype(int).tolist()
                real_numbers.extend(slot_numbers)

            print(f"   Loaded {len(real_numbers)} real numbers for near-miss analysis")
            return real_numbers
        except Exception as e:
            print(f"⚠️  Error loading real numbers history: {e}")
            return []

    # ✅ NEW METHOD: Intraday scoring with family multipliers
    def calculate_enhanced_scores_intraday(self, numbers_list, slot, history, target_date, family_multipliers):
        """
        Intraday scoring with pattern family multipliers
        PRESERVES ALL EXISTING v1 LOGIC + adds family multipliers
        """
        if not numbers_list:
            return []
            
        scored_numbers = []
        for rank, number in enumerate(numbers_list, 1):
            base_score = 1.0 / rank
            digit_tags = []
            history_key = (slot, number)
            
            # ✅ PRESERVE EXACT v1 QUANTUM BOOST LOGIC
            quantum_boosted_score, quantum_debug = self.apply_pattern_boost(slot, number, base_score, target_date)
            pattern_bonus = 0.0
            
            # ✅ ENHANCED: Apply family multipliers to pattern bonuses
            if PATTERN_PACKS_AVAILABLE and self.pattern_config.get('s40_enabled', True):
                if pattern_packs.is_s40(number):
                    s40_bonus = self.pattern_config.get('pattern_weights', {}).get('s40_bonus', self.S40_BONUS)
                    # Apply S40 family multiplier
                    if 'S40' in family_multipliers:
                        s40_bonus *= family_multipliers['S40']
                    pattern_bonus += s40_bonus
                    
                if self.pattern_config.get('digit_packs_enabled', True):
                    digit_tags = pattern_packs.get_digit_pack_tags(number)
                    # ✅ UPDATED: Family-level bonuses with multipliers
                    families = self._get_family_categories(digit_tags)
                    for fam in families:
                        fam_bonus = self.DIGIT_PACK_BONUS
                        if fam in family_multipliers:
                            fam_bonus *= family_multipliers[fam]
                        pattern_bonus += fam_bonus
            
            # ✅ PRESERVE EXACT v1 MEMORY BONUS LOGIC
            memory_bonus = 0.0
            if self.pattern_config.get('memory_bonus_enabled', True) and history_key in history:
                h = history[history_key]
                memory_bonus = (
                    self.W_DIRECT * h['direct_hits'] +
                    self.W_CROSS * h['cross_hits'] +
                    self.W_S40_HIT * h['s40_hits']
                )
                memory_bonus = memory_bonus / max(1, self.N_DAYS / 10)
            
            # ✅ PRESERVE EXACT v1 CROSS-SLOT BOOST LOGIC
            for other_slot in self.slots:
                if other_slot != slot:
                    cross_boost = self.get_cross_slot_boost(other_slot, slot, number)
                    pattern_bonus += cross_boost
            
            # ✅ PRESERVE EXACT v1 TIME BOOST LOGIC
            time_boost = self.get_time_awareness_boost(slot, target_date)
            if time_boost > 1.0:
                pattern_bonus += (time_boost - 1.0) * 0.5
            
            # ✅ PRESERVE EXACT v1 DIGIT PREFERENCE LOGIC
            digit_boost = self.get_digit_preference_boost(slot, number)
            pattern_bonus += digit_boost
            
            max_pattern = self.pattern_config.get('pattern_weights', {}).get('max_pattern_bonus', self.MAX_PATTERN_BONUS)
            pattern_bonus = min(pattern_bonus, max_pattern)
            
            # ✅ FINAL SCORE COMBINATION (PRESERVED FROM v1)
            pattern_multiplier = self._compute_pattern_multiplier(slot, number)
            final_score = (quantum_boosted_score + pattern_bonus + memory_bonus) * pattern_multiplier
            
            direct_hits = history[history_key]['direct_hits'] if history_key in history else 0
            cross_hits = history[history_key]['cross_hits'] if history_key in history else 0
            s40_hits = history[history_key]['s40_hits'] if history_key in history else 0
            
            scored_numbers.append({
                'number': number,
                'rank': rank,
                'base_score': base_score,
                'pattern_bonus': pattern_bonus,
                'memory_bonus': memory_bonus,
                'final_score': final_score,
                'is_s40': PATTERN_PACKS_AVAILABLE and pattern_packs.is_s40(number),
                'digit_pack_tags': ','.join(digit_tags),
                'direct_hits_30d': direct_hits,
                'cross_hits_30d': cross_hits,
                's40_hits_30d': s40_hits,
                'quantum_boosted_score': quantum_boosted_score,
                'quantum_boost_components': quantum_debug,
                'pattern_multiplier': pattern_multiplier,
            })

        return scored_numbers

    # ✅ MODIFIED METHOD: Added intraday support parameters with safe defaults
    def generate_enhanced_bet_plan(self, df, target_rows, target_date, history, 
                                   slot_filter=None, family_multipliers=None, mode="normal"):
        """
        Enhanced with intraday recalculation support
        PRESERVES EXACT v1 BEHAVIOR WHEN NO NEW PARAMETERS PROVIDED
        """
        
        # ✅ PRESERVE EXACT v1 BEHAVIOR - NO CHANGES TO EXISTING LOGIC
        bets_data = []
        summary_data = []
        diagnostic_data = []
        quantum_debug_data = []
        ultra_debug_data = []
        explainability_data = []

        self.history = history
        print("🔍 Loading real numbers history for near-miss learning...")
        self.real_numbers_history = self.load_real_numbers_history(NEAR_MISS_WINDOW_DAYS_DEFAULT, target_date)

        self.topn_shortlist_profile = self._load_topn_shortlist_profile()
        self.pattern_regime_summary = self._load_pattern_regime_summary()
        self.topn_policy = self._load_topn_policy()
        self.slot_health_snapshot = self._load_slot_health_snapshot()

        regime_line_parts = []
        per_slot_regimes = self.pattern_regime_summary.get("per_slot", {}) if isinstance(self.pattern_regime_summary, dict) else {}
        for slot in self.slots:
            slot_key = str(slot).upper()
            slot_regimes = per_slot_regimes.get(slot_key, {}) if isinstance(per_slot_regimes, dict) else {}
            s40_regime = str(slot_regimes.get("S40", "NORMAL")).upper()
            fam_regime = str(slot_regimes.get("FAMILY_164950", "NORMAL")).upper()
            s40_mult = PATTERN_REGIME_MULTIPLIERS.get(s40_regime, 1.0)
            fam_mult = PATTERN_REGIME_MULTIPLIERS.get(fam_regime, 1.0)
            regime_line_parts.append(f"{slot_key}: S40={s40_mult:.1f}, 164950={fam_mult:.1f}")
        if regime_line_parts:
            print(f"PatternFactors {'; '.join(regime_line_parts)}")
        
        long_df = self.convert_wide_to_long_format(df, target_rows)
        target_df = long_df.copy()
        target_df['date_parsed'] = target_df['date'].apply(
            lambda x: pd.to_datetime(x).date() if isinstance(x, str) else x.date() if isinstance(x, datetime) else x
        )
        target_df = target_df[target_df['date_parsed'] == target_date]
        
        if target_df.empty:
            raise ValueError(f"No predictions found for target date {target_date}")
        
        # ✅ INTRADAY SUPPORT: Filter slots if provided (preserves v1 behavior when None)
        slots_to_process = slot_filter if slot_filter is not None else self.slots
        
        for slot in slots_to_process:
            if slot not in self.slots:
                print(f"⚠️  Skipping invalid slot: {slot}")
                continue
                
            slot_data = target_df[target_df['slot'] == slot]
            if slot_data.empty:
                print(f"⚠️  No data found for slot {slot} on {target_date}")
                continue
            
            numbers_list = slot_data['numbers'].iloc[0]
            
            # ✅ INTRADAY SUPPORT: Apply family multipliers if provided
            print(f"🎯 Processing {slot}: {len(numbers_list)} numbers")
            scored_numbers, slot_debug = self.calculate_quant_scores(numbers_list, slot)
            self.bet_engine_debug.setdefault(slot, {})["candidates"] = slot_debug
            
            if not scored_numbers:
                print(f"   Empty scored list - using fallback")
                continue
            
            policy_block = (self.topn_policy or {}).get(slot, {}) if isinstance(self.topn_policy, dict) else {}
            n_star = policy_block.get("final_best_n") or policy_block.get("best_n_exact") or DEFAULT_BEST_N
            try:
                n_star_int = int(n_star)
            except Exception:
                n_star_int = DEFAULT_BEST_N
            n_main = min(max(n_star_int, 1), 5)
            n_main = min(n_main, len(scored_numbers)) if scored_numbers else n_main
            shortlist_k = max(1, n_main)

            shortlist, all_scored = self.build_dynamic_shortlist(scored_numbers, desired_k=shortlist_k)
            tiers = self.assign_tiers(shortlist)
            policy_roi = policy_block.get("roi_final_best") if isinstance(policy_block, dict) else None
            print(
                f"   Top-N policy: final_best_n={n_star_int}, roi_final_best={policy_roi:+.1f}%"
                if policy_roi is not None
                else f"   Top-N policy: final_best_n={n_star_int}"
            )

            slot_debug_numbers = []
            
            for item in all_scored:
                diagnostic_data.append({
                    'date': target_date,
                    'slot': slot,
                    'number': item['number'],
                    'base_rank': item['rank'],
                    'base_score': item['base_score'],
                    'is_s40': item['is_s40'],
                    'digit_pack_tags': item['digit_pack_tags'],
                    'direct_hits_30d': item['direct_hits_30d'],
                    'cross_hits_30d': item['cross_hits_30d'],
                    's40_hits_30d': item['s40_hits_30d'],
                    'pattern_bonus': item['pattern_bonus'],
                    'memory_bonus': item['memory_bonus'],
                    'final_score': item['final_score'],
                    'shortlisted': item['shortlisted'],
                    'quantum_boosted_score': item.get('quantum_boosted_score', 0),
                    'pattern_multiplier': item.get('pattern_multiplier', 1.0),
                })
                
                quantum_components = item.get('quantum_boost_components', {})
                quantum_debug_data.append({
                    'date': target_date,
                    'slot': slot,
                    'number': item['number'],
                    'base_score': quantum_components.get('base_score', 0),
                    'digit_match_boost': quantum_components.get('digit_match_boost', 0),
                    'hero_number_boost': quantum_components.get('hero_number_boost', 0),
                    'time_boost': quantum_components.get('time_boost', 0),
                    'cross_slot_boost': quantum_components.get('cross_slot_boost', 0),
                    'near_miss_boost': quantum_components.get('near_miss_boost', 0),
                    'mirror_boost': quantum_components.get('mirror_boost', 0),
                    'digital_root_boost': quantum_components.get('digital_root_boost', 0),
                    'total_quantum_boost': quantum_components.get('total_quantum_boost', 0),
                    'final_quantum_score': item.get('quantum_boosted_score', 0)
                })
            
            if shortlist:
                shortlist_numbers = [item['number'] for item in shortlist]
                print(f"   Shortlist ({len(shortlist)}): {', '.join(str(n) for n in shortlist_numbers)}")
            else:
                print(f"   Empty shortlist - using fallback")
                shortlist = scored_numbers[:3]
                tiers = self.assign_tiers(shortlist)

            for item in shortlist:
                num_str = to_2d_str(item['number'])
                reasons = (self.bet_engine_debug.get(slot, {}) or {}).get("candidates", {})
                candidate_debug = reasons.get(num_str, {}) if isinstance(reasons, dict) else {}
                pattern_tags = candidate_debug.get("pattern_tags") or self._get_number_families(item['number'])
                regimes = candidate_debug.get("regimes", {}) if isinstance(candidate_debug, dict) else {}
                slot_debug_numbers.append({
                    "number": num_str,
                    "tier": tiers.get(item['number'], 'C'),
                    "final_score": float(item.get('final_score', 0.0)),
                    "pattern_tags": pattern_tags,
                    "regimes": {k: v for k, v in regimes.items() if v},
                })
            
            andar_digit, bahar_digit = self.get_andar_bahar(shortlist)
            main_stake_total = 0
            main_max_return = 0

            base_main_total = BASE_MAIN_TOTAL_UNITS * self.base_unit
            use_weighted_pattern = len(shortlist) > 3
            weights: List[float] = []
            if use_weighted_pattern:
                weights = list(range(len(shortlist), 0, -1))
                weight_sum = sum(weights) or 1
                scaled_weights = [w * base_main_total / weight_sum for w in weights]
            else:
                scaled_weights = []

            for idx, item in enumerate(shortlist):
                number = item['number']
                tier = tiers.get(number, 'C')
                if use_weighted_pattern:
                    stake = round(scaled_weights[idx], 2)
                else:
                    if tier == 'A':
                        stake = 2 * self.base_unit
                    elif tier == 'B':
                        stake = 1 * self.base_unit
                    else:
                        stake = 0.5 * self.base_unit

                pattern_factor = self.get_pattern_multiplier(slot, number)
                stake = round(stake * pattern_factor, 2)

                potential_return = stake * 90
                main_stake_total += stake
                main_max_return += potential_return

                bets_data.append({
                    'date': target_date,
                    'slot': slot,
                    'layer_type': 'Main',
                    'number_or_digit': to_2d_str(number),
                    'tier': tier,
                    'stake': stake,
                    'potential_return': potential_return,
                    'source_rank': item['rank'],
                    'notes': f"ULTRA v5 scoring: final_score={item['final_score']:.3f}",
                    'actual_result': '',
                    'hit_flag_main': '',
                    'hit_flag_andar': '',
                    'hit_flag_bahar': '',
                    'net_pnl': '',
                    'families': item.get('pattern_tags', ''),
                    'score_base': item.get('base_score', 0.0),
                    'score_pattern': item.get('pattern_score', 0.0),
                    'score_slot_health': item.get('slot_health_multiplier', 1.0),
                    'score_final': item.get('final_score', 0.0),
                    'drift_families': item.get('drift_families', ''),
                })
            
            if andar_digit is not None:
                bets_data.append({
                    'date': target_date,
                    'slot': slot,
                    'layer_type': 'ANDAR',
                    'number_or_digit': str(andar_digit),
                    'tier': 'NA',
                    'stake': self.base_unit,
                    'potential_return': 90,
                    'source_rank': '',
                    'notes': f"most frequent tens digit",
                    'actual_result': '',
                    'hit_flag_main': '',
                    'hit_flag_andar': '',
                    'hit_flag_bahar': '',
                    'net_pnl': ''
                })
            
            if bahar_digit is not None:
                bets_data.append({
                    'date': target_date,
                    'slot': slot,
                    'layer_type': 'BAHAR',
                    'number_or_digit': str(bahar_digit),
                    'tier': 'NA',
                    'stake': self.base_unit,
                    'potential_return': 90,
                    'source_rank': '',
                    'notes': f"most frequent ones digit",
                    'actual_result': '',
                    'hit_flag_main': '',
                    'hit_flag_andar': '',
                    'hit_flag_bahar': '',
                    'net_pnl': ''
                })
            
            for layer_type in ['S36', 'PackCore', 'PackBooster']:
                bets_data.append({
                    'date': target_date,
                    'slot': slot,
                    'layer_type': layer_type,
                    'number_or_digit': '',
                    'tier': 'NA',
                    'stake': 0.0,
                    'potential_return': 0.0,
                    'source_rank': '',
                    'notes': 'ULTRA v5 implementation',
                    'actual_result': '',
                    'hit_flag_main': '',
                    'hit_flag_andar': '',
                    'hit_flag_bahar': '',
                    'net_pnl': ''
                })
            
            andar_stake = self.base_unit if andar_digit is not None else 0
            bahar_stake = self.base_unit if bahar_digit is not None else 0
            total_stake = main_stake_total + andar_stake + bahar_stake
            max_total_return = main_max_return + (90 if andar_digit else 0) + (90 if bahar_digit else 0)
            
            summary_data.append({
                'date': target_date,
                'slot': slot,
                'main_count': len(shortlist),
                'main_stake': main_stake_total,
                'main_max_return': main_max_return,
                'andar_stake': andar_stake,
                'bahar_stake': bahar_stake,
                'total_stake': total_stake,
                'max_total_return': max_total_return,
                'enhanced_patterns': True,
                'ultra_mode': True,
                'dynamic_top_n': True
            })

            self.bet_engine_debug.setdefault(slot, {})['numbers'] = slot_debug_numbers
            self.bet_engine_debug.setdefault(slot, {})['target_date'] = target_date.strftime('%Y-%m-%d')
            
            print(f"   Main: {len(shortlist)} numbers, {fmt_rupees(main_stake_total)} stake")
            print(f"   ANDAR: {andar_digit}, BAHAR: {bahar_digit}")
            print(
                f"   Total stake: {fmt_rupees(total_stake)}, Max return: {fmt_rupees(max_total_return)}"
            )

            # Compact family focus line for transparency
            family_focus_parts = []
            for item in shortlist:
                fams = item.get('pattern_tags', '')
                fam_label = fams if fams else 'single'
                family_focus_parts.append(f"{to_2d_str(item['number'])}[{fam_label}]")
            if family_focus_parts:
                print(f"{slot} family-focus: {', '.join(family_focus_parts)}")
        
        bets_df = pd.DataFrame(bets_data)
        summary_df = pd.DataFrame(summary_data)
        diagnostic_df = pd.DataFrame(diagnostic_data)
        quantum_debug_df = pd.DataFrame(quantum_debug_data)
        ultra_debug_df = pd.DataFrame(ultra_debug_data)
        explainability_df = pd.DataFrame(explainability_data)

        # Build slot health map and per-slot multipliers
        slot_health_map: Dict[str, SlotHealth] = {}
        slot_multipliers: Dict[str, float] = {}
        slot_level_map: Dict[str, str] = {}

        topn_profile = self._load_topn_roi_profile(window_days=ROI_WINDOW_DAYS_DEFAULT)
        topn_per_slot = topn_profile.get("per_slot", {}) if isinstance(topn_profile, dict) else {}

        roi_map: Dict[str, float] = {}
        for slot in self.slots:
            key = str(slot).upper()
            snapshot = (self.slot_health_snapshot or {}).get(key, {}) if isinstance(self.slot_health_snapshot, dict) else {}
            base_health = get_slot_health(key)
            roi_val = float(snapshot.get("roi_30", snapshot.get("roi_percent", getattr(base_health, "roi_percent", 0.0))) or getattr(base_health, "roi_percent", 0.0))
            roi_map[key] = roi_val

        for slot in self.slots:
            key = str(slot).upper()
            # Always ask quant_slot_health for the latest SlotHealth
            health = get_slot_health(key)

            if health is None:
                # Fallback neutral health if nothing is available
                health = SlotHealth(
                    slot=key,
                    roi_percent=0.0,
                    wins=0,
                    losses=0,
                    hit_rate=0.0,
                    current_streak=0,
                    slump=False,
                    roi_bucket="UNKNOWN",
                )

            slot_snapshot = (self.slot_health_snapshot or {}).get(key, {}) if isinstance(self.slot_health_snapshot, dict) else {}
            if isinstance(slot_snapshot, dict) and "slump" in slot_snapshot:
                health.slump = bool(slot_snapshot.get("slump", health.slump))

            slot_health_map[key] = health
            mult, level = self._compute_slot_multiplier(key, health)
            mult *= self._slot_roi_dampener(key, health, topn_per_slot)
            slot_multipliers[key] = mult
            slot_level_map[key] = level

            print(
                f"[QUANT-SIGNALS] Slot {key}: "
                f"slump={health.slump}, roi_bucket={health.roi_bucket}, slot_multiplier={mult:.2f}, level={level}"
            )

        layer_multipliers = self._compute_layer_multipliers()
        print("Layer multipliers (30d):")
        for layer_key in ["MAIN", "ANDAR", "BAHAR"]:
            print(f"   {layer_key}: {layer_multipliers.get(layer_key, 1.0):.2f}x")

        active_overlays = [lt for lt, pol in OVERLAY_POLICY.items() if pol.get("stake_units", 0) > 0]
        print(f"[OVERLAYS] Active (config-driven): {', '.join(active_overlays) if active_overlays else 'None'}")

        try:
            # --- Apply multipliers to bets_df (numeric-safe) ---
            if not bets_df.empty and "slot" in bets_df.columns:
                bets_df["slot_key"] = bets_df["slot"].astype(str).str.upper()
                bets_df["slot_multiplier"] = (
                    bets_df["slot_key"].map(slot_multipliers).fillna(1.0).astype(float)
                )

                # Coerce stake to numeric
                stake_numeric = pd.to_numeric(bets_df["stake"], errors="coerce")
                numeric_mask = stake_numeric.notna()

                # Scale only numeric stakes (leave blanks like S36/Pack rows untouched)
                scaled_stake = (
                    stake_numeric[numeric_mask]
                    * bets_df.loc[numeric_mask, "slot_multiplier"]
                ).round(2)
                bets_df.loc[numeric_mask, "stake"] = scaled_stake

                # Recompute potential_return only where it is numeric
                potential_numeric = pd.to_numeric(
                    bets_df["potential_return"], errors="coerce"
                )
                potential_mask = potential_numeric.notna() & numeric_mask

                updated_returns = (
                    pd.to_numeric(
                        bets_df.loc[potential_mask, "stake"], errors="coerce"
                    )
                    * 90
                ).round(2)
                bets_df.loc[potential_mask, "potential_return"] = updated_returns

                # Attach slot slump / ROI bucket flags
                bets_df = bets_df.assign(
                    slot_slump_flag=bets_df["slot_key"].apply(
                        lambda s: (slot_health_map.get(s) or get_slot_health(s)).slump
                    ),
                    slot_roi_bucket=bets_df["slot_key"].apply(
                        lambda s: (slot_health_map.get(s) or get_slot_health(s)).roi_bucket
                    ),
                )

            # --- Apply multipliers to summary_df (numeric-safe) ---
            if not summary_df.empty and "slot" in summary_df.columns:
                summary_df["slot_key"] = summary_df["slot"].astype(str).str.upper()
                summary_df["slot_multiplier"] = (
                    summary_df["slot_key"].map(slot_multipliers).fillna(1.0).astype(float)
                )

                for col in [
                    "main_stake",
                    "andar_stake",
                    "bahar_stake",
                    "total_stake",
                    "max_total_return",
                    "main_max_return",
                ]:
                    if col in summary_df.columns:
                        col_numeric = pd.to_numeric(
                            summary_df[col], errors="coerce"
                        ).fillna(0.0)
                        summary_df[col] = (col_numeric * summary_df["slot_multiplier"]).round(2)

                summary_df = summary_df.assign(
                    slot_slump_flag=summary_df["slot_key"].apply(
                        lambda s: (slot_health_map.get(s) or get_slot_health(s)).slump
                    ),
                    slot_roi_bucket=summary_df["slot_key"].apply(
                        lambda s: (slot_health_map.get(s) or get_slot_health(s)).roi_bucket
                    ),
                )

        except Exception as e:
            print(f"❌ Error in slot-multiplier scaling: {e}")
            # Fail-safe: if anything goes wrong, we keep original bets_df/summary_df
            # and continue without crashing.

        # Apply layer multipliers (MAIN / ANDAR / BAHAR)
        if not bets_df.empty and "layer_type" in bets_df.columns:
            bets_df["layer_key"] = bets_df["layer_type"].astype(str).str.upper()
            for layer_key, mult in layer_multipliers.items():
                layer_mask = bets_df["layer_key"] == layer_key
                if not layer_mask.any():
                    continue
                stake_numeric = pd.to_numeric(bets_df.loc[layer_mask, "stake"], errors="coerce")
                valid_mask = layer_mask & stake_numeric.notna()
                bets_df.loc[valid_mask, "stake"] = (stake_numeric[valid_mask] * float(mult)).round(2)

                pot_numeric = pd.to_numeric(bets_df.loc[layer_mask, "potential_return"], errors="coerce")
                pot_mask = layer_mask & pot_numeric.notna()
                bets_df.loc[pot_mask, "potential_return"] = (
                    pd.to_numeric(bets_df.loc[pot_mask, "stake"], errors="coerce") * 90
                ).round(2)

        # Apply slot-level gating after quant slot multipliers
        if not bets_df.empty:
            if "slot_key" not in bets_df.columns:
                bets_df["slot_key"] = bets_df["slot"].astype(str).str.upper()
            for slot in self.slots:
                slot_key = str(slot).upper()
                level = slot_level_map.get(slot_key, "MID")
                level_mult = SLOT_LEVEL_MULTIPLIERS.get(level, 1.0)
                slot_mask = bets_df["slot_key"] == slot_key

                if level == "OFF":
                    main_mask = slot_mask & (bets_df["layer_type"] == "Main")
                    bets_df.loc[main_mask, ["stake", "potential_return"]] = 0.0

                    ab_mask = slot_mask & bets_df["layer_type"].isin(["ANDAR", "BAHAR"])
                    bets_df.loc[ab_mask, "stake"] = 0.0
                    bets_df.loc[ab_mask, "potential_return"] = 0.0
                    print(f"{slot_key} in OFF mode – main stakes suppressed")
                else:
                    stake_numeric = pd.to_numeric(bets_df.loc[slot_mask, "stake"], errors="coerce")
                    numeric_mask = slot_mask & stake_numeric.notna()
                    bets_df.loc[numeric_mask, "stake"] = (stake_numeric[numeric_mask] * level_mult).round(2)

                    potential_numeric = pd.to_numeric(bets_df.loc[slot_mask, "potential_return"], errors="coerce")
                    potential_mask = slot_mask & potential_numeric.notna()
                    bets_df.loc[potential_mask, "potential_return"] = (
                        pd.to_numeric(bets_df.loc[potential_mask, "stake"], errors="coerce") * 90
                    ).round(2)

        # Recompute summary totals to reflect slot-level and BAHAR adjustments
        if not summary_df.empty and "slot" in summary_df.columns and not bets_df.empty:
            summary_df["slot_key"] = summary_df["slot"].astype(str).str.upper()
            for slot in self.slots:
                slot_key = str(slot).upper()
                slot_mask = bets_df["slot_key"] == slot_key
                summary_mask = summary_df["slot_key"] == slot_key

                main_total = pd.to_numeric(bets_df.loc[slot_mask & (bets_df["layer_type"] == "Main"), "stake"], errors="coerce").sum()
                andar_total = pd.to_numeric(bets_df.loc[slot_mask & (bets_df["layer_type"] == "ANDAR"), "stake"], errors="coerce").sum()
                bahar_total = pd.to_numeric(bets_df.loc[slot_mask & (bets_df["layer_type"] == "BAHAR"), "stake"], errors="coerce").sum()
                max_total_return = pd.to_numeric(bets_df.loc[slot_mask, "potential_return"], errors="coerce").sum()

                summary_df.loc[summary_mask, "main_stake"] = main_total
                summary_df.loc[summary_mask, "andar_stake"] = andar_total
                summary_df.loc[summary_mask, "bahar_stake"] = bahar_total
                summary_df.loc[summary_mask, "total_stake"] = main_total + andar_total + bahar_total
                summary_df.loc[summary_mask, "max_total_return"] = max_total_return
                summary_df.loc[summary_mask, "slot_level"] = slot_level_map.get(slot_key, "MID")

        for slot in self.slots:
            slot_key = str(slot).upper()
            total_slot_stake = 0.0
            if not summary_df.empty and "slot_key" in summary_df.columns:
                slot_mask = summary_df["slot_key"] == slot_key
                total_slot_stake = float(summary_df.loc[slot_mask, "total_stake"].sum())
            print(f"[STAKE-LEVEL] {slot_key}: level={slot_level_map.get(slot_key, 'MID')} total_stake={fmt_rupees(total_slot_stake)}")

        if not bets_df.empty:
            print(
                f"BAHAR defensive mode: stakes scaled by {BAHAR_EXTRA_MULTIPLIER} due to negative historical ROI."
            )

        bets_df = apply_overlay_policy_to_bets(bets_df, slot_health_map, self.base_unit)

        if not bets_df.empty:
            pack_mask = bets_df["layer_type"].isin(["PackCore", "PackBooster"])
            numeric_mask = bets_df["number_or_digit"].apply(is_valid_2d_number)
            adjustable_mask = pack_mask & numeric_mask

            for idx, row in bets_df.loc[adjustable_mask].iterrows():
                slot_key = row.get("slot")
                number = row.get("number_or_digit")
                factor = self.get_pattern_multiplier(slot_key, number)
                try:
                    stake_val = float(row.get("stake", 0.0))
                except Exception:
                    continue
                updated_stake = round(stake_val * factor, 2)
                bets_df.at[idx, "stake"] = updated_stake
                bets_df.at[idx, "potential_return"] = round(updated_stake * 90, 2)

        return bets_df, summary_df, diagnostic_df, quantum_debug_df, ultra_debug_df, explainability_df

    def save_bet_plan(self, bets_df, summary_df, diagnostic_df, quantum_debug_df, ultra_debug_df, explainability_df, target_date):
        output_dir = Path(__file__).resolve().parent / "predictions" / "bet_engine"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"bet_plan_master_{target_date.strftime('%Y%m%d')}.xlsx"
        file_path = output_dir / filename
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            bets_df.to_excel(writer, sheet_name='bets', index=False)
            summary_df.to_excel(writer, sheet_name='summary', index=False)
            diagnostic_df.to_excel(writer, sheet_name='diagnostic_scores', index=False)
            quantum_debug_df.to_excel(writer, sheet_name='quantum_debug', index=False)
            ultra_debug_df.to_excel(writer, sheet_name='ultra_debug', index=False)
            explainability_df.to_excel(writer, sheet_name='explainability', index=False)
        
        explain_json = output_dir / f"bet_engine_explainability_{target_date.strftime('%Y%m%d')}.json"
        explain_summary = {
            "timestamp": datetime.now().isoformat(),
            "target_date": target_date.strftime('%Y-%m-%d'),
            "total_numbers_analyzed": len(explainability_df),
            "dynamic_top_n_used": True,
            "avg_final_score": explainability_df['final_score'].mean() if not explainability_df.empty else 0,
        }
        
        with open(explain_json, 'w') as f:
            json.dump(explain_summary, f, indent=2)

        return file_path

    def write_debug_payload(self, target_date: datetime.date):
        try:
            debug_path = Path(__file__).resolve().parent / "data" / "bet_engine_debug.json"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "generated_at": datetime.now().isoformat(),
                "target_date": target_date.strftime('%Y-%m-%d'),
                "slots": {},
            }
            for slot, info in (self.bet_engine_debug or {}).items():
                payload["slots"][slot] = {
                    "numbers": info.get("numbers", []),
                    "target_date": info.get("target_date", target_date.strftime('%Y-%m-%d')),
                }
            with open(debug_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            print(f"⚠️  Could not write bet_engine_debug.json: {exc}")


def analyze_near_miss_history(days: int = 30):
    """Analyze near-miss candidates using the same logic as the bet engine."""
    try:
        results_df = quant_data_core.load_results_dataframe()
    except Exception as exc:
        print(f"⚠️ Unable to load results for near-miss analysis: {exc}")
        return {}

    if results_df is None or results_df.empty:
        print("⚠️ No historical results available for near-miss analysis")
        return {}

    results_df['DATE'] = pd.to_datetime(results_df['DATE'], errors='coerce')
    results_df = results_df.dropna(subset=['DATE'])
    results_df['DATE_ONLY'] = results_df['DATE'].dt.date

    latest_real_date = quant_data_core.get_latest_result_date(results_df)
    if not latest_real_date:
        print("⚠️ Could not determine latest result date")
        return {}

    start_date = latest_real_date - timedelta(days=days - 1)
    filtered_df = results_df[(results_df['DATE_ONLY'] >= start_date) & (results_df['DATE_ONLY'] <= latest_real_date)]

    slots = ["FRBD", "GZBD", "GALI", "DSWR"]
    summary = {}
    aggregate_counter = Counter()

    for slot in slots:
        if slot not in filtered_df.columns:
            continue
        numbers = pd.to_numeric(filtered_df[slot], errors='coerce').dropna().astype(int).tolist()
        counter = Counter()
        for num in numbers:
            neighbors = [num - 1, num + 1]
            for candidate in neighbors:
                counter[candidate % 100] += 1
                aggregate_counter[candidate % 100] += 1

        summary[slot] = {
            "recent_draws": len(numbers),
            "top_near_miss_candidates": counter.most_common(5),
        }

    summary["aggregate"] = {
        "top_near_miss_candidates": aggregate_counter.most_common(10)
    }

    output_path = Path(__file__).resolve().parent / "logs" / "performance" / "near_miss_report.json"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Near-miss report saved to {output_path}")
    except Exception as exc:
        print(f"⚠️ Unable to save near-miss report: {exc}")

    print("\n🔍 Near-miss candidates (last {days} days):".format(days=days))
    for slot, info in summary.items():
        if slot == "aggregate":
            continue
        print(f" • {slot}: {info['top_near_miss_candidates']}")
    print(f" • Aggregate: {summary.get('aggregate', {}).get('top_near_miss_candidates', [])}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Precise Bet Engine v5 Rocket - Ultra clear output')
    parser.add_argument('--target', choices=['today', 'tomorrow', 'auto'], default='tomorrow')
    parser.add_argument('--source', choices=['scr9', 'fusion'], default='scr9')
    
    args = parser.parse_args()
    
    try:
        engine = PreciseBetEngine()
        
        print(f"🔍 Locating latest {args.source.upper()} predictions...")
        latest_file = engine.find_latest_predictions_file(args.source)
        df = engine.load_ultimate_predictions(latest_file)
        
        target_rows = engine.select_target_data(df, args.target)
        long_df = engine.convert_wide_to_long_format(df, target_rows)
        target_date = engine.get_target_date(long_df)
        
        # ✅ ROCKET MODE: Clear date mapping
        system_today = datetime.now().date()
        print(f"\n📅 DATE MAPPING (CRYSTAL CLEAR):")
        print(f"   🖥️  System Date: {system_today}")
        print(f"   🎯 Target Date: {target_date}") 
        print(f"   📊 Mode: {args.target.upper()}")
        
        print("🧠 Loading script hit memory...")
        memory_df = engine.load_script_hit_memory(target_date)
        history = engine.build_history_table(memory_df)

        print("🎲 Generating ULTRA bet plan...")
        bets_df, summary_df, diagnostic_df, quantum_debug_df, ultra_debug_df, explainability_df = engine.generate_enhanced_bet_plan(df, target_rows, target_date, history)

        if bets_df.empty:
            print("❌ No bets generated")
            return 1

        bets_df, summary_df = engine.apply_stake_overlays(bets_df, summary_df, target_date)
        engine.write_debug_payload(target_date)

        output_path = engine.save_bet_plan(bets_df, summary_df, diagnostic_df, quantum_debug_df, ultra_debug_df, explainability_df, target_date)
        print(f"💾 Bet plan saved: {output_path}")
        
        # ✅ ROCKET MODE: Ultra clear summary
        engine.print_rocket_summary(bets_df, summary_df, target_date, latest_file, args.target)
        
        print(f"✅ ULTRA v5 ROCKET MODE completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())