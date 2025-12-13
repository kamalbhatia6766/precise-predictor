# deepseek_scr9.py - MODIFIED WITH SPEED MODE
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional
import warnings
import os
import glob
import subprocess
import re
import time
import json
from pathlib import Path
import quant_data_core
from quant_core import execution_core, hit_core
from script_hit_metrics import (
    SLOTS,
    build_script_league,
    build_script_weights_by_slot,
    get_metrics_table,
    compute_script_metrics,
    hero_weak_table,
)
from script_hit_memory_utils import load_script_hit_memory
import pattern_packs
warnings.filterwarnings('ignore')

USE_SCRIPT_WEIGHTS = True
SCRIPT_WEIGHTS_WINDOW_DAYS = 30
SCRIPT_WEIGHTS_MIN_SCORE = 0.01
LEARN_WINDOW_DAYS = 90
LEARN_WEIGHT_EXACT = 1.0
LEARN_WEIGHT_NEAR = 0.3
LEARN_SCORE_ALPHA = 0.1
SCRIPT_SCORE_A = 3.0
SCRIPT_SCORE_B = 1.0
SCRIPT_SCORE_C = 0.5
SCRIPT_WEIGHT_MIN = 0.5
SCRIPT_WEIGHT_MAX = 1.5
BOOSTER_WINDOW_DAYS = 90
BOOST_ALPHA = 3
BOOST_BETA = 1
BOOST_GAMMA = 2
BOOST_DELTA = 1
# Controls how noisy backtest printing is in FULL mode
VERBOSE_BACKTEST = False  # default: compact output


def _normalize_slot(slot_val):
    mapping = {"1": "FRBD", "2": "GZBD", "3": "GALI", "4": "DSWR"}
    if pd.isna(slot_val):
        return None
    s = str(slot_val).strip().upper()
    return mapping.get(s, s if s else None)


def _choose_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["result_date", "date", "predict_date"]:
        if col in df.columns:
            return col
    return None


def _normalize_number(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(str(value).zfill(2)) % 100
    except Exception:
        return None


def load_learning_scores(base_dir, window_days=LEARN_WINDOW_DAYS):
    """Load recent hit/near-hit memory and build per-slot learning scores and weights."""

    df = load_script_hit_memory(base_dir=Path(base_dir))
    if df is None or df.empty:
        print("[LEARNING] Hit-memory file not found or empty; proceeding without learning boost.")
        return {}, {}, {"window": window_days, "rows": 0}

    df = df.copy()
    date_col = _choose_date_column(df)
    if not date_col:
        print("[LEARNING] No valid date column in hit-memory; proceeding without learning boost.")
        return {}, {}, {"window": window_days, "rows": 0}

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    cutoff = datetime.now() - timedelta(days=window_days)
    df = df[df[date_col] >= cutoff]
    if df.empty:
        print("[LEARNING] No recent hit-memory data within window; skipping learning boost.")
        return {}, {}, {"window": window_days, "rows": 0}

    df["slot"] = df.get("slot").apply(_normalize_slot)
    df["script_id"] = df.get("script_id", "").astype(str).str.upper()
    df["hit_type"] = df.get("hit_type", "MISS").astype(str).str.upper()
    df["predicted_number"] = df.get("predicted_number", df.get("number")).apply(_normalize_number)
    df["is_exact_hit"] = df.get("is_exact_hit", False).fillna(False).astype(bool)
    df["is_near_hit"] = df.get("is_near_hit", False).fillna(False).astype(bool)
    df["is_near_miss"] = df.get("is_near_miss", False).fillna(False).astype(bool)

    df = df.dropna(subset=["slot", "script_id", "predicted_number"])

    learning_scores = defaultdict(dict)
    slot_script_weights = defaultdict(dict)
    near_hits = {"NEAR", "NEIGHBOR", "MIRROR", "CROSS_SLOT", "CROSS_DAY"}

    for slot in SLOTS:
        slot_df = df[df["slot"] == slot]
        if slot_df.empty:
            continue

        for num, group in slot_df.groupby("predicted_number"):
            count_exact = group["is_exact_hit"].sum()
            count_near = group["is_near_hit"].sum()
            if "hit_type" in group:
                counts = Counter(group["hit_type"])
                count_exact = max(count_exact, counts.get("EXACT", 0) + counts.get("DIRECT", 0))
                count_near = max(
                    count_near,
                    sum(counts.get(ht, 0) for ht in near_hits) + group["is_near_miss"].sum(),
                )
            learning_scores[slot][int(num)] = LEARN_WEIGHT_EXACT * count_exact + LEARN_WEIGHT_NEAR * count_near

        for script, group in slot_df.groupby("script_id"):
            total_preds = len(group)
            if total_preds == 0:
                continue
            hit_types = group["hit_type"].astype(str)
            exact_hits = group["is_exact_hit"].sum() + hit_types.isin({"EXACT", "DIRECT"}).sum()
            near_total = group["is_near_hit"].sum() + group["is_near_miss"].sum() + hit_types.isin(near_hits).sum()
            hit_rate_exact = exact_hits / total_preds
            near_miss_rate = near_total / total_preds
            blind_miss_rate = max(0.0, 1.0 - (hit_rate_exact + near_miss_rate))
            score = SCRIPT_SCORE_A * hit_rate_exact + SCRIPT_SCORE_B * near_miss_rate - SCRIPT_SCORE_C * blind_miss_rate
            slot_script_weights[slot][str(script).upper()] = score

    normalized_weights = defaultdict(dict)
    for slot, weights in slot_script_weights.items():
        if not weights:
            continue
        scores = list(weights.values())
        min_score, max_score = min(scores), max(scores)
        if np.isclose(min_score, max_score):
            normalized_weights[slot] = {s: 1.0 for s in weights.keys()}
        else:
            for script, score in weights.items():
                scaled = SCRIPT_WEIGHT_MIN + (score - min_score) * (SCRIPT_WEIGHT_MAX - SCRIPT_WEIGHT_MIN) / (max_score - min_score)
                normalized_weights[slot][script] = max(SCRIPT_WEIGHT_MIN, min(SCRIPT_WEIGHT_MAX, scaled))

    total_rows = len(df)
    if learning_scores:
        print(
            f"[LEARNING] Window={window_days}d, rows={total_rows}, "
            + ", ".join(
                f"{slot}: "
                + ", ".join(
                    f"{k}={v:.2f}" for k, v in sorted((normalized_weights.get(slot) or {}).items())
                )
                if normalized_weights.get(slot)
                else f"{slot}: none"
                for slot in SLOTS
            )
        )
    else:
        print("[LEARNING] No learning scores computed; proceeding without boost.")

    return learning_scores, normalized_weights, {"window": window_days, "rows": total_rows}

class UltimatePredictionEngine:
    """
    ULTIMATE PREDICTION ENGINE - WITH SPEED MODE
    ✓ Full mode: All scripts (24min) 
    ✓ Fast mode: Critical scripts only (10-12min)
    """
    
    def __init__(self, speed_mode='full'):
        self.slot_names = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
        self.speed_mode = speed_mode  # 'full' or 'fast'
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_weights = {}
        self.slot_script_weights = None
        self.script_weight_metrics = None
        self.script_weight_metrics_df = None
        self.metrics_weight_map = {}
        self.learning_scores, self.learning_slot_weights, self.learning_meta = load_learning_scores(self.base_dir)
        self.latest_score_details = {}
        self.s40_numbers = set(getattr(pattern_packs, "S40", []))
        self.family_164950 = {0, 1, 4, 5, 6, 9}
        # Cache to avoid rerunning heavy scripts (e.g., SCR6) for the same date/mode within a run
        self.script_prediction_cache = {}
        self.setup_directories()
        self.load_script_weight_preview()

    def _set_equal_weights_for_slot(self, slot: str):
        self.slot_script_weights = self.slot_script_weights or {}
        scripts = []
        if self.script_weight_metrics_df is not None and not self.script_weight_metrics_df.empty:
            slot_df = self.script_weight_metrics_df[self.script_weight_metrics_df.get("slot") == slot]
            if not slot_df.empty and "script_id" in slot_df.columns:
                scripts = [self._script_key(str(s)) for s in slot_df["script_id"].tolist()]
        self.slot_script_weights[slot] = {script: 1.0 for script in scripts}

    def setup_directories(self):
        """Create output folders"""
        base_dir = self.base_dir
        folders = [
            'outputs/predictions', 
            'outputs/analysis', 
            'logs/performance', 
            'logs/prediction_results',
            'predictions/deepseek_scr9',
            'predictions/deepseek_scr1',
            'predictions/deepseek_scr2', 
            'predictions/deepseek_scr3',
            'predictions/deepseek_scr4',
            'predictions/deepseek_scr5',
            'predictions/deepseek_scr6',
            'predictions/deepseek_scr7',
            'predictions/deepseek_scr8',
        ]
        for folder in folders:
            full_path = os.path.join(base_dir, folder)
            os.makedirs(full_path, exist_ok=True)

        self.move_old_files_to_logs()

    def load_script_weights(self):
        """Load performance-based script weights (defaults to 1.0)."""
        weight_files = [
            os.path.join(self.base_dir, "logs", "performance", "script_weights.json"),
            os.path.join(self.base_dir, "logs", "performance", "script_performance_summary.json"),
        ]
        weights = {}
        loaded = False

        for path in weight_files:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                script_rows = data.get("scripts") if isinstance(data, dict) else None
                if isinstance(data, dict) and not script_rows:
                    # Allow direct mapping of name → weight
                    weights.update({str(k).upper(): float(v) for k, v in data.items() if v is not None})
                    loaded = True
                    break
                for row in script_rows or []:
                    name = str(row.get("name", "")).upper()
                    weight_val = row.get("weight", 1.0)
                    if not weight_val:
                        weight_val = 1.0
                    weights[name] = weight_val
                if weights:
                    loaded = True
                    break
            except Exception:
                continue

        if not loaded:
            print("⚠️  Script weight file missing; using uniform weights.")
        return weights

    def load_script_weight_preview(self):
        """Load script metrics and build preview weights when enabled."""

        try:
            base_learning_weights = self.learning_slot_weights if hasattr(self, "learning_slot_weights") else {}

            metrics_df, summary = get_metrics_table(
                window_days=SCRIPT_WEIGHTS_WINDOW_DAYS,
                base_dir=Path(self.base_dir),
                mode="per_slot",
            )

            self.script_weight_metrics_df = metrics_df
            self.script_weight_metrics = {
                "rows": len(metrics_df) if metrics_df is not None else 0,
                "window_days": (summary or {}).get("effective_window_days", SCRIPT_WEIGHTS_WINDOW_DAYS),
            }

            if metrics_df is None or metrics_df.empty:
                print("Script metrics warming up – skipping script-wise performance overlay.")
                for slot in SLOTS:
                    print(f"[WARN] No hero/weak metrics for slot {slot}, using equal weights")
                    self._set_equal_weights_for_slot(slot)
                self.slot_script_weights = self.slot_script_weights or {}
            else:
                weights = build_script_weights_by_slot(
                    metrics_df,
                    min_samples=30,
                    min_score=SCRIPT_WEIGHTS_MIN_SCORE,
                )
                if not weights:
                    print("[WARN] No eligible script weights derived; using equal weights per slot")
                    for slot in SLOTS:
                        self._set_equal_weights_for_slot(slot)
                    self.slot_script_weights = self.slot_script_weights or {}
                else:
                    self.slot_script_weights = weights

            self._print_weight_preview()
        except Exception as e:
            print(f"⚠️  Script weight preview unavailable: {e}")
            self.slot_script_weights = {}
            self.script_weight_metrics = None
            self.script_weight_metrics_df = None

        if base_learning_weights:
            merged = {}
            for slot in SLOTS:
                learning_weights = base_learning_weights.get(slot, {})
                current_weights = (self.slot_script_weights or {}).get(slot, {})
                if learning_weights:
                    if current_weights:
                        combo = {}
                        for script in set(learning_weights.keys()) | set(current_weights.keys()):
                            combo[script] = learning_weights.get(script, 1.0) * current_weights.get(script, 1.0)
                        total = sum(combo.values()) or 1.0
                        merged[slot] = {k: v / total for k, v in combo.items()}
                    else:
                        merged[slot] = learning_weights
            if merged:
                self.slot_script_weights = merged

    def _compute_slot_weights(self, metrics_for_slot):
        weights = {}
        scores = {}
        for script, metrics in metrics_for_slot.items():
            hit_rate = metrics.get("hit_rate", 0.0)
            blind_miss_rate = metrics.get("blind_miss_rate", 1.0)
            base = max(0.1, hit_rate)
            penalty = max(0.0, blind_miss_rate - 0.5) * 0.5
            score = max(SCRIPT_WEIGHTS_MIN_SCORE, base - penalty)
            scores[script] = score
        total = sum(scores.values()) or 1.0
        for script, score in scores.items():
            weights[script] = score / total
        return weights

    def _print_weight_preview(self):
        if self.script_weight_metrics_df is None or self.script_weight_metrics_df.empty:
            for slot in SLOTS:
                print(f"  No hero/weak for slot {slot}; using equal weights.")
            return
        slot_bands = hero_weak_table(self.script_weight_metrics_df)
        window_used = SCRIPT_WEIGHTS_WINDOW_DAYS
        if isinstance(self.script_weight_metrics, dict):
            window_used = self.script_weight_metrics.get("window_days", window_used)
        status = "ENABLED" if USE_SCRIPT_WEIGHTS else "DISABLED (preview only)"
        print(f"SCRIPT WEIGHT PREVIEW (last {window_used} days) [{status}]:")
        if slot_bands is None or slot_bands.empty:
            for slot in SLOTS:
                print(f"[WARN] No hero/weak metrics for slot {slot}, using equal weights")
                self._set_equal_weights_for_slot(slot)
        else:
            self._apply_hero_weak_tilt(slot_bands)
            for _, row in slot_bands.iterrows():
                hero = row.get("hero_script") or "n/a"
                weak = row.get("weak_script") or "n/a"
                slot = row.get("slot")
                if hero == "n/a" and weak == "n/a":
                    print(f"[WARN] No hero/weak metrics for slot {slot}, using equal weights")
                    self._set_equal_weights_for_slot(slot)
                    continue
                print(f"  {slot}: hero=[{hero}] weak=[{weak}] window={window_used}d")
                if self.slot_script_weights and slot in self.slot_script_weights:
                    weight_line = ", ".join(
                        f"{k}={v:.2f}" for k, v in self.slot_script_weights.get(slot, {}).items()
                    )
                    print(f"     weights: {weight_line}")
        league = build_script_league(self.script_weight_metrics_df, min_predictions=5)
        if league:
            overall = league.get("overall") or {}
            hero_overall = overall.get("hero", {}) if isinstance(overall, dict) else {}
            weak_overall = overall.get("weak", {}) if isinstance(overall, dict) else {}
            hero_label = hero_overall.get("script_id") or "n/a"
            weak_label = weak_overall.get("script_id") or "n/a"
            print(f"  Overall heroes=[{hero_label}] weak=[{weak_label}] window_rows={league.get('window_rows')}")

    def _apply_hero_weak_tilt(self, slot_bands: pd.DataFrame) -> None:
        if not USE_SCRIPT_WEIGHTS:
            return
        if self.slot_script_weights is None:
            return

        tilt_factors = {"hero": 1.3, "weak": 0.7}
        for _, row in slot_bands.iterrows():
            slot = str(row.get("slot")) if row.get("slot") is not None else None
            if not slot:
                continue
            slot = slot.upper()
            weights = self.slot_script_weights.get(slot)
            if not weights:
                continue
            hero_script = str(row.get("hero_script") or "").upper()
            weak_script = str(row.get("weak_script") or "").upper()
            adjusted = {}
            for script, weight in weights.items():
                factor = 1.0
                if script.upper() == hero_script:
                    factor = tilt_factors["hero"]
                elif script.upper() == weak_script:
                    factor = tilt_factors["weak"]
                adjusted[script] = weight * factor

            total = sum(adjusted.values()) or 1.0
            self.slot_script_weights[slot] = {k: v / total for k, v in adjusted.items()}

    def _script_key(self, script_name: str) -> str:
        match = re.search(r"scr(\d+)", script_name, re.IGNORECASE)
        if match:
            return f"SCR{int(match.group(1))}"
        return script_name.upper()
    
    def move_old_files_to_logs(self):
        """Move old prediction files to organized folders"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        movement_rules = [
            (['ultimate_*.xlsx', 'ultimate_*.txt', 'ultimate_*.csv'], 'logs/prediction_results'),
            (['ultimate_predictions.xlsx', 'ultimate_detailed_predictions.xlsx', 'bet_plan.xlsx'], 'predictions/deepseek_scr2'),
            (['advanced_analysis_*.txt', 'advanced_detailed_*.xlsx', 'prediction_diagnostic*.xlsx'], 'predictions/deepseek_scr7'),
            (['scr10_predictions_*.xlsx', 'scr10_detailed_*.xlsx', 'scr10_analysis_*.txt', 'scr10_diagnostic*.xlsx'], 'predictions/deepseek_scr8'),
            (['scr10_performance*.csv'], 'logs/performance'),
            (['ultimate_performance*.csv'], 'logs/performance'),
        ]
        
        print("🧹 Organizing files into structured folders...")
        
        for patterns, dest_folder in movement_rules:
            dest_dir = os.path.join(base_dir, dest_folder)
            os.makedirs(dest_dir, exist_ok=True)
            
            for pattern in patterns:
                for file_path in glob.glob(os.path.join(base_dir, pattern)):
                    if os.path.isfile(file_path):
                        try:
                            filename = os.path.basename(file_path)
                            dest_path = os.path.join(dest_dir, filename)
                            
                            if os.path.exists(dest_path):
                                name, ext = os.path.splitext(filename)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                dest_path = os.path.join(dest_dir, f"{name}_{timestamp}{ext}")
                            
                            os.rename(file_path, dest_path)
                            print(f"📁 Moved: {filename} -> {dest_folder}/")
                        except Exception as e:
                            print(f"⚠️ Could not move {file_path}: {e}")
    
    def load_data(self, file_path):
        """Load Excel data using shared quant_data_core loader"""
        print("📂 Loading Excel file via quant_data_core...")
        try:
            df = quant_data_core.load_results_dataframe()
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
            return None

        required_cols = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Results DataFrame missing required columns: {missing}")
            return None

        slot_mapping = [
            ("FRBD", 1),
            ("GZBD", 2),
            ("GALI", 3),
            ("DSWR", 4),
        ]

        all_data = []

        for _, row in df.iterrows():
            date_val = row["DATE"]
            if pd.isna(date_val):
                continue

            try:
                date_val = pd.to_datetime(date_val)
            except Exception:
                continue

            for col_name, slot_idx in slot_mapping:
                raw_val = row.get(col_name)
                if pd.isna(raw_val):
                    continue

                s = str(raw_val).strip()
                if not s or s.upper() == "XX":
                    continue

                try:
                    num = int(float(s)) % 100
                except Exception:
                    continue

                all_data.append({
                    "date": date_val,
                    "slot": slot_idx,
                    "number": num,
                })

        df_clean = pd.DataFrame(all_data)
        if df_clean.empty:
            print("❌ No valid data found after parsing results DataFrame")
            return None

        df_clean["date"] = pd.to_datetime(df_clean["date"])
        df_clean = df_clean.sort_values(["date", "slot"]).reset_index(drop=True)

        start_date = df_clean["date"].min().strftime("%Y-%m-%d")
        end_date = df_clean["date"].max().strftime("%Y-%m-%d")
        print(f"✅ Loaded {len(df_clean)} records from {start_date} to {end_date}")

        return df_clean
    
    def get_opposite(self, n):
        """Get opposite number"""
        if n < 10:
            return n * 10
        else:
            return (n % 10) * 10 + (n // 10)
    
    def run_child_script(self, script_name, target_date=None, mode=None):
        """Run a child script and parse its predictions with optional caching"""
        try:
            cache_key = None
            timeout_seconds = 300

            if script_name.lower() == 'deepseek_scr6.py':
                # Heavier model ensemble – cache per target_date + mode within the same run
                cache_key = (script_name, str(target_date) if target_date else None, mode or 'default')
                if cache_key in self.script_prediction_cache:
                    print(
                        "   🔁 Using cached predictions for deepseek_scr6.py "
                        f"({cache_key[1]}, mode={cache_key[2]})"
                    )
                    cached_preds = self.script_prediction_cache[cache_key]
                    return {slot: list(nums) for slot, nums in cached_preds.items()}
                timeout_seconds = None  # Allow generous runtime for SCR6 without killing the process

            result = subprocess.run(
                ['py', '-3.12', script_name],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            
            predictions = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
            output_lines = result.stdout.split('\n')
            
            for line in output_lines:
                line = line.strip()
                
                for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
                    if slot in line and (':' in line or '(' in line):
                        if ':' in line:
                            numbers_part = line.split(':', 1)[1]
                        elif '(' in line:
                            numbers_part = line.split(')', 1)[1]
                        else:
                            continue
                            
                        numbers = re.findall(r'\b\d{2}\b', numbers_part)
                        for num_str in numbers:
                            try:
                                num = int(num_str)
                                if 0 <= num <= 99 and num not in predictions[slot]:
                                    predictions[slot].append(num)
                            except:
                                continue
                
                if "Hot numbers:" in line:
                    hot_numbers = re.findall(r"'(\d+)", line)
                    for num_str in hot_numbers:
                        try:
                            num = int(num_str) % 100
                            for slot in predictions:
                                if num not in predictions[slot]:
                                    predictions[slot].append(num)
                        except:
                            continue
            
            if cache_key:
                # Store a shallow copy to protect cache from accidental mutation
                cached_copy = {slot: list(nums) for slot, nums in predictions.items()}
                self.script_prediction_cache[cache_key] = cached_copy

            return predictions

        except subprocess.TimeoutExpired:
            print(f"   ⚠️  {script_name} timed out after {timeout_seconds} seconds")
            return {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        except Exception as e:
            print(f"   ⚠️  {script_name} failed: {e}")
            return {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}

    def collect_all_script_predictions(self, target_date=None, mode=None):
        """Collect predictions from all available scripts - OPTIMIZED FOR SPEED MODE"""
        if VERBOSE_BACKTEST:
            print("🎯 Collecting predictions from scripts...")

            if self.speed_mode == 'fast':
                print("   ⚡ FAST MODE: Running SCR1–SCR8 once for tomorrow only")
            else:
                print("   🐢 FULL MODE: Running all scripts")

        scripts = [
            'deepseek_scr1.py',
            'deepseek_scr2.py',
            'deepseek_scr3.py',
            'deepseek_scr4.py',
            'deepseek_scr5.py',
            'deepseek_scr6.py',
            'deepseek_scr7.py',
            'deepseek_scr8.py',
        ]
        
        all_predictions = {}
        script_times = {}
        
        for script in scripts:
            if os.path.exists(script):
                start_time = time.time()
                preds = self.run_child_script(script, target_date=target_date, mode=mode)
                end_time = time.time()
                script_times[script] = end_time - start_time
                all_predictions[script] = preds
            else:
                if VERBOSE_BACKTEST:
                    print(f"   ⚠️  {script} not found")

        # Print timing summary
        if VERBOSE_BACKTEST and script_times:
            print("   ⏰ Script execution times:")
            for script, exec_time in script_times.items():
                print(f"     {script}: {exec_time:.1f}s")
        
        return all_predictions
    
    def build_slot_scores(self, all_preds_for_slot, history_df_for_slot, slot_name=None):
        """Build ensemble scores for a slot - OPTIMIZED"""
        base_scores = Counter()
        use_weights = bool(USE_SCRIPT_WEIGHTS and self.slot_script_weights)

        # Process each script's predictions
        for script_name, predictions in all_preds_for_slot.items():
            slot_key = slot_name.upper() if slot_name else None
            weight = 1.0
            if use_weights and slot_key:
                slot_weights = self.slot_script_weights.get(slot_key) or {}
                weight = slot_weights.get(self._script_key(script_name), 1.0)

            for rank, number in enumerate(predictions, 1):
                if rank == 1:
                    rank_weight = 5
                elif rank == 2:
                    rank_weight = 4
                elif rank == 3:
                    rank_weight = 3
                elif rank == 4:
                    rank_weight = 2
                else:
                    rank_weight = 1

                base_scores[number] += rank_weight * weight
        
        # Frequency bonus (optimized for speed)
        frequency_bonus = Counter()
        for script_name, predictions in all_preds_for_slot.items():
            for number in predictions[:8]:  # Reduced from 10 to 8 in fast mode
                frequency_bonus[number] += 1
        
        for number, freq in frequency_bonus.items():
            if freq >= 2:  # Reduced threshold in fast mode
                base_scores[number] += freq * 2

        # Opposite number bonus
        high_score_numbers = [num for num, score in base_scores.most_common(8)]  # Reduced from 10
        for number in high_score_numbers:
            opposite = self.get_opposite(number)
            base_scores[opposite] += 3

        # Recent actual results bonus
        if len(history_df_for_slot) > 0:
            recent_numbers = history_df_for_slot['number'].tail(3).tolist()
            for number in recent_numbers:
                base_scores[number] += 2
                base_scores[self.get_opposite(number)] += 1

        slot_key = slot_name.upper() if slot_name else None
        learning_map = self.learning_scores.get(slot_key, {}) if slot_key else {}
        final_scores = Counter()
        score_details = {}

        for number, base_score in base_scores.items():
            learning_score = learning_map.get(number, 0.0)
            final_score = base_score + LEARN_SCORE_ALPHA * learning_score
            final_scores[number] = final_score
            score_details[number] = {
                "base_score": base_score,
                "learning_score": learning_score,
                "final_score": final_score,
            }

        return final_scores, score_details

    def apply_number_booster(self, slot_name, candidate_scores, score_details, df_history, target_date):
        if not candidate_scores:
            return candidate_scores, score_details

        slot_idx = next((k for k, v in self.slot_names.items() if v == slot_name), None)
        if slot_idx is None:
            return candidate_scores, score_details

        window_start = target_date - timedelta(days=BOOSTER_WINDOW_DAYS)
        hist_window = df_history[(df_history["date"] >= window_start) & (df_history["date"] < target_date)]
        if hist_window.empty:
            return candidate_scores, score_details

        slot_hist = hist_window[hist_window["slot"] == slot_idx]
        other_slots = hist_window[hist_window["slot"] != slot_idx]
        slot_counts = Counter(slot_hist.get("number", []))
        cross_slot_dates = other_slots[other_slots.get("number").isin(list(candidate_scores.keys()))]

        adjusted_scores = {}
        boosters = {}

        for num, base_score in candidate_scores.items():
            mirror = int(str(num).zfill(2)[::-1])
            neighbors = {(num + 1) % 100, (num - 1) % 100}
            exact_hits = slot_counts.get(num, 0)
            near_hits = slot_counts.get(mirror, 0) + sum(slot_counts.get(n, 0) for n in neighbors)
            cross_hits = 0
            if not cross_slot_dates.empty:
                cross_hits = len(cross_slot_dates[cross_slot_dates.get("number") == num])

            is_s40 = 1 if num in self.s40_numbers else 0
            digits = {num // 10, num % 10}
            is_family = 1 if digits.issubset(self.family_164950) else 0

            booster = (
                BOOST_ALPHA * exact_hits
                + BOOST_BETA * (near_hits + cross_hits)
                + BOOST_GAMMA * is_s40
                + BOOST_DELTA * is_family
            )

            final_score = base_score + booster
            adjusted_scores[num] = final_score
            boosters[num] = booster
            detail = score_details.get(num, {}) if score_details else {}
            detail.update({"booster": booster, "final_score": final_score, "base_score": base_score})
            score_details[num] = detail

        if boosters and VERBOSE_BACKTEST:
            top_boosted = sorted(boosters.items(), key=lambda x: x[1], reverse=True)[:3]
            boost_line = ", ".join(
                f"{num:02d} (+{boosters[num]:.1f}→{adjusted_scores[num]:.1f})" for num, _ in top_boosted
            )
            print(f"[BOOST] {slot_name}: top boosters {boost_line}")

        return adjusted_scores, score_details
    
    def predict_for_target_date(self, df_history, target_date):
        """Generate predictions for a specific target date - OPTIMIZED"""
        print(f"[SCR11-BACKTEST] Testing {target_date.date()}... (SCR1–SCR8 bundle)")

        # Collect predictions from all scripts
        all_script_preds = self.collect_all_script_predictions(target_date=target_date, mode='predict_for_date')

        print(f"   [SCR11] {target_date.date()}: SCR1–SCR8 executed.")
        
        # Reorganize by slot
        slot_predictions = {}
        for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
            slot_preds = {}
            for script_name, predictions in all_script_preds.items():
                slot_preds[script_name] = predictions[slot]
            slot_predictions[slot] = slot_preds
        
        # Build final predictions for each slot
        final_predictions = {}
        slot_score_details = {}
        
        for slot_name in ["FRBD", "GZBD", "GALI", "DSWR"]:
            slot_data = df_history[df_history['slot'] == list(self.slot_names.keys())[list(self.slot_names.values()).index(slot_name)]]
            numbers = slot_data['number'].tolist()
            
            if len(numbers) < 5:
                # Fallback: simple frequency
                freq = Counter(numbers)
                final_predictions[slot_name] = [num for num, count in freq.most_common(15)]
            else:
                # Use ensemble scoring
                scores, score_details = self.build_slot_scores(
                    slot_predictions[slot_name], slot_data, slot_name=slot_name
                )
                slot_score_details[slot_name] = score_details
                
                # Dynamic top-k selection (optimized)
                if scores:
                    top_scores = [score for num, score in scores.most_common(8)]  # Reduced from 10
                    if len(top_scores) > 1:
                        score_ratio = top_scores[0] / top_scores[1] if top_scores[1] > 0 else 10
                        if score_ratio > 2.0:
                            top_k = 5
                        elif score_ratio > 1.5:
                            top_k = 10
                        else:
                            top_k = 15
                    else:
                        top_k = 15
                else:
                    top_k = 15
                
                # Apply range diversity
                candidates = [num for num, score in scores.most_common(top_k * 2)]
                candidate_scores = {
                    num: score_details.get(num, {}).get("final_score", scores[num])
                    for num in candidates
                }
                boosted_scores, score_details = self.apply_number_booster(
                    slot_name, candidate_scores, score_details, df_history, target_date
                )
                final_pred = self.apply_diversity_filter(boosted_scores, top_k)
                final_predictions[slot_name] = final_pred

        self.latest_score_details = slot_score_details
        return final_predictions
    
    def apply_diversity_filter(self, scores, top_k):
        """Ensure diversity across ranges"""
        if not scores:
            return list(range(top_k))
        
        sorted_preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [num for num, _ in sorted_preds[:top_k * 2]]
        
        # Group by ranges
        ranges = {
            'low': [n for n in candidates if 0 <= n <= 33],
            'medium': [n for n in candidates if 34 <= n <= 66],
            'high': [n for n in candidates if 67 <= n <= 99]
        }
        
        selected = []
        
        # Take top from each range
        for rng in ['low', 'medium', 'high']:
            if ranges[rng]:
                selected.append(ranges[rng][0])
        
        # Fill remaining
        for num in candidates:
            if num not in selected and len(selected) < top_k:
                selected.append(num)
        
        return selected[:top_k]
    
    def backtest_recent_days(self, df, days=30):
        """Backtest on recent complete days - OPTIMIZED FOR SPEED MODE"""
        print(f"\n📊 Running backtest for last {days} days...")
        
        # SPEED MODE OPTIMIZATION: Reduce backtest days
        if self.speed_mode == 'fast':
            days = 1  # Only test most recent day in fast mode
            print("   🚀 SPEED MODE: Testing only most recent day")
        
        try:
            # Find dates with all four slots
            date_counts = df.groupby('date')['slot'].nunique()
            complete_dates = date_counts[date_counts == 4].index.tolist()
            complete_dates.sort()
            
            if len(complete_dates) < days:
                print(f"⚠️  Only {len(complete_dates)} complete dates found, using all")
                test_dates = complete_dates[-days:] if len(complete_dates) > 0 else []
            else:
                test_dates = complete_dates[-days:]
            
            if not test_dates:
                print("❌ No complete dates found for backtesting")
                return pd.DataFrame()
            
            backtest_results = []
            
            for test_date in test_dates:
                test_date_str = test_date.strftime('%Y-%m-%d')
                print(f"   → {test_date.date()}")
                
                # Split data
                history_df = df[df['date'] < test_date]
                actual_df = df[df['date'] == test_date]
                
                if len(history_df) == 0:
                    continue
                
                # Get predictions for this test date
                predictions = self.predict_for_target_date(history_df, test_date)
                
                # Compare with actuals
                for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
                    actual_data = actual_df[actual_df['slot'] == list(self.slot_names.keys())[list(self.slot_names.values()).index(slot)]]
                    if not actual_data.empty:
                        actual_number = actual_data['number'].iloc[0]
                        pred_numbers = predictions.get(slot, [])
                        
                        # Find rank of actual number
                        rank_missing_reason = ""
                        rank = None
                        if actual_number in pred_numbers:
                            rank = pred_numbers.index(actual_number) + 1
                        else:
                            rank = 999
                            rank_missing_reason = "not_in_predictions"
                        
                        # Check hits
                        hit_top1 = rank == 1 if rank else False
                        hit_top5 = rank <= 5 if rank else False
                        hit_top10 = rank <= 10 if rank else False
                        hit_top15 = rank <= 15 if rank else False
                        
                        backtest_results.append({
                            'date': test_date_str,
                            'slot': slot,
                            'actual': actual_number,
                            'predictions': ','.join(map(str, pred_numbers)),
                            'rank': rank,
                            'rank_missing_reason': rank_missing_reason,
                            'hit_top1': hit_top1,
                            'hit_top5': hit_top5,
                            'hit_top10': hit_top10,
                            'hit_top15': hit_top15
                        })

                if not VERBOSE_BACKTEST:
                    print(
                        f"[SCR11] Backtest {test_date.date()}: done (FRBD,GZBD,GALI,DSWR boosts applied internally)"
                    )
            
            results_df = pd.DataFrame(backtest_results)
            
            # Save to performance log
            perf_file = 'logs/performance/ultimate_performance.csv'
            if not results_df.empty:
                if "rank" in results_df.columns:
                    missing_mask = results_df["rank"].isna()
                    results_df["rank_missing"] = missing_mask.astype(int)
                    results_df["rank"] = results_df["rank"].fillna(999).astype(int)
                if "rank_missing_reason" in results_df.columns:
                    results_df["rank_missing_reason"] = results_df["rank_missing_reason"].fillna("")
                if os.path.exists(perf_file):
                    existing_df = pd.read_csv(perf_file)
                    if "rank" in existing_df.columns:
                        existing_df["rank"] = existing_df["rank"].fillna(999).astype(int)
                    if "rank_missing" in existing_df.columns:
                        existing_df["rank_missing"] = existing_df["rank_missing"].fillna(0).astype(int)
                    else:
                        existing_df["rank_missing"] = existing_df.get("rank", pd.Series(dtype=int)).isna().astype(int)
                    if "rank_missing_reason" in existing_df.columns:
                        existing_df["rank_missing_reason"] = existing_df["rank_missing_reason"].fillna("")
                    combined_df = pd.concat([existing_df, results_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['date', 'slot'], keep='last')
                    combined_df.to_csv(perf_file, index=False)
                else:
                    results_df.to_csv(perf_file, index=False)
                print(f"✅ Backtest results saved to {perf_file}")
            else:
                print("⚠️ No backtest rows to save")
            
            return results_df
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return pd.DataFrame()
    
    def detect_status(self, df):
        """Detect current date status"""
        latest_date = df['date'].max()
        today_data = df[df['date'] == latest_date]
        filled_slots = set(today_data['slot'].tolist())
        empty_slots = [s for s in [1, 2, 3, 4] if s not in filled_slots]
        return latest_date, filled_slots, empty_slots
    
    def generate_predictions(self, df):
        """Generate complete predictions"""
        latest_date, filled_slots, empty_slots = self.detect_status(df)
        
        print(f"\n📅 Latest Date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"✅ Filled: {[self.slot_names[s] for s in filled_slots]}")
        print(f"❌ Empty: {[self.slot_names[s] for s in empty_slots]}")
        
        predictions = []
        
        # TODAY'S EMPTY SLOTS (skip in FAST mode)
        if self.speed_mode != 'fast' and empty_slots:
            print(f"\n🎯 Predicting TODAY's empty slots...")
            date_str = latest_date.strftime('%Y-%m-%d')

            for slot in empty_slots:
                slot_name = self.slot_names[slot]
                slot_data = df[df['slot'] == slot]
                numbers = slot_data['number'].tolist()

                # Use ensemble prediction
                all_script_preds = self.collect_all_script_predictions(target_date=latest_date, mode='today_empty')
                slot_preds = {}
                for script_name, preds in all_script_preds.items():
                    slot_preds[script_name] = preds[slot_name]

                scores, score_details = self.build_slot_scores(slot_preds, slot_data, slot_name=slot_name)
                pred_nums = [num for num, score in scores.most_common(15)]
                opposites = [self.get_opposite(n) for n in pred_nums[:3]]

                for rank, num in enumerate(pred_nums, 1):
                    predictions.append({
                        'date': date_str,
                        'slot': slot_name,
                        'number': f"{num:02d}",
                        'rank': rank,
                        'type': 'TODAY_EMPTY',
                        'base_score': score_details.get(num, {}).get('base_score'),
                        'learning_score': score_details.get(num, {}).get('learning_score'),
                        'final_score': score_details.get(num, {}).get('final_score'),
                        'opposites': ', '.join([f"{n:02d}" for n in opposites]) if rank == 1 else ''
                    })
        elif self.speed_mode == 'fast' and empty_slots:
            print("\n⚡ FAST MODE: Skipping today's empty-slot predictions")
        
        # TOMORROW'S ALL SLOTS
        tomorrow = latest_date + timedelta(days=1)
        date_str = tomorrow.strftime('%Y-%m-%d')
        print(f"\n🎯 Predicting TOMORROW ({date_str})...")
        
        # Use the main prediction function
        tomorrow_preds = self.predict_for_target_date(df, tomorrow)
        
        for slot_name, pred_numbers in tomorrow_preds.items():
            opposites = [self.get_opposite(n) for n in pred_numbers[:3]]
            slot_details = self.latest_score_details.get(slot_name, {}) if hasattr(self, 'latest_score_details') else {}

            for rank, num in enumerate(pred_numbers, 1):
                details = slot_details.get(num, {})
                predictions.append({
                    'date': date_str,
                    'slot': slot_name,
                    'number': f"{num:02d}",
                    'rank': rank,
                    'type': 'TOMORROW',
                    'base_score': details.get('base_score'),
                    'learning_score': details.get('learning_score'),
                    'final_score': details.get('final_score'),
                    'opposites': ', '.join([f"{n:02d}" for n in opposites]) if rank == 1 else ''
                })
        
        return pd.DataFrame(predictions)
    
    def create_outputs(self, predictions_df, df, backtest_results):
        """Create output files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        scr9_pred_dir = os.path.join(base_dir, "predictions", "deepseek_scr9")
        
        # Ensure the directory exists
        os.makedirs(scr9_pred_dir, exist_ok=True)
        
        # Wide format
        wide_data = []
        for date in predictions_df['date'].unique():
            date_data = {
                'date': date,
                'type': predictions_df[predictions_df['date'] == date]['type'].iloc[0]
            }
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                slot_data = predictions_df[
                    (predictions_df['date'] == date) & 
                    (predictions_df['slot'] == slot)
                ]
                if not slot_data.empty:
                    nums = slot_data['number'].tolist()[:15]
                    date_data[slot] = ', '.join(nums)
                    opp = slot_data['opposites'].iloc[0]
                    if pd.notna(opp) and opp:
                        date_data[f'{slot}_OPP'] = opp
            wide_data.append(date_data)
        
        wide_df = pd.DataFrame(wide_data)
        
        # Define file paths in the dedicated SCR9 folder
        pred_file = os.path.join(scr9_pred_dir, f'ultimate_predictions_{timestamp}.xlsx')
        detail_file = os.path.join(scr9_pred_dir, f'ultimate_detailed_{timestamp}.xlsx')
        diag_file = os.path.join(scr9_pred_dir, f'ultimate_diagnostic_{timestamp}.xlsx')
        analysis_file = os.path.join(scr9_pred_dir, f'ultimate_analysis_{timestamp}.txt')
        
        # Save files to dedicated folder
        wide_df.to_excel(pred_file, index=False)
        predictions_df.to_excel(detail_file, index=False)
        
        # Create diagnostic file
        diagnostic_data = []
        for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
            slot_data = df[df['slot'] == list(self.slot_names.keys())[list(self.slot_names.values()).index(slot)]]
            if len(slot_data) > 0:
                recent = slot_data['number'].tail(10).tolist()
                freq = Counter(slot_data['number'])
                hot_numbers = [num for num, count in freq.most_common(5)]
                
                diagnostic_data.append({
                    'slot': slot,
                    'recent_numbers': ', '.join([f'{n:02d}' for n in recent]),
                    'hot_numbers': ', '.join([f'{n:02d}' for n in hot_numbers]),
                    'total_records': len(slot_data)
                })
        
        diag_df = pd.DataFrame(diagnostic_data)
        diag_df.to_excel(diag_file, index=False)
        
        # Report
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("  🎯 SCR11 ULTIMATE PREDICTION ENGINE - REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Speed Mode: {self.speed_mode.upper()}\n")
            f.write(f"Records: {len(df)}\n")
            f.write(f"Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n\n")
            
            # Backtest summary
            if not backtest_results.empty:
                f.write("BACKTEST SUMMARY:\n")
                f.write("-" * 50 + "\n")
                for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    slot_results = backtest_results[backtest_results['slot'] == slot]
                    if not slot_results.empty:
                        total = len(slot_results)
                        hit_top5 = slot_results['hit_top5'].sum()
                        f.write(f"  {slot}: {hit_top5}/{total} hit_top5 ({hit_top5/total*100:.1f}%)\n")
                f.write("\n")
            
            # Predictions
            f.write("PREDICTIONS:\n")
            f.write("-" * 50 + "\n")
            for date in wide_df['date'].unique():
                row = wide_df[wide_df['date'] == date].iloc[0]
                f.write(f"\n{date} ({row['type']}):\n")
                for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                    if slot in row:
                        f.write(f"  {slot}: {row[slot]}\n")
                        if f'{slot}_OPP' in row:
                            f.write(f"       Opp: {row[f'{slot}_OPP']}\n")

            # Script weights used
            f.write("\nSCRIPT WEIGHTS USED:\n")
            f.write("-" * 50 + "\n")
            for idx in range(1, 10):
                key = f"SCR{idx}"
                weight_val = self.script_weights.get(key, 1.0)
                f.write(f"  {key}: {weight_val:.2f}\n")

        # Return file paths for console display
        return wide_df, pred_file, detail_file, diag_file, analysis_file
    
    def run(self, file_path):
        """Main execution"""
        print("=" * 70)
        print(f"  🎯 SCR11 ULTIMATE PREDICTION ENGINE - {self.speed_mode.upper()} MODE")
        print("  ✓ TRUE SCR1-10 Integration + Backtesting + Organized Output")
        print(f"  Mode: {self.speed_mode.upper()}")
        print("=" * 70)
        
        start_time = time.time()
        
        df = self.load_data(file_path)
        if df is None:
            return
        
        # Add slot names for analysis
        df['slot_name'] = df['slot'].map(self.slot_names)
        
        print(f"\n📊 Data Summary:")
        for slot in [1, 2, 3, 4]:
            slot_data = df[df['slot'] == slot]
            print(f"   {self.slot_names[slot]}: {len(slot_data)} records")
        
        # Run backtest (skip entirely in FAST mode)
        if self.speed_mode == 'fast':
            print("\n🏎 FAST MODE: Skipping 30-day backtest (using existing reality metrics)")
            backtest_results = pd.DataFrame()
        else:
            backtest_results = self.backtest_recent_days(df, days=30)

        # Print backtest summary
        if not backtest_results.empty:
            print(f"\n📊 Backtest (last {len(backtest_results['date'].unique())} days):")
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                slot_results = backtest_results[backtest_results['slot'] == slot]
                if not slot_results.empty:
                    total = len(slot_results)
                    hit_top5 = slot_results['hit_top5'].sum()
                    print(f"   {slot}: {hit_top5}/{total} hit_top5 ({hit_top5/total*100:.1f}%)")
        elif self.speed_mode == 'fast':
            print("\n📊 Backtest metrics unavailable in FAST mode (skipped to save time)")

        if self.speed_mode == 'fast':
            print("\n⚡ FAST MODE: Only generating TOMORROW predictions (no historical backtest)")
        else:
            print("\n🐢 FULL MODE: Running backtest + TOMORROW predictions")

        # Generate predictions
        predictions_df = self.generate_predictions(df)
        wide_df, pred_file, detail_file, diag_file, analysis_file = self.create_outputs(predictions_df, df, backtest_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print("  📊 RESULTS")
        print("=" * 70)
        
        for date in wide_df['date'].unique():
            row = wide_df[wide_df['date'] == date].iloc[0]
            print(f"\n📅 {date} ({row['type']}):")
            print("-" * 70)
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                if slot in row:
                    print(f"  {slot:5s}: {row[slot]}")
                    if f'{slot}_OPP' in row:
                        print(f"         Opp: {row[f'{slot}_OPP']}")
        
        print("\n" + "=" * 70)
        print("✅ Files saved:")
        print(f"   - {os.path.relpath(pred_file)}")
        print(f"   - {os.path.relpath(detail_file)}")
        print(f"   - {os.path.relpath(diag_file)}")
        print(f"   - {os.path.relpath(analysis_file)}")
        print(f"   - logs/performance/ultimate_performance.csv")
        print(f"\n⏰ Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print("=" * 70)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SCR11 Ultimate Prediction Engine')
    parser.add_argument('--speed-mode', choices=['full', 'fast'], default='full',
                       help='Speed mode: full (backtest + tomorrow) or fast (tomorrow only)')
    parser.add_argument(
        '--verbose-backtest',
        action='store_true',
        help='Show detailed BOOST logs during backtest',
    )

    args = parser.parse_args()
    VERBOSE_BACKTEST = args.verbose_backtest

    predictor = UltimatePredictionEngine(speed_mode=args.speed_mode)
    predictor.run('number prediction learn.xlsx')
