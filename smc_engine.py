"""A+ SMC 4-Rule Engine (Revized & Flexible).

Rule 1: M15 Liquidity Sweep (Sessions + Major M15 Swing Levels)
Rule 2: M5 CHOCH (Expanded 50-candle Lookback)
Rule 3: M5 FVG Detection
Rule 4: R:R Validation (>= 1:2)
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SWEEP_BUFFER_PCT = 0.00005  # Standardize buffer
CHOCH_LOOKBACK = 50        # Extended to 50 candles (approx 4 hours on M5)
MIN_RR = 2.0

# Dynamic/Realistic FVG min width for Gold
FVG_MIN_WIDTH: Dict[str, float] = {
    "XAUUSD": 0.20,
    "XAUUSD=X": 0.20,
    "GC=F": 0.20,
}

SESSIONS_UTC = [
    ("London", 7, 0, 11, 0),
    ("NewYork", 12, 30, 17, 0),
]


@dataclass
class SetupResult:
    passed: bool = False
    direction: str = ""
    reason: str = ""
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    rr: float = 0.0
    rejection_reason: str = ""
    sweep_price: float = 0.0
    sweep_level_name: str = ""
    fvg_high: float = 0.0
    fvg_low: float = 0.0


@dataclass
class PendingSetup:
    pair_name: str = ""
    direction: str = ""
    sweep_price: float = 0.0
    sweep_level_name: str = ""
    fvg_low: float = 0.0
    fvg_high: float = 0.0
    choch_idx: int = 0
    detected_at: float = 0.0


def is_within_trading_session() -> Tuple[bool, str]:
    now = datetime.now(timezone.utc)
    current_minutes = now.hour * 60 + now.minute

    for name, sh, sm, eh, em in SESSIONS_UTC:
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        if start_min <= current_minutes < end_min:
            return True, name

    return False, f"Outside session (UTC {now.hour:02d}:{now.minute:02d})"


# ── Rule 1: M15 Liquidity Sweep ──────────────────────────────────────────────

def _compute_all_levels(m15: pd.DataFrame) -> dict:
    """Computes Session levels + Recent M15 Major Swing Highs/Lows."""
    levels = {}
    if m15 is None or len(m15) < 30:
        return levels

    # 1. Recent M15 Structural Swings (Last 30 candles)
    recent = m15.iloc[-30:]
    levels["M15_Swing_High"] = float(recent["High"].max())
    levels["M15_Swing_Low"] = float(recent["Low"].min())

    # 2. Previous Day High/Low
    m15_dates = m15.index.date
    unique_dates = sorted(set(m15_dates))
    if len(unique_dates) >= 2:
        prev_day = unique_dates[-2]
        prev_data = m15[m15.index.date == prev_day]
        if not prev_data.empty:
            levels["PDH"] = float(prev_data["High"].max())
            levels["PDL"] = float(prev_data["Low"].min())

    return levels


def check_liquidity_sweep(m15: pd.DataFrame) -> Optional[Tuple[str, float, str]]:
    if m15 is None or len(m15) < 5:
        return None

    levels = _compute_all_levels(m15)
    if not levels:
        return None

    # Check last 3 completed candles for sweep
    for idx in range(-3, 0):
        candle = m15.iloc[idx]
        c_high = float(candle["High"])
        c_low = float(candle["Low"])
        c_close = float(candle["Close"])

        # Bullish Sweep (Sell-side sweep)
        low_levels = {k: v for k, v in levels.items() if "Low" in k or k == "PDL"}
        for name, level in low_levels.items():
            if c_low < level and c_close > level:
                logger.info("Rule 1 PASS: Bullish sweep on %s (Low: %.2f < Level: %.2f)", name, c_low, level)
                return ("bullish", c_low, name)

        # Bearish Sweep (Buy-side sweep)
        high_levels = {k: v for k, v in levels.items() if "High" in k or k == "PDH"}
        for name, level in high_levels.items():
            if c_high > level and c_close < level:
                logger.info("Rule 1 PASS: Bearish sweep on %s (High: %.2f > Level: %.2f)", name, c_high, level)
                return ("bearish", c_high, name)

    return None


# ── Rule 2: M5 CHOCH Detection ──────────────────────────────────────────────

def check_choch(m5: pd.DataFrame, direction: str) -> Optional[int]:
    if m5 is None or len(m5) < CHOCH_LOOKBACK:
        return None

    lookback = m5.iloc[-CHOCH_LOOKBACK:]
    
    if direction == "bullish":
        # Local High in recent history
        local_high = float(lookback.iloc[:-3]["High"].max())
        for j in range(len(m5) - 10, len(m5)):
            if float(m5.iloc[j]["Close"]) > local_high:
                logger.info("Rule 2 PASS: Bullish CHOCH at index %d (Close > Local High %.2f)", j, local_high)
                return j
    else:
        # Local Low in recent history
        local_low = float(lookback.iloc[:-3]["Low"].min())
        for j in range(len(m5) - 10, len(m5)):
            if float(m5.iloc[j]["Close"]) < local_low:
                logger.info("Rule 2 PASS: Bearish CHOCH at index %d (Close < Local Low %.2f)", j, local_low)
                return j

    return None


# ── Rule 3: FVG Detection ──────────────────────────────────────────

def detect_fvg(m5: pd.DataFrame, choch_idx: int, direction: str, pair_name: str = "") -> Optional[Tuple[float, float]]:
    if m5 is None or len(m5) < 3:
        return None

    # Check last 3 candles for gap
    c1 = m5.iloc[-3]
    c3 = m5.iloc[-1]
    min_width = FVG_MIN_WIDTH.get(pair_name, 0.10)

    if direction == "bullish":
        fvg_low = float(c1["High"])
        fvg_high = float(c3["Low"])
        if fvg_high > fvg_low and (fvg_high - fvg_low) >= min_width:
            logger.info("Rule 3 PASS: Bullish FVG [%.2f, %.2f]", fvg_low, fvg_high)
            return (fvg_low, fvg_high)
    else:
        fvg_high = float(c1["Low"])
        fvg_low = float(c3["High"])
        if fvg_high > fvg_low and (fvg_high - fvg_low) >= min_width:
            logger.info("Rule 3 PASS: Bearish FVG [%.2f, %.2f]", fvg_low, fvg_high)
            return (fvg_low, fvg_high)

    return None


def check_fvg_retest(current_price: float, fvg_low: float, fvg_high: float, direction: str) -> bool:
    return fvg_low <= current_price <= fvg_high


def is_fvg_still_valid(m5: pd.DataFrame, fvg_low: float, fvg_high: float, direction: str) -> bool:
    if m5 is None or m5.empty:
        return False
    recent = m5.iloc[-5:]
    for _, row in recent.iterrows():
        if direction == "bullish" and float(row["Close"]) < fvg_low:
            return False
        if direction == "bearish" and float(row["Close"]) > fvg_high:
            return False
    return True


# ── Rule 4: R:R Validation ───────────────────────────────────────────────────

def check_rr(direction: str, fvg_low: float, fvg_high: float, sweep_price: float, levels: dict) -> Optional[Tuple[float, float, float, float]]:
    entry = (fvg_low + fvg_high) / 2.0
    
    if direction == "bullish":
        sl = sweep_price
        high_levels = [v for k, v in levels.items() if "High" in k and v > entry]
        tp = min(high_levels) if high_levels else entry + (abs(entry - sl) * 2.5)
    else:
        sl = sweep_price
        low_levels = [v for k, v in levels.items() if "Low" in k and v < entry]
        tp = max(low_levels) if low_levels else entry - (abs(entry - sl) * 2.5)

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0:
        return None
    rr = reward / risk

    if rr < MIN_RR:
        return None

    return (entry, sl, tp, rr)


def detect_setup(m15: pd.DataFrame, m5: pd.DataFrame, pair_name: str) -> Optional[PendingSetup]:
    sweep = check_liquidity_sweep(m15)
    if sweep is None:
        return None

    direction, sweep_price, level_name = sweep

    choch_idx = check_choch(m5, direction)
    if choch_idx is None:
        return None

    fvg = detect_fvg(m5, choch_idx, direction, pair_name)
    if fvg is None:
        return None
        
    fvg_low, fvg_high = fvg

    import time
    return PendingSetup(
        pair_name=pair_name,
        direction=direction,
        sweep_price=sweep_price,
        sweep_level_name=level_name,
        fvg_low=fvg_low,
        fvg_high=fvg_high,
        choch_idx=choch_idx,
        detected_at=time.time(),
    )


def check_retest_and_rr(pending: PendingSetup, m15: pd.DataFrame, m5: pd.DataFrame) -> Optional[SetupResult]:
    if m5 is None or m5.empty:
        return None

    current_price = float(m5.iloc[-1]["Close"])

    if not is_fvg_still_valid(m5, pending.fvg_low, pending.fvg_high, pending.direction):
        return None

    if not check_fvg_retest(current_price, pending.fvg_low, pending.fvg_high, pending.direction):
        return None

    levels = _compute_all_levels(m15)
    rr_result = check_rr(pending.direction, pending.fvg_low, pending.fvg_high, pending.sweep_price, levels)
    if rr_result is None:
        result = SetupResult()
        result.rejection_reason = "Rule 4 FAIL: R:R < 2.0"
        return result

    entry, sl, tp, rr = rr_result
    return SetupResult(
        passed=True,
        direction=pending.direction,
        entry=entry,
        sl=sl,
        tp=tp,
        rr=rr,
        sweep_price=pending.sweep_price,
        sweep_level_name=pending.sweep_level_name,
        fvg_low=pending.fvg_low,
        fvg_high=pending.fvg_high,
        reason=f"M15 {pending.sweep_level_name} Swept + M5 CHOCH + FVG Retest @ {current_price:.2f} (RR {rr:.1f})",
    )
