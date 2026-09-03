"""A+ SMC 4-Rule Engine.

Rule 1: M15 Liquidity Sweep
Rule 2: M5 CHOCH (Change of Character)
Rule 3: M5 FVG (Fair Value Gap)
Rule 4: R:R Validation (>= 1:2)

Additional filters:
- Session filter: London (07:00-11:00 UTC) and New York (12:30-17:00 UTC) only
- FVG size threshold: min 0.50 USD for XAUUSD
- Retest-only trigger: alert when price first enters FVG, not when FVG is created
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SWEEP_BUFFER_PCT = 0.0001  # 0.01%
CHOCH_LOOKBACK = 20
DISPLACEMENT_BODY_MULT = 1.5
DISPLACEMENT_BODY_RATIO = 0.6
MIN_RR = 2.0

# FVG minimum width thresholds (in price units) per pair
FVG_MIN_WIDTH: Dict[str, float] = {
    "XAUUSD": 0.50,  # $0.50 minimum for gold
}

# Session windows (UTC hours).  Only London + New York allowed.
SESSIONS_UTC = [
    ("London", 7, 0, 11, 0),     # 07:00 – 11:00 UTC
    ("NewYork", 12, 30, 17, 0),   # 12:30 – 17:00 UTC
]


@dataclass
class SetupResult:
    """Result of SMC analysis for a pair."""
    passed: bool = False
    direction: str = ""  # "bullish" or "bearish"
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
    """An FVG detected via Rules 1-3, waiting for price retest."""
    pair_name: str = ""
    direction: str = ""
    sweep_price: float = 0.0
    sweep_level_name: str = ""
    fvg_low: float = 0.0
    fvg_high: float = 0.0
    choch_idx: int = 0
    detected_at: float = 0.0  # unix timestamp


# ── Session Filter ────────────────────────────────────────────────────────────

def is_within_trading_session() -> Tuple[bool, str]:
    """Return (allowed, reason).  Only London and NewYork sessions pass."""
    now = datetime.now(timezone.utc)
    h, m = now.hour, now.minute
    current_minutes = h * 60 + m

    for name, sh, sm, eh, em in SESSIONS_UTC:
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        if start_min <= current_minutes < end_min:
            return True, name

    return False, f"Outside session (UTC {h:02d}:{m:02d})"


# ── Rule 1: M15 Liquidity Sweep ──────────────────────────────────────────────

def _compute_session_levels(m15: pd.DataFrame) -> dict:
    """Compute session highs/lows and PDH/PDL."""
    levels = {}

    # Previous Day High/Low
    m15_dates = m15.index.date
    unique_dates = sorted(set(m15_dates))
    if len(unique_dates) >= 2:
        prev_day = unique_dates[-2]
        prev_data = m15[m15.index.date == prev_day]
        if not prev_data.empty:
            levels["PDH"] = float(prev_data["High"].max())
            levels["PDL"] = float(prev_data["Low"].min())

    # Session levels from today
    today = unique_dates[-1] if unique_dates else None
    if today is None:
        return levels

    today_data = m15[m15.index.date == today]
    if today_data.empty:
        return levels

    sessions = {
        "Asian": (0, 8),
        "London": (7, 16),
        "NewYork": (12, 21),
    }
    for name, (start_h, end_h) in sessions.items():
        mask = (today_data.index.hour >= start_h) & (today_data.index.hour < end_h)
        sess = today_data[mask]
        if not sess.empty:
            levels[f"{name}_High"] = float(sess["High"].max())
            levels[f"{name}_Low"] = float(sess["Low"].min())

    return levels


def check_liquidity_sweep(m15: pd.DataFrame) -> Optional[Tuple[str, float, str]]:
    """Check if the latest M15 candle swept a key level.

    Returns (direction, sweep_price, level_name) or None.
    direction: 'bullish' (swept SSL, expect BUY) or 'bearish' (swept BSL, expect SELL)
    """
    if m15 is None or len(m15) < 5:
        return None

    levels = _compute_session_levels(m15)
    if not levels:
        return None

    candle = m15.iloc[-1]
    c_high = float(candle["High"])
    c_low = float(candle["Low"])
    c_close = float(candle["Close"])
    price = c_close  # reference price for buffer
    buf = price * SWEEP_BUFFER_PCT

    # Check bullish sweep (SSL - sell-side liquidity swept)
    # Wick below a key low, close back above
    low_levels = {k: v for k, v in levels.items() if "Low" in k or k == "PDL"}
    for name, level in low_levels.items():
        if c_low < (level - buf) and c_close > level:
            logger.info("Bullish sweep: candle low %.5f < %s %.5f, close %.5f > level", c_low, name, level, c_close)
            return ("bullish", c_low, name)

    # Check bearish sweep (BSL - buy-side liquidity swept)
    # Wick above a key high, close back below
    high_levels = {k: v for k, v in levels.items() if "High" in k or k == "PDH"}
    for name, level in high_levels.items():
        if c_high > (level + buf) and c_close < level:
            logger.info("Bearish sweep: candle high %.5f > %s %.5f, close %.5f < level", c_high, name, level, c_close)
            return ("bearish", c_high, name)

    return None


# ── Rule 2: M5 CHOCH Detection ──────────────────────────────────────────────

def _find_swing_high(df: pd.DataFrame, idx: int) -> bool:
    """Check if candle at idx is a swing high (higher high than 2 candles on each side)."""
    if idx < 2 or idx >= len(df) - 2:
        return False
    h = float(df.iloc[idx]["High"])
    return all(h > float(df.iloc[idx + d]["High"]) for d in [-2, -1, 1, 2])


def _find_swing_low(df: pd.DataFrame, idx: int) -> bool:
    if idx < 2 or idx >= len(df) - 2:
        return False
    lo = float(df.iloc[idx]["Low"])
    return all(lo < float(df.iloc[idx + d]["Low"]) for d in [-2, -1, 1, 2])


def _is_displacement(candle, avg_body: float) -> bool:
    """Check if a candle is a displacement candle."""
    o, h, lo, c = float(candle["Open"]), float(candle["High"]), float(candle["Low"]), float(candle["Close"])
    body = abs(c - o)
    rng = h - lo
    if rng == 0:
        return False
    return body > DISPLACEMENT_BODY_MULT * avg_body and (body / rng) > DISPLACEMENT_BODY_RATIO


def check_choch(m5: pd.DataFrame, direction: str) -> Optional[int]:
    """Check for CHOCH on M5 after a sweep.

    direction: 'bullish' (look for bullish CHOCH = break above swing high)
               'bearish' (look for bearish CHOCH = break below swing low)

    Returns the index (iloc) of the CHOCH candle, or None.
    """
    if m5 is None or len(m5) < CHOCH_LOOKBACK:
        return None

    lookback = m5.iloc[-CHOCH_LOOKBACK:]
    # Compute average body for displacement check
    bodies = [abs(float(lookback.iloc[i]["Close"]) - float(lookback.iloc[i]["Open"])) for i in range(len(lookback))]
    avg_body = np.mean(bodies[-10:]) if len(bodies) >= 10 else np.mean(bodies)
    if avg_body == 0:
        return None

    start_iloc = len(m5) - CHOCH_LOOKBACK

    if direction == "bullish":
        # Find most recent swing high, then check if a later candle broke above it with displacement
        swing_highs = []
        for i in range(len(lookback)):
            abs_i = start_iloc + i
            if _find_swing_high(m5, abs_i):
                swing_highs.append((abs_i, float(m5.iloc[abs_i]["High"])))
        if not swing_highs:
            return None
        # Take the most recent swing high
        sh_idx, sh_price = swing_highs[-1]
        # Look for candles after this swing high that break above it
        for j in range(sh_idx + 3, len(m5)):
            candle = m5.iloc[j]
            if float(candle["High"]) > sh_price and float(candle["Close"]) > sh_price:
                if _is_displacement(candle, avg_body):
                    logger.info("Bullish CHOCH at index %d, broke swing high %.5f", j, sh_price)
                    return j
    else:  # bearish
        swing_lows = []
        for i in range(len(lookback)):
            abs_i = start_iloc + i
            if _find_swing_low(m5, abs_i):
                swing_lows.append((abs_i, float(m5.iloc[abs_i]["Low"])))
        if not swing_lows:
            return None
        sl_idx, sl_price = swing_lows[-1]
        for j in range(sl_idx + 3, len(m5)):
            candle = m5.iloc[j]
            if float(candle["Low"]) < sl_price and float(candle["Close"]) < sl_price:
                if _is_displacement(candle, avg_body):
                    logger.info("Bearish CHOCH at index %d, broke swing low %.5f", j, sl_price)
                    return j

    return None


# ── Rule 3: FVG Detection (detect only, NOT retest) ──────────────────────────

def detect_fvg(m5: pd.DataFrame, choch_idx: int, direction: str,
               pair_name: str = "") -> Optional[Tuple[float, float]]:
    """Detect if the CHOCH candle created an FVG.

    Unlike the old check_fvg, this does NOT check price retracement.
    It only validates the 3-candle gap exists and is unmitigated.

    Returns (fvg_low, fvg_high) or None.
    """
    if m5 is None or choch_idx < 2 or choch_idx >= len(m5):
        return None

    i = choch_idx
    candle_i = m5.iloc[i]
    candle_i2 = m5.iloc[i - 2]

    if direction == "bullish":
        # Bullish FVG: candle[i].low > candle[i-2].high
        fvg_low = float(candle_i2["High"])
        fvg_high = float(candle_i["Low"])
        if fvg_high <= fvg_low:
            return None
        # FVG size threshold check
        fvg_width = fvg_high - fvg_low
        min_width = FVG_MIN_WIDTH.get(pair_name, 0.0)
        if fvg_width < min_width:
            logger.info("%s: Bullish FVG too narrow (%.4f < %.4f), skipping",
                        pair_name, fvg_width, min_width)
            return None
        # Check unmitigated: no candle after i fully closed below fvg_low
        for k in range(i + 1, len(m5)):
            if float(m5.iloc[k]["Close"]) < fvg_low:
                logger.info("Bullish FVG mitigated at index %d", k)
                return None
        logger.info("Bullish FVG detected: [%.5f, %.5f] (width=%.4f)",
                    fvg_low, fvg_high, fvg_width)
        return (fvg_low, fvg_high)

    else:  # bearish
        # Bearish FVG: candle[i].high < candle[i-2].low
        fvg_high = float(candle_i2["Low"])
        fvg_low = float(candle_i["High"])
        if fvg_high <= fvg_low:
            return None
        fvg_width = fvg_high - fvg_low
        min_width = FVG_MIN_WIDTH.get(pair_name, 0.0)
        if fvg_width < min_width:
            logger.info("%s: Bearish FVG too narrow (%.4f < %.4f), skipping",
                        pair_name, fvg_width, min_width)
            return None
        for k in range(i + 1, len(m5)):
            if float(m5.iloc[k]["Close"]) > fvg_high:
                logger.info("Bearish FVG mitigated at index %d", k)
                return None
        logger.info("Bearish FVG detected: [%.5f, %.5f] (width=%.4f)",
                    fvg_low, fvg_high, fvg_width)
        return (fvg_low, fvg_high)


def check_fvg_retest(current_price: float, fvg_low: float, fvg_high: float,
                     direction: str) -> bool:
    """Check if price is retracing INTO the FVG zone (first touch / entry).

    For BUY:  price drops into the bullish FVG from above
    For SELL: price rises into the bearish FVG from below
    """
    if direction == "bullish":
        return fvg_low <= current_price <= fvg_high
    else:
        return fvg_low <= current_price <= fvg_high


def is_fvg_still_valid(m5: pd.DataFrame, fvg_low: float, fvg_high: float,
                       direction: str) -> bool:
    """Check that the FVG has not been fully mitigated by recent candles."""
    if m5 is None or m5.empty:
        return False
    # Check last 10 candles for mitigation
    recent = m5.iloc[-10:]
    for _, row in recent.iterrows():
        if direction == "bullish" and float(row["Close"]) < fvg_low:
            return False
        if direction == "bearish" and float(row["Close"]) > fvg_high:
            return False
    return True


# ── Rule 4: R:R Validation ───────────────────────────────────────────────────

def check_rr(direction: str, fvg_low: float, fvg_high: float,
             sweep_price: float, levels: dict) -> Optional[Tuple[float, float, float, float]]:
    """Validate minimum 1:2 R:R.

    Returns (entry, sl, tp, rr) or None.
    """
    entry = (fvg_low + fvg_high) / 2.0
    buf = entry * SWEEP_BUFFER_PCT * 10  # wider buffer for SL

    if direction == "bullish":
        sl = sweep_price - buf
        # TP = next resistance above entry
        high_levels = [v for k, v in levels.items() if ("High" in k or k == "PDH") and v > entry]
        if not high_levels:
            return None
        tp = min(high_levels)  # nearest resistance
    else:
        sl = sweep_price + buf
        low_levels = [v for k, v in levels.items() if ("Low" in k or k == "PDL") and v < entry]
        if not low_levels:
            return None
        tp = max(low_levels)  # nearest support

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0:
        return None
    rr = reward / risk

    if rr < MIN_RR:
        logger.info("R:R %.2f < %.1f, setup invalidated", rr, MIN_RR)
        return None

    logger.info("R:R validated: entry=%.5f, SL=%.5f, TP=%.5f, RR=%.2f", entry, sl, tp, rr)
    return (entry, sl, tp, rr)


# ── Phase 1: Detect pending setups (Rules 1-3, no retest yet) ────────────────

def detect_setup(m15: pd.DataFrame, m5: pd.DataFrame,
                 pair_name: str) -> Optional[PendingSetup]:
    """Run Rules 1-3 to detect an FVG that hasn't been retested yet.

    Does NOT check retest or R:R — those happen in Phase 2 when price enters FVG.
    """
    # Rule 1: Liquidity Sweep
    sweep = check_liquidity_sweep(m15)
    if sweep is None:
        return None

    direction, sweep_price, level_name = sweep
    logger.info("%s Rule 1 PASS: %s sweep at %s (%.5f)",
                pair_name, direction, level_name, sweep_price)

    # Rule 2: CHOCH
    choch_idx = check_choch(m5, direction)
    if choch_idx is None:
        logger.info("%s Rule 2 FAIL: No %s CHOCH on M5", pair_name, direction)
        return None
    logger.info("%s Rule 2 PASS: CHOCH at M5 index %d", pair_name, choch_idx)

    # Rule 3: FVG detection (NOT retest)
    fvg = detect_fvg(m5, choch_idx, direction, pair_name)
    if fvg is None:
        logger.info("%s Rule 3 FAIL: No valid FVG from CHOCH candle", pair_name)
        return None
    fvg_low, fvg_high = fvg
    logger.info("%s Rule 3 PASS: FVG [%.5f, %.5f] — awaiting retest",
                pair_name, fvg_low, fvg_high)

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


# ── Phase 2: Check retest of a pending FVG + R:R ─────────────────────────────

def check_retest_and_rr(pending: PendingSetup, m15: pd.DataFrame,
                        m5: pd.DataFrame) -> Optional[SetupResult]:
    """Given a pending setup, check if price is now retesting the FVG
    and if R:R >= 1:2.  Returns a full SetupResult if all conditions met."""
    if m5 is None or m5.empty:
        return None

    current_price = float(m5.iloc[-1]["Close"])

    # Confirm FVG still unmitigated
    if not is_fvg_still_valid(m5, pending.fvg_low, pending.fvg_high, pending.direction):
        logger.info("%s: Pending FVG [%.5f, %.5f] has been mitigated — removing",
                    pending.pair_name, pending.fvg_low, pending.fvg_high)
        return None  # caller should remove from pending

    # Check retest
    if not check_fvg_retest(current_price, pending.fvg_low, pending.fvg_high,
                            pending.direction):
        return None  # not retesting yet, keep pending

    logger.info("%s: Price %.5f is retesting FVG [%.5f, %.5f]!",
                pending.pair_name, current_price, pending.fvg_low, pending.fvg_high)

    # Rule 4: R:R validation
    levels = _compute_session_levels(m15)
    rr_result = check_rr(pending.direction, pending.fvg_low, pending.fvg_high,
                         pending.sweep_price, levels)
    if rr_result is None:
        logger.info("%s: FVG retested but R:R < 2.0 — no alert", pending.pair_name)
        result = SetupResult()
        result.rejection_reason = "Rule 4 FAIL: R:R < 2.0 or no TP level found"
        return result  # passed=False signals removal

    entry, sl, tp, rr = rr_result
    logger.info("%s Rule 4 PASS: RR=%.2f — ALL RULES MET!", pending.pair_name, rr)

    side = "BUY" if pending.direction == "bullish" else "SELL"
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
        reason=(f"M15 {pending.sweep_level_name} Swept + M5 Displacement CHOCH "
                f"+ M5 FVG Retested @ {current_price:.5f} (RR {rr:.1f})"),
    )
