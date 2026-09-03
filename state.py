"""State management for alert deduplication and pending FVG tracking."""
import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Use /tmp for state in production (read-only filesystem).
# Fallback to project dir for local dev.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_FILE = os.path.join("/tmp", "smc_state.json")
RESET_HOURS = 12
PENDING_EXPIRY_HOURS = 4  # pending setups expire after 4 hours


def _load() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            logger.exception("Failed to load state file")
    return {"alerts": {}, "pending": {}}


def _save(state: dict):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        logger.exception("Failed to save state file")


def _ensure_keys(state: dict) -> dict:
    """Ensure the state dict has all required keys."""
    if "alerts" not in state:
        state["alerts"] = state.copy()
        for k in list(state.keys()):
            if k not in ("alerts", "pending"):
                del state[k]
    if "pending" not in state:
        state["pending"] = {}
    return state


# ── Alert State (deduplication) ───────────────────────────────────────────────

def can_alert(pair: str) -> bool:
    """Check if we can send an alert for this pair (not in cooldown)."""
    state = _ensure_keys(_load())
    entry = state["alerts"].get(pair)
    if entry is None:
        return True
    last_alert = entry.get("last_alert_time", 0)
    if time.time() - last_alert > RESET_HOURS * 3600:
        return True
    return False


def record_alert(pair: str, direction: str, entry: float, sl: float, tp: float):
    """Record that an alert was sent."""
    state = _ensure_keys(_load())
    state["alerts"][pair] = {
        "last_alert_time": time.time(),
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
    }
    # Clear pending setup once alert is sent
    state["pending"].pop(pair, None)
    _save(state)
    logger.info("Alert state recorded for %s", pair)


def check_and_reset(pair: str, current_price: float) -> Optional[str]:
    """Check if price hit TP or SL and reset state if so.
    Returns 'tp', 'sl', or None.
    """
    state = _ensure_keys(_load())
    entry = state["alerts"].get(pair)
    if entry is None:
        return None

    direction = entry.get("direction", "")
    tp = entry.get("tp", 0)
    sl = entry.get("sl", 0)

    hit = None
    if direction == "bullish":
        if current_price >= tp:
            hit = "tp"
        elif current_price <= sl:
            hit = "sl"
    elif direction == "bearish":
        if current_price <= tp:
            hit = "tp"
        elif current_price >= sl:
            hit = "sl"

    if hit:
        del state["alerts"][pair]
        _save(state)
        logger.info("%s hit %s — state reset", pair, hit.upper())

    return hit


# ── Pending Setups (FVGs awaiting retest) ─────────────────────────────────────

def get_pending(pair: str) -> Optional[dict]:
    """Get the pending setup for a pair, or None."""
    state = _ensure_keys(_load())
    pending = state["pending"].get(pair)
    if pending is None:
        return None
    # Check expiry
    if time.time() - pending.get("detected_at", 0) > PENDING_EXPIRY_HOURS * 3600:
        logger.info("%s: Pending setup expired (>%dh), removing", pair, PENDING_EXPIRY_HOURS)
        del state["pending"][pair]
        _save(state)
        return None
    return pending


def save_pending(pair: str, setup_data: dict):
    """Save a pending setup (FVG detected, awaiting retest)."""
    state = _ensure_keys(_load())
    state["pending"][pair] = setup_data
    _save(state)
    logger.info("%s: Pending setup saved — FVG [%.5f, %.5f] awaiting retest",
                pair, setup_data.get("fvg_low", 0), setup_data.get("fvg_high", 0))


def remove_pending(pair: str):
    """Remove a pending setup (mitigated or triggered)."""
    state = _ensure_keys(_load())
    if pair in state["pending"]:
        del state["pending"][pair]
        _save(state)
        logger.info("%s: Pending setup removed", pair)


def get_all_pending() -> Dict[str, dict]:
    """Get all pending setups."""
    state = _ensure_keys(_load())
    now = time.time()
    # Clean expired
    expired = [k for k, v in state["pending"].items()
               if now - v.get("detected_at", 0) > PENDING_EXPIRY_HOURS * 3600]
    for k in expired:
        logger.info("%s: Pending setup expired, removing", k)
        del state["pending"][k]
    if expired:
        _save(state)
    return dict(state["pending"])
