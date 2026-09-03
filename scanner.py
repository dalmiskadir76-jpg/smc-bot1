"""Main scanner — two-phase approach:
Phase 1: Detect new FVGs (Rules 1-3) and store as pending.
Phase 2: Check if price retests pending FVGs, validate R:R (Rule 4), and alert.

Additional filters applied:
- Session filter: only London (07:00-11:00 UTC) and New York (12:30-17:00 UTC)
- FVG size threshold: min $0.50 for XAUUSD
- Retest-only trigger: alert ONLY when price first enters FVG zone
"""
import logging
import time

import data_feed, smc_engine, state, telegram
from .data_feed import PAIRS
from .smc_engine import PendingSetup, is_within_trading_session

logger = logging.getLogger(__name__)


def scan_all_pairs() -> dict:
    """Run a full scan cycle on all pairs.

    Returns a summary dict for API response.
    """
    start = time.time()
    logger.info("=== Scan cycle started ===")

    # ── Session filter ────────────────────────────────────────────────────
    session_ok, session_info = is_within_trading_session()
    if not session_ok:
        elapsed = time.time() - start
        logger.info("Session filter: %s — skipping scan", session_info)
        return {
            "pairs": {},
            "session_filter": session_info,
            "skipped": True,
            "elapsed_seconds": round(elapsed, 1),
        }

    logger.info("Active session: %s", session_info)
    results = {}

    for pair_name, ticker in PAIRS.items():
        try:
            logger.info("Scanning %s (%s)", pair_name, ticker)

            # Fetch data
            m15 = data_feed.get_m15(ticker)
            m5 = data_feed.get_m5(ticker)

            if m15 is None or m5 is None:
                logger.warning("%s: Failed to fetch data, skipping", pair_name)
                results[pair_name] = {"status": "data_error"}
                continue

            # ── Check active alerts for TP/SL hit ─────────────────────────
            current_price = float(m5.iloc[-1]["Close"])
            hit = state.check_and_reset(pair_name, current_price)
            if hit:
                emoji = "\U0001f389" if hit == "tp" else "\U0001f6d1"
                outcome = "TP HIT \u2014 Setup won!" if hit == "tp" else "SL HIT \u2014 Setup lost."
                msg = f"{emoji} {pair_name}: {outcome} Price: {current_price:.5f}"
                telegram.send_message(msg)

            # ── Phase 2: Check pending FVG retests FIRST ──────────────────
            pending_data = state.get_pending(pair_name)
            if pending_data is not None:
                pending = PendingSetup(
                    pair_name=pending_data["pair_name"],
                    direction=pending_data["direction"],
                    sweep_price=pending_data["sweep_price"],
                    sweep_level_name=pending_data["sweep_level_name"],
                    fvg_low=pending_data["fvg_low"],
                    fvg_high=pending_data["fvg_high"],
                    choch_idx=pending_data.get("choch_idx", 0),
                    detected_at=pending_data.get("detected_at", 0),
                )

                # Check if FVG is still valid
                if not smc_engine.is_fvg_still_valid(
                        m5, pending.fvg_low, pending.fvg_high, pending.direction):
                    logger.info("%s: Pending FVG mitigated — removing", pair_name)
                    state.remove_pending(pair_name)
                    results[pair_name] = {"status": "pending_mitigated"}
                    continue

                # Check retest + R:R
                retest_result = smc_engine.check_retest_and_rr(pending, m15, m5)
                if retest_result is not None and retest_result.passed:
                    # All 4 rules satisfied!
                    if not state.can_alert(pair_name):
                        logger.info("%s: Setup triggered but in cooldown", pair_name)
                        results[pair_name] = {
                            "status": "alert_cooldown",
                            "direction": retest_result.direction,
                        }
                        continue

                    logger.info("%s: ALL 4 RULES PASSED (retest) — sending alert!",
                                pair_name)
                    sent = telegram.send_alert(
                        pair_name=pair_name,
                        direction=retest_result.direction,
                        reason=retest_result.reason,
                        entry=retest_result.entry,
                        sl=retest_result.sl,
                        tp=retest_result.tp,
                    )
                    if sent:
                        state.record_alert(
                            pair_name, retest_result.direction,
                            retest_result.entry, retest_result.sl, retest_result.tp,
                        )
                    results[pair_name] = {
                        "status": "alert_sent" if sent else "alert_failed",
                        "trigger": "fvg_retest",
                        "direction": retest_result.direction,
                        "entry": retest_result.entry,
                        "sl": retest_result.sl,
                        "tp": retest_result.tp,
                        "rr": retest_result.rr,
                    }
                    continue

                elif retest_result is not None and not retest_result.passed:
                    # Retest happened but R:R failed — invalidate
                    logger.info("%s: FVG retested but R:R failed — removing pending",
                                pair_name)
                    state.remove_pending(pair_name)
                    results[pair_name] = {
                        "status": "rr_failed_on_retest",
                        "reason": retest_result.rejection_reason,
                    }
                    continue
                else:
                    # Not retesting yet — keep pending
                    results[pair_name] = {
                        "status": "pending_fvg",
                        "fvg": [pending.fvg_low, pending.fvg_high],
                        "direction": pending.direction,
                        "waiting_for": "price_retest",
                    }
                    continue

            # ── Phase 1: Detect new setups (Rules 1-3) ───────────────────
            # Only if no alert is active and no pending setup exists
            if not state.can_alert(pair_name):
                results[pair_name] = {"status": "alert_active"}
                continue

            setup = smc_engine.detect_setup(m15, m5, pair_name)
            if setup is not None:
                # Store as pending — do NOT alert yet
                state.save_pending(pair_name, {
                    "pair_name": setup.pair_name,
                    "direction": setup.direction,
                    "sweep_price": setup.sweep_price,
                    "sweep_level_name": setup.sweep_level_name,
                    "fvg_low": setup.fvg_low,
                    "fvg_high": setup.fvg_high,
                    "choch_idx": setup.choch_idx,
                    "detected_at": setup.detected_at,
                })
                results[pair_name] = {
                    "status": "fvg_detected_pending_retest",
                    "direction": setup.direction,
                    "fvg": [setup.fvg_low, setup.fvg_high],
                    "sweep": setup.sweep_level_name,
                }
            else:
                results[pair_name] = {"status": "no_setup"}

        except Exception:
            logger.exception("Error scanning %s", pair_name)
            results[pair_name] = {"status": "error"}

    elapsed = time.time() - start
    logger.info("=== Scan cycle completed in %.1fs ===", elapsed)
    return {
        "pairs": results,
        "session": session_info,
        "elapsed_seconds": round(elapsed, 1),
    }
