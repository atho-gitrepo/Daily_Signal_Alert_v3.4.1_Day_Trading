#!/usr/bin/env python3
"""
AI Trading Bot v3.4.1 - Main Entry Point
Super TDI + MACD + Super Bollinger Bands Strategy
ALIGNED WITH MANUAL STRATEGY: RSI (TDI) primary + MACD secondary + BB entry
FIXED: UnboundLocalError and duplicate prevention
"""

import os
import sys
import time
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import signal as signal_module  # ← FIXED: Import as different name

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========== IMPORTS ==========

# Settings
from settings import config

# Data Fetcher
from utils.binance_data_client import binance_client

# MongoDB
from utils.mongodb_client import mongodb_client

# Indicators
from utils import calculate_all_indicators
from utils.indicators import Indicators

# Signal Manager
from utils.signal_manager import signal_manager

# Telegram
from utils.telegram_bot import telegram_bot

# Strategy - Super TDI + MACD + Super Bollinger Bands
from strategy.signal_engine import SignalEngine
from strategy.ai_analyzer import ai_analyzer
from strategy.cheat_sheet import SignalCheatSheet
from utils.logger import get_logger

# ========== LOGGING ==========

# Create logs directory
os.makedirs('logs', exist_ok=True)

logger = get_logger(__name__)

# Emoji indicators
EMOJI = {
    "START": "🚀", "STOP": "🛑", "SUCCESS": "✅", "ERROR": "❌",
    "WARNING": "⚠️", "INFO": "ℹ️", "SIGNAL": "📡", "HEALTH": "💚",
    "AI": "🤖", "DB": "💾", "TELEGRAM": "📨", "SYNC": "🔄",
    "PROFIT": "💰", "LOSS": "💸", "REJECT": "🚫", "WAIT": "⏳",
    "LOCK": "🔒", "UNLOCK": "🔓", "MONITOR": "📊", "RESULT": "📈",
    "LTF": "⏱️", "HTF": "📊", "SCORE": "🎯", "GRADE_A": "🏆",
    "GRADE_B": "🥈", "GRADE_C": "🥉", "DIVERGENCE": "↩️",
    "PATTERN": "🕯️", "S_R": "📊", "SESSION": "🌍", "STRUCTURE": "🏗️",
    "REGIME": "📈", "STATE": "🔄", "FETCH": "📊", "TDI": "📈",
    "BB": "📊", "MACD": "📊", "CANDLE": "🕯️", "CHEAT": "📋",
    "EXIT": "⏰", "RESTORE": "♻️", "ACTIVE": "🔴",
}


# ========== GLOBAL STATE ==========

running = True
bot_stats = {
    'status': 'initializing',
    'start_time': datetime.now().isoformat(),
    'signals_generated': 0,
    'ai_approved': 0,
    'ai_rejected': 0,
    'ai_waited': 0,
    'active_signals': 0,
    'exits': {
        'profit': 0,
        'loss': 0,
        'break_even': 0,
        'exit_1h': 0,
        'exit_target': 0,
    },
    'total_pnl': 0.0,
    'errors': 0,
    'version': '3.4.5',
    'strategy': 'Super TDI + MACD + Super Bollinger Bands',
    'ai_enabled': ai_analyzer.enabled if ai_analyzer else False,
    'mongodb_enabled': mongodb_client.is_available() if mongodb_client else False,
    'features': {
        'tdi_zones': True,
        'macd_confirmation': getattr(config.strategy, 'require_macd_confirmation', True),
        'bb_touch': True,
        'candle_shrinking': True,
        'reversal_confirmation': True,
        'divergence': getattr(config.strategy, 'enable_divergence', True),
        'candle_patterns': getattr(config.strategy, 'enable_candle_patterns', True),
        'support_resistance': getattr(config.strategy, 'enable_support_resistance', True),
        'bb_squeeze': getattr(config.strategy, 'enable_bb_squeeze', True),
        'session_filtering': getattr(config.strategy, 'enable_session_filtering', True),
        'ai_validation': ai_analyzer.enabled if ai_analyzer else False,
        'htf_trend_alignment': getattr(config.strategy, 'require_htf_alignment', False),
    },
}


# ========== DATA FETCHER WRAPPER ==========

class DataFetcherWrapper:
    """Wrapper for binance_data_client."""

    def __init__(self):
        self._client = binance_client
        if self._client is not None:
            logger.info(f"{EMOJI['SUCCESS']} Data fetcher: Binance client loaded")
        else:
            logger.warning(f"{EMOJI['WARNING']} Data fetcher not available")

    def fetch_klines(self, symbol: str, interval: str = '5m', limit: int = 200):
        """Fetch klines data."""
        if self._client is not None:
            try:
                df = self._client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    heikin_ashi=True
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.debug(f"Data fetch error: {e}")
        return None

    def fetch_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current price."""
        if self._client is not None:
            try:
                return self._client.get_current_price(symbol)
            except Exception as e:
                logger.debug(f"Price fetch error: {e}")
        return None

    def cleanup(self):
        """Clean up resources."""
        if self._client is not None:
            try:
                self._client.cleanup()
            except Exception as e:
                logger.debug(f"Cleanup error: {e}")


# Create singleton
data_client = DataFetcherWrapper()


# ========== RECOVERY FUNCTIONS ==========

def recover_and_sync_signals():
    """Recover active signals from database on startup."""
    if not mongodb_client.is_available():
        logger.warning(f"{EMOJI['WARNING']} MongoDB not available - skipping recovery")
        return
    
    try:
        logger.info(f"{EMOJI['RESTORE']} Syncing active signals with database...")
        
        # Sync signal manager with database
        signal_manager.sync_with_database()
        
        # Also check for any orphaned signals in DB
        active_db = mongodb_client.get_active_signals()
        if active_db:
            logger.info(f"{EMOJI['INFO']} Found {len(active_db)} active signals in database")
            
            # Ensure all DB signals are in memory
            for doc_id, db_signal in active_db.items():
                symbol = db_signal.get('symbol')
                if symbol and not signal_manager.is_symbol_locked(symbol):
                    signal_manager._recover_from_db(symbol)
        
        # Clean up orphaned signals
        cleaned = mongodb_client.cleanup_orphaned_active_signals()
        if cleaned > 0:
            logger.info(f"{EMOJI['SUCCESS']} Cleaned up {cleaned} orphaned signals")
            
        # Log recovered signals
        active = signal_manager.get_all_active_signals()
        if active:
            logger.info(f"{EMOJI['ACTIVE']} Active signals after recovery: {len(active)}")
            for symbol, signal in active.items():
                logger.info(f"  {symbol}: entry={signal.entry_price:.4f}, age={signal.get_age_minutes():.1f}min, doc={signal.db_doc_id}")
            
    except Exception as e:
        logger.error(f"{EMOJI['ERROR']} Recovery failed: {e}")
        import traceback
        traceback.print_exc()


# ========== MAIN PROCESSING FUNCTIONS ==========

def fetch_data(symbol: str, timeframe: str = '5m') -> Optional[pd.DataFrame]:
    """Fetch historical data with indicators."""
    try:
        df = data_client.fetch_klines(symbol, timeframe, 200)
        if df is None or df.empty:
            return None

        # Calculate all indicators (includes MACD now)
        df = calculate_all_indicators(df)
        return df

    except Exception as e:
        logger.error(f"Data fetch error for {symbol}: {e}")
        return None


def check_conditions(signal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check signal conditions and return detailed status."""
    if not signal_data:
        return {'valid': False, 'conditions_met': 0, 'conditions_total': 5}

    conditions = {
        'condition_1_tdi_zone': signal_data.get('condition_1_tdi_zone', False),
        'condition_2_tdi_cross': signal_data.get('condition_2_tdi_cross', False),
        'condition_3_bb_touch': signal_data.get('condition_3_bb_touch', False),
        'condition_4_candles_shrinking': signal_data.get('condition_4_candles_shrinking', False),
        'condition_5_reversal_confirm': signal_data.get('condition_5_reversal_confirm', False),
    }

    conditions_met = sum(1 for v in conditions.values() if v)
    conditions_total = 5

    min_conditions = getattr(config.strategy, 'min_conditions_for_signal', 3)
    is_valid = conditions_met >= min_conditions

    return {
        'valid': is_valid,
        'conditions': conditions,
        'conditions_met': conditions_met,
        'conditions_total': conditions_total,
    }


def process_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Process a single symbol with duplicate prevention."""
    
    if signal_manager is None:
        logger.warning(f"{EMOJI['WARNING']} Signal manager not available for {symbol}")
        return None

    try:
        # ===== STEP 1: Check lock (memory + DB) =====
        if signal_manager.is_symbol_locked(symbol):
            return None

        # ===== STEP 2: Fetch data =====
        df = fetch_data(symbol, config.market.timeframe)
        if df is None:
            return None

        # Fetch HTF data for context only
        htf_df = fetch_data(symbol, config.market.htf_timeframe)
        if htf_df is None:
            logger.debug(f"{symbol}: No HTF data available, continuing without context")
            htf_df = None

        # ===== STEP 3: Process signal =====
        signal_engine = SignalEngine(use_ai=ai_analyzer.enabled if ai_analyzer else False)
        signal_data = signal_engine.process(df, symbol, htf_df=htf_df)  # ← Renamed to signal_data

        if signal_data is None or signal_data.get('signal') == 'NO_TRADE':
            return None

        # Get current price
        current_price = data_client.fetch_current_price(symbol)
        if current_price:
            signal_data['entry_price'] = current_price
        elif not df.empty:
            signal_data['entry_price'] = df['close'].iloc[-1]

        # ===== STEP 4: Check conditions =====
        condition_check = check_conditions(signal_data)

        signal_data['conditions_met'] = condition_check['conditions_met']
        signal_data['conditions_total'] = condition_check['conditions_total']
        signal_data['condition_1_tdi_zone'] = condition_check['conditions'].get('condition_1_tdi_zone', False)
        signal_data['condition_2_tdi_cross'] = condition_check['conditions'].get('condition_2_tdi_cross', False)
        signal_data['condition_3_bb_touch'] = condition_check['conditions'].get('condition_3_bb_touch', False)
        signal_data['condition_4_candles_shrinking'] = condition_check['conditions'].get('condition_4_candles_shrinking', False)
        signal_data['condition_5_reversal_confirm'] = condition_check['conditions'].get('condition_5_reversal_confirm', False)

        # ===== STEP 5: AI Decision =====
        ai_decision = signal_data.get('ai_decision', 'NONE')

        # ===== STEP 6: Build cheat sheet =====
        cheat_sheet_obj = SignalCheatSheet()
        if signal_data.get('direction') == 'BUY':
            signal_data['cheat_sheet'] = cheat_sheet_obj.generate_buy_cheat_sheet(signal_data)
        elif signal_data.get('direction') == 'SELL':
            signal_data['cheat_sheet'] = cheat_sheet_obj.generate_sell_cheat_sheet(signal_data)
        else:
            signal_data['cheat_sheet'] = cheat_sheet_obj.generate_wait_cheat_sheet(signal_data)

        # ===== STEP 7: Lock symbol if valid =====
        if condition_check['valid']:
            if ai_decision == 'APPROVE' or not ai_analyzer.enabled:
                # Double-check lock before locking (prevents race conditions)
                if signal_manager.is_symbol_locked(symbol):
                    logger.debug(f"{symbol}: Symbol was locked during processing - skipping")
                    return None

                if signal_manager.lock_symbol(
                    symbol=symbol,
                    signal_type=signal_data.get('direction', 'UNKNOWN'),
                    entry_price=signal_data.get('entry_price', df['close'].iloc[-1] if not df.empty else 0),
                    raw_data=signal_data,
                ):
                    bot_stats['signals_generated'] += 1
                    if ai_decision == 'APPROVE':
                        bot_stats['ai_approved'] += 1

                    # Send Telegram notification
                    if telegram_bot is not None and telegram_bot.enabled:
                        try:
                            telegram_bot.send_signal(
                                symbol=symbol,
                                signal_type=signal_data.get('direction', 'UNKNOWN'),
                                entry_price=signal_data.get('entry_price', 0),
                                stop_loss=signal_data.get('stop_loss', 0),
                                take_profit=signal_data.get('take_profit', 0),
                                confidence=signal_data.get('confidence', 0.7),
                                ai_decision=ai_decision,
                                ai_confidence=signal_data.get('ai_confidence', 0.8),
                                ai_reasoning=signal_data.get('ai_reasoning', ''),
                                rrr=signal_data.get('rrr', 2.0),
                                total_score=signal_data.get('quality_score', 75),
                                grade=signal_data.get('grade', 'B'),
                                signal_strength=signal_data.get('signal_strength', 'SOFT'),
                                risk_multiplier=signal_data.get('risk_multiplier', 1.0),
                                tdi_level=signal_data.get('tdi_level', 50),
                                tdi_zone=signal_data.get('tdi_zone', 'NEUTRAL'),
                                conditions_met=condition_check['conditions_met'],
                                conditions_total=condition_check['conditions_total'],
                                condition_1_tdi_zone=condition_check['conditions'].get('condition_1_tdi_zone', False),
                                condition_2_tdi_cross=condition_check['conditions'].get('condition_2_tdi_cross', False),
                                condition_3_bb_touch=condition_check['conditions'].get('condition_3_bb_touch', False),
                                condition_4_candles_shrinking=condition_check['conditions'].get('condition_4_candles_shrinking', False),
                                condition_5_reversal_confirm=condition_check['conditions'].get('condition_5_reversal_confirm', False),
                                divergence_detected=signal_data.get('divergence_detected', False),
                                candle_pattern=signal_data.get('candle_pattern', 'NONE'),
                                sr_confirmed=signal_data.get('sr_confirmed', False),
                                bb_squeeze=signal_data.get('bb_squeeze', False),
                                session=signal_data.get('session', 'UNKNOWN'),
                                macd_bullish=signal_data.get('macd_bullish', False),
                                macd_histogram=signal_data.get('macd_histogram', 0.0),
                                cheat_sheet=signal_data.get('cheat_sheet', ''),
                                max_hold_minutes=60,
                            )
                        except Exception as e:
                            logger.warning(f"Telegram error for {symbol}: {e}")

                    # Print cheat sheet
                    cheat_sheet = signal_data.get('cheat_sheet', '')
                    if cheat_sheet:
                        print("\n" + "=" * 70)
                        print(cheat_sheet)
                        if ai_decision == 'APPROVE':
                            print(f"\n🤖 AI Decision: {ai_decision} (Confidence: {signal_data.get('ai_confidence', 0.8)*100:.0f}%)")
                        macd_bullish = signal_data.get('macd_bullish', False)
                        macd_histogram = signal_data.get('macd_histogram', 0.0)
                        print(f"\n📊 MACD: {'BULLISH ✅' if macd_bullish else 'NEUTRAL'} (Histogram: {macd_histogram:.4f})")
                        print("=" * 70 + "\n")

                    logger.info(f"{EMOJI['SIGNAL']} {signal_data.get('direction')} {symbol} @ {signal_data.get('entry_price', 0):.4f} | Conditions: {condition_check['conditions_met']}/{condition_check['conditions_total']} | AI: {ai_decision} | Grade: {signal_data.get('grade', 'C')} | Score: {signal_data.get('quality_score', 0)}")
                    return signal_data

            elif ai_decision == 'REJECT':
                bot_stats['ai_rejected'] += 1
                logger.info(f"{EMOJI['REJECT']} AI REJECTED: {symbol} - {signal_data.get('ai_reasoning', 'No reason')}")

            elif ai_decision == 'WAIT':
                bot_stats['ai_waited'] += 1
                logger.info(f"{EMOJI['WAIT']} AI WAIT: {symbol} - {signal_data.get('ai_reasoning', 'No reason')}")
        else:
            logger.debug(f"{EMOJI['REJECT']} {symbol}: Only {condition_check['conditions_met']}/{condition_check['conditions_total']} conditions met")

        return None

    except Exception as e:
        logger.error(f"Process error for {symbol}: {e}")
        bot_stats['errors'] += 1
        import traceback
        traceback.print_exc()
        return None


def check_active_signals():
    """Check and update active signals with enhanced logging."""
    if signal_manager is None:
        return

    active = signal_manager.get_all_active_signals()
    bot_stats['active_signals'] = len(active)

    if not active:
        return

    # Monitor each active signal
    for symbol, signal_obj in active.items():  # ← Renamed to signal_obj to avoid conflict
        try:
            current_price = data_client.fetch_current_price(symbol)
            if current_price is None:
                continue

            # Get current TDI from last candle for target exits
            last_candle = {}
            df = fetch_data(symbol, config.market.timeframe)
            if df is not None and not df.empty:
                last_candle['tdi_slow_ma'] = df['tdi_slow_ma'].iloc[-1] if 'tdi_slow_ma' in df.columns else 50

            # Check signal
            status, diff, updated = signal_manager.check_active_signal(
                symbol, current_price, last_candle
            )

            # Update stats based on exit status
            if status != "ACTIVE" and updated is not None:
                exit_message = ""
                if status == "PROFIT":
                    bot_stats['exits']['profit'] += 1
                    exit_message = f"PROFIT +${updated.pnl:.2f} ({updated.pnl_percent:.2f}%)"
                elif status == "LOSS":
                    bot_stats['exits']['loss'] += 1
                    exit_message = f"LOSS ${updated.pnl:.2f} ({updated.pnl_percent:.2f}%)"
                elif status == "BREAK_EVEN":
                    bot_stats['exits']['break_even'] += 1
                    exit_message = "BREAK-EVEN"
                elif status == "EXIT_1H":
                    bot_stats['exits']['exit_1h'] += 1
                    exit_message = f"1H FORCED EXIT ${updated.pnl:.2f}"
                elif status == "EXIT_TARGET":
                    bot_stats['exits']['exit_target'] += 1
                    exit_message = f"TDI TARGET EXIT ${updated.pnl:.2f}"

                bot_stats['total_pnl'] += updated.pnl

                logger.info(f"{EMOJI['EXIT']} {symbol}: {exit_message} | DB Doc: {updated.db_doc_id} | Conditions: {updated.conditions_met}/{updated.conditions_total}")

                # Send Telegram result
                if telegram_bot is not None and telegram_bot.enabled:
                    try:
                        telegram_bot.send_result(
                            symbol=symbol,
                            signal_type=signal_obj.signal_type,
                            entry_price=signal_obj.entry_price,
                            exit_price=current_price,
                            pnl=updated.pnl,
                            pnl_percent=updated.pnl_percent,
                            status=status,
                            bars_held=updated.bar_count,
                            fees=updated.fees,
                            confidence=updated.confidence,
                            tdi_level=updated.tdi_level,
                            rrr=updated.rrr,
                            signal_strength=updated.signal_strength,
                            risk_multiplier=updated.risk_multiplier,
                            total_score=updated.total_score if updated.total_score else updated.quality_score,
                            grade=updated.grade,
                            conditions_met=getattr(updated, 'conditions_met', 0),
                            conditions_total=getattr(updated, 'conditions_total', 5),
                            divergence_detected=getattr(updated, 'divergence_detected', False),
                            candle_pattern=getattr(updated, 'candle_pattern', 'NONE'),
                            sr_confirmed=getattr(updated, 'sr_confirmed', False),
                            session=getattr(updated, 'session', 'UNKNOWN'),
                            entry_time=updated.entry_time,
                            exit_time=updated.exit_time,
                        )
                    except Exception as e:
                        logger.warning(f"Telegram result error for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Active signal check error for {symbol}: {e}")
            import traceback
            traceback.print_exc()


# ========== HEALTH SERVER ==========

def run_health_server():
    """Run health check server."""
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                active_signals = signal_manager.get_all_active_signals() if signal_manager else {}
                
                status_data = {
                    'status': 'running' if running else 'stopped',
                    'version': '3.4.5',
                    'strategy': 'Super TDI + MACD + Super Bollinger Bands',
                    'timestamp': datetime.now().isoformat(),
                    'stats': bot_stats,
                    'active_signals': len(active_signals),
                    'active_signal_details': {
                        symbol: {
                            'entry': signal.entry_price,
                            'age_minutes': signal.get_age_minutes(),
                            'status': signal.status,
                            'grade': signal.grade,
                            'conditions': f"{signal.conditions_met}/{signal.conditions_total}",
                            'db_doc_id': signal.db_doc_id,
                        }
                        for symbol, signal in active_signals.items()
                    },
                    'ai_enabled': ai_analyzer.enabled if ai_analyzer else False,
                    'mongodb_enabled': mongodb_client.is_available() if mongodb_client else False,
                    'exit_rules': {
                        'max_hold_minutes': 60,
                        'min_hold_minutes': 15,
                        'buy_exit_target': 70,
                        'sell_exit_target': 30,
                    },
                    'features': bot_stats['features'],
                    'macd_settings': {
                        'fast': getattr(config.strategy, 'macd_fast', 12),
                        'slow': getattr(config.strategy, 'macd_slow', 26),
                        'signal': getattr(config.strategy, 'macd_signal', 9),
                        'required': getattr(config.strategy, 'require_macd_confirmation', True),
                    },
                    'db_stats': mongodb_client.metrics if mongodb_client else {},
                }
                self.wfile.write(json.dumps(status_data, indent=2).encode())
            elif self.path == '/force_check':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                result = {"message": "Force check initiated"}
                self.wfile.write(json.dumps(result).encode())
            else:
                self.send_response(404)
                self.end_headers()

    try:
        port = getattr(config.deployment, 'port', 8080) if hasattr(config, 'deployment') else 8080
        server = HTTPServer(('0.0.0.0', port), HealthHandler)
        logger.info(f"{EMOJI['HEALTH']} Health server running on port {port}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Health server error: {e}")


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global running
    logger.info(f"{EMOJI['STOP']} Shutting down...")
    running = False


# ========== MAIN ==========

def main():
    """Main entry point."""
    global running

    logger.info("=" * 70)
    logger.info(f"{EMOJI['START']} AI TRADING BOT v3.4.5")
    logger.info("📊 SUPER TDI + MACD + SUPER BOLLINGER BANDS STRATEGY")
    logger.info("=" * 70)
    logger.info("📋 Strategy Features:")
    logger.info("  - Super TDI Zones (25/35/50/65/75) - Your RSI")
    logger.info("  - MACD Confirmation (Fast=12, Slow=26, Signal=9)")
    logger.info("  - Super Bollinger Bands (34 period, 1.75 dev)")
    logger.info("  - 5-Step Entry Checklist")
    logger.info("  - Cheat Sheet Explanations")
    logger.info("  - Divergence Detection")
    logger.info("  - Candle Pattern Recognition")
    logger.info("  - Support/Resistance Levels")
    logger.info("  - BB Squeeze Detection")
    logger.info("  - Session Filtering")
    logger.info(f"  - AI Validation: {'✅' if ai_analyzer.enabled else '❌'}")
    logger.info(f"  - MongoDB: {'✅' if mongodb_client.is_available() else '❌'}")
    logger.info("=" * 70)
    logger.info("⏰ Exit Rules:")
    logger.info("  - Max Hold: 60 minutes (1 hour)")
    logger.info("  - Min Hold: 15 minutes")
    logger.info("  - TP/SL: Always priority")
    logger.info("  - TDI Target: BUY → 70, SELL → 30")
    logger.info("=" * 70)
    logger.info(f"📈 Symbols: {config.market.symbols}")
    logger.info(f"⏱️ Timeframe: {config.market.timeframe}")
    logger.info(f"📋 HTF: {config.market.htf_timeframe} (Context Only)")
    logger.info(f"🎯 Run Mode: {config.deployment.run_mode.value if hasattr(config.deployment, 'run_mode') else 'UNKNOWN'}")
    min_conditions = getattr(config.strategy, 'min_conditions_for_signal', 3)
    htf_required = getattr(config.strategy, 'require_htf_alignment', False)
    macd_required = getattr(config.strategy, 'require_macd_confirmation', True)
    logger.info(f"📊 Min Conditions: {min_conditions}/5")
    logger.info(f"📊 MACD Required: {'✅' if macd_required else '❌'}")
    logger.info(f"📊 HTF Alignment Required: {'✅' if htf_required else '❌'} (DISABLED - Context Only)")
    logger.info("=" * 70)

    # Setup signal handlers - FIXED: Use signal_module instead of signal
    signal_module.signal(signal_module.SIGINT, signal_handler)
    signal_module.signal(signal_module.SIGTERM, signal_handler)

    # Start health server
    threading.Thread(target=run_health_server, daemon=True).start()

    # ===== RECOVER ACTIVE SIGNALS FROM DATABASE =====
    recover_and_sync_signals()

    # ===== LOG ACTIVE SIGNALS =====
    active = signal_manager.get_all_active_signals()
    if active:
        logger.info(f"{EMOJI['ACTIVE']} Active signals: {len(active)}")
        for symbol, signal_obj in active.items():  # ← Renamed to signal_obj
            logger.info(f"  {symbol}: entry={signal_obj.entry_price:.4f}, age={signal_obj.get_age_minutes():.1f}min, doc={signal_obj.db_doc_id}")

    # Symbols to monitor
    symbols = config.market.symbols
    logger.info(f"{EMOJI['MONITOR']} Monitoring {len(symbols)} symbols: {', '.join(symbols[:5])}...")

    # Send startup message
    if telegram_bot.enabled:
        telegram_bot.send_startup_message(
            symbols=symbols,
            config_info={
                'environment': config.deployment.environment.value if hasattr(config.deployment, 'environment') else 'unknown',
                'timeframe': config.market.timeframe,
                'ltf_timeframe': getattr(config.market, 'ltf_timeframe', '1m'),
                'htf_timeframe': config.market.htf_timeframe,
                'ai_enabled': ai_analyzer.enabled,
                'min_conditions': min_conditions,
                'rrr_range': f"{getattr(config.strategy, 'min_rrr', 1.5)}-{getattr(config.strategy, 'max_rrr', 4.0)}",
                'bb_period': getattr(config.strategy, 'bb_period', 34),
                'bb_deviation': getattr(config.strategy, 'bb_deviation', 1.75),
                'max_hold_minutes': 60,
                'htf_alignment_required': htf_required,
                'macd_required': macd_required,
                'macd_settings': {
                    'fast': getattr(config.strategy, 'macd_fast', 12),
                    'slow': getattr(config.strategy, 'macd_slow', 26),
                    'signal': getattr(config.strategy, 'macd_signal', 9),
                }
            }
        )

    # Main loop
    cycle = 0
    while running:
        try:
            cycle += 1
            start_time = time.time()

            # Process each symbol
            for symbol in symbols:
                if not running:
                    break
                try:
                    process_symbol(symbol)
                except Exception as e:
                    logger.error(f"Symbol {symbol} error: {e}")

            # Check active signals (this handles exits)
            check_active_signals()

            # Log status every 10 cycles
            if cycle % 10 == 0:
                logger.info(
                    f"{EMOJI['MONITOR']} Cycle {cycle}: "
                    f"{bot_stats['signals_generated']} signals, "
                    f"{bot_stats['ai_approved']} approved, "
                    f"{bot_stats['ai_rejected']} rejected, "
                    f"{bot_stats['active_signals']} active, "
                    f"Exits: {bot_stats['exits']['profit']}P/{bot_stats['exits']['loss']}L/{bot_stats['exits']['break_even']}BE/{bot_stats['exits']['exit_1h']}1h/{bot_stats['exits']['exit_target']}T, "
                    f"PnL: ${bot_stats['total_pnl']:.2f}, "
                    f"{bot_stats['errors']} errors"
                )

            # Sleep
            elapsed = time.time() - start_time
            polling_interval = getattr(config.market, 'polling_interval_seconds', 15)
            sleep_time = max(0, polling_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

    # Cleanup
    logger.info(f"{EMOJI['STOP']} Shutting down...")
    try:
        if mongodb_client is not None:
            mongodb_client.cleanup()
        data_client.cleanup()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")

    # Send shutdown message
    if telegram_bot.enabled:
        telegram_bot.send_shutdown_message({
            'signals_generated': bot_stats['signals_generated'],
            'ai_approved': bot_stats['ai_approved'],
            'ai_rejected': bot_stats['ai_rejected'],
            'total_pnl': bot_stats['total_pnl'],
            'avg_rrr': 0,
            'exits': bot_stats['exits'],
        })

    logger.info(f"{EMOJI['RESULT']} Final stats:")
    logger.info(f"  Signals Generated: {bot_stats['signals_generated']}")
    logger.info(f"  AI Approved: {bot_stats['ai_approved']}")
    logger.info(f"  AI Rejected: {bot_stats['ai_rejected']}")
    logger.info(f"  Total PnL: ${bot_stats['total_pnl']:.2f}")
    logger.info(f"  Exits: {bot_stats['exits']['profit']}P / {bot_stats['exits']['loss']}L / {bot_stats['exits']['break_even']}BE / {bot_stats['exits']['exit_1h']}1h / {bot_stats['exits']['exit_target']}T")
    logger.info(f"{EMOJI['SUCCESS']} Bot stopped")


if __name__ == "__main__":
    main()
