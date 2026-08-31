"""
Signal Manager - Handles signal lifecycle, debouncing, and duplicate prevention
ALIGNED WITH YOUR MANUAL STRATEGY:
- ENTRY: TDI below 50 (BUY) or above 50 (SELL)
- EXIT: BUY → TDI 70, SELL → TDI 30
- TP/SL: Always priority
- Max hold: Force exit
Version: 3.4.1 - FIXED: Added missing pandas import
"""

import logging
import time
import pandas as pd  # ← ADD THIS IMPORT
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback

try:
    from utils.mongodb_client import mongodb_client as db_client, convert_numpy_types
except ImportError:
    db_client = None
    convert_numpy_types = None
    logging.warning("No database client available. Signal persistence disabled.")

logger = logging.getLogger(__name__)

EMOJI = {
    "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "INFO": "ℹ️",
    "SIGNAL": "📡", "PROFIT": "💰", "LOSS": "💸", "LOCK": "🔒",
    "UNLOCK": "🔓", "MONITOR": "📊", "WAIT": "⏳", "REJECT": "🚫",
    "RESTORE": "♻️", "BREAK": "⏹️", "LTF": "⏱️", "HTF": "📊",
    "CONFLICT": "⚔️", "ACTIVE": "🟢", "RESOLVED": "✅", "SCORE": "🎯",
    "GRADE_A": "🏆", "GRADE_B": "🥈", "GRADE_C": "🥉",
    "APPROVED": "✅", "DB": "💾", "MONGODB": "🍃",
    "DIVERGENCE": "↩️", "PATTERN": "🕯️", "S_R": "📊", "SESSION": "🌍",
    "TDI": "📈", "BB": "📊", "MACD": "📊", "CHEAT": "📋",
    "AI": "🤖", "EXIT": "⏰", "CONTINUATION": "🚀",
}


class TradeLifecycle(str, Enum):
    ACTIVE = "ACTIVE"
    PROFIT = "PROFIT"
    LOSS = "LOSS"
    BREAK_EVEN = "BREAK_EVEN"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXIT_TARGET = "EXIT_TARGET"  # TDI reached target (70 for BUY, 30 for SELL)
    EXIT_1H = "EXIT_1H"


@dataclass
class SignalData:
    """Signal data structure aligned with YOUR strategy."""
    symbol: str
    signal_type: str
    entry_price: float
    entry_time: str
    stop_loss: float
    take_profit: float
    confidence: float
    status: str = "ACTIVE"
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0

    # TDI fields
    tdi_level: float = 0.0
    tdi_zone: str = "NEUTRAL"
    tdi_fast: float = 0.0
    tdi_slow: float = 0.0

    # YOUR strategy specific
    entry_tdi: float = 0.0
    target_tdi: float = 0.0  # 70 for BUY, 30 for SELL
    strategy_type: str = "REVERSAL_TO_TARGET"

    # BB fields
    bb_position: float = 0.5
    bb_touch_lower: bool = False
    bb_touch_upper: bool = False

    # MACD fields
    macd_bullish: bool = False
    macd_bearish: bool = False
    macd_histogram: float = 0.0

    # Conditions
    conditions_met: int = 0
    conditions_total: int = 4
    condition_1_tdi_zone: bool = False
    condition_2_tdi_cross: bool = False
    condition_3_bb_touch: bool = False
    condition_4_candles_shrinking: bool = False

    # Quality
    quality_score: int = 50
    total_score: int = 0
    grade: str = ""

    # AI
    ai_decision: str = "PENDING"
    ai_confidence: float = 0.0
    ai_reasoning: str = ""

    # Risk
    rrr: float = 0.0
    signal_strength: str = "SOFT"
    risk_multiplier: float = 1.0

    raw_data: Dict = field(default_factory=dict)
    bar_count: int = 0
    max_hold_minutes: int = 60
    min_hold_minutes: int = 15
    highest_price: Optional[float] = None
    lowest_price: Optional[float] = None
    db_doc_id: Optional[str] = None
    last_checked_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "fees": self.fees,
            "tdi_level": self.tdi_level,
            "tdi_zone": self.tdi_zone,
            "entry_tdi": self.entry_tdi,
            "target_tdi": self.target_tdi,
            "bb_position": self.bb_position,
            "macd_bullish": self.macd_bullish,
            "macd_bearish": self.macd_bearish,
            "macd_histogram": self.macd_histogram,
            "conditions_met": self.conditions_met,
            "quality_score": self.quality_score,
            "total_score": self.total_score if self.total_score else self.quality_score,
            "grade": self.grade,
            "rrr": self.rrr,
            "signal_strength": self.signal_strength,
        }

    def get_age_minutes(self) -> float:
        try:
            entry_time = datetime.fromisoformat(self.entry_time)
            return (datetime.now() - entry_time).total_seconds() / 60
        except (ValueError, TypeError):
            return 0

    def get_age_seconds(self) -> float:
        try:
            entry_time = datetime.fromisoformat(self.entry_time)
            return (datetime.now() - entry_time).total_seconds()
        except (ValueError, TypeError):
            return 0


class SignalManager:
    """
    Signal Manager aligned with YOUR actual strategy.

    YOUR STRATEGY:
    - BUY: Enter TDI below 50 → Exit at TDI 70
    - SELL: Enter TDI above 50 → Exit at TDI 30
    - TP/SL: Always priority
    - Max hold: Force exit
    """

    def __init__(self):
        self.active_signals: Dict[str, SignalData] = {}
        self.signal_history: List[SignalData] = []
        self.max_history = 1000
        self.symbol_last_signal: Dict[str, datetime] = {}
        self.symbol_signal_count: Dict[str, int] = {}
        self.global_signal_timestamps: List[datetime] = []

        # Cooldown settings
        self.SYMBOL_COOLDOWN_MINUTES = 30
        self.MAX_SIGNALS_PER_HOUR = 8

        # Exit settings (YOUR STRATEGY)
        self.MAX_HOLD_MINUTES = 60
        self.MIN_HOLD_MINUTES = 15
        self.BUY_EXIT_TARGET = 70.0   # TDI target for BUY
        self.SELL_EXIT_TARGET = 30.0  # TDI target for SELL

        # Grade thresholds
        self.GRADE_A_THRESHOLD = 80
        self.GRADE_B_THRESHOLD = 60
        self.GRADE_C_THRESHOLD = 50
        self.MIN_SIGNAL_SCORE = 50

        self.db_client = db_client
        self.db_enabled = self.db_client is not None and self.db_client.is_available() if self.db_client is not None else False

        logger.info(f"✅ SIGNAL_MANAGER v3.4.4: Initialized")
        logger.info(f"  - Strategy: BUY → TDI 70, SELL → TDI 30")
        logger.info(f"  - Max Hold: {self.MAX_HOLD_MINUTES} minutes")
        logger.info(f"  - Min Hold: {self.MIN_HOLD_MINUTES} minutes")
        logger.info(f"  - Grade A+: 90+ | A: 80+ | B+: 72+ | B: 60+ | C: 50+")
        logger.info(f"  - Min Signal Score: {self.MIN_SIGNAL_SCORE}")

    def _get_grade(self, score: int) -> str:
        if score >= 90:
            return "A+"
        elif score >= self.GRADE_A_THRESHOLD:
            return "A"
        elif score >= 72:
            return "B+"
        elif score >= self.GRADE_B_THRESHOLD:
            return "B"
        elif score >= self.GRADE_C_THRESHOLD:
            return "C"
        else:
            return "D"

    def is_symbol_locked(self, symbol: str) -> bool:
        if symbol in self.active_signals:
            signal = self.active_signals[symbol]
            if signal.status == "ACTIVE":
                return True
            else:
                del self.active_signals[symbol]
        return False

    def lock_symbol(self, symbol: str, signal_type: str, entry_price: float,
                    raw_data: Dict, **kwargs) -> bool:
        """Lock a symbol with a new signal."""
        try:
            quality_score = raw_data.get('quality_score', 0)
            grade = self._get_grade(quality_score)

            # Check minimum score
            if quality_score < self.MIN_SIGNAL_SCORE:
                logger.warning(f"{EMOJI['REJECT']} Score {quality_score} below minimum for {symbol}")
                return False

            # Check cooldown
            if symbol in self.symbol_last_signal:
                time_since = (datetime.now() - self.symbol_last_signal[symbol]).total_seconds() / 60
                if time_since < self.SYMBOL_COOLDOWN_MINUTES:
                    logger.warning(f"{EMOJI['REJECT']} Cooldown: {self.SYMBOL_COOLDOWN_MINUTES - time_since:.1f}min remaining for {symbol}")
                    return False

            # Check global limit
            now = datetime.now()
            self.global_signal_timestamps = [ts for ts in self.global_signal_timestamps if (now - ts).total_seconds() < 3600]
            if len(self.global_signal_timestamps) >= self.MAX_SIGNALS_PER_HOUR:
                logger.warning(f"{EMOJI['REJECT']} Global limit: {self.MAX_SIGNALS_PER_HOUR}/hour")
                return False

            # Create signal
            signal = SignalData(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry_price,
                entry_time=datetime.now().isoformat(),
                stop_loss=raw_data.get('stop_loss', 0),
                take_profit=raw_data.get('take_profit', 0),
                confidence=raw_data.get('confidence', 0.5),

                # TDI
                tdi_level=raw_data.get('tdi_level', 50),
                tdi_zone=raw_data.get('tdi_zone', 'NEUTRAL'),
                tdi_fast=raw_data.get('tdi_fast', 50),
                tdi_slow=raw_data.get('tdi_slow', 50),

                # YOUR strategy specific
                entry_tdi=raw_data.get('entry_tdi', 50),
                target_tdi=raw_data.get('target_tdi', 50),
                strategy_type=raw_data.get('strategy_type', 'REVERSAL_TO_TARGET'),

                # BB
                bb_position=raw_data.get('bb_position', 0.5),
                bb_touch_lower=raw_data.get('touch_lower', False),
                bb_touch_upper=raw_data.get('touch_upper', False),

                # MACD
                macd_bullish=raw_data.get('macd_bullish', False),
                macd_bearish=raw_data.get('macd_bearish', False),
                macd_histogram=raw_data.get('macd_histogram', 0.0),

                # Conditions
                conditions_met=raw_data.get('conditions_met', 0),
                conditions_total=raw_data.get('conditions_total', 4),
                condition_1_tdi_zone=raw_data.get('condition_1_tdi_zone', False),
                condition_2_tdi_cross=raw_data.get('condition_2_tdi_cross', False),
                condition_3_bb_touch=raw_data.get('condition_3_bb_touch', False),
                condition_4_candles_shrinking=raw_data.get('condition_4_candles_shrinking', False),

                # Quality
                quality_score=quality_score,
                total_score=quality_score,
                grade=grade,

                # Risk
                rrr=raw_data.get('rrr', 0),
                signal_strength=raw_data.get('signal_strength', 'SOFT'),
                risk_multiplier=raw_data.get('risk_multiplier', 1.0),

                raw_data=raw_data,
                max_hold_minutes=self.MAX_HOLD_MINUTES,
                highest_price=entry_price,
                lowest_price=entry_price,
            )

            # Save to DB
            if self.db_enabled and convert_numpy_types:
                try:
                    signal_dict = signal.to_dict()
                    # Convert numpy types before saving
                    signal_dict = convert_numpy_types(signal_dict)
                    doc_id = self._save_to_db(signal_dict)
                    if doc_id:
                        signal.db_doc_id = doc_id
                        raw_data['db_doc_id'] = doc_id
                except Exception as e:
                    logger.warning(f"{EMOJI['WARNING']} Failed to save signal to DB: {e}")

            self.active_signals[symbol] = signal
            self.symbol_last_signal[symbol] = datetime.now()
            self.symbol_signal_count[symbol] = self.symbol_signal_count.get(symbol, 0) + 1
            self.global_signal_timestamps.append(datetime.now())

            logger.info(
                f"{EMOJI['LOCK']} {signal_type} {symbol} @ {entry_price:.4f} | "
                f"TDI: {signal.entry_tdi:.1f} → Target: {signal.target_tdi:.1f} | "
                f"Grade: {grade} | Score: {quality_score}"
            )
            return True

        except Exception as e:
            logger.error(f"{EMOJI['ERROR']} Error locking {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_to_db(self, signal_data: Dict) -> Optional[str]:
        if not self.db_enabled:
            return None
        try:
            return self.db_client.save_signal(signal_data)
        except Exception as e:
            logger.warning(f"{EMOJI['WARNING']} Failed to save signal: {e}")
            return None

    def get_all_active_signals(self) -> Dict[str, SignalData]:
        return self.active_signals.copy()

    def check_active_signal(self, symbol: str, current_price: float,
                           last_candle: Dict) -> Tuple[str, float, Optional[SignalData]]:
        """
        Check active signal - ALIGNED WITH YOUR ACTUAL STRATEGY.

        YOUR EXIT RULES:
        - BUY: Exit when TDI reaches 70.0
        - SELL: Exit when TDI reaches 30.0
        - TP/SL: Always priority
        - Max hold: Force exit
        """
        if symbol not in self.active_signals:
            return "NO_SIGNAL", 0, None

        signal = self.active_signals[symbol]
        signal.bar_count += 1
        signal.last_checked_at = datetime.now().isoformat()

        # Update highest/lowest
        if signal.highest_price is None or current_price > signal.highest_price:
            signal.highest_price = current_price
        if signal.lowest_price is None or current_price < signal.lowest_price:
            signal.lowest_price = current_price

        age_minutes = signal.get_age_minutes()

        # ===== MINIMUM HOLD TIME =====
        if age_minutes < self.MIN_HOLD_MINUTES:
            return "ACTIVE", current_price - signal.entry_price, signal

        # ===== RULE 1: CHECK STOP LOSS (HIGHEST PRIORITY) =====
        if signal.signal_type == "BUY":
            if current_price <= signal.stop_loss:
                updated = self._unlock_symbol(symbol, TradeLifecycle.LOSS, current_price)
                logger.info(f"{EMOJI['LOSS']} {symbol}: SL HIT at ${current_price:.4f}")
                return TradeLifecycle.LOSS.value, current_price - signal.entry_price, updated
        else:
            if current_price >= signal.stop_loss:
                updated = self._unlock_symbol(symbol, TradeLifecycle.LOSS, current_price)
                logger.info(f"{EMOJI['LOSS']} {symbol}: SL HIT at ${current_price:.4f}")
                return TradeLifecycle.LOSS.value, signal.entry_price - current_price, updated

        # ===== RULE 2: CHECK TAKE PROFIT =====
        if signal.signal_type == "BUY":
            if current_price >= signal.take_profit:
                updated = self._unlock_symbol(symbol, TradeLifecycle.PROFIT, current_price)
                logger.info(f"{EMOJI['PROFIT']} {symbol}: TP HIT at ${current_price:.4f}")
                return TradeLifecycle.PROFIT.value, current_price - signal.entry_price, updated
        else:
            if current_price <= signal.take_profit:
                updated = self._unlock_symbol(symbol, TradeLifecycle.PROFIT, current_price)
                logger.info(f"{EMOJI['PROFIT']} {symbol}: TP HIT at ${current_price:.4f}")
                return TradeLifecycle.PROFIT.value, signal.entry_price - current_price, updated

        # ===== RULE 3: TDI TARGET EXIT (YOUR STRATEGY) =====
        # Get current TDI from last_candle
        current_tdi = last_candle.get('tdi_slow_ma', signal.tdi_level)

        if signal.signal_type == "BUY":
            # BUY: Entered below 50, exit when TDI reaches 70
            if current_tdi >= self.BUY_EXIT_TARGET:
                updated = self._unlock_symbol(symbol, TradeLifecycle.EXIT_TARGET, current_price)
                logger.info(f"{EMOJI['EXIT']} {symbol}: TDI reached {self.BUY_EXIT_TARGET} - EXIT at ${current_price:.4f}")
                return "EXIT_TARGET", current_price - signal.entry_price, updated

        elif signal.signal_type == "SELL":
            # SELL: Entered above 50, exit when TDI reaches 30
            if current_tdi <= self.SELL_EXIT_TARGET:
                updated = self._unlock_symbol(symbol, TradeLifecycle.EXIT_TARGET, current_price)
                logger.info(f"{EMOJI['EXIT']} {symbol}: TDI reached {self.SELL_EXIT_TARGET} - EXIT at ${current_price:.4f}")
                return "EXIT_TARGET", signal.entry_price - current_price, updated

        # ===== RULE 4: MAX HOLD REACHED =====
        if age_minutes >= signal.max_hold_minutes:
            updated = self._unlock_symbol(symbol, TradeLifecycle.EXIT_1H, current_price)
            logger.info(f"{EMOJI['EXIT']} {symbol}: MAX HOLD ({age_minutes:.1f}min) - EXIT at ${current_price:.4f}")
            return "EXIT_1H", current_price - signal.entry_price, updated

        # ===== ACTIVE =====
        return "ACTIVE", current_price - signal.entry_price, signal

    def _unlock_symbol(self, symbol: str, status: TradeLifecycle, exit_price: float) -> Optional[SignalData]:
        """Unlock a symbol and update its status."""
        if symbol not in self.active_signals:
            return None

        signal = self.active_signals[symbol]
        status_str = status.value if hasattr(status, 'value') else str(status)

        signal.status = status_str
        signal.exit_time = datetime.now().isoformat()
        signal.exit_price = exit_price

        # Calculate PnL
        if signal.signal_type == "BUY":
            signal.pnl = exit_price - signal.entry_price
        else:
            signal.pnl = signal.entry_price - exit_price

        if getattr(signal, "total_score", 0) == 0:
            signal.total_score = signal.quality_score

        # Fees
        fee = signal.entry_price * 0.0011 + exit_price * 0.0011
        signal.fees = fee
        signal.pnl = signal.pnl - fee
        signal.pnl_percent = (signal.pnl / signal.entry_price) * 100 if signal.entry_price > 0 else 0

        # Update DB
        if signal.db_doc_id and self.db_enabled:
            try:
                update_data = signal.to_dict()
                if convert_numpy_types:
                    update_data = convert_numpy_types(update_data)
                self._update_in_db(signal.db_doc_id, status_str, update_data)
            except Exception as e:
                logger.warning(f"{EMOJI['WARNING']} Failed to update signal in DB: {e}")

        self.signal_history.append(signal)
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)

        del self.active_signals[symbol]

        hold_minutes = signal.get_age_minutes()
        logger.info(
            f"{EMOJI['UNLOCK']} {symbol}: {status_str} | "
            f"PnL: ${signal.pnl:.2f} ({signal.pnl_percent:.2f}%) | "
            f"Hold: {hold_minutes:.1f}min | "
            f"TDI: {signal.entry_tdi:.1f} → {signal.tdi_level:.1f}"
        )
        return signal

    def _update_in_db(self, doc_id: str, status: str, update_data: Dict) -> bool:
        if not self.db_enabled or not doc_id:
            return False
        try:
            return self.db_client.update_signal_status(doc_id, status, update_data)
        except Exception as e:
            logger.warning(f"{EMOJI['WARNING']} Failed to update signal: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get signal manager statistics."""
        active_count = len(self.active_signals)
        return {
            "active": active_count,
            "history": len(self.signal_history),
            "active_symbols": list(self.active_signals.keys()),
            "exit_rules": {
                "max_hold_minutes": self.MAX_HOLD_MINUTES,
                "min_hold_minutes": self.MIN_HOLD_MINUTES,
                "buy_exit_target": self.BUY_EXIT_TARGET,
                "sell_exit_target": self.SELL_EXIT_TARGET,
            },
            "version": "3.4.4"
        }


# Singleton
signal_manager = SignalManager()
