"""
Signal Engine - Super TDI + MACD + Super Bollinger Bands
ALIGNED WITH YOUR ACTUAL STRATEGY:
- BUY: Enter below 50 → Exit at 70
- SELL: Enter above 50 → Exit at 30
Version: 3.4.1 - FIXED: Exit targets (70 for BUY, 30 for SELL)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from strategy.tdi_detector import TDIDetector
from strategy.bb_detector import BBDetector
from strategy.cheat_sheet import SignalCheatSheet
from strategy.ai_analyzer import ai_analyzer
from settings import config

logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Signal Engine aligned with YOUR actual strategy.

    YOUR STRATEGY:
    - BUY: Enter TDI below 50 → Exit at TDI 70 (Soft Sell zone)
    - SELL: Enter TDI above 50 → Exit at TDI 30 (Soft Buy zone)
    - PRIMARY: TDI (RSI) - Entry and exit based on TDI levels
    - SECONDARY: MACD - Confirmation
    - CONFIRMATION: BB - Last thing to confirm
    """

    def __init__(self, use_ai: bool = True):
        self.tdi_detector = TDIDetector()
        self.bb_detector = BBDetector()
        self.cheat_sheet = SignalCheatSheet()

        # AI
        self.use_ai = use_ai and ai_analyzer.enabled if ai_analyzer else False

        # Strategy parameters
        self.tdi_center_line = getattr(config.strategy, 'tdi_center_line', 50.0)
        self.tdi_soft_buy = getattr(config.strategy, 'tdi_soft_buy', 35.0)
        self.tdi_hard_buy = getattr(config.strategy, 'tdi_oversold', 25.0)
        self.tdi_soft_sell = getattr(config.strategy, 'tdi_soft_sell', 65.0)
        self.tdi_hard_sell = getattr(config.strategy, 'tdi_overbought', 75.0)

        # YOUR EXIT TARGETS
        self.BUY_EXIT_TARGET = 70.0   # Between Soft Sell (65) and Hard Sell (75)
        self.SELL_EXIT_TARGET = 30.0  # Between Soft Buy (35) and Hard Buy (25)

        self.min_rrr = getattr(config.strategy, 'min_rrr', 1.5)
        self.default_rrr = getattr(config.strategy, 'default_rrr', 2.0)
        self.max_rrr = getattr(config.strategy, 'max_rrr', 4.0)
        self.min_quality_score = getattr(config.strategy, 'min_quality_score', 50)

        # Exit parameters
        self.max_hold_minutes = 60  # Max hold
        self.extension_minutes = 30  # Extra hold if breakout

        # Grade thresholds
        self.grade_a_threshold = getattr(config.strategy, 'grade_a_threshold', 80)
        self.grade_b_threshold = getattr(config.strategy, 'grade_b_threshold', 60)
        self.grade_c_threshold = getattr(config.strategy, 'grade_c_threshold', 55)

        logger.info(f"🔧 Signal Engine v3.4.1 initialized - AI: {'✅' if self.use_ai else '❌'}")
        logger.info(f"   RRR Range: {self.min_rrr} - {self.max_rrr}")
        logger.info(f"   Default RRR: {self.default_rrr}")
        logger.info(f"   TDI Center Line: {self.tdi_center_line}")
        logger.info(f"   🎯 BUY Exit Target: TDI {self.BUY_EXIT_TARGET}")
        logger.info(f"   🎯 SELL Exit Target: TDI {self.SELL_EXIT_TARGET}")
        logger.info(f"   Strategy: Enter at extremes → Exit at opposite extremes")

    def process(self, df: pd.DataFrame, symbol: str = "UNKNOWN", htf_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Process signal aligned with YOUR actual strategy.
        """
        if df is None or df.empty:
            return self._no_signal(symbol, "No data")

        # ===== STEP 1: GET TDI OPPORTUNITY (PRIMARY) =====
        tdi_result = self.tdi_detector.detect_opportunity(df)

        logger.debug(f"📊 {symbol} TDI: {tdi_result.get('opportunity')} | Zone: {tdi_result.get('tdi_zone')} | Direction: {tdi_result.get('direction')} | TDI: {tdi_result.get('tdi_level', 50):.1f}")

        # ===== STEP 2: GET BB INTERACTION (CONFIRMATION) =====
        bb_result = self.bb_detector.detect_bb_interaction(df)

        logger.debug(f"📊 {symbol} BB: Position: {bb_result.get('position', 0.5):.2f} | Touch Lower: {bb_result.get('touch_lower', False)} | Touch Upper: {bb_result.get('touch_upper', False)}")

        # ===== STEP 3: GET MACD (SECONDARY) =====
        macd_result = self._check_macd(df)

        if macd_result:
            logger.debug(f"📊 {symbol} MACD: Histogram={macd_result.get('histogram', 0):.4f}, Bullish={macd_result.get('bullish', False)}")

        # ===== STEP 4: DETERMINE DIRECTION (YOUR STRATEGY) =====
        direction = self._determine_direction(tdi_result, bb_result, macd_result)

        if direction == "NONE":
            tdi_level = tdi_result.get('tdi_level', 50)
            tdi_zone = tdi_result.get('tdi_zone', 'UNKNOWN')
            return self._no_signal(symbol, f"TDI: {tdi_level:.1f} ({tdi_zone})")

        # ===== STEP 5: CHECK CONDITIONS =====
        conditions_ok, conditions_reason, conditions_met = self._check_conditions(
            tdi_result, bb_result, macd_result, direction
        )

        if not conditions_ok:
            logger.debug(f"📊 {symbol} {direction} rejected: {conditions_reason}")
            return self._no_signal(symbol, conditions_reason)

        # ===== STEP 6: GENERATE SIGNAL =====
        if direction == "BUY":
            signal_data = self._generate_buy_signal(symbol, df, tdi_result, bb_result, macd_result)
            logger.info(f"🟢 {symbol}: BUY signal - TDI: {tdi_result.get('tdi_level', 50):.1f} → Target: {self.BUY_EXIT_TARGET}")
        else:
            signal_data = self._generate_sell_signal(symbol, df, tdi_result, bb_result, macd_result)
            logger.info(f"🔴 {symbol}: SELL signal - TDI: {tdi_result.get('tdi_level', 50):.1f} → Target: {self.SELL_EXIT_TARGET}")

        if signal_data is None:
            return self._no_signal(symbol, "Signal generation failed")

        # ===== STEP 7: QUALITY CHECK =====
        if signal_data.get('quality_score', 0) < self.min_quality_score:
            logger.debug(f"🚫 {symbol}: Quality score {signal_data.get('quality_score', 0)} below minimum")
            return self._no_signal(symbol, "Quality score too low")

        # ===== STEP 8: AI VALIDATION =====
        if self.use_ai:
            signal_data = self._apply_ai_validation(symbol, signal_data)
            if signal_data.get('signal') == 'NO_TRADE':
                return signal_data

        # ===== STEP 9: CHEAT SHEET =====
        signal_data['cheat_sheet'] = self._generate_cheat_sheet(signal_data)

        self.last_signal = signal_data
        self.last_signal_time = datetime.now()

        exit_target = self.BUY_EXIT_TARGET if direction == "BUY" else self.SELL_EXIT_TARGET
        logger.info(f"✅ {symbol}: {direction} signal - TDI: {tdi_result.get('tdi_level', 50):.1f} → Target: {exit_target}")
        return signal_data

    def _determine_direction(self, tdi_data: Dict, bb_data: Dict, macd_data: Dict) -> str:
        """
        Determine direction based on YOUR actual strategy.

        YOUR STRATEGY:
        - BUY: TDI below 50 + Green above Red → Trade to 70
        - SELL: TDI above 50 + Green below Red → Trade down to 30
        """
        tdi_level = tdi_data.get('tdi_level', 50)
        green_above_red = tdi_data.get('green_above_red', False)

        # BUY: TDI below center line (50) with bullish cross
        if tdi_level < self.tdi_center_line and green_above_red:
            return "BUY"

        # SELL: TDI above center line (50) with bearish cross
        if tdi_level > self.tdi_center_line and not green_above_red:
            return "SELL"

        return "NONE"

    def _check_conditions(self, tdi_data: Dict, bb_data: Dict, macd_data: Dict, direction: str) -> Tuple[bool, str, int]:
        """
        Check conditions based on YOUR strategy.
        """
        conditions_met = 0
        reasons = []

        tdi_zone = tdi_data.get('tdi_zone', '')
        tdi_level = tdi_data.get('tdi_level', 50)

        if direction == "BUY":
            # Condition 1: TDI in buyer zone (REQUIRED)
            if tdi_zone in ['OVERSOLD', 'SOFT_BUY', 'BUY_ZONE']:
                conditions_met += 1
                reasons.append(f"TDI in {tdi_zone} ({tdi_level:.1f})")
            else:
                return False, f"TDI not in buyer zone: {tdi_zone}", 0

            # Condition 2: Green above Red (REQUIRED)
            if tdi_data.get('green_above_red', False):
                conditions_met += 1
                reasons.append("Green above Red ✅")
            else:
                return False, "Green not above Red", 0

            # Condition 3: MACD (SECONDARY)
            if macd_data and macd_data.get('bullish', False):
                conditions_met += 1
                reasons.append("MACD bullish ✅")

            # Condition 4: BB (CONFIRMATION)
            if bb_data.get('touch_lower', False) or bb_data.get('position', 0.5) < 0.35:
                conditions_met += 1
                reasons.append("BB near lower ✅")
            else:
                reasons.append("BB waiting (optional)")

        else:  # SELL
            # Condition 1: TDI in seller zone (REQUIRED)
            if tdi_zone in ['SOFT_SELL', 'OVERBOUGHT', 'SELL_ZONE']:
                conditions_met += 1
                reasons.append(f"TDI in {tdi_zone} ({tdi_level:.1f})")
            else:
                return False, f"TDI not in seller zone: {tdi_zone}", 0

            # Condition 2: Green below Red (REQUIRED)
            if not tdi_data.get('green_above_red', True):
                conditions_met += 1
                reasons.append("Green below Red ✅")
            else:
                return False, "Green not below Red", 0

            # Condition 3: MACD (SECONDARY)
            if macd_data and macd_data.get('bearish', False):
                conditions_met += 1
                reasons.append("MACD bearish ✅")

            # Condition 4: BB (CONFIRMATION)
            if bb_data.get('touch_upper', False) or bb_data.get('position', 0.5) > 0.65:
                conditions_met += 1
                reasons.append("BB near upper ✅")
            else:
                reasons.append("BB waiting (optional)")

        # Minimum 2 conditions (TDI zone + cross) = valid signal
        return True, " | ".join(reasons), conditions_met

    def _check_macd(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check MACD for confirmation."""
        try:
            if 'macd' not in df.columns or 'macd_signal' not in df.columns:
                return None

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            macd = last.get('macd', 0)
            signal = last.get('macd_signal', 0)
            histogram = last.get('macd_histogram', 0)
            prev_histogram = prev.get('macd_histogram', 0)

            return {
                'macd': macd,
                'signal': signal,
                'histogram': histogram,
                'prev_histogram': prev_histogram,
                'bullish': macd > signal and histogram > prev_histogram,
                'bearish': macd < signal and histogram < prev_histogram,
                'above_signal': macd > signal,
                'hist_rising': histogram > prev_histogram,
            }
        except Exception as e:
            logger.debug(f"MACD check error: {e}")
            return None

    def _generate_buy_signal(self, symbol: str, df: pd.DataFrame,
                             tdi_data: Dict, bb_data: Dict, macd_data: Dict) -> Dict[str, Any]:
        """Generate BUY signal - Enter below 50 → Exit at 70."""
        last = df.iloc[-1]
        current_price = last.get('close', 0)
        atr = self._calculate_atr(df)

        # Entry TDI level
        entry_tdi = tdi_data.get('tdi_level', 50)

        # YOUR Target is TDI 70 (Soft Sell zone)
        target_tdi = self.BUY_EXIT_TARGET  # 70.0

        # Calculate SL/TP based on ATR
        stop_loss = current_price - (1.5 * atr) if atr > 0 else current_price * 0.985
        take_profit = current_price + (self.default_rrr * (current_price - stop_loss))

        # Calculate RRR
        risk = current_price - stop_loss
        reward = take_profit - current_price
        rrr = reward / risk if risk > 0 else self.default_rrr
        rrr = min(self.max_rrr, max(self.min_rrr, rrr))

        # Quality score
        quality_score = self._calculate_quality_score(tdi_data, bb_data, macd_data, "BUY")
        grade = self._get_grade(quality_score)

        return {
            'symbol': symbol,
            'direction': 'BUY',
            'signal': 'BUY',
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rrr': rrr,
            'confidence': 0.7,
            'quality_score': quality_score,
            'grade': grade,
            # TDI
            'tdi_level': entry_tdi,
            'tdi_zone': tdi_data.get('tdi_zone', 'UNKNOWN'),
            'tdi_zone_description': tdi_data.get('tdi_zone_description', ''),
            'tdi_bullish_cross': tdi_data.get('bullish_cross', False),
            'tdi_fast': tdi_data.get('tdi_fast', 50),
            'tdi_slow': tdi_data.get('tdi_slow', 50),
            'green_above_red': tdi_data.get('green_above_red', False),
            # BB
            'bb_position': bb_data.get('position', 0.5),
            'touch_lower': bb_data.get('touch_lower', False),
            'touch_upper': bb_data.get('touch_upper', False),
            'candles_shrinking': bb_data.get('candles_shrinking', False),
            # MACD
            'macd_bullish': macd_data.get('bullish', False) if macd_data else False,
            'macd_histogram': macd_data.get('histogram', 0) if macd_data else 0,
            'macd_above_signal': macd_data.get('above_signal', False) if macd_data else False,
            # YOUR strategy specific
            'entry_tdi': entry_tdi,
            'target_tdi': target_tdi,
            'strategy_type': 'BUY_TO_70',
            # Conditions
            'conditions_met': 0,
            'conditions_total': 4,
            'condition_1_tdi_zone': tdi_data.get('tdi_zone') in ['OVERSOLD', 'SOFT_BUY', 'BUY_ZONE'],
            'condition_2_tdi_cross': tdi_data.get('green_above_red', False),
            'condition_3_bb_touch': bb_data.get('touch_lower', False) or bb_data.get('position', 0.5) < 0.35,
            'condition_4_candles_shrinking': bb_data.get('candles_shrinking', False),
            # Timestamp
            'timestamp': datetime.now().isoformat(),
            'cheat_sheet': None,
            'ai_decision': 'PENDING',
            'ai_reasoning': 'Awaiting AI analysis...',
            'ai_confidence': 0.0,
            'ai_analysis': {},
            # Exit tracking
            'tdi_target_reached': False,
            'max_hold_minutes': 60,
        }

    def _generate_sell_signal(self, symbol: str, df: pd.DataFrame,
                              tdi_data: Dict, bb_data: Dict, macd_data: Dict) -> Dict[str, Any]:
        """Generate SELL signal - Enter above 50 → Exit at 30."""
        last = df.iloc[-1]
        current_price = last.get('close', 0)
        atr = self._calculate_atr(df)

        # Entry TDI level
        entry_tdi = tdi_data.get('tdi_level', 50)

        # YOUR Target is TDI 30 (Soft Buy zone)
        target_tdi = self.SELL_EXIT_TARGET  # 30.0

        # Calculate SL/TP based on ATR
        stop_loss = current_price + (1.5 * atr) if atr > 0 else current_price * 1.015
        take_profit = current_price - (self.default_rrr * (stop_loss - current_price))

        # Calculate RRR
        risk = stop_loss - current_price
        reward = current_price - take_profit
        rrr = reward / risk if risk > 0 else self.default_rrr
        rrr = min(self.max_rrr, max(self.min_rrr, rrr))

        # Quality score
        quality_score = self._calculate_quality_score(tdi_data, bb_data, macd_data, "SELL")
        grade = self._get_grade(quality_score)

        return {
            'symbol': symbol,
            'direction': 'SELL',
            'signal': 'SELL',
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rrr': rrr,
            'confidence': 0.7,
            'quality_score': quality_score,
            'grade': grade,
            # TDI
            'tdi_level': entry_tdi,
            'tdi_zone': tdi_data.get('tdi_zone', 'UNKNOWN'),
            'tdi_zone_description': tdi_data.get('tdi_zone_description', ''),
            'tdi_bearish_cross': tdi_data.get('bearish_cross', False),
            'tdi_fast': tdi_data.get('tdi_fast', 50),
            'tdi_slow': tdi_data.get('tdi_slow', 50),
            'green_above_red': tdi_data.get('green_above_red', False),
            # BB
            'bb_position': bb_data.get('position', 0.5),
            'touch_lower': bb_data.get('touch_lower', False),
            'touch_upper': bb_data.get('touch_upper', False),
            'candles_shrinking': bb_data.get('candles_shrinking', False),
            # MACD
            'macd_bearish': macd_data.get('bearish', False) if macd_data else False,
            'macd_histogram': macd_data.get('histogram', 0) if macd_data else 0,
            'macd_below_signal': not macd_data.get('above_signal', True) if macd_data else False,
            # YOUR strategy specific
            'entry_tdi': entry_tdi,
            'target_tdi': target_tdi,
            'strategy_type': 'SELL_TO_30',
            # Conditions
            'conditions_met': 0,
            'conditions_total': 4,
            'condition_1_tdi_zone': tdi_data.get('tdi_zone') in ['SOFT_SELL', 'OVERBOUGHT', 'SELL_ZONE'],
            'condition_2_tdi_cross': not tdi_data.get('green_above_red', True),
            'condition_3_bb_touch': bb_data.get('touch_upper', False) or bb_data.get('position', 0.5) > 0.65,
            'condition_4_candles_shrinking': bb_data.get('candles_shrinking', False),
            # Timestamp
            'timestamp': datetime.now().isoformat(),
            'cheat_sheet': None,
            'ai_decision': 'PENDING',
            'ai_reasoning': 'Awaiting AI analysis...',
            'ai_confidence': 0.0,
            'ai_analysis': {},
            # Exit tracking
            'tdi_target_reached': False,
            'max_hold_minutes': 120,
        }

    def _calculate_quality_score(self, tdi_data: Dict, bb_data: Dict, macd_data: Dict, direction: str) -> int:
        """Calculate quality score."""
        score = 50  # Base score

        # TDI zone bonus
        zone = tdi_data.get('tdi_zone', '')
        if zone in ['OVERSOLD', 'OVERBOUGHT']:
            score += 20
        elif zone in ['SOFT_BUY', 'SOFT_SELL']:
            score += 15
        elif zone in ['BUY_ZONE', 'SELL_ZONE']:
            score += 10

        # Cross bonus
        if tdi_data.get('bullish_cross', False) or tdi_data.get('bearish_cross', False):
            score += 15

        # MACD bonus
        if macd_data:
            if (direction == "BUY" and macd_data.get('bullish', False)) or \
               (direction == "SELL" and macd_data.get('bearish', False)):
                score += 10

        # BB bonus
        position = bb_data.get('position', 0.5)
        if (direction == "BUY" and position < 0.25) or (direction == "SELL" and position > 0.75):
            score += 10

        return min(100, max(0, score))

    def _get_grade(self, score: int) -> str:
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 72:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 55:
            return "C"
        else:
            return "D"

    def _generate_cheat_sheet(self, signal_data: Dict[str, Any]) -> str:
        """Generate cheat sheet for YOUR strategy."""
        direction = signal_data.get('direction', 'UNKNOWN')
        symbol = signal_data.get('symbol', 'UNKNOWN')
        tdi_level = signal_data.get('tdi_level', 50)
        target_tdi = signal_data.get('target_tdi', 50)

        lines = []
        lines.append(f"{'🟢' if direction == 'BUY' else '🔴'} <b>{direction} SIGNAL</b> - {symbol}")
        lines.append("📋 <b>Strategy: Trade Across the Range</b>")
        lines.append("")

        if direction == 'BUY':
            lines.append(f"📊 TDI Entry: <b>{tdi_level:.1f}</b> (below 50 - Buyer Zone)")
            lines.append(f"🎯 Exit Target: <b>TDI {target_tdi:.1f}</b> (Soft Sell Zone)")
            lines.append(f"📈 Trade: {tdi_level:.1f} → {target_tdi:.1f}")
            lines.append(f"✅ Green line crossed ABOVE Red (Bulls taking over)")
            lines.append(f"📈 MACD: {'✅ BULLISH' if signal_data.get('macd_bullish', False) else '⏳ NEUTRAL'}")
            lines.append(f"📊 BB: {'✅ Near lower band' if signal_data.get('touch_lower', False) or signal_data.get('bb_position', 0.5) < 0.35 else '⏳ Waiting'}")
            lines.append("")
            lines.append("⏰ <b>Exit Rules:</b>")
            lines.append(f"• Exit when TDI reaches <b>{target_tdi:.1f}</b> (Soft Sell zone)")
            lines.append("• Max hold: 120 minutes")

        else:  # SELL
            lines.append(f"📊 TDI Entry: <b>{tdi_level:.1f}</b> (above 50 - Seller Zone)")
            lines.append(f"🎯 Exit Target: <b>TDI {target_tdi:.1f}</b> (Soft Buy Zone)")
            lines.append(f"📉 Trade: {tdi_level:.1f} → {target_tdi:.1f}")
            lines.append(f"✅ Green line crossed BELOW Red (Bears taking over)")
            lines.append(f"📉 MACD: {'✅ BEARISH' if signal_data.get('macd_bearish', False) else '⏳ NEUTRAL'}")
            lines.append(f"📊 BB: {'✅ Near upper band' if signal_data.get('touch_upper', False) or signal_data.get('bb_position', 0.5) > 0.65 else '⏳ Waiting'}")
            lines.append("")
            lines.append("⏰ <b>Exit Rules:</b>")
            lines.append(f"• Exit when TDI reaches <b>{target_tdi:.1f}</b> (Soft Buy zone)")
            lines.append("• Max hold: 60 minutes")

        # Trade details
        lines.append("")
        lines.append("📊 <b>Trade Details</b>")
        lines.append(f"• Entry: ${signal_data.get('entry_price', 0):.4f}")
        lines.append(f"• SL: ${signal_data.get('stop_loss', 0):.4f}")
        lines.append(f"• TP: ${signal_data.get('take_profit', 0):.4f}")
        lines.append(f"• RRR: {signal_data.get('rrr', 0):.1f}")
        lines.append(f"• Grade: {signal_data.get('grade', 'C')}")

        return "\n".join(lines)

    def _apply_ai_validation(self, symbol: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AI validation."""
        if not self.use_ai:
            return signal_data

        try:
            logger.info(f"🤖 {symbol}: Requesting AI analysis...")
            ai_result = ai_analyzer.analyze_signal(signal_data)

            signal_data['ai_decision'] = ai_result.decision
            signal_data['ai_reasoning'] = ai_result.reasoning
            signal_data['ai_confidence'] = ai_result.confidence

            if ai_result.decision == 'REJECT':
                signal_data['signal'] = 'NO_TRADE'
                signal_data['direction'] = 'NONE'
                logger.info(f"🚫 {symbol}: AI REJECTED - {ai_result.reasoning}")
                return signal_data

            if ai_result.decision == 'WAIT':
                signal_data['signal'] = 'NO_TRADE'
                signal_data['direction'] = 'NONE'
                logger.info(f"⏳ {symbol}: AI WAIT - {ai_result.reasoning}")
                return signal_data

            logger.info(f"✅ {symbol}: AI APPROVED - Confidence: {ai_result.confidence:.0%}")

        except Exception as e:
            logger.error(f"❌ {symbol}: AI validation error: {e}")
            signal_data['ai_decision'] = 'ERROR'

        return signal_data

    def _no_signal(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Return no signal result."""
        return {
            'symbol': symbol,
            'signal': 'NO_TRADE',
            'direction': 'NONE',
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'cheat_sheet': f"⏳ WAIT - {symbol}\n💬 {reason}",
            'ai_decision': 'NONE',
            'ai_reasoning': 'No signal to analyze',
            'ai_confidence': 0.0,
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        try:
            if df is None or len(df) < period:
                return df['close'].iloc[-1] * 0.01 if not df.empty else 0

            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))

            return np.mean(tr[-period:])
        except Exception as e:
            logger.debug(f"ATR calculation error: {e}")
            return df['close'].iloc[-1] * 0.01 if not df.empty else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'version': '3.4.1',
            'strategy': 'Trade Across the Range',
            'buy_exit_target': self.BUY_EXIT_TARGET,
            'sell_exit_target': self.SELL_EXIT_TARGET,
            'use_ai': self.use_ai,
            'min_rrr': self.min_rrr,
            'default_rrr': self.default_rrr,
            'max_rrr': self.max_rrr,
            'tdi_center_line': self.tdi_center_line,
        }
