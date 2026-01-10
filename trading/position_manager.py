"""
Position Manager for tracking and managing trading positions.

Handles position state, P&L calculations, and risk monitoring.
"""

from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from loguru import logger


class PositionSide(Enum):
    """Position side."""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    order_ids: List[int] = field(default_factory=list)

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT

    @property
    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT

    @property
    def pnl_points(self) -> float:
        """P&L in points."""
        if self.is_flat:
            return 0.0
        direction = 1 if self.is_long else -1
        return direction * (self.current_price - self.entry_price)

    @property
    def mae(self) -> float:
        """Maximum Adverse Excursion (worst drawdown during trade)."""
        if self.is_long:
            return max(0, self.entry_price - self.lowest_price)
        elif self.is_short:
            return max(0, self.highest_price - self.entry_price)
        return 0.0

    @property
    def mfe(self) -> float:
        """Maximum Favorable Excursion (best unrealized gain during trade)."""
        if self.is_long:
            return max(0, self.highest_price - self.entry_price)
        elif self.is_short:
            return max(0, self.entry_price - self.lowest_price)
        return 0.0

    @property
    def holding_time_minutes(self) -> float:
        """Time in position in minutes."""
        return (datetime.now() - self.entry_time).total_seconds() / 60.0


@dataclass
class ClosedPosition:
    """Record of a closed position."""
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    holding_time_minutes: float
    mae: float
    mfe: float


class PositionManager:
    """
    Manages position tracking and P&L calculation.
    """

    def __init__(
        self,
        symbol: str,
        point_value: float = 20.0,  # NQ point value
        commission_per_side: float = 2.25
    ):
        """
        Args:
            symbol: Trading symbol
            point_value: Dollar value per point
            commission_per_side: Commission per contract per side
        """
        self.symbol = symbol
        self.point_value = point_value
        self.commission_per_side = commission_per_side

        # Current position
        self.position: Optional[Position] = None

        # Position history
        self.closed_positions: List[ClosedPosition] = []

        # Daily statistics
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.daily_start_equity: float = 0.0

        # Account state
        self.account_equity: float = 100000.0
        self.peak_equity: float = 100000.0

    def open_position(
        self,
        side: PositionSide,
        quantity: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        order_ids: List[int] = None
    ) -> Position:
        """
        Open a new position.

        Args:
            side: LONG or SHORT
            quantity: Number of contracts
            entry_price: Entry fill price
            stop_loss: Stop loss price
            take_profit: Take profit price
            order_ids: Associated order IDs

        Returns:
            Position object
        """
        if self.position and not self.position.is_flat:
            logger.warning("Opening position while already in position")

        commission = self.commission_per_side * quantity

        self.position = Position(
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price,
            commission=commission,
            order_ids=order_ids or []
        )

        logger.info(
            f"Opened {side.name} position: {quantity} @ {entry_price}, "
            f"SL: {stop_loss}, TP: {take_profit}"
        )

        return self.position

    def update_price(self, price: float):
        """
        Update current price and P&L.

        Args:
            price: Current market price
        """
        if not self.position or self.position.is_flat:
            return

        self.position.current_price = price
        self.position.highest_price = max(self.position.highest_price, price)
        self.position.lowest_price = min(self.position.lowest_price, price)

        # Calculate unrealized P&L
        pnl_points = self.position.pnl_points
        self.position.unrealized_pnl = pnl_points * self.point_value * self.position.quantity

    def close_position(
        self,
        exit_price: float,
        quantity: Optional[int] = None
    ) -> Optional[ClosedPosition]:
        """
        Close the current position.

        Args:
            exit_price: Exit fill price
            quantity: Quantity to close (None = all)

        Returns:
            ClosedPosition record
        """
        if not self.position or self.position.is_flat:
            logger.warning("No position to close")
            return None

        close_qty = quantity or self.position.quantity
        exit_commission = self.commission_per_side * close_qty

        # Calculate P&L
        direction = 1 if self.position.is_long else -1
        pnl_points = direction * (exit_price - self.position.entry_price)
        pnl = pnl_points * self.point_value * close_qty
        total_commission = self.position.commission + exit_commission
        net_pnl = pnl - total_commission

        # Create closed position record
        closed = ClosedPosition(
            symbol=self.symbol,
            side=self.position.side,
            quantity=close_qty,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            entry_time=self.position.entry_time,
            exit_time=datetime.now(),
            pnl=net_pnl,
            commission=total_commission,
            holding_time_minutes=self.position.holding_time_minutes,
            mae=self.position.mae,
            mfe=self.position.mfe
        )

        self.closed_positions.append(closed)

        # Update daily stats
        self.daily_pnl += net_pnl
        self.daily_trades += 1

        # Update account
        self.account_equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.account_equity)

        logger.info(
            f"Closed {closed.side.name} position: {close_qty} @ {exit_price}, "
            f"P&L: ${net_pnl:.2f}"
        )

        # Partial or full close
        if close_qty >= self.position.quantity:
            self.position = None
        else:
            self.position.quantity -= close_qty
            self.position.commission = 0  # Commission already accounted

        return closed

    def check_stops(self, current_price: float) -> Optional[str]:
        """
        Check if stops are hit.

        Args:
            current_price: Current market price

        Returns:
            'stop_loss', 'take_profit', or None
        """
        if not self.position or self.position.is_flat:
            return None

        self.update_price(current_price)

        # Check stop loss
        if self.position.stop_loss:
            if self.position.is_long and current_price <= self.position.stop_loss:
                return 'stop_loss'
            elif self.position.is_short and current_price >= self.position.stop_loss:
                return 'stop_loss'

        # Check take profit
        if self.position.take_profit:
            if self.position.is_long and current_price >= self.position.take_profit:
                return 'take_profit'
            elif self.position.is_short and current_price <= self.position.take_profit:
                return 'take_profit'

        return None

    def modify_stops(
        self,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        """Modify position stops."""
        if self.position:
            if stop_loss is not None:
                self.position.stop_loss = stop_loss
            if take_profit is not None:
                self.position.take_profit = take_profit
            logger.info(f"Modified stops: SL={stop_loss}, TP={take_profit}")

    def trail_stop(self, trail_points: float):
        """
        Trail the stop loss.

        Args:
            trail_points: Points to trail behind current price
        """
        if not self.position or self.position.is_flat:
            return

        if self.position.is_long:
            new_stop = self.position.current_price - trail_points
            if self.position.stop_loss is None or new_stop > self.position.stop_loss:
                self.position.stop_loss = new_stop
        else:
            new_stop = self.position.current_price + trail_points
            if self.position.stop_loss is None or new_stop < self.position.stop_loss:
                self.position.stop_loss = new_stop

    def get_position_state(self) -> Dict:
        """
        Get current position state for model input.

        Returns:
            Dict with position information
        """
        if not self.position or self.position.is_flat:
            return {
                'has_position': False,
                'side': 0,
                'quantity': 0,
                'unrealized_pnl': 0.0,
                'holding_time': 0.0,
                'pnl_pct': 0.0
            }

        return {
            'has_position': True,
            'side': 1 if self.position.is_long else -1,
            'quantity': self.position.quantity,
            'unrealized_pnl': self.position.unrealized_pnl,
            'holding_time': self.position.holding_time_minutes,
            'pnl_pct': self.position.unrealized_pnl / self.account_equity
        }

    def get_risk_state(self) -> Dict:
        """
        Get risk state for model input.

        Returns:
            Dict with risk information
        """
        current_drawdown = (self.peak_equity - self.account_equity) / self.peak_equity
        daily_pnl_pct = self.daily_pnl / self.daily_start_equity if self.daily_start_equity > 0 else 0

        position_exposure = 0.0
        if self.position and not self.position.is_flat:
            position_exposure = abs(self.position.quantity) / 2  # Normalized by max position

        return {
            'current_drawdown': current_drawdown,
            'daily_pnl': daily_pnl_pct,
            'position_exposure': position_exposure,
            'account_equity': self.account_equity
        }

    def new_day(self):
        """Reset daily statistics."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_equity = self.account_equity

    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        if not self.closed_positions:
            return {}

        pnls = [p.pnl for p in self.closed_positions]
        wins = [p for p in self.closed_positions if p.pnl > 0]
        losses = [p for p in self.closed_positions if p.pnl < 0]

        return {
            'total_trades': len(self.closed_positions),
            'total_pnl': sum(pnls),
            'win_rate': len(wins) / len(self.closed_positions) if self.closed_positions else 0,
            'avg_win': np.mean([w.pnl for w in wins]) if wins else 0,
            'avg_loss': np.mean([l.pnl for l in losses]) if losses else 0,
            'profit_factor': abs(sum(w.pnl for w in wins) / sum(l.pnl for l in losses)) if losses else float('inf'),
            'avg_holding_time': np.mean([p.holding_time_minutes for p in self.closed_positions]),
            'current_equity': self.account_equity,
            'peak_equity': self.peak_equity,
            'max_drawdown': (self.peak_equity - min(self.account_equity, self.peak_equity)) / self.peak_equity
        }
