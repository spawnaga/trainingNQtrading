"""
Paper Trading Module for testing strategies without real capital.

Simulates order execution with realistic fills, slippage, and commissions.
"""

from datetime import datetime
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass
import random
import numpy as np
from loguru import logger

from .order_manager import OrderSide, OrderType
from .position_manager import PositionManager, PositionSide, Position


@dataclass
class PaperOrder:
    """Paper trading order."""
    order_id: int
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "Pending"
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    parent_id: Optional[int] = None  # For bracket orders


class PaperTrader:
    """
    Paper trading simulator for strategy testing.

    Features:
    - Realistic order execution with slippage
    - Commission tracking
    - Position management
    - P&L calculation
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_side: float = 2.25,
        slippage_points: float = 0.25,
        point_value: float = 20.0,  # NQ point value
        max_position: int = 2
    ):
        """
        Args:
            initial_capital: Starting capital
            commission_per_side: Commission per contract per side
            slippage_points: Slippage in points
            point_value: Dollar value per point
            max_position: Maximum contracts allowed
        """
        self.initial_capital = initial_capital
        self.commission = commission_per_side
        self.slippage = slippage_points
        self.point_value = point_value
        self.max_position = max_position

        # Position manager
        self.position_manager = PositionManager(
            symbol="NQ",
            point_value=point_value,
            commission_per_side=commission_per_side
        )
        self.position_manager.account_equity = initial_capital
        self.position_manager.peak_equity = initial_capital
        self.position_manager.daily_start_equity = initial_capital

        # Order tracking
        self.orders: Dict[int, PaperOrder] = {}
        self.next_order_id = 1

        # Current market state
        self.current_price = 0.0
        self.current_bid = 0.0
        self.current_ask = 0.0

        # Trade history
        self.trade_history: List[Dict] = []

        # Callbacks
        self._fill_callbacks: List[Callable] = []

    def update_market(self, price: float, bid: float = None, ask: float = None):
        """
        Update market prices.

        Args:
            price: Last trade price
            bid: Bid price (default: price - 0.25)
            ask: Ask price (default: price + 0.25)
        """
        self.current_price = price
        self.current_bid = bid if bid else price - 0.25
        self.current_ask = ask if ask else price + 0.25

        # Update position P&L
        self.position_manager.update_price(price)

        # Check for stop triggers
        self._check_pending_orders()

    def _check_pending_orders(self):
        """Check and execute pending orders based on current price."""
        for order_id, order in list(self.orders.items()):
            if order.status != "Pending":
                continue

            triggered = False
            fill_price = None

            if order.order_type == OrderType.STOP:
                # Stop order logic
                if order.side == OrderSide.BUY:
                    if self.current_price >= order.stop_price:
                        triggered = True
                        fill_price = order.stop_price + self.slippage
                else:
                    if self.current_price <= order.stop_price:
                        triggered = True
                        fill_price = order.stop_price - self.slippage

            elif order.order_type == OrderType.LIMIT:
                # Limit order logic
                if order.side == OrderSide.BUY:
                    if self.current_price <= order.limit_price:
                        triggered = True
                        fill_price = order.limit_price
                else:
                    if self.current_price >= order.limit_price:
                        triggered = True
                        fill_price = order.limit_price

            if triggered:
                self._execute_order(order, fill_price)

    def _execute_order(self, order: PaperOrder, fill_price: float):
        """Execute an order at the given price."""
        order.status = "Filled"
        order.fill_price = fill_price
        order.fill_time = datetime.now()

        logger.info(f"Paper fill: {order.side.value} {order.quantity} @ {fill_price}")

        # Handle position changes
        current_pos = self.position_manager.position

        if order.side == OrderSide.BUY:
            if current_pos and current_pos.is_short:
                # Closing short
                self.position_manager.close_position(fill_price, order.quantity)
                # Cancel related bracket orders
                self._cancel_bracket_orders(order.parent_id)
            else:
                # Opening or adding to long
                self.position_manager.open_position(
                    PositionSide.LONG,
                    order.quantity,
                    fill_price,
                    order_ids=[order.order_id]
                )
        else:
            if current_pos and current_pos.is_long:
                # Closing long
                self.position_manager.close_position(fill_price, order.quantity)
                self._cancel_bracket_orders(order.parent_id)
            else:
                # Opening or adding to short
                self.position_manager.open_position(
                    PositionSide.SHORT,
                    order.quantity,
                    fill_price,
                    order_ids=[order.order_id]
                )

        # Record trade
        self.trade_history.append({
            'order_id': order.order_id,
            'time': order.fill_time,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'type': order.order_type.value
        })

        # Notify callbacks
        for callback in self._fill_callbacks:
            callback(order)

    def _cancel_bracket_orders(self, parent_id: Optional[int]):
        """Cancel child orders of a filled bracket order."""
        if parent_id is None:
            return

        for order_id, order in self.orders.items():
            if order.parent_id == parent_id and order.status == "Pending":
                order.status = "Cancelled"
                logger.debug(f"Cancelled bracket order {order_id}")

    def place_market_order(
        self,
        side: OrderSide,
        quantity: int
    ) -> int:
        """
        Place a market order (executes immediately).

        Args:
            side: BUY or SELL
            quantity: Number of contracts

        Returns:
            Order ID
        """
        # Check position limits
        current_qty = 0
        if self.position_manager.position:
            current_qty = self.position_manager.position.quantity
            if self.position_manager.position.is_short:
                current_qty = -current_qty

        if side == OrderSide.BUY:
            new_qty = current_qty + quantity
        else:
            new_qty = current_qty - quantity

        if abs(new_qty) > self.max_position:
            logger.warning(f"Order rejected: would exceed max position of {self.max_position}")
            return -1

        order_id = self.next_order_id
        self.next_order_id += 1

        # Calculate fill price with slippage
        if side == OrderSide.BUY:
            fill_price = self.current_ask + self.slippage
        else:
            fill_price = self.current_bid - self.slippage

        order = PaperOrder(
            order_id=order_id,
            symbol="NQ",
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            status="Filled",
            fill_price=fill_price,
            fill_time=datetime.now()
        )

        self.orders[order_id] = order
        self._execute_order(order, fill_price)

        return order_id

    def place_limit_order(
        self,
        side: OrderSide,
        quantity: int,
        limit_price: float,
        parent_id: Optional[int] = None
    ) -> int:
        """
        Place a limit order.

        Args:
            side: BUY or SELL
            quantity: Number of contracts
            limit_price: Limit price
            parent_id: Parent order ID (for bracket orders)

        Returns:
            Order ID
        """
        order_id = self.next_order_id
        self.next_order_id += 1

        order = PaperOrder(
            order_id=order_id,
            symbol="NQ",
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            parent_id=parent_id
        )

        self.orders[order_id] = order
        logger.debug(f"Placed limit order: {side.value} {quantity} @ {limit_price}")

        # Check for immediate fill
        self._check_pending_orders()

        return order_id

    def place_stop_order(
        self,
        side: OrderSide,
        quantity: int,
        stop_price: float,
        parent_id: Optional[int] = None
    ) -> int:
        """
        Place a stop order.

        Args:
            side: BUY or SELL
            quantity: Number of contracts
            stop_price: Stop trigger price
            parent_id: Parent order ID (for bracket orders)

        Returns:
            Order ID
        """
        order_id = self.next_order_id
        self.next_order_id += 1

        order = PaperOrder(
            order_id=order_id,
            symbol="NQ",
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            parent_id=parent_id
        )

        self.orders[order_id] = order
        logger.debug(f"Placed stop order: {side.value} {quantity} @ STP {stop_price}")

        return order_id

    def place_bracket_order(
        self,
        side: OrderSide,
        quantity: int,
        stop_loss: float,
        take_profit: float
    ) -> Dict[str, int]:
        """
        Place a bracket order (entry + stop loss + take profit).

        Args:
            side: BUY or SELL
            quantity: Number of contracts
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Dict with order IDs
        """
        # Entry order (market)
        entry_id = self.place_market_order(side, quantity)

        if entry_id < 0:
            return {'entry': -1, 'stop': -1, 'target': -1}

        # Exit side is opposite
        exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

        # Stop loss
        stop_id = self.place_stop_order(exit_side, quantity, stop_loss, entry_id)

        # Take profit
        target_id = self.place_limit_order(exit_side, quantity, take_profit, entry_id)

        # Update position with stops
        if self.position_manager.position:
            self.position_manager.position.stop_loss = stop_loss
            self.position_manager.position.take_profit = take_profit

        logger.info(f"Placed bracket: {side.value} {quantity}, SL: {stop_loss}, TP: {take_profit}")

        return {
            'entry': entry_id,
            'stop': stop_id,
            'target': target_id
        }

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a pending order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == "Pending":
                order.status = "Cancelled"
                logger.debug(f"Cancelled order {order_id}")
                return True
        return False

    def cancel_all_orders(self):
        """Cancel all pending orders."""
        for order_id, order in self.orders.items():
            if order.status == "Pending":
                order.status = "Cancelled"

    def close_position(self) -> Optional[int]:
        """Close current position with market order."""
        pos = self.position_manager.position
        if not pos or pos.is_flat:
            return None

        side = OrderSide.SELL if pos.is_long else OrderSide.BUY
        return self.place_market_order(side, pos.quantity)

    def get_equity(self) -> float:
        """Get current account equity."""
        return self.position_manager.account_equity

    def get_position(self) -> Optional[Position]:
        """Get current position."""
        return self.position_manager.position

    def get_statistics(self) -> Dict:
        """Get trading statistics."""
        stats = self.position_manager.get_statistics()
        stats['initial_capital'] = self.initial_capital
        stats['total_return'] = (self.get_equity() - self.initial_capital) / self.initial_capital
        return stats

    def reset(self):
        """Reset paper trader to initial state."""
        self.position_manager = PositionManager(
            symbol="NQ",
            point_value=self.point_value,
            commission_per_side=self.commission
        )
        self.position_manager.account_equity = self.initial_capital
        self.position_manager.peak_equity = self.initial_capital
        self.position_manager.daily_start_equity = self.initial_capital

        self.orders.clear()
        self.next_order_id = 1
        self.trade_history.clear()

        logger.info("Paper trader reset")
