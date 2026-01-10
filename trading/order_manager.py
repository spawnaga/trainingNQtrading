"""
Order Manager for executing trades via IB Gateway.

Handles order creation, submission, modification, and monitoring.
"""

from datetime import datetime
from typing import Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from ib_insync import (
    IB, Contract, Order, Trade, OrderStatus,
    MarketOrder, LimitOrder, StopOrder
)
from loguru import logger


class OrderType(Enum):
    """Order types."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderRequest:
    """Order request parameters."""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, DAY, IOC
    order_ref: str = ""


@dataclass
class OrderResult:
    """Order execution result."""
    order_id: int
    status: str
    filled_qty: int
    avg_fill_price: float
    commission: float
    timestamp: datetime
    error_message: str = ""


@dataclass
class BracketOrderResult:
    """Bracket order result (entry + stop + target)."""
    entry_order: OrderResult
    stop_order_id: int
    target_order_id: int


class OrderManager:
    """
    Manages order creation and execution via IB.
    """

    def __init__(self, ib: IB, contract: Contract):
        """
        Args:
            ib: Connected IB instance
            contract: Trading contract
        """
        self.ib = ib
        self.contract = contract

        # Order tracking
        self.pending_orders: Dict[int, Trade] = {}
        self.filled_orders: Dict[int, OrderResult] = {}
        self.cancelled_orders: Dict[int, OrderResult] = {}

        # Callbacks
        self._fill_callbacks: List[Callable[[OrderResult], None]] = []
        self._status_callbacks: List[Callable[[int, str], None]] = []

        # Setup event handlers
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details

    def _on_order_status(self, trade: Trade):
        """Handle order status updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status

        logger.debug(f"Order {order_id} status: {status}")

        for callback in self._status_callbacks:
            callback(order_id, status)

        if status == 'Filled':
            result = OrderResult(
                order_id=order_id,
                status=status,
                filled_qty=int(trade.orderStatus.filled),
                avg_fill_price=trade.orderStatus.avgFillPrice,
                commission=sum(f.commissionReport.commission for f in trade.fills if f.commissionReport),
                timestamp=datetime.now()
            )
            self.filled_orders[order_id] = result

            for callback in self._fill_callbacks:
                callback(result)

        elif status in ['Cancelled', 'ApiCancelled']:
            result = OrderResult(
                order_id=order_id,
                status=status,
                filled_qty=int(trade.orderStatus.filled),
                avg_fill_price=trade.orderStatus.avgFillPrice,
                commission=0,
                timestamp=datetime.now()
            )
            self.cancelled_orders[order_id] = result

    def _on_exec_details(self, trade: Trade, fill):
        """Handle execution details."""
        logger.info(f"Fill: {fill.execution.shares} @ {fill.execution.price}")

    def add_fill_callback(self, callback: Callable[[OrderResult], None]):
        """Add callback for order fills."""
        self._fill_callbacks.append(callback)

    def add_status_callback(self, callback: Callable[[int, str], None]):
        """Add callback for status updates."""
        self._status_callbacks.append(callback)

    def place_market_order(
        self,
        side: OrderSide,
        quantity: int,
        order_ref: str = ""
    ) -> Trade:
        """
        Place a market order.

        Args:
            side: BUY or SELL
            quantity: Number of contracts
            order_ref: Reference string

        Returns:
            Trade object
        """
        order = MarketOrder(
            action=side.value,
            totalQuantity=quantity,
            orderRef=order_ref
        )

        trade = self.ib.placeOrder(self.contract, order)
        self.pending_orders[trade.order.orderId] = trade

        logger.info(f"Placed market order: {side.value} {quantity} @ MKT, ID: {trade.order.orderId}")

        return trade

    def place_limit_order(
        self,
        side: OrderSide,
        quantity: int,
        limit_price: float,
        order_ref: str = ""
    ) -> Trade:
        """
        Place a limit order.

        Args:
            side: BUY or SELL
            quantity: Number of contracts
            limit_price: Limit price
            order_ref: Reference string

        Returns:
            Trade object
        """
        order = LimitOrder(
            action=side.value,
            totalQuantity=quantity,
            lmtPrice=limit_price,
            orderRef=order_ref
        )

        trade = self.ib.placeOrder(self.contract, order)
        self.pending_orders[trade.order.orderId] = trade

        logger.info(f"Placed limit order: {side.value} {quantity} @ {limit_price}, ID: {trade.order.orderId}")

        return trade

    def place_stop_order(
        self,
        side: OrderSide,
        quantity: int,
        stop_price: float,
        order_ref: str = ""
    ) -> Trade:
        """
        Place a stop order.

        Args:
            side: BUY or SELL
            quantity: Number of contracts
            stop_price: Stop trigger price
            order_ref: Reference string

        Returns:
            Trade object
        """
        order = StopOrder(
            action=side.value,
            totalQuantity=quantity,
            stopPrice=stop_price,
            orderRef=order_ref
        )

        trade = self.ib.placeOrder(self.contract, order)
        self.pending_orders[trade.order.orderId] = trade

        logger.info(f"Placed stop order: {side.value} {quantity} @ STP {stop_price}, ID: {trade.order.orderId}")

        return trade

    def place_bracket_order(
        self,
        side: OrderSide,
        quantity: int,
        entry_price: Optional[float],
        stop_loss: float,
        take_profit: float,
        use_market_entry: bool = True
    ) -> Tuple[Trade, Trade, Trade]:
        """
        Place a bracket order (entry + stop loss + take profit).

        Args:
            side: BUY or SELL
            quantity: Number of contracts
            entry_price: Entry limit price (ignored if market entry)
            stop_loss: Stop loss price
            take_profit: Take profit price
            use_market_entry: Use market order for entry

        Returns:
            Tuple of (entry_trade, stop_trade, profit_trade)
        """
        parent_id = self.ib.client.getReqId()

        # Entry order
        if use_market_entry:
            entry_order = Order(
                orderId=parent_id,
                action=side.value,
                orderType='MKT',
                totalQuantity=quantity,
                transmit=False
            )
        else:
            entry_order = Order(
                orderId=parent_id,
                action=side.value,
                orderType='LMT',
                lmtPrice=entry_price,
                totalQuantity=quantity,
                transmit=False
            )

        # Exit side is opposite
        exit_side = 'SELL' if side == OrderSide.BUY else 'BUY'

        # Take profit order
        profit_order = Order(
            orderId=self.ib.client.getReqId(),
            action=exit_side,
            orderType='LMT',
            lmtPrice=take_profit,
            totalQuantity=quantity,
            parentId=parent_id,
            transmit=False
        )

        # Stop loss order
        stop_order = Order(
            orderId=self.ib.client.getReqId(),
            action=exit_side,
            orderType='STP',
            stopPrice=stop_loss,
            totalQuantity=quantity,
            parentId=parent_id,
            transmit=True  # Last order triggers transmission
        )

        # Place orders
        entry_trade = self.ib.placeOrder(self.contract, entry_order)
        profit_trade = self.ib.placeOrder(self.contract, profit_order)
        stop_trade = self.ib.placeOrder(self.contract, stop_order)

        # Track orders
        self.pending_orders[entry_trade.order.orderId] = entry_trade
        self.pending_orders[profit_trade.order.orderId] = profit_trade
        self.pending_orders[stop_trade.order.orderId] = stop_trade

        logger.info(
            f"Placed bracket order: {side.value} {quantity}, "
            f"SL: {stop_loss}, TP: {take_profit}"
        )

        return entry_trade, stop_trade, profit_trade

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancellation submitted
        """
        if order_id in self.pending_orders:
            trade = self.pending_orders[order_id]
            self.ib.cancelOrder(trade.order)
            logger.info(f"Cancellation submitted for order {order_id}")
            return True
        return False

    def cancel_all_orders(self):
        """Cancel all pending orders."""
        for order_id in list(self.pending_orders.keys()):
            self.cancel_order(order_id)
        logger.info("All orders cancelled")

    def modify_order(
        self,
        order_id: int,
        new_price: Optional[float] = None,
        new_quantity: Optional[int] = None
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            new_price: New limit/stop price
            new_quantity: New quantity

        Returns:
            True if modification submitted
        """
        if order_id not in self.pending_orders:
            return False

        trade = self.pending_orders[order_id]
        order = trade.order

        if new_price is not None:
            if order.orderType == 'LMT':
                order.lmtPrice = new_price
            elif order.orderType == 'STP':
                order.stopPrice = new_price

        if new_quantity is not None:
            order.totalQuantity = new_quantity

        self.ib.placeOrder(self.contract, order)
        logger.info(f"Modified order {order_id}: price={new_price}, qty={new_quantity}")

        return True

    def get_order_status(self, order_id: int) -> Optional[str]:
        """Get current status of an order."""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id].orderStatus.status
        elif order_id in self.filled_orders:
            return 'Filled'
        elif order_id in self.cancelled_orders:
            return 'Cancelled'
        return None

    def wait_for_fill(
        self,
        order_id: int,
        timeout: float = 30.0
    ) -> Optional[OrderResult]:
        """
        Wait for an order to fill.

        Args:
            order_id: Order ID to wait for
            timeout: Maximum wait time in seconds

        Returns:
            OrderResult if filled, None if timeout
        """
        start = time.time()

        while time.time() - start < timeout:
            if order_id in self.filled_orders:
                return self.filled_orders[order_id]

            status = self.get_order_status(order_id)
            if status in ['Cancelled', 'ApiCancelled', 'Error']:
                return None

            self.ib.sleep(0.1)

        logger.warning(f"Timeout waiting for order {order_id}")
        return None

    def close_position(
        self,
        quantity: int,
        current_position: float
    ) -> Optional[Trade]:
        """
        Close a position.

        Args:
            quantity: Quantity to close
            current_position: Current position (positive=long, negative=short)

        Returns:
            Trade object or None
        """
        if current_position > 0:
            # Long position - sell to close
            return self.place_market_order(OrderSide.SELL, quantity, "CLOSE")
        elif current_position < 0:
            # Short position - buy to close
            return self.place_market_order(OrderSide.BUY, quantity, "CLOSE")

        return None
