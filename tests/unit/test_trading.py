"""
Unit tests for trading module.

Tests:
- IBConnector
- OrderManager
- PositionManager
- PaperTrader
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading.ib_connector import IBConnector, ConnectionConfig, MarketData, DataStreamer
from trading.order_manager import OrderManager, OrderSide, OrderType, OrderRequest, OrderResult
from trading.position_manager import PositionManager, Position, PositionSide, ClosedPosition
from trading.paper_trader import PaperTrader, PaperOrder


class TestConnectionConfig:
    """Tests for ConnectionConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = ConnectionConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 4002
        assert config.client_id == 1

    def test_custom_values(self):
        """Test custom configuration."""
        config = ConnectionConfig(
            host="192.168.1.100",
            port=4001,
            client_id=5
        )

        assert config.host == "192.168.1.100"
        assert config.port == 4001
        assert config.client_id == 5


class TestMarketData:
    """Tests for MarketData dataclass."""

    def test_creation(self):
        """Test MarketData creation."""
        data = MarketData(
            timestamp=datetime.now(),
            open=15000.0,
            high=15050.0,
            low=14950.0,
            close=15025.0,
            volume=1000
        )

        assert data.open == 15000.0
        assert data.close == 15025.0
        assert data.volume == 1000


class TestIBConnector:
    """Tests for IBConnector (without actual connection)."""

    def test_initialization(self):
        """Test connector initialization."""
        connector = IBConnector(paper_trading=True)

        assert connector.paper_trading == True
        assert connector.connected == False

    def test_paper_trading_client_id(self):
        """Test paper trading uses different client ID."""
        config = ConnectionConfig(client_id=1)
        connector = IBConnector(config, paper_trading=True)

        assert connector.config.client_id == 101  # Original + 100

    def test_contract_cache(self):
        """Test contract caching."""
        connector = IBConnector()

        assert connector._contracts == {}


class TestDataStreamer:
    """Tests for DataStreamer."""

    def test_initialization(self):
        """Test streamer initialization."""
        connector = Mock()
        streamer = DataStreamer(connector)

        assert streamer.data_buffer == []

    def test_on_new_bar(self):
        """Test handling new bar data."""
        connector = Mock()
        streamer = DataStreamer(connector)

        data = MarketData(
            timestamp=datetime.now(),
            open=15000.0,
            high=15050.0,
            low=14950.0,
            close=15025.0,
            volume=1000
        )

        streamer.on_new_bar(data)

        assert len(streamer.data_buffer) == 1
        assert streamer.data_buffer[0].close == 15025.0

    def test_buffer_limit(self):
        """Test buffer size limit."""
        connector = Mock()
        streamer = DataStreamer(connector)
        streamer.max_buffer_size = 10

        for i in range(20):
            data = MarketData(
                timestamp=datetime.now(),
                open=15000.0 + i,
                high=15050.0,
                low=14950.0,
                close=15025.0,
                volume=1000
            )
            streamer.on_new_bar(data)

        assert len(streamer.data_buffer) == 10

    def test_get_recent_data(self):
        """Test getting recent data as DataFrame."""
        connector = Mock()
        streamer = DataStreamer(connector)

        for i in range(5):
            data = MarketData(
                timestamp=datetime.now() + timedelta(minutes=i),
                open=15000.0 + i,
                high=15050.0,
                low=14950.0,
                close=15025.0 + i,
                volume=1000
            )
            streamer.on_new_bar(data)

        df = streamer.get_recent_data(3)

        assert len(df) == 3


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_values(self):
        """Test enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


class TestOrderType:
    """Tests for OrderType enum."""

    def test_values(self):
        """Test enum values."""
        assert OrderType.MARKET.value == "MKT"
        assert OrderType.LIMIT.value == "LMT"
        assert OrderType.STOP.value == "STP"


class TestPositionSide:
    """Tests for PositionSide enum."""

    def test_values(self):
        """Test enum values."""
        assert PositionSide.FLAT.value == 0
        assert PositionSide.LONG.value == 1
        assert PositionSide.SHORT.value == -1


class TestPosition:
    """Tests for Position class."""

    def test_creation(self):
        """Test position creation."""
        pos = Position(
            symbol="NQ",
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0,
            entry_time=datetime.now()
        )

        assert pos.symbol == "NQ"
        assert pos.is_long == True
        assert pos.is_short == False
        assert pos.is_flat == False

    def test_pnl_points_long(self):
        """Test P&L calculation for long position."""
        pos = Position(
            symbol="NQ",
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0,
            entry_time=datetime.now(),
            current_price=15100.0
        )

        assert pos.pnl_points == 100.0

    def test_pnl_points_short(self):
        """Test P&L calculation for short position."""
        pos = Position(
            symbol="NQ",
            side=PositionSide.SHORT,
            quantity=1,
            entry_price=15000.0,
            entry_time=datetime.now(),
            current_price=14900.0
        )

        assert pos.pnl_points == 100.0

    def test_mae_mfe(self):
        """Test MAE/MFE calculation."""
        pos = Position(
            symbol="NQ",
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0,
            entry_time=datetime.now(),
            current_price=15050.0,
            highest_price=15200.0,
            lowest_price=14800.0
        )

        assert pos.mae == 200.0  # Entry - Low
        assert pos.mfe == 200.0  # High - Entry


class TestPositionManager:
    """Tests for PositionManager."""

    def test_initialization(self, position_manager):
        """Test manager initialization."""
        assert position_manager.symbol == "NQ"
        assert position_manager.point_value == 20.0
        assert position_manager.position is None

    def test_open_position(self, position_manager):
        """Test opening a position."""
        pos = position_manager.open_position(
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0
        )

        assert pos is not None
        assert pos.side == PositionSide.LONG
        assert pos.quantity == 1
        assert position_manager.position == pos

    def test_update_price(self, position_manager):
        """Test price update."""
        position_manager.open_position(
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0
        )

        position_manager.update_price(15100.0)

        assert position_manager.position.current_price == 15100.0
        assert position_manager.position.unrealized_pnl == 100.0 * 20.0  # 100 pts * $20

    def test_close_position(self, position_manager):
        """Test closing a position."""
        position_manager.open_position(
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0
        )

        closed = position_manager.close_position(exit_price=15100.0)

        assert closed is not None
        assert isinstance(closed, ClosedPosition)
        assert position_manager.position is None
        assert len(position_manager.closed_positions) == 1

    def test_pnl_calculation(self, position_manager):
        """Test P&L calculation on close."""
        position_manager.open_position(
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0
        )

        closed = position_manager.close_position(exit_price=15100.0)

        # PnL = 100 pts * $20 - (2 * $2.25 commission) = $2000 - $4.50 = $1995.50
        expected_pnl = 100 * 20 - 2 * 2.25
        assert closed.pnl == pytest.approx(expected_pnl, rel=0.01)

    def test_check_stops_long(self, position_manager):
        """Test stop checking for long position."""
        position_manager.open_position(
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0,
            stop_loss=14900.0,
            take_profit=15200.0
        )

        # Price above stop - no trigger
        result = position_manager.check_stops(15050.0)
        assert result is None

        # Price hits stop loss
        result = position_manager.check_stops(14850.0)
        assert result == 'stop_loss'

    def test_check_stops_short(self, position_manager):
        """Test stop checking for short position."""
        position_manager.open_position(
            side=PositionSide.SHORT,
            quantity=1,
            entry_price=15000.0,
            stop_loss=15100.0,
            take_profit=14800.0
        )

        # Price hits stop
        result = position_manager.check_stops(15150.0)
        assert result == 'stop_loss'

        # Reset and check take profit
        position_manager.close_position(15150.0)
        position_manager.open_position(
            side=PositionSide.SHORT,
            quantity=1,
            entry_price=15000.0,
            stop_loss=15100.0,
            take_profit=14800.0
        )

        result = position_manager.check_stops(14750.0)
        assert result == 'take_profit'

    def test_modify_stops(self, position_manager):
        """Test stop modification."""
        position_manager.open_position(
            side=PositionSide.LONG,
            quantity=1,
            entry_price=15000.0,
            stop_loss=14900.0
        )

        position_manager.modify_stops(stop_loss=14950.0, take_profit=15300.0)

        assert position_manager.position.stop_loss == 14950.0
        assert position_manager.position.take_profit == 15300.0

    def test_get_statistics(self, position_manager):
        """Test statistics calculation."""
        # Make some trades
        position_manager.open_position(PositionSide.LONG, 1, 15000.0)
        position_manager.close_position(15100.0)

        position_manager.open_position(PositionSide.SHORT, 1, 15100.0)
        position_manager.close_position(15050.0)

        stats = position_manager.get_statistics()

        assert stats['total_trades'] == 2
        assert stats['win_rate'] == 1.0  # Both trades profitable


class TestPaperTrader:
    """Tests for PaperTrader."""

    def test_initialization(self, paper_trader):
        """Test paper trader initialization."""
        assert paper_trader.initial_capital == 100000.0
        assert paper_trader.get_equity() == 100000.0

    def test_update_market(self, paper_trader):
        """Test market price update."""
        paper_trader.update_market(15000.0)

        assert paper_trader.current_price == 15000.0
        assert paper_trader.current_bid == 14999.75
        assert paper_trader.current_ask == 15000.25

    def test_place_market_order_buy(self, paper_trader):
        """Test placing market buy order."""
        paper_trader.update_market(15000.0)

        order_id = paper_trader.place_market_order(OrderSide.BUY, 1)

        assert order_id > 0
        assert paper_trader.get_position() is not None
        assert paper_trader.get_position().is_long

    def test_place_market_order_sell(self, paper_trader):
        """Test placing market sell order."""
        paper_trader.update_market(15000.0)

        order_id = paper_trader.place_market_order(OrderSide.SELL, 1)

        assert order_id > 0
        assert paper_trader.get_position() is not None
        assert paper_trader.get_position().is_short

    def test_bracket_order(self, paper_trader):
        """Test bracket order placement."""
        paper_trader.update_market(15000.0)

        result = paper_trader.place_bracket_order(
            OrderSide.BUY,
            quantity=1,
            stop_loss=14900.0,
            take_profit=15200.0
        )

        assert result['entry'] > 0
        assert result['stop'] > 0
        assert result['target'] > 0
        assert paper_trader.get_position().stop_loss == 14900.0
        assert paper_trader.get_position().take_profit == 15200.0

    def test_stop_trigger(self, paper_trader):
        """Test stop loss triggering."""
        paper_trader.update_market(15000.0)

        paper_trader.place_bracket_order(
            OrderSide.BUY,
            quantity=1,
            stop_loss=14900.0,
            take_profit=15200.0
        )

        # Move price to trigger stop
        paper_trader.update_market(14850.0)

        # Position should be closed
        assert paper_trader.get_position() is None

    def test_take_profit_trigger(self, paper_trader):
        """Test take profit triggering."""
        paper_trader.update_market(15000.0)

        paper_trader.place_bracket_order(
            OrderSide.BUY,
            quantity=1,
            stop_loss=14900.0,
            take_profit=15200.0
        )

        # Move price to trigger take profit
        paper_trader.update_market(15250.0)

        # Position should be closed
        assert paper_trader.get_position() is None

    def test_max_position_limit(self, paper_trader):
        """Test max position enforcement."""
        paper_trader.update_market(15000.0)

        # Try to exceed max position (2)
        paper_trader.place_market_order(OrderSide.BUY, 2)
        result = paper_trader.place_market_order(OrderSide.BUY, 1)

        assert result == -1  # Should be rejected

    def test_close_position(self, paper_trader):
        """Test closing position."""
        paper_trader.update_market(15000.0)
        paper_trader.place_market_order(OrderSide.BUY, 1)

        paper_trader.update_market(15100.0)
        paper_trader.close_position()

        assert paper_trader.get_position() is None

    def test_statistics(self, paper_trader):
        """Test statistics calculation."""
        paper_trader.update_market(15000.0)
        paper_trader.place_market_order(OrderSide.BUY, 1)

        paper_trader.update_market(15100.0)
        paper_trader.close_position()

        stats = paper_trader.get_statistics()

        assert stats['total_trades'] == 1
        assert stats['total_pnl'] > 0
        assert stats['total_return'] > 0

    def test_reset(self, paper_trader):
        """Test paper trader reset."""
        paper_trader.update_market(15000.0)
        paper_trader.place_market_order(OrderSide.BUY, 1)

        paper_trader.reset()

        assert paper_trader.get_position() is None
        assert paper_trader.get_equity() == 100000.0
        assert len(paper_trader.trade_history) == 0

    def test_cancel_order(self, paper_trader):
        """Test order cancellation."""
        paper_trader.update_market(15000.0)

        # Place limit order that won't fill immediately
        order_id = paper_trader.place_limit_order(
            OrderSide.BUY,
            quantity=1,
            limit_price=14900.0  # Below current price
        )

        result = paper_trader.cancel_order(order_id)

        assert result == True
        assert paper_trader.orders[order_id].status == "Cancelled"

    def test_commission_tracking(self, paper_trader):
        """Test commission is tracked."""
        paper_trader.update_market(15000.0)
        paper_trader.place_market_order(OrderSide.BUY, 1)

        paper_trader.update_market(15000.0)  # Same price
        paper_trader.close_position()

        # With same entry/exit price, only commission affects P&L
        # Entry: $2.25, Exit: $2.25, plus slippage
        stats = paper_trader.get_statistics()
        assert stats['total_pnl'] < 0  # Should be negative due to costs
