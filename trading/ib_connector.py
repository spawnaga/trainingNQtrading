"""
Interactive Brokers Gateway Connector using ib_insync.

Handles connection management, data streaming, and order execution
for NQ futures trading.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger

from ib_insync import (
    IB, Contract, Future, Order, Trade, BarData,
    MarketOrder, LimitOrder, StopOrder, BracketOrder,
    util
)


@dataclass
class ConnectionConfig:
    """IB Gateway connection configuration."""
    host: str = "127.0.0.1"
    port: int = 4002  # IB Gateway port
    client_id: int = 1
    timeout: int = 30
    readonly: bool = False


@dataclass
class MarketData:
    """Container for market data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0
    bar_count: int = 0


class IBConnector:
    """
    Interactive Brokers Gateway connector.

    Manages connection, market data, and order execution.
    """

    def __init__(
        self,
        config: Optional[ConnectionConfig] = None,
        paper_trading: bool = True
    ):
        """
        Args:
            config: Connection configuration
            paper_trading: If True, use paper trading client ID
        """
        self.config = config or ConnectionConfig()
        self.paper_trading = paper_trading

        # Use different client ID for paper trading
        if paper_trading:
            self.config.client_id = self.config.client_id + 100

        self.ib = IB()
        self.connected = False

        # Contract cache
        self._contracts: Dict[str, Contract] = {}

        # Data callbacks
        self._bar_callbacks: List[Callable[[MarketData], None]] = []
        self._tick_callbacks: List[Callable[[Dict], None]] = []

        # Active subscriptions
        self._bar_subscriptions: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """
        Connect to IB Gateway.

        Returns:
            True if connection successful
        """
        try:
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly
            )

            self.connected = self.ib.isConnected()

            if self.connected:
                logger.info(f"Connected to IB Gateway at {self.config.host}:{self.config.port}")
                logger.info(f"Client ID: {self.config.client_id}, Paper Trading: {self.paper_trading}")

                # Setup event handlers
                self.ib.errorEvent += self._on_error
                self.ib.disconnectedEvent += self._on_disconnect

            return self.connected

        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            return False

    def connect_sync(self) -> bool:
        """Synchronous connect."""
        try:
            self.ib.connect(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly
            )
            self.connected = self.ib.isConnected()

            if self.connected:
                logger.info(f"Connected to IB Gateway")
                self.ib.errorEvent += self._on_error
                self.ib.disconnectedEvent += self._on_disconnect

            return self.connected
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from IB Gateway."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB Gateway")

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract):
        """Handle IB errors."""
        # Ignore certain non-critical errors
        non_critical = [2104, 2106, 2158]  # Market data connection messages
        if errorCode not in non_critical:
            logger.warning(f"IB Error {errorCode}: {errorString}")

    def _on_disconnect(self):
        """Handle disconnection."""
        self.connected = False
        logger.warning("Disconnected from IB Gateway")

    def get_nq_contract(self) -> Future:
        """
        Get NQ futures contract (continuous front month).

        Returns:
            NQ futures Contract
        """
        if 'NQ' in self._contracts:
            return self._contracts['NQ']

        # NQ continuous futures
        contract = Future(
            symbol='NQ',
            exchange='CME',
            currency='USD'
        )

        # Qualify to get exact contract details
        qualified = self.ib.qualifyContracts(contract)

        if qualified:
            self._contracts['NQ'] = qualified[0]
            logger.info(f"NQ Contract: {qualified[0].localSymbol}")
            return qualified[0]
        else:
            raise ValueError("Could not qualify NQ contract")

    def get_specific_contract(
        self,
        symbol: str = 'NQ',
        expiry: str = None,  # YYYYMM format
        exchange: str = 'CME'
    ) -> Future:
        """
        Get specific futures contract.

        Args:
            symbol: Contract symbol
            expiry: Expiration date (YYYYMM)
            exchange: Exchange

        Returns:
            Qualified contract
        """
        contract = Future(
            symbol=symbol,
            lastTradeDateOrContractMonth=expiry,
            exchange=exchange,
            currency='USD'
        )

        qualified = self.ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
        else:
            raise ValueError(f"Could not qualify contract {symbol} {expiry}")

    def get_historical_data(
        self,
        contract: Contract,
        duration: str = "1 Y",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        end_datetime: str = ""
    ) -> pd.DataFrame:
        """
        Get historical bar data.

        Args:
            contract: Contract to get data for
            duration: Duration string (e.g., "1 Y", "6 M", "1 W")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour")
            what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)
            use_rth: Use regular trading hours only
            end_datetime: End date/time (empty for now)

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching historical data: {duration} of {bar_size} bars")

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1
        )

        if not bars:
            logger.warning("No historical data returned")
            return pd.DataFrame()

        # Convert to DataFrame
        df = util.df(bars)
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime')
        df = df[['open', 'high', 'low', 'close', 'volume']]

        logger.info(f"Retrieved {len(df)} bars from {df.index.min()} to {df.index.max()}")

        return df

    def subscribe_realtime_bars(
        self,
        contract: Contract,
        callback: Callable[[MarketData], None],
        bar_size: int = 5  # seconds
    ):
        """
        Subscribe to real-time bar updates.

        Args:
            contract: Contract to subscribe
            callback: Callback function for new bars
            bar_size: Bar size in seconds (5 second minimum)
        """
        self._bar_callbacks.append(callback)

        def on_bar_update(bars, hasNewBar):
            if hasNewBar and bars:
                bar = bars[-1]
                data = MarketData(
                    timestamp=bar.time,
                    open=bar.open_,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    vwap=bar.wap,
                    bar_count=bar.barCount
                )
                for cb in self._bar_callbacks:
                    cb(data)

        bars = self.ib.reqRealTimeBars(
            contract,
            barSize=bar_size,
            whatToShow='TRADES',
            useRTH=False
        )

        bars.updateEvent += on_bar_update
        self._bar_subscriptions[contract.symbol] = bars

        logger.info(f"Subscribed to real-time bars for {contract.symbol}")

    def unsubscribe_realtime_bars(self, symbol: str):
        """Unsubscribe from real-time bars."""
        if symbol in self._bar_subscriptions:
            self.ib.cancelRealTimeBars(self._bar_subscriptions[symbol])
            del self._bar_subscriptions[symbol]
            logger.info(f"Unsubscribed from {symbol}")

    def get_current_price(self, contract: Contract) -> Optional[float]:
        """
        Get current market price.

        Returns:
            Current price or None
        """
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(1)  # Wait for data

        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)

        return price if not np.isnan(price) else None

    def get_account_summary(self) -> Dict[str, float]:
        """
        Get account summary.

        Returns:
            Dict with account values
        """
        summary = {}
        account_values = self.ib.accountSummary()

        for av in account_values:
            if av.tag in ['NetLiquidation', 'AvailableFunds', 'BuyingPower',
                         'TotalCashValue', 'UnrealizedPnL', 'RealizedPnL']:
                summary[av.tag] = float(av.value)

        return summary

    def get_positions(self) -> List[Dict]:
        """
        Get current positions.

        Returns:
            List of position dicts
        """
        positions = []
        for pos in self.ib.positions():
            positions.append({
                'symbol': pos.contract.symbol,
                'position': pos.position,
                'avg_cost': pos.avgCost,
                'market_value': pos.position * pos.avgCost
            })
        return positions

    def run(self):
        """Run the IB event loop."""
        self.ib.run()

    def sleep(self, seconds: float):
        """Sleep while processing events."""
        self.ib.sleep(seconds)


class DataStreamer:
    """
    Real-time data streaming manager.
    """

    def __init__(self, connector: IBConnector):
        self.connector = connector
        self.data_buffer: List[MarketData] = []
        self.max_buffer_size = 10000

    def on_new_bar(self, data: MarketData):
        """Handle new bar data."""
        self.data_buffer.append(data)

        # Trim buffer if too large
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]

    def get_recent_data(self, n_bars: int) -> pd.DataFrame:
        """Get recent bars as DataFrame."""
        if not self.data_buffer:
            return pd.DataFrame()

        bars = self.data_buffer[-n_bars:]
        return pd.DataFrame([{
            'datetime': b.timestamp,
            'open': b.open,
            'high': b.high,
            'low': b.low,
            'close': b.close,
            'volume': b.volume
        } for b in bars]).set_index('datetime')

    def start_streaming(self, contract: Contract):
        """Start streaming data."""
        self.connector.subscribe_realtime_bars(contract, self.on_new_bar)

    def stop_streaming(self, symbol: str):
        """Stop streaming data."""
        self.connector.unsubscribe_realtime_bars(symbol)
