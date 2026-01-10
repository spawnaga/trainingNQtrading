"""
Logging utilities for the trading system.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger


def setup_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    console: bool = True,
    file: bool = True,
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None:
    """
    Setup loguru logger with console and file handlers.

    Args:
        log_dir: Directory for log files
        log_level: Minimum log level
        console: Enable console output
        file: Enable file output
        rotation: Log file rotation size
        retention: Log file retention period
    """
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    if console:
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )

    # File handler
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Main log file
        logger.add(
            str(log_path / "trading_{time:YYYY-MM-DD}.log"),
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

        # Error log file
        logger.add(
            str(log_path / "errors_{time:YYYY-MM-DD}.log"),
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

        # Trade log file
        logger.add(
            str(log_path / "trades_{time:YYYY-MM-DD}.log"),
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
            filter=lambda record: "trade" in record["extra"],
            rotation="1 day",
            retention="90 days"
        )


def get_logger(name: str = None):
    """
    Get a logger instance.

    Args:
        name: Optional logger name for context

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


def log_trade(
    action: str,
    symbol: str,
    quantity: int,
    price: float,
    pnl: Optional[float] = None,
    **kwargs
) -> None:
    """
    Log a trade execution.

    Args:
        action: Trade action (BUY, SELL, CLOSE)
        symbol: Trading symbol
        quantity: Number of contracts
        price: Execution price
        pnl: Realized P&L (if closing)
        **kwargs: Additional trade details
    """
    trade_logger = logger.bind(trade=True)

    msg = f"{action} {quantity} {symbol} @ {price:.2f}"
    if pnl is not None:
        msg += f" | P&L: ${pnl:.2f}"

    for key, value in kwargs.items():
        msg += f" | {key}: {value}"

    trade_logger.info(msg)


def log_performance(metrics: dict) -> None:
    """
    Log performance metrics.

    Args:
        metrics: Dictionary of performance metrics
    """
    logger.info("=" * 50)
    logger.info("Performance Report")
    logger.info("=" * 50)

    for key, value in metrics.items():
        if isinstance(value, float):
            if 'rate' in key.lower() or 'return' in key.lower() or 'drawdown' in key.lower():
                logger.info(f"{key}: {value:.2%}")
            else:
                logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    logger.info("=" * 50)


class TradeLogger:
    """
    Context manager for logging trade operations.
    """

    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting {self.operation}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is not None:
            logger.error(f"{self.operation} failed after {duration:.2f}s: {exc_val}")
            return False

        logger.info(f"{self.operation} completed in {duration:.2f}s")
        return True
