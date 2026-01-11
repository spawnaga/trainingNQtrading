"""
Trading Monitor Dashboard
=========================
Real-time monitoring of the live trading system.
Displays signals, positions, and P&L.

Usage:
    python trading_monitor.py --paper
    python trading_monitor.py --live
"""

import argparse
import time
import sys
from datetime import datetime
from pathlib import Path

from ib_insync import IB, Future

sys.path.insert(0, str(Path(__file__).parent))


def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='')


def format_pnl(value: float) -> str:
    """Format P&L with color."""
    if value > 0:
        return f'\033[92m+${value:,.2f}\033[0m'  # Green
    elif value < 0:
        return f'\033[91m-${abs(value):,.2f}\033[0m'  # Red
    return f'${value:,.2f}'


def main():
    parser = argparse.ArgumentParser(description='Trading Monitor')
    parser.add_argument('--paper', action='store_true', help='Paper trading')
    parser.add_argument('--live', action='store_true', help='Live trading')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--client-id', type=int, default=2)
    args = parser.parse_args()

    if not args.paper and not args.live:
        print("Specify --paper or --live")
        sys.exit(1)

    port = 7497 if args.paper else 7496
    mode = "PAPER" if args.paper else "LIVE"

    ib = IB()

    try:
        print(f"Connecting to IB ({mode})...")
        ib.connect(args.host, port, clientId=args.client_id)

        # Create NQ contract
        contract = Future('NQ', exchange='CME', currency='USD')
        contracts = ib.qualifyContracts(contract)
        if contracts:
            contract = contracts[0]
        else:
            print("Could not qualify NQ contract")
            sys.exit(1)

        # Subscribe to ticker
        ticker = ib.reqMktData(contract)

        print("Connected. Monitoring...")
        time.sleep(2)

        while True:
            ib.sleep(1)

            clear_screen()

            print("=" * 60)
            print(f"  NQ TRADING MONITOR - {mode}")
            print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            # Market data
            print("\n--- MARKET DATA ---")
            print(f"  Last Price:  ${ticker.last:,.2f}" if ticker.last else "  Last Price:  N/A")
            print(f"  Bid:         ${ticker.bid:,.2f}" if ticker.bid else "  Bid:         N/A")
            print(f"  Ask:         ${ticker.ask:,.2f}" if ticker.ask else "  Ask:         N/A")
            print(f"  Volume:      {ticker.volume:,.0f}" if ticker.volume else "  Volume:      N/A")

            # Positions
            print("\n--- POSITIONS ---")
            positions = ib.positions()
            nq_position = 0
            nq_avg_cost = 0
            for pos in positions:
                if pos.contract.symbol == 'NQ':
                    nq_position = int(pos.position)
                    nq_avg_cost = pos.avgCost

            print(f"  NQ Position: {nq_position}")
            if nq_position != 0:
                print(f"  Avg Cost:    ${nq_avg_cost:,.2f}")
                if ticker.last:
                    unrealized = (ticker.last - nq_avg_cost) * nq_position * 20  # $20 per point
                    print(f"  Unrealized:  {format_pnl(unrealized)}")

            # Account summary
            print("\n--- ACCOUNT ---")
            account = ib.accountSummary()
            for item in account:
                if item.tag in ['NetLiquidation', 'TotalCashValue', 'UnrealizedPnL', 'RealizedPnL']:
                    value = float(item.value)
                    if 'PnL' in item.tag:
                        print(f"  {item.tag}: {format_pnl(value)}")
                    else:
                        print(f"  {item.tag}: ${value:,.2f}")

            # Open orders
            print("\n--- OPEN ORDERS ---")
            orders = ib.openOrders()
            if orders:
                for order in orders:
                    print(f"  {order.action} {order.totalQuantity} @ {order.orderType}")
            else:
                print("  No open orders")

            # Recent fills
            print("\n--- RECENT FILLS ---")
            fills = ib.fills()
            recent_fills = [f for f in fills if f.contract.symbol == 'NQ'][-5:]
            if recent_fills:
                for fill in recent_fills:
                    print(f"  {fill.execution.side} {fill.execution.shares} @ ${fill.execution.price:,.2f}")
            else:
                print("  No recent fills")

            print("\n" + "=" * 60)
            print("  Press Ctrl+C to exit")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ib.disconnect()


if __name__ == '__main__':
    main()
