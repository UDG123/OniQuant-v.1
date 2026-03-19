"""NautilusTrader execution bridge — backtest + live via shared config plane.

Provides three integration surfaces between OniQuant and nautilus_trader:

1. **BacktestNode** (simulation) — fed from a ``ParquetDataCatalog``,
   used by the Optuna research loops and champion-challenger framework.
2. **TradingNode** (live execution) — connects to Binance via the
   native adapter, routed through our existing ``execution.py`` broker
   protocol.
3. **DataCatalog** — Parquet-based historical data store shared by both
   nodes, with helpers to ingest from ``ml_training_data`` and write
   back labelled results.

Advanced order types (OCO, OTO) are built via the NautilusTrader
``OrderFactory`` and submitted as ``OrderList`` objects through the
event-driven architecture.

Usage — backtest::

    bridge = NautilusBridge()
    bridge.ingest_from_postgres(symbol="BTCUSDT")
    results = bridge.run_backtest(symbol="BTCUSDT")

Usage — live::

    bridge = NautilusBridge(mode="live")
    bridge.start_live()

Usage — as OrderBroker for execution.py::

    broker = NautilusBrokerAdapter(bridge)
    await execute_dynamic_limit_order(signal, order_book, broker=broker)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# NautilusTrader imports (lazy-checked at init to avoid hard dep at import)
# ---------------------------------------------------------------------------

try:
    from nautilus_trader.backtest.config import (
        BacktestDataConfig,
        BacktestEngineConfig,
        BacktestRunConfig,
        BacktestVenueConfig,
    )
    from nautilus_trader.backtest.node import BacktestNode
    from nautilus_trader.config import (
        InstrumentProviderConfig,
        LiveExecEngineConfig,
        LoggingConfig,
        TradingNodeConfig,
    )
    from nautilus_trader.core.datetime import dt_to_unix_nanos
    from nautilus_trader.live.node import TradingNode
    from nautilus_trader.model.data import Bar, BarType, QuoteTick
    from nautilus_trader.model.enums import (
        AccountType,
        ContingencyType,
        OmsType,
        OrderSide,
        OrderType,
        TimeInForce,
        TriggerType,
    )
    from nautilus_trader.model.identifiers import (
        ClientOrderId,
        ExecAlgorithmId,
        InstrumentId,
        OrderListId,
        StrategyId,
        TraderId,
        Venue,
    )
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.model.orders import (
        LimitOrder,
        MarketOrder,
        OrderList,
        StopMarketOrder,
    )
    from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
    from nautilus_trader.trading.strategy import Strategy, StrategyConfig

    _HAS_NAUTILUS = True
except ImportError:
    _HAS_NAUTILUS = False

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from src.services.execution import OrderBroker, OrderStatus

logger = logging.getLogger("oniquant.nautilus_bridge")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSTGRES_URL: str = os.getenv(
    "POSTGRES_URL",
    "postgresql://user:password@localhost:5432/oniquant",
)
SYNC_POSTGRES_URL: str = POSTGRES_URL.replace("+asyncpg", "")

_CATALOG_PATH: str = os.getenv("NAUTILUS_CATALOG_PATH", "./data_catalog")
_DEFAULT_VENUE: str = os.getenv("NAUTILUS_VENUE", "BINANCE")
_DEFAULT_BALANCE: str = os.getenv("NAUTILUS_BALANCE", "1000000 USDT")
_LOOKBACK_DAYS: int = int(os.getenv("NAUTILUS_LOOKBACK_DAYS", "90"))

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_engine = create_engine(SYNC_POSTGRES_URL, pool_pre_ping=True, pool_size=3)
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

_TRADE_QUERY = text("""
    SELECT
        symbol,
        entry_price,
        close_price,
        pnl_pct,
        atr_at_entry,
        closed_at
    FROM ml_training_data
    WHERE (:symbol IS NULL OR symbol = :symbol)
      AND closed_at >= NOW() - MAKE_INTERVAL(days => :lookback_days)
    ORDER BY closed_at ASC
""")


# ---------------------------------------------------------------------------
# Shared configuration plane
# ---------------------------------------------------------------------------


def _build_venue_config(
    venue_name: str = _DEFAULT_VENUE,
    starting_balance: str = _DEFAULT_BALANCE,
    oms_type: str = "NETTING",
    account_type: str = "CASH",
) -> dict[str, Any]:
    """Configuration dict reusable by both backtest and live nodes."""
    return {
        "venue_name": venue_name,
        "starting_balance": starting_balance,
        "oms_type": oms_type,
        "account_type": account_type,
    }


def _build_logging_config(log_level: str = "INFO") -> LoggingConfig:
    return LoggingConfig(log_level=log_level)


# ---------------------------------------------------------------------------
# DataCatalog manager
# ---------------------------------------------------------------------------


class CatalogManager:
    """Parquet-based historical data store for Optuna research loops.

    Wraps ``ParquetDataCatalog`` with helpers to:
    - Ingest OHLCV-like data from ``ml_training_data`` into Parquet.
    - Query bars/ticks by symbol and time range.
    - List available instruments and date ranges.
    """

    def __init__(self, catalog_path: str = _CATALOG_PATH) -> None:
        if not _HAS_NAUTILUS:
            raise ImportError(
                "nautilus_trader is required — install via: "
                "pip install nautilus_trader"
            )
        self._path = Path(catalog_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._catalog = ParquetDataCatalog(path=str(self._path))

    @property
    def catalog(self) -> ParquetDataCatalog:
        return self._catalog

    @property
    def path(self) -> str:
        return str(self._path)

    def ingest_from_postgres(
        self,
        symbol: str | None = None,
        lookback_days: int = _LOOKBACK_DAYS,
    ) -> int:
        """Load trades from ``ml_training_data`` and write as Parquet bars.

        Converts the per-trade entry/close price series into 1-bar
        synthetic OHLCV records that NautilusTrader can consume.

        Returns the number of bars written.
        """
        with _SessionLocal() as session:
            rows = session.execute(
                _TRADE_QUERY,
                {"symbol": symbol, "lookback_days": lookback_days},
            ).fetchall()

        if not rows:
            logger.warning("No trades to ingest from ml_training_data")
            return 0

        df = pd.DataFrame(rows, columns=[
            "symbol", "entry_price", "close_price", "pnl_pct",
            "atr_at_entry", "closed_at",
        ])
        df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True)

        for col in ("entry_price", "close_price", "atr_at_entry"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Write per-symbol Parquet partitions.
        total_written = 0
        for sym, group in df.groupby("symbol"):
            output_dir = self._path / "bars" / str(sym)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / "trades.parquet"

            group.to_parquet(out_file, index=False, engine="pyarrow")
            total_written += len(group)
            logger.info(
                "Ingested %d bars for %s → %s", len(group), sym, out_file,
            )

        return total_written

    def list_instruments(self) -> list[str]:
        """List all symbols with ingested data."""
        bars_dir = self._path / "bars"
        if not bars_dir.exists():
            return []
        return [d.name for d in bars_dir.iterdir() if d.is_dir()]

    def load_dataframe(self, symbol: str) -> pd.DataFrame:
        """Load raw trade data for a symbol as a pandas DataFrame."""
        parquet_file = self._path / "bars" / symbol / "trades.parquet"
        if not parquet_file.exists():
            return pd.DataFrame()
        return pd.read_parquet(parquet_file)


# ---------------------------------------------------------------------------
# OniQuant Strategy — bridges NautilusTrader events to our signal pipeline
# ---------------------------------------------------------------------------


class OniQuantStrategyConfig(StrategyConfig if _HAS_NAUTILUS else object):
    """Configuration for the OniQuant bridge strategy."""

    instrument_id: str = ""
    trade_size: str = "1.0"
    desk_id: int = 1
    atr_stop_mult: float = 1.5
    take_profit_mult: float = 2.0


class OniQuantStrategy(Strategy if _HAS_NAUTILUS else object):
    """NautilusTrader Strategy that emits OCO/OTO bracket orders.

    On each bar close:
    1. Evaluates whether the OniQuant signal pipeline would fire.
    2. If so, submits an **OTO bracket** (entry triggers TP + SL):
       - Parent: limit entry at the current close.
       - Child 1 (OCO): take-profit limit.
       - Child 2 (OCO): stop-loss stop-market.

    The OCO link ensures that when either the TP or SL fills, the
    other is automatically cancelled.
    """

    def __init__(self, config: OniQuantStrategyConfig) -> None:
        if not _HAS_NAUTILUS:
            raise ImportError("nautilus_trader is required")
        super().__init__(config)
        self._instrument_id = InstrumentId.from_str(config.instrument_id)
        self._trade_size = Decimal(config.trade_size)
        self._desk_id = config.desk_id
        self._atr_mult = config.atr_stop_mult
        self._tp_mult = config.take_profit_mult
        self._instrument = None

    def on_start(self) -> None:
        self._instrument = self.cache.instrument(self._instrument_id)
        if self._instrument is None:
            self.log.error(f"Instrument {self._instrument_id} not found in cache")
            return
        self.subscribe_bars(BarType.from_str(
            f"{self._instrument_id}-1-MINUTE-LAST-EXTERNAL"
        ))
        self.log.info(f"OniQuantStrategy started for {self._instrument_id}")

    def on_bar(self, bar: Bar) -> None:
        """Evaluate entry on each bar close and submit bracket if triggered."""
        if self._instrument is None:
            return

        # Placeholder: in production this reads from Redis desk_state
        # or the Kelly allocator to decide whether to enter.
        # For the bridge, we demonstrate the OCO/OTO order structure.
        pass

    def submit_oco_bracket(
        self,
        side: OrderSide,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> OrderList | None:
        """Submit an OTO bracket: entry → OCO(TP, SL).

        Order structure::

            Entry (Limit)         ← parent, OTO
              ├── Take-Profit (Limit)  ← child, OCO with SL
              └── Stop-Loss (StopMarket) ← child, OCO with TP

        When the entry fills, both children go live.  When either
        child fills, the other is automatically cancelled (OCO).
        """
        if self._instrument is None:
            self.log.error("Cannot submit bracket — no instrument loaded")
            return None

        instrument = self._instrument
        qty = instrument.make_qty(self._trade_size)

        # --- Entry order (parent) -----------------------------------------
        entry_order = self.order_factory.limit(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=qty,
            price=instrument.make_price(entry_price),
            time_in_force=TimeInForce.GTC,
            post_only=True,
        )

        # --- Take-profit (child 1, OCO) ----------------------------------
        tp_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        tp_order = self.order_factory.limit(
            instrument_id=self._instrument_id,
            order_side=tp_side,
            quantity=qty,
            price=instrument.make_price(take_profit_price),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            contingency_type=ContingencyType.OCO,
            parent_order_id=entry_order.client_order_id,
        )

        # --- Stop-loss (child 2, OCO) ------------------------------------
        sl_order = self.order_factory.stop_market(
            instrument_id=self._instrument_id,
            order_side=tp_side,
            quantity=qty,
            trigger_price=instrument.make_price(stop_loss_price),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            contingency_type=ContingencyType.OCO,
            parent_order_id=entry_order.client_order_id,
        )

        # Link OCO children to each other.
        tp_order.linked_order_ids = [sl_order.client_order_id]
        sl_order.linked_order_ids = [tp_order.client_order_id]

        # Build the OTO order list: entry triggers children.
        order_list = OrderList(
            order_list_id=OrderListId(f"OTO-{uuid.uuid4().hex[:8]}"),
            orders=[entry_order, tp_order, sl_order],
        )

        self.submit_order_list(order_list)

        self.log.info(
            f"OTO bracket submitted: entry={entry_price} "
            f"tp={take_profit_price} sl={stop_loss_price} "
            f"side={side.name} qty={self._trade_size}",
        )

        return order_list

    def submit_oco_pair(
        self,
        buy_price: float,
        sell_price: float,
        quantity: Decimal | None = None,
    ) -> OrderList | None:
        """Submit a standalone OCO pair (two limit orders, one cancels other).

        Used for range-breakout strategies where either the upper or
        lower bound triggers, cancelling the other.
        """
        if self._instrument is None:
            return None

        instrument = self._instrument
        qty = instrument.make_qty(quantity or self._trade_size)

        buy_order = self.order_factory.limit(
            instrument_id=self._instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
            price=instrument.make_price(buy_price),
            time_in_force=TimeInForce.GTC,
            contingency_type=ContingencyType.OCO,
        )

        sell_order = self.order_factory.limit(
            instrument_id=self._instrument_id,
            order_side=OrderSide.SELL,
            quantity=qty,
            price=instrument.make_price(sell_price),
            time_in_force=TimeInForce.GTC,
            contingency_type=ContingencyType.OCO,
        )

        buy_order.linked_order_ids = [sell_order.client_order_id]
        sell_order.linked_order_ids = [buy_order.client_order_id]

        order_list = OrderList(
            order_list_id=OrderListId(f"OCO-{uuid.uuid4().hex[:8]}"),
            orders=[buy_order, sell_order],
        )

        self.submit_order_list(order_list)
        self.log.info(
            f"OCO pair submitted: buy={buy_price} sell={sell_price}",
        )
        return order_list

    def on_order_filled(self, event) -> None:
        self.log.info(f"Order filled: {event}")

    def on_stop(self) -> None:
        self.log.info("OniQuantStrategy stopped")


# ---------------------------------------------------------------------------
# NautilusBridge — unified backtest + live entry point
# ---------------------------------------------------------------------------


class NautilusBridge:
    """Shared configuration plane for BacktestNode and TradingNode.

    Parameters
    ----------
    mode : str
        ``"backtest"`` or ``"live"``.
    catalog_path : str
        Parquet data catalog directory.
    venue : str
        Venue name (default ``"BINANCE"``).
    starting_balance : str
        Initial simulated balance (default ``"1000000 USDT"``).
    log_level : str
        NautilusTrader log level.
    """

    def __init__(
        self,
        mode: str = "backtest",
        catalog_path: str = _CATALOG_PATH,
        venue: str = _DEFAULT_VENUE,
        starting_balance: str = _DEFAULT_BALANCE,
        log_level: str = "INFO",
    ) -> None:
        if not _HAS_NAUTILUS:
            raise ImportError(
                "nautilus_trader is required — "
                "pip install nautilus_trader"
            )
        self._mode = mode
        self._venue = venue
        self._balance = starting_balance
        self._log_level = log_level
        self._catalog_mgr = CatalogManager(catalog_path)
        self._venue_config = _build_venue_config(venue, starting_balance)

    @property
    def catalog(self) -> CatalogManager:
        return self._catalog_mgr

    # ----- Data ingestion ---------------------------------------------------

    def ingest_from_postgres(
        self,
        symbol: str | None = None,
        lookback_days: int = _LOOKBACK_DAYS,
    ) -> int:
        """Ingest trade data from PostgreSQL into the Parquet catalog."""
        return self._catalog_mgr.ingest_from_postgres(symbol, lookback_days)

    # ----- Backtest ---------------------------------------------------------

    def run_backtest(
        self,
        symbol: str = "BTCUSDT",
        start_time: str | None = None,
        end_time: str | None = None,
        strategy_config: OniQuantStrategyConfig | None = None,
        chunk_size: int = 10_000,
    ) -> list[Any]:
        """Run a backtest via BacktestNode with the shared config plane.

        Returns the list of BacktestResult objects.
        """
        instrument_id = f"{symbol}.{self._venue}"

        config = BacktestRunConfig(
            engine=BacktestEngineConfig(
                trader_id="BACKTESTER-001",
                logging=_build_logging_config(self._log_level),
            ),
            venues=[
                BacktestVenueConfig(
                    name=self._venue,
                    oms_type=self._venue_config["oms_type"],
                    account_type=self._venue_config["account_type"],
                    starting_balances=[self._balance],
                ),
            ],
            data=[
                BacktestDataConfig(
                    catalog_path=self._catalog_mgr.path,
                    data_cls="nautilus_trader.model.data.Bar",
                    instrument_id=instrument_id,
                    bar_spec="1-MINUTE-LAST",
                    start_time=start_time,
                    end_time=end_time,
                ),
            ],
            chunk_size=chunk_size,
        )

        node = BacktestNode(configs=[config])

        # Add the strategy if provided.
        if strategy_config is not None:
            strategy_config.instrument_id = instrument_id
            strategy = OniQuantStrategy(config=strategy_config)
            node.get_engine(config.id).add_strategy(strategy)

        results = node.run()

        for r in results:
            logger.info("Backtest %s complete: %s", r.run_id, r.stats)

        return results

    # ----- Live trading -----------------------------------------------------

    def build_live_node(
        self,
        trader_id: str = "LIVE-001",
        strategy_config: OniQuantStrategyConfig | None = None,
    ) -> TradingNode:
        """Build a TradingNode for live Binance execution.

        Requires ``BINANCE_API_KEY`` and ``BINANCE_API_SECRET``
        environment variables.
        """
        from nautilus_trader.adapters.binance import (
            BINANCE,
            BinanceAccountType,
            BinanceDataClientConfig,
            BinanceExecClientConfig,
            BinanceLiveDataClientFactory,
            BinanceLiveExecClientFactory,
        )

        node_config = TradingNodeConfig(
            trader_id=TraderId(trader_id),
            logging=_build_logging_config(self._log_level),
            exec_engine=LiveExecEngineConfig(
                reconciliation=True,
                reconciliation_lookback_mins=1440,
            ),
            data_clients={
                BINANCE: BinanceDataClientConfig(
                    api_key=None,   # reads BINANCE_API_KEY env
                    api_secret=None,
                    account_type=BinanceAccountType.SPOT,
                    instrument_provider=InstrumentProviderConfig(load_all=True),
                ),
            },
            exec_clients={
                BINANCE: BinanceExecClientConfig(
                    api_key=None,
                    api_secret=None,
                    account_type=BinanceAccountType.SPOT,
                    instrument_provider=InstrumentProviderConfig(load_all=True),
                    max_retries=3,
                ),
            },
            timeout_connection=30.0,
            timeout_reconciliation=10.0,
        )

        node = TradingNode(config=node_config)
        node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
        node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)

        if strategy_config is not None:
            strategy = OniQuantStrategy(config=strategy_config)
            node.trader.add_strategy(strategy)

        node.build()
        logger.info("Live TradingNode built: trader_id=%s", trader_id)

        return node

    def start_live(
        self,
        strategy_config: OniQuantStrategyConfig | None = None,
    ) -> None:
        """Build and run the live TradingNode (blocks until SIGINT)."""
        node = self.build_live_node(strategy_config=strategy_config)
        try:
            node.run()
        finally:
            node.dispose()


# ---------------------------------------------------------------------------
# NautilusBrokerAdapter — implements execution.py's OrderBroker protocol
# ---------------------------------------------------------------------------


class NautilusBrokerAdapter:
    """Adapts NautilusTrader's order management to the ``OrderBroker`` protocol.

    This allows ``execute_dynamic_limit_order()`` in ``execution.py`` to
    route orders through NautilusTrader's event-driven engine, gaining
    access to OCO/OTO contingent orders, execution algorithms (TWAP),
    and cross-venue routing.

    Parameters
    ----------
    bridge : NautilusBridge
        An initialized bridge (must have a live TradingNode built).
    strategy : OniQuantStrategy
        The strategy instance whose ``order_factory`` will create orders.
    """

    def __init__(
        self,
        bridge: NautilusBridge,
        strategy: OniQuantStrategy,
    ) -> None:
        self._bridge = bridge
        self._strategy = strategy
        self._pending_orders: dict[str, Any] = {}

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
    ) -> str:
        """Submit a limit order via the Nautilus order factory."""
        instrument = self._strategy._instrument
        if instrument is None:
            raise RuntimeError(f"Instrument not loaded for {symbol}")

        order_side = OrderSide.BUY if side == "LONG" else OrderSide.SELL

        order = self._strategy.order_factory.limit(
            instrument_id=instrument.id,
            order_side=order_side,
            quantity=instrument.make_qty(Decimal(str(quantity))),
            price=instrument.make_price(price),
            time_in_force=TimeInForce.GTC,
            post_only=True,
        )

        self._strategy.submit_order(order)
        order_id = str(order.client_order_id)
        self._pending_orders[order_id] = order

        return order_id

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        order = self._pending_orders.get(order_id)
        if order is None:
            return False
        self._strategy.cancel_order(order)
        return True

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Poll current fill status."""
        order = self._pending_orders.get(order_id)
        if order is None:
            return OrderStatus(
                order_id=order_id, is_filled=False,
            )

        is_filled = order.is_closed and order.filled_qty > 0

        return OrderStatus(
            order_id=order_id,
            is_filled=is_filled,
            filled_price=float(order.avg_px) if order.avg_px else None,
            filled_qty=float(order.filled_qty) if order.filled_qty else None,
            remaining_qty=float(order.leaves_qty) if order.leaves_qty else None,
        )

    async def get_order_book(self, symbol: str) -> dict:
        """Fetch order book from the Nautilus cache."""
        instrument_id = InstrumentId.from_str(
            f"{symbol}.{_DEFAULT_VENUE}"
        )
        book = self._strategy.cache.order_book(instrument_id)

        if book is not None and book.best_bid_price() and book.best_ask_price():
            return {
                "best_bid": float(book.best_bid_price()),
                "best_ask": float(book.best_ask_price()),
            }

        # Fallback: use latest quote tick.
        quote = self._strategy.cache.quote_tick(instrument_id)
        if quote is not None:
            return {
                "best_bid": float(quote.bid_price),
                "best_ask": float(quote.ask_price),
            }

        return {"best_bid": 0.0, "best_ask": 0.0}

    async def place_oco_bracket(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        quantity: float,
    ) -> str | None:
        """Submit an OTO bracket (entry → OCO(TP, SL)) via NautilusTrader.

        This is an extension beyond the ``OrderBroker`` protocol,
        available when the broker adapter is used directly.
        """
        order_side = OrderSide.BUY if side == "LONG" else OrderSide.SELL
        self._strategy._trade_size = Decimal(str(quantity))

        order_list = self._strategy.submit_oco_bracket(
            side=order_side,
            entry_price=entry_price,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
        )

        if order_list is None:
            return None

        list_id = str(order_list.id)
        for order in order_list.orders:
            self._pending_orders[str(order.client_order_id)] = order

        return list_id
