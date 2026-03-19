# OniQuant v1 — Architecture Overview

## What Is This

OniQuant is an autonomous algorithmic trading platform built on FastAPI, Celery, PostgreSQL, and Redis. It runs five independent quantitative trading desks, each with a specialized strategy. Signals arrive via TradingView webhooks, pass through an 8-stage processing pipeline (including AI consensus scoring via Claude), and are executed with a dynamic limit-order chase engine. A nightly Optuna optimizer tunes all strategy parameters against the historical Sortino ratio.

**Total codebase: ~7,860 lines across 22 Python files.**

---

## Directory Structure

```
OniQuant-v.1/
├── Dockerfile                           # Python 3.11-slim, no CMD (Railway start commands)
├── requirements.txt                     # 12 production dependencies
├── .env.example                         # Environment variable template
│
└── src/
    ├── main.py                          # FastAPI app, health endpoint, Optuna dashboard mount
    │
    ├── core/
    │   ├── models.py                    # Async SQLAlchemy ORM (TradeLog, VetoLog, MLTrainingData)
    │   └── optimizer.py                 # RecursiveOptimizer — autonomous Optuna engine
    │
    ├── api/
    │   └── webhooks.py                  # TradingView webhook router (8-stage pipeline)
    │
    ├── strategies/
    │   ├── desk1_scalping.py            # Order Flow Imbalance (OFI) micro-structure scalping
    │   ├── desk2_fx.py                  # Kalman-filtered mean-reversion on FX pairs
    │   ├── desk3_swing.py               # Lorentzian distance k-NN classifier
    │   ├── desk4_gold.py                # XAU/USD volatility breakout with volume confirmation
    │   └── desk5_crypto.py              # CVD z-score momentum with ATR regime filter
    │
    ├── services/
    │   ├── execution.py                 # Dynamic limit-order engine with cancel-replace chase
    │   ├── signal_hydration.py          # Merge webhook + desk state for AI evaluation
    │   ├── redis_manager.py             # Async Redis: dedup, volatility lock, desk state, pending signals
    │   └── telegram_broadcaster.py      # Fire-and-forget Telegram trade notifications
    │
    └── workers/
        ├── trailing_stop.py             # Celery Beat (15s): trailing-stop sweep + pending order trigger
        └── optimizer_job.py             # Celery Beat (03:00 UTC): nightly Optuna parameter tuning
```

---

## Signal Processing Pipeline

```
TradingView Alert (webhook POST)
    │
    ▼
[1] Pydantic Validation
    │
    ▼
[2] Redis Deduplication (60s window, SET NX)
    │
    ▼
[3] Volatility Lock Check
    │
    ▼
[4] Fetch Desk State from Redis
    ├──→ Quantitative Strategy Evaluation (desk-specific)
    │       ├─ Desk 1: OFI score + price divergence
    │       ├─ Desk 2: Kalman z-score + RSI confirmation
    │       ├─ Desk 3: Lorentzian k-NN vote
    │       ├─ Desk 4: Breakout + volume confirmation
    │       └─ Desk 5: CVD z-score + regime filter
    │
    ▼
[5] Signal Hydration (merge webhook + desk state + system prompt)
    │
    ▼
[6] Claude API Dispatch (consensus_score 1–10, threshold ≥ 7.0)
    │
    ▼
[7] Execution Gate
    ├──→ Immediate: execute_dynamic_limit_order() with chase logic
    └──→ Deferred: enqueue to Redis ZSET (pending signals)
    │
    ▼
[8] Telegram Broadcast (fire-and-forget)
```

---

## The Five Trading Desks

### Desk 1 — OFI Scalping (`desk1_scalping.py`)

Detects order-flow imbalance divergences on 1–5 min data. When cumulative OFI is strong (normalised score > 0.60) but price is flat, it signals a divergence trade.

| Parameter | Default | Range |
|-----------|---------|-------|
| `ofi_window` | 14 | 8–30 |
| `ofi_threshold` | 0.60 | 0.30–0.85 |
| `price_flat_pct` | 0.15% | 0.05–0.40% |
| `atr_stop_mult` | 1.5 | 1.0–3.0 |

**Output:** `ScalpingResult` — signal, ofi_score, ofi_raw, price_delta_pct, dynamic_stop, is_valid_setup

---

### Desk 2 — Kalman FX (`desk2_fx.py`)

Scalar Kalman filter extracts smoothed price on major FX pairs. When the prediction-error z-score breaches ±2.0 and RSI confirms (oversold/overbought), it signals mean-reversion.

| Parameter | Default | Range |
|-----------|---------|-------|
| `z_entry` | 2.0 | 1.0–3.5 |
| `process_var` (Q) | 1e-5 | 1e-6–1e-3 (log) |
| `measurement_var` (R) | 1e-3 | 1e-4–1e-1 (log) |
| `rsi_oversold` | 35.0 | 20–40 |
| `rsi_overbought` | 65.0 | 60–80 |

**Output:** `FXResult` — signal, kalman_z_score, kalman_price, rsi, atr, dynamic_stop, is_valid_setup

---

### Desk 3 — Lorentzian Swing (`desk3_swing.py`)

k-NN classifier using Lorentzian distance `Σ log(1 + |aᵢ - bᵢ|)` over a feature matrix of normalised RSI, ADX, CCI, and price/EMA ratio. Bullish if neighbour consensus exceeds the threshold.

| Parameter | Default | Range |
|-----------|---------|-------|
| `neighbours` | 8 | 3–20 |
| `lookback` | 200 | 80–500 |
| `threshold` | 0.6 | 0.35–0.85 |
| `adx_period` | 14 | — |
| `cci_period` | 20 | — |
| `ema_period` | 50 | 20–100 |

**Output:** `SwingResult` — is_valid_setup, lorentzian_score, trend_direction, rsi, adx

---

### Desk 4 — Gold Breakout (`desk4_gold.py`)

Volatility-driven breakout on XAU/USD. Confirms via EMA trend alignment and volume surge above the rolling average multiplied by the volume factor.

| Parameter | Default | Range |
|-----------|---------|-------|
| `atr_period` | 14 | 7–28 |
| `ema_period` | 50 | 20–120 |
| `volume_factor` | 1.5 | 1.0–3.5 |
| `rr_multiplier` | 2.0 | 1.0–5.0 |

**Output:** `BreakoutResult` — is_valid_breakout, risk_reward_ratio, atr, trend_direction

---

### Desk 5 — CVD Crypto (`desk5_crypto.py`)

Cumulative Volume Delta z-score momentum. CVD is approximated via tick-rule proxy: `volume × (2×(close−low)/(high−low) − 1)`. Triggers on z-score spikes when ATR is in the "expanding" volatility regime.

| Parameter | Default | Range |
|-----------|---------|-------|
| `cvd_window` | 20 | 10–40 |
| `cvd_threshold` | 2.0 | 0.5–3.5 |
| `atr_regime_mult` | 1.25 | 1.05–1.80 |
| `atr_stop_mult` | 2.0 | 1.0–4.0 |
| `rsi_period` | 14 | — |

**Output:** `CryptoResult` — signal, cvd_zscore, volatility_regime, rsi, atr, dynamic_stop, is_valid_setup

---

## Database Schema

### `trade_log`

Full lifecycle of every simulated position (SIM_OPEN → SIM_CLOSED).

| Column | Type | Notes |
|--------|------|-------|
| desk_id | int | Trading desk 1–5 |
| symbol | varchar(32) | e.g. BTCUSDT, XAUUSD |
| side | varchar(8) | BUY / SELL |
| status | varchar(16) | SIM_OPEN / SIM_CLOSED |
| entry_price / exit_price | numeric(18,8) | |
| highest_price / lowest_price | numeric(18,8) | Trailing-stop bookkeeping |
| stop_price | numeric(18,8) | Current trailing stop level |
| mfe / mae | numeric(18,8) | Max Favourable / Adverse Excursion |
| mfe_mae_ratio | float | Trade quality score |
| pnl_pct | float | Profit/loss at close |

### `veto_log`

Every signal rejected by the Redis hot-path risk gates.

| Column | Type | Notes |
|--------|------|-------|
| signal_id | varchar(128) | Original webhook signal ID |
| reason | varchar(64) | volatility_lock, max_exposure, cooldown_active, duplicate_signal |
| raw_payload | text | Full webhook JSON for post-mortem |

### `ml_training_data`

Feature snapshot per closed trade — consumed by the Optuna optimizer.

| Column | Type | Notes |
|--------|------|-------|
| desk_id | int | Source desk |
| entry_price / close_price | numeric(18,8) | |
| pnl_pct | float | Trade return |
| atr_at_entry | float | ATR at signal time |
| volume_ratio | float | Volume vs rolling avg |
| rsi_at_entry / adx_at_entry / cci_at_entry | float | Technical indicators |
| ema_distance | float | Price distance from EMA (%) |
| cvd_zscore | float | CVD z-score at entry |
| lorentzian_score | float | k-NN vote fraction |
| consensus_score | float | AI ensemble agreement (0–1) |

---

## Optimization Engine

### Celery Worker (`optimizer_job.py`)

Runs nightly at 03:00 UTC. Loads the last 1,000 closed trades, runs 60 Optuna trials, and maximises the annualised Sortino ratio. Study state is persisted in PostgreSQL for distributed parallelism.

### RecursiveOptimizer (`core/optimizer.py`)

Standalone class decoupled from Celery — usable in notebooks, CLI scripts, or the Celery beat schedule.

**37 parameters** across all 5 desks + 1 cross-desk consensus gate, all via `trial.suggest_float()` / `trial.suggest_int()` (define-by-run API).

| Feature | Detail |
|---------|--------|
| Sampler | TPE with `multivariate=True` (correlated parameter modelling) |
| Pruner | MedianPruner (10 startup trials) |
| Storage | PostgreSQL RDB (distributed, persistent) |
| Fitness | Annualised Sortino ratio (MAR = 0) |
| Penalty | −10.0 if fewer than 30 trades survive filtering |
| Extras | Parameter importance (fANOVA), per-desk param extraction |

**Sortino formula:**

```
sortino = (mean_return / downside_std) × √252
```

Only negative returns contribute to downside deviation.

---

## Execution Engine (`services/execution.py`)

Dynamic limit-order placement with aggressive chase logic:

```
Place limit @ best bid (LONG) / best ask (SHORT)
    → Poll 30s (1s cadence)
    → If not filled: cancel-replace at new best price
    → Repeat up to 10 chase cycles (max ~5 min)
    → If still unfilled: mark EXPIRED
```

Implements the `OrderBroker` protocol (abstract interface) so exchange adapters are pluggable.

---

## Redis State Management

| Key Pattern | Type | TTL | Purpose |
|-------------|------|-----|---------|
| `signal:dedup:{id}` | String | 60s | Webhook deduplication |
| `global:volatility_lock` | String | varies | Halt all trading |
| `desk:{id}:state` | Hash | varies | Live OHLCV + indicators |
| `signals:pending` | ZSET | 15m | Deferred signals (score = target_price) |
| `price:latest:{SYMBOL}` | String | varies | Cached latest price |

---

## Deployment (Docker + Railway)

Single `Dockerfile` (python:3.11-slim), no `CMD`/`ENTRYPOINT`. Railway launches three services from the same image:

| Service | Start Command |
|---------|---------------|
| **Web** | `uvicorn src.main:app --host 0.0.0.0 --port $PORT` |
| **Trailing Stop** | `celery -A src.workers.trailing_stop worker --beat --loglevel=info` |
| **Optimizer** | `celery -A src.workers.optimizer_job worker --beat --loglevel=info` |

---

## Environment Variables

```bash
POSTGRES_URL=postgresql+asyncpg://user:password@localhost:5432/oniquant
REDIS_URL=redis://localhost:6379/0
CLAUDE_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxx
BINANCE_API_KEY=your_binance_api_key_here
TWELVEDATA_API_KEY=your_twelvedata_api_key_here
TRAILING_STOP_PCT=0.02          # 2% trailing stop
OPTIMIZER_TRADE_LOOKBACK=1000   # trades to evaluate
OPTIMIZER_N_TRIALS=80           # trials per optimization run
CONSENSUS_THRESHOLD=7.0         # minimum AI score to execute
TELEGRAM_BOT_TOKEN=...          # optional
TELEGRAM_CHAT_ID=...            # optional
```

---

## Dependencies

```
fastapi>=0.110.0          uvicorn[standard]>=0.29.0
redis>=5.0.0              asyncpg>=0.29.0
sqlalchemy[asyncio]>=2.0  orjson>=3.10.0
celery>=5.3.0             pandas>=2.2.0
pandas-ta>=0.3.14b1       optuna>=3.6.0
optuna-dashboard>=0.15.0  httpx>=0.27.0
```
