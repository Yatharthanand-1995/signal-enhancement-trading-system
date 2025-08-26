-- Trading System Database Schema
-- Optimized for time-series financial data storage and retrieval

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Securities master table for top 100 US stocks
CREATE TABLE securities (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    sp500_weight DECIMAL(5,4),
    is_active BOOLEAN DEFAULT true,
    date_added DATE DEFAULT CURRENT_DATE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimized daily price data with partitioning
CREATE TABLE daily_prices (
    symbol_id INT REFERENCES securities(id),
    trade_date DATE NOT NULL,
    open DECIMAL(19,4) NOT NULL,
    high DECIMAL(19,4) NOT NULL,
    low DECIMAL(19,4) NOT NULL,
    close DECIMAL(19,4) NOT NULL,
    adj_close DECIMAL(19,4) NOT NULL,
    volume BIGINT NOT NULL,
    dollar_volume DECIMAL(24,2) GENERATED ALWAYS AS (close * volume) STORED,
    PRIMARY KEY (symbol_id, trade_date)
) PARTITION BY RANGE (trade_date);

-- Create monthly partitions for 2024 and 2023
CREATE TABLE daily_prices_2024_01 PARTITION OF daily_prices
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE daily_prices_2024_02 PARTITION OF daily_prices
FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE daily_prices_2024_03 PARTITION OF daily_prices
FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE daily_prices_2024_04 PARTITION OF daily_prices
FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE daily_prices_2024_05 PARTITION OF daily_prices
FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE daily_prices_2024_06 PARTITION OF daily_prices
FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE daily_prices_2024_07 PARTITION OF daily_prices
FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE daily_prices_2024_08 PARTITION OF daily_prices
FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE daily_prices_2024_09 PARTITION OF daily_prices
FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE daily_prices_2024_10 PARTITION OF daily_prices
FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE daily_prices_2024_11 PARTITION OF daily_prices
FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE daily_prices_2024_12 PARTITION OF daily_prices
FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

-- Create 2023 partitions
CREATE TABLE daily_prices_2023 PARTITION OF daily_prices
FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

-- Create 2022 partitions
CREATE TABLE daily_prices_2022 PARTITION OF daily_prices
FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');

-- Create 2021 partitions
CREATE TABLE daily_prices_2021 PARTITION OF daily_prices
FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');

-- Create 2020 partitions
CREATE TABLE daily_prices_2020 PARTITION OF daily_prices
FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');

-- Pre-calculated technical indicators table
CREATE TABLE technical_indicators (
    symbol_id INT REFERENCES securities(id),
    trade_date DATE NOT NULL,
    rsi_9 DECIMAL(5,2),
    rsi_14 DECIMAL(5,2),
    macd_value DECIMAL(19,4),
    macd_signal DECIMAL(19,4),
    macd_histogram DECIMAL(19,4),
    bb_upper DECIMAL(19,4),
    bb_middle DECIMAL(19,4),
    bb_lower DECIMAL(19,4),
    sma_20 DECIMAL(19,4),
    sma_50 DECIMAL(19,4),
    ema_12 DECIMAL(19,4),
    ema_26 DECIMAL(19,4),
    atr_14 DECIMAL(19,4),
    volume_sma_20 BIGINT,
    stoch_k DECIMAL(5,2),
    stoch_d DECIMAL(5,2),
    williams_r DECIMAL(5,2),
    PRIMARY KEY (symbol_id, trade_date),
    FOREIGN KEY (symbol_id, trade_date) REFERENCES daily_prices(symbol_id, trade_date)
);

-- Market regime detection results
CREATE TABLE market_regimes (
    id SERIAL PRIMARY KEY,
    detection_date DATE NOT NULL,
    regime_type INT NOT NULL CHECK (regime_type IN (0, 1, 2)),
    regime_name VARCHAR(50),
    volatility_level DECIMAL(5,4),
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance tracking and signals
CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    symbol_id INT REFERENCES securities(id),
    signal_date TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    direction VARCHAR(10) CHECK (direction IN ('BUY', 'SELL', 'HOLD')),
    strength DECIMAL(3,2),
    price_at_signal DECIMAL(19,4),
    predicted_price DECIMAL(19,4),
    confidence DECIMAL(3,2),
    model_version VARCHAR(20),
    technical_score DECIMAL(3,2),
    ml_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ML model predictions storage
CREATE TABLE ml_predictions (
    id SERIAL PRIMARY KEY,
    symbol_id INT REFERENCES securities(id),
    prediction_date DATE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction_value DECIMAL(19,4),
    prediction_direction VARCHAR(10),
    confidence_score DECIMAL(3,2),
    feature_vector JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Backtesting results
CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital DECIMAL(15,2),
    final_value DECIMAL(15,2),
    total_return DECIMAL(5,4),
    sharpe_ratio DECIMAL(5,3),
    max_drawdown DECIMAL(5,4),
    total_trades INT,
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(5,3),
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual trades from backtests
CREATE TABLE backtest_trades (
    id SERIAL PRIMARY KEY,
    backtest_id INT REFERENCES backtest_results(id),
    symbol_id INT REFERENCES securities(id),
    entry_date DATE NOT NULL,
    exit_date DATE,
    entry_price DECIMAL(19,4),
    exit_price DECIMAL(19,4),
    quantity INT,
    direction VARCHAR(10),
    pnl DECIMAL(15,2),
    holding_days INT,
    signal_strength DECIMAL(3,2)
);

-- Create optimized indexes
CREATE INDEX idx_daily_prices_symbol_date ON daily_prices(symbol_id, trade_date DESC);
CREATE INDEX idx_daily_prices_date ON daily_prices(trade_date DESC);
CREATE INDEX idx_technical_indicators_date ON technical_indicators(trade_date DESC);
CREATE INDEX idx_trading_signals_date ON trading_signals(signal_date DESC);
CREATE INDEX idx_trading_signals_symbol ON trading_signals(symbol_id, signal_date DESC);
CREATE INDEX idx_securities_active ON securities(is_active) WHERE is_active = true;
CREATE INDEX idx_ml_predictions_symbol_date ON ml_predictions(symbol_id, prediction_date DESC);
CREATE INDEX idx_backtest_results_strategy ON backtest_results(strategy_name, created_at DESC);

-- Create materialized view for latest prices and indicators
CREATE MATERIALIZED VIEW mv_latest_prices AS
SELECT 
    s.symbol,
    s.company_name,
    s.sector,
    dp.trade_date,
    dp.close,
    dp.volume,
    dp.dollar_volume,
    ti.rsi_14,
    ti.macd_histogram,
    ti.bb_upper,
    ti.bb_middle,
    ti.bb_lower,
    LAG(dp.close, 1) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date) as prev_close,
    (dp.close - LAG(dp.close, 1) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date)) / 
     NULLIF(LAG(dp.close, 1) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date), 0) * 100 as daily_return
FROM securities s
JOIN daily_prices dp ON s.id = dp.symbol_id
LEFT JOIN technical_indicators ti ON dp.symbol_id = ti.symbol_id AND dp.trade_date = ti.trade_date
WHERE dp.trade_date >= CURRENT_DATE - INTERVAL '60 days'
  AND s.is_active = true
WITH DATA;

-- Create index on materialized view
CREATE UNIQUE INDEX mv_latest_prices_symbol_date ON mv_latest_prices(symbol, trade_date);

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_materialized_views() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_latest_prices;
END;
$$ LANGUAGE plpgsql;

-- Function to create new monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partition(year_month TEXT) RETURNS void AS $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    start_date := (year_month || '-01')::DATE;
    end_date := (start_date + INTERVAL '1 month')::DATE;
    partition_name := 'daily_prices_' || REPLACE(year_month, '-', '_');
    
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I PARTITION OF daily_prices
        FOR VALUES FROM (%L) TO (%L)',
        partition_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;

-- Views for performance monitoring
CREATE VIEW v_table_stats AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    n_live_tup AS row_count,
    n_dead_tup AS dead_rows,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- View for slow queries
CREATE VIEW v_slow_queries AS
SELECT 
    query,
    calls,
    round(total_exec_time::numeric, 2) AS total_time_ms,
    round(mean_exec_time::numeric, 2) AS mean_time_ms,
    round(stddev_exec_time::numeric, 2) AS stddev_time_ms,
    rows
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean_exec_time DESC
LIMIT 20;

-- View for index usage statistics
CREATE VIEW v_index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan;

-- Insert some initial data for top 100 stocks (placeholder)
INSERT INTO securities (symbol, company_name, sector, industry, market_cap) VALUES
('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics', 2800000000000),
('MSFT', 'Microsoft Corporation', 'Technology', 'Software', 2700000000000),
('GOOGL', 'Alphabet Inc.', 'Communication Services', 'Internet Content & Information', 1700000000000),
('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 'Internet & Direct Marketing Retail', 1500000000000),
('NVDA', 'NVIDIA Corporation', 'Technology', 'Semiconductors', 1200000000000),
('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Automobiles', 800000000000),
('META', 'Meta Platforms Inc.', 'Communication Services', 'Interactive Media & Services', 750000000000),
('BRK.B', 'Berkshire Hathaway Inc.', 'Financial Services', 'Insurance - Diversified', 700000000000),
('UNH', 'UnitedHealth Group Incorporated', 'Healthcare', 'Healthcare Plans', 500000000000),
('JNJ', 'Johnson & Johnson', 'Healthcare', 'Drug Manufacturers - General', 450000000000);

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;