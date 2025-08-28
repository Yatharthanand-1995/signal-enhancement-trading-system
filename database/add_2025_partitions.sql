-- Add 2025 monthly partitions to daily_prices table
-- This fixes the missing partitions issue identified in the analysis

-- January 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_01 PARTITION OF daily_prices
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- February 2025  
CREATE TABLE IF NOT EXISTS daily_prices_2025_02 PARTITION OF daily_prices
FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- March 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_03 PARTITION OF daily_prices
FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- April 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_04 PARTITION OF daily_prices
FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');

-- May 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_05 PARTITION OF daily_prices
FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');

-- June 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_06 PARTITION OF daily_prices
FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');

-- July 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_07 PARTITION OF daily_prices
FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

-- August 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_08 PARTITION OF daily_prices
FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');

-- September 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_09 PARTITION OF daily_prices
FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');

-- October 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_10 PARTITION OF daily_prices
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

-- November 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_11 PARTITION OF daily_prices
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- December 2025
CREATE TABLE IF NOT EXISTS daily_prices_2025_12 PARTITION OF daily_prices
FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Add indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_01_symbol_date ON daily_prices_2025_01(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_02_symbol_date ON daily_prices_2025_02(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_03_symbol_date ON daily_prices_2025_03(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_04_symbol_date ON daily_prices_2025_04(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_05_symbol_date ON daily_prices_2025_05(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_06_symbol_date ON daily_prices_2025_06(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_07_symbol_date ON daily_prices_2025_07(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_08_symbol_date ON daily_prices_2025_08(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_09_symbol_date ON daily_prices_2025_09(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_10_symbol_date ON daily_prices_2025_10(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_11_symbol_date ON daily_prices_2025_11(symbol_id, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_2025_12_symbol_date ON daily_prices_2025_12(symbol_id, trade_date DESC);