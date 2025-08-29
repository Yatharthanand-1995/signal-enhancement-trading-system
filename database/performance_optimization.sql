-- Database Performance Optimization Scripts
-- Strategic indexing, query optimization, and performance tuning

-- =======================
-- PERFORMANCE MONITORING
-- =======================

-- Enable query statistics collection
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET track_functions = 'all';
ALTER SYSTEM SET log_statement_stats = on;

-- Reload configuration
SELECT pg_reload_conf();

-- =======================
-- STRATEGIC INDEXING
-- =======================

-- Securities table indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_securities_symbol 
ON securities (symbol);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_securities_active 
ON securities (is_active) WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_securities_sector 
ON securities (sector);

-- Daily prices indexes (critical for trading queries)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_daily_prices_symbol_date 
ON daily_prices (symbol_id, trade_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_daily_prices_date_volume 
ON daily_prices (trade_date, volume) WHERE volume > 1000000;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_daily_prices_close_change 
ON daily_prices (close, (close - open)) WHERE (close - open) != 0;

-- Composite index for most common query pattern
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_daily_prices_symbol_date_ohlcv 
ON daily_prices (symbol_id, trade_date) INCLUDE (open, high, low, close, volume);

-- Trading signals indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trading_signals_symbol_date 
ON trading_signals (symbol, signal_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trading_signals_type_strength 
ON trading_signals (signal_type, strength DESC) WHERE strength > 0.5;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trading_signals_direction 
ON trading_signals (direction, confidence DESC) WHERE direction != 'HOLD';

-- ML predictions indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_symbol_date 
ON ml_predictions (symbol, prediction_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ml_predictions_model_confidence 
ON ml_predictions (model_name, confidence DESC) WHERE confidence > 0.6;

-- Performance metrics indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_date_type 
ON performance_metrics (metric_date DESC, metric_type);

-- Portfolio holdings indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_holdings_date 
ON portfolio_holdings (as_of_date DESC);

-- System logs indexes (for monitoring)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_logs_timestamp 
ON system_logs (log_timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_logs_level_component 
ON system_logs (log_level, component) WHERE log_level >= 'WARNING';

-- =======================
-- PARTITIONING OPTIMIZATION
-- =======================

-- Add hash partitioning for high-volume trading data
-- This is for future scaling when data grows significantly

-- Create partitioned trading_signals table (example for future use)
-- CREATE TABLE trading_signals_partitioned (
--     LIKE trading_signals INCLUDING ALL
-- ) PARTITION BY HASH (symbol);

-- Create partitions
-- CREATE TABLE trading_signals_part_0 PARTITION OF trading_signals_partitioned
--     FOR VALUES WITH (MODULUS 4, REMAINDER 0);
-- CREATE TABLE trading_signals_part_1 PARTITION OF trading_signals_partitioned
--     FOR VALUES WITH (MODULUS 4, REMAINDER 1);
-- CREATE TABLE trading_signals_part_2 PARTITION OF trading_signals_partitioned
--     FOR VALUES WITH (MODULUS 4, REMAINDER 2);
-- CREATE TABLE trading_signals_part_3 PARTITION OF trading_signals_partitioned
--     FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- =======================
-- QUERY OPTIMIZATION VIEWS
-- =======================

-- Materialized view for frequently accessed market data statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_market_summary AS
SELECT 
    dp.trade_date,
    COUNT(*) as securities_traded,
    SUM(dp.volume) as total_volume,
    SUM(dp.dollar_volume) as total_dollar_volume,
    AVG(dp.close) as avg_close_price,
    AVG(CASE WHEN dp.close > dp.open THEN 1 ELSE 0 END) as pct_gainers,
    MIN(dp.low) as market_low,
    MAX(dp.high) as market_high
FROM daily_prices dp
WHERE dp.trade_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY dp.trade_date;

-- Create index on materialized view
CREATE UNIQUE INDEX idx_mv_daily_market_summary_date 
ON mv_daily_market_summary (trade_date DESC);

-- Refresh schedule for materialized view (run daily)
-- This would be handled by a scheduled job in production

-- View for active trading opportunities
CREATE VIEW v_active_trading_opportunities AS
SELECT 
    s.symbol,
    s.company_name,
    dp.close as current_price,
    dp.volume,
    ts.signal_type,
    ts.direction,
    ts.strength,
    ts.confidence,
    mp.predicted_return,
    mp.confidence as ml_confidence
FROM securities s
JOIN daily_prices dp ON s.id = dp.symbol_id 
    AND dp.trade_date = (
        SELECT MAX(trade_date) 
        FROM daily_prices 
        WHERE symbol_id = s.id
    )
LEFT JOIN trading_signals ts ON s.symbol = ts.symbol 
    AND ts.signal_date >= CURRENT_DATE - INTERVAL '1 day'
    AND ts.strength > 0.6
LEFT JOIN ml_predictions mp ON s.symbol = mp.symbol 
    AND mp.prediction_date >= CURRENT_DATE - INTERVAL '1 day'
    AND mp.confidence > 0.7
WHERE s.is_active = true
    AND dp.volume > 100000
    AND (ts.strength > 0.6 OR mp.confidence > 0.7);

-- =======================
-- PERFORMANCE STATISTICS
-- =======================

-- Function to get table statistics
CREATE OR REPLACE FUNCTION get_table_stats()
RETURNS TABLE(
    table_name TEXT,
    row_count BIGINT,
    table_size TEXT,
    index_size TEXT,
    total_size TEXT,
    last_analyze TIMESTAMP WITH TIME ZONE,
    last_vacuum TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        n_tup_ins + n_tup_upd - n_tup_del as row_count,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) + pg_indexes_size(schemaname||'.'||tablename)) as total_size,
        last_analyze,
        last_vacuum
    FROM pg_stat_user_tables 
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get slow queries
CREATE OR REPLACE FUNCTION get_slow_queries(min_duration_ms INTEGER DEFAULT 1000)
RETURNS TABLE(
    query TEXT,
    calls BIGINT,
    total_time_ms NUMERIC,
    mean_time_ms NUMERIC,
    rows_returned BIGINT,
    stddev_time_ms NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        LEFT(query, 100) as query,
        calls,
        ROUND(total_exec_time, 2) as total_time_ms,
        ROUND(mean_exec_time, 2) as mean_time_ms,
        rows,
        ROUND(stddev_exec_time, 2) as stddev_time_ms
    FROM pg_stat_statements 
    WHERE mean_exec_time > min_duration_ms
    ORDER BY mean_exec_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- Function to get index usage statistics
CREATE OR REPLACE FUNCTION get_index_usage()
RETURNS TABLE(
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    idx_scan BIGINT,
    idx_tup_read BIGINT,
    idx_tup_fetch BIGINT,
    size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        psi.schemaname,
        psi.tablename,
        psi.indexname,
        psi.idx_scan,
        psi.idx_tup_read,
        psi.idx_tup_fetch,
        pg_size_pretty(pg_relation_size(psi.schemaname||'.'||psi.indexname)) as size
    FROM pg_stat_user_indexes psi
    JOIN pg_indexes pi ON psi.schemaname = pi.schemaname 
        AND psi.tablename = pi.tablename 
        AND psi.indexname = pi.indexname
    ORDER BY psi.idx_scan DESC;
END;
$$ LANGUAGE plpgsql;

-- =======================
-- CONNECTION OPTIMIZATION
-- =======================

-- Optimize connection settings for trading workload
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Optimize for read-heavy workloads (common in trading)
ALTER SYSTEM SET seq_page_cost = 1.0;
ALTER SYSTEM SET cpu_tuple_cost = 0.01;
ALTER SYSTEM SET cpu_index_tuple_cost = 0.005;
ALTER SYSTEM SET cpu_operator_cost = 0.0025;

-- Enable parallel query processing
ALTER SYSTEM SET max_parallel_workers = 4;
ALTER SYSTEM SET max_parallel_workers_per_gather = 2;
ALTER SYSTEM SET parallel_tuple_cost = 0.1;
ALTER SYSTEM SET parallel_setup_cost = 1000.0;

-- =======================
-- MAINTENANCE PROCEDURES
-- =======================

-- Procedure for automated maintenance
CREATE OR REPLACE FUNCTION automated_maintenance()
RETURNS TEXT AS $$
DECLARE
    result TEXT := 'Automated maintenance completed:' || chr(10);
BEGIN
    -- Update table statistics
    ANALYZE securities;
    ANALYZE daily_prices;
    ANALYZE trading_signals;
    ANALYZE ml_predictions;
    result := result || '- Statistics updated' || chr(10);
    
    -- Refresh materialized views
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_market_summary;
    result := result || '- Materialized views refreshed' || chr(10);
    
    -- Vacuum analyze high-activity tables
    VACUUM ANALYZE trading_signals;
    VACUUM ANALYZE daily_prices;
    result := result || '- Vacuum completed on active tables' || chr(10);
    
    -- Clean up old log entries (keep last 30 days)
    DELETE FROM system_logs 
    WHERE log_timestamp < CURRENT_DATE - INTERVAL '30 days';
    result := result || '- Old log entries cleaned up' || chr(10);
    
    -- Clean up old performance metrics (keep last 90 days)
    DELETE FROM performance_metrics 
    WHERE metric_date < CURRENT_DATE - INTERVAL '90 days';
    result := result || '- Old performance metrics cleaned up' || chr(10);
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- =======================
-- PERFORMANCE MONITORING QUERIES
-- =======================

-- Query to identify missing indexes
CREATE VIEW v_missing_indexes AS
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
    AND n_distinct > 100
    AND correlation < 0.1
    AND tablename NOT IN (
        SELECT DISTINCT tablename 
        FROM pg_indexes 
        WHERE schemaname = 'public'
    );

-- Query to find unused indexes
CREATE VIEW v_unused_indexes AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(schemaname||'.'||indexname)) as size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
    AND schemaname = 'public'
ORDER BY pg_relation_size(schemaname||'.'||indexname) DESC;

-- Real-time performance dashboard query
CREATE VIEW v_performance_dashboard AS
SELECT 
    'Database Size' as metric,
    pg_size_pretty(pg_database_size(current_database())) as value,
    'bytes' as unit
UNION ALL
SELECT 
    'Active Connections',
    count(*)::text,
    'connections'
FROM pg_stat_activity 
WHERE state = 'active'
UNION ALL
SELECT 
    'Cache Hit Ratio',
    round(
        100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2
    )::text || '%',
    'percentage'
FROM pg_stat_database
UNION ALL
SELECT 
    'Transactions Per Second',
    round(
        sum(xact_commit + xact_rollback) / 
        EXTRACT(epoch FROM (now() - stats_reset)), 2
    )::text,
    'tps'
FROM pg_stat_database
WHERE datname = current_database();

-- =======================
-- CLEANUP AND FINALIZE
-- =======================

-- Apply all system configuration changes
SELECT pg_reload_conf();

-- Update all table statistics
ANALYZE;

-- Display optimization summary
SELECT 
    'Performance optimization completed!' as status,
    (SELECT count(*) FROM pg_indexes WHERE schemaname = 'public') as indexes_created,
    (SELECT count(*) FROM pg_matviews WHERE schemaname = 'public') as materialized_views,
    (SELECT count(*) FROM pg_views WHERE schemaname = 'public') as views_created;

-- Show current database performance metrics
SELECT * FROM v_performance_dashboard;

COMMIT;