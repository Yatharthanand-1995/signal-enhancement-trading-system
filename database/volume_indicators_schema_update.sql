-- Volume Indicators Schema Updates
-- Add volume-based indicators to existing technical_indicators table
-- Research-backed volume indicators for enhanced signal generation

-- Add new volume indicator columns to technical_indicators table
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS obv BIGINT;
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS cmf DECIMAL(5,4);
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS mfi DECIMAL(5,2);
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS vwap DECIMAL(19,4);
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS accumulation_distribution BIGINT;
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS price_volume_trend DECIMAL(19,4);
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS volume_profile JSONB;

-- Add volume-specific indicators for enhanced analysis
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS volume_ratio DECIMAL(8,4);
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS volume_sma_10 BIGINT;
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS volume_ema_20 BIGINT;
ALTER TABLE technical_indicators ADD COLUMN IF NOT EXISTS unusual_volume_flag BOOLEAN DEFAULT FALSE;

-- Create indexes for new volume indicators to optimize queries
CREATE INDEX IF NOT EXISTS idx_ti_obv ON technical_indicators(obv);
CREATE INDEX IF NOT EXISTS idx_ti_cmf ON technical_indicators(cmf);
CREATE INDEX IF NOT EXISTS idx_ti_mfi ON technical_indicators(mfi);
CREATE INDEX IF NOT EXISTS idx_ti_vwap ON technical_indicators(vwap);
CREATE INDEX IF NOT EXISTS idx_ti_volume_ratio ON technical_indicators(volume_ratio);
CREATE INDEX IF NOT EXISTS idx_ti_unusual_volume ON technical_indicators(unusual_volume_flag) WHERE unusual_volume_flag = true;

-- Create volume profile analysis table for advanced volume analysis
CREATE TABLE IF NOT EXISTS volume_profile_analysis (
    id SERIAL PRIMARY KEY,
    symbol_id INT REFERENCES securities(id),
    analysis_date DATE NOT NULL,
    price_levels JSONB NOT NULL,
    volume_at_price JSONB NOT NULL,
    poc_price DECIMAL(19,4),  -- Point of Control (highest volume price)
    value_area_high DECIMAL(19,4),
    value_area_low DECIMAL(19,4),
    volume_node_strength DECIMAL(5,4),
    support_resistance_levels JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol_id, analysis_date)
);

-- Create index for volume profile queries
CREATE INDEX idx_volume_profile_symbol_date ON volume_profile_analysis(symbol_id, analysis_date DESC);

-- Create volume signals table for tracking volume-based trading signals
CREATE TABLE IF NOT EXISTS volume_signals (
    id SERIAL PRIMARY KEY,
    symbol_id INT REFERENCES securities(id),
    signal_date TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    direction VARCHAR(10) CHECK (direction IN ('BUY', 'SELL', 'HOLD')),
    strength DECIMAL(3,2),
    confidence DECIMAL(3,2),
    volume_indicator VARCHAR(20),  -- OBV, CMF, MFI, etc.
    volume_value DECIMAL(19,4),
    volume_threshold DECIMAL(19,4),
    supporting_indicators JSONB,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for volume signals
CREATE INDEX idx_volume_signals_symbol_date ON volume_signals(symbol_id, signal_date DESC);
CREATE INDEX idx_volume_signals_type ON volume_signals(signal_type, signal_date DESC);
CREATE INDEX idx_volume_signals_strength ON volume_signals(strength DESC) WHERE strength >= 0.7;

-- Update the materialized view to include volume indicators
DROP MATERIALIZED VIEW IF EXISTS mv_latest_prices;
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
    -- New volume indicators
    ti.obv,
    ti.cmf,
    ti.mfi,
    ti.vwap,
    ti.volume_ratio,
    ti.unusual_volume_flag,
    LAG(dp.close, 1) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date) as prev_close,
    (dp.close - LAG(dp.close, 1) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date)) / 
     NULLIF(LAG(dp.close, 1) OVER (PARTITION BY s.symbol ORDER BY dp.trade_date), 0) * 100 as daily_return
FROM securities s
JOIN daily_prices dp ON s.id = dp.symbol_id
LEFT JOIN technical_indicators ti ON dp.symbol_id = ti.symbol_id AND dp.trade_date = ti.trade_date
WHERE dp.trade_date >= CURRENT_DATE - INTERVAL '60 days'
  AND s.is_active = true
WITH DATA;

-- Recreate unique index for materialized view
CREATE UNIQUE INDEX mv_latest_prices_symbol_date ON mv_latest_prices(symbol, trade_date);

-- Create function to identify unusual volume activity
CREATE OR REPLACE FUNCTION detect_unusual_volume() RETURNS void AS $$
BEGIN
    -- Update unusual_volume_flag based on volume analysis
    UPDATE technical_indicators SET unusual_volume_flag = true
    WHERE volume_ratio >= 2.0  -- 2x normal volume
      AND trade_date >= CURRENT_DATE - INTERVAL '5 days';
      
    -- Log unusual volume detection
    INSERT INTO volume_signals (symbol_id, signal_date, signal_type, direction, 
                               strength, confidence, volume_indicator, explanation)
    SELECT 
        ti.symbol_id,
        NOW(),
        'VOLUME_SPIKE',
        CASE 
            WHEN dp.close > dp.open THEN 'BUY'
            WHEN dp.close < dp.open THEN 'SELL' 
            ELSE 'HOLD'
        END,
        LEAST(ti.volume_ratio / 2.0, 1.0)::DECIMAL(3,2),
        0.75,
        'VOLUME_RATIO',
        'Unusual volume detected: ' || ti.volume_ratio || 'x normal volume'
    FROM technical_indicators ti
    JOIN daily_prices dp ON ti.symbol_id = dp.symbol_id AND ti.trade_date = dp.trade_date
    WHERE ti.unusual_volume_flag = true
      AND ti.trade_date >= CURRENT_DATE - INTERVAL '1 day'
    ON CONFLICT DO NOTHING;
END;
$$ LANGUAGE plpgsql;

-- Create view for volume signal analysis
CREATE OR REPLACE VIEW v_volume_signal_summary AS
SELECT 
    s.symbol,
    vs.signal_date,
    vs.signal_type,
    vs.direction,
    vs.strength,
    vs.confidence,
    vs.volume_indicator,
    vs.explanation,
    dp.close as price_at_signal,
    dp.volume,
    ti.volume_ratio
FROM volume_signals vs
JOIN securities s ON vs.symbol_id = s.id
LEFT JOIN daily_prices dp ON vs.symbol_id = dp.symbol_id 
    AND DATE(vs.signal_date) = dp.trade_date
LEFT JOIN technical_indicators ti ON vs.symbol_id = ti.symbol_id 
    AND DATE(vs.signal_date) = ti.trade_date
WHERE vs.signal_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY vs.signal_date DESC;

-- Add comments for documentation
COMMENT ON COLUMN technical_indicators.obv IS 'On-Balance Volume - Granville (1963) accumulation/distribution indicator';
COMMENT ON COLUMN technical_indicators.cmf IS 'Chaikin Money Flow - 20-period money flow oscillator';
COMMENT ON COLUMN technical_indicators.mfi IS 'Money Flow Index - Volume-weighted RSI equivalent';
COMMENT ON COLUMN technical_indicators.vwap IS 'Volume-Weighted Average Price - Institutional trading reference';
COMMENT ON COLUMN technical_indicators.volume_profile IS 'JSON containing volume distribution at different price levels';
COMMENT ON TABLE volume_signals IS 'Volume-based trading signals with research-backed thresholds';
COMMENT ON FUNCTION detect_unusual_volume() IS 'Automated detection of unusual volume activity for signal generation';

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE volume_profile_analysis TO trading_user;
GRANT ALL PRIVILEGES ON TABLE volume_signals TO trading_user;
GRANT ALL PRIVILEGES ON SEQUENCE volume_profile_analysis_id_seq TO trading_user;
GRANT ALL PRIVILEGES ON SEQUENCE volume_signals_id_seq TO trading_user;