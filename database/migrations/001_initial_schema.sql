-- Migration: 001_initial_schema.sql
-- Create initial database schema for options pricing engine

-- Market snapshots table
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    underlying_symbol VARCHAR(10) NOT NULL,
    spot_price DECIMAL(12, 4) NOT NULL,
    risk_free_rate DECIMAL(6, 5) NOT NULL,
    dividend_yield DECIMAL(6, 5) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    INDEX idx_symbol_timestamp (underlying_symbol, timestamp)
);

-- Options market data table
CREATE TABLE IF NOT EXISTS options_market_data (
    option_id SERIAL PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES market_snapshots(snapshot_id) ON DELETE CASCADE,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
    strike DECIMAL(12, 4) NOT NULL,
    maturity_date DATE NOT NULL,
    time_to_maturity DECIMAL(10, 6) NOT NULL,
    bid_price DECIMAL(12, 6),
    ask_price DECIMAL(12, 6),
    mid_price DECIMAL(12, 6) NOT NULL,
    implied_volatility DECIMAL(8, 6),
    volume INTEGER,
    open_interest INTEGER,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
    INDEX idx_snapshot (snapshot_id),
    INDEX idx_strike_maturity (strike, time_to_maturity)
);

-- Model calibrations table
CREATE TABLE IF NOT EXISTS model_calibrations (
    calibration_id SERIAL PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES market_snapshots(snapshot_id) ON DELETE CASCADE,
    model_type VARCHAR(20) NOT NULL CHECK (model_type IN ('HESTON', 'MERTON', 'HESTON_MERTON')),
    
    -- Heston parameters
    v0 DECIMAL(10, 6),              -- Initial variance
    kappa DECIMAL(10, 6),           -- Mean reversion speed
    theta DECIMAL(10, 6),           -- Long-term variance
    sigma DECIMAL(10, 6),           -- Vol of vol
    rho DECIMAL(6, 4),              -- Correlation
    
    -- Merton parameters
    lambda DECIMAL(10, 6),          -- Jump intensity
    mu_j DECIMAL(6, 4),             -- Jump mean
    sigma_j DECIMAL(6, 4),          -- Jump std
    
    -- Calibration quality metrics
    rmse DECIMAL(12, 8) NOT NULL,
    mae DECIMAL(12, 8),
    max_error DECIMAL(12, 8),
    num_options INTEGER NOT NULL,
    
    -- Performance metrics
    calibration_time_ms INTEGER NOT NULL,
    optimization_method VARCHAR(20),
    success BOOLEAN NOT NULL,
    
    -- Metadata
    calibrated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    calibrated_by VARCHAR(50),
    notes TEXT,
    
    INDEX idx_model_type (model_type),
    INDEX idx_calibrated_at (calibrated_at),
    INDEX idx_snapshot (snapshot_id)
);

-- Pricing cache table (for audit trail)
CREATE TABLE IF NOT EXISTS pricing_cache (
    cache_id SERIAL PRIMARY KEY,
    cache_key VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(20) NOT NULL,
    option_type VARCHAR(4) NOT NULL,
    spot DECIMAL(12, 4) NOT NULL,
    strike DECIMAL(12, 4) NOT NULL,
    rate DECIMAL(6, 5) NOT NULL,
    dividend DECIMAL(6, 5) NOT NULL,
    maturity DECIMAL(10, 6) NOT NULL,
    price DECIMAL(12, 6) NOT NULL,
    method VARCHAR(10) NOT NULL,
    
    -- Model parameters (JSON for flexibility)
    model_params JSONB,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMP NOT NULL DEFAULT NOW(),
    access_count INTEGER DEFAULT 1,
    
    INDEX idx_cache_key (cache_key),
    INDEX idx_created_at (created_at)
);

-- Greeks cache table
CREATE TABLE IF NOT EXISTS greeks_cache (
    greeks_id SERIAL PRIMARY KEY,
    option_id INTEGER,
    calibration_id INTEGER REFERENCES model_calibrations(calibration_id),
    
    spot DECIMAL(12, 4) NOT NULL,
    strike DECIMAL(12, 4) NOT NULL,
    maturity DECIMAL(10, 6) NOT NULL,
    
    delta DECIMAL(10, 6) NOT NULL,
    gamma DECIMAL(10, 8) NOT NULL,
    vega DECIMAL(10, 6) NOT NULL,
    theta DECIMAL(10, 6) NOT NULL,
    rho DECIMAL(10, 6),
    
    price DECIMAL(12, 6) NOT NULL,
    
    computed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    INDEX idx_option (option_id),
    INDEX idx_calibration (calibration_id)
);

-- API request log table
CREATE TABLE IF NOT EXISTS api_requests (
    request_id SERIAL PRIMARY KEY,
    endpoint VARCHAR(50) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    request_body JSONB,
    response_body JSONB,
    error_message TEXT,
    client_ip VARCHAR(45),
    user_agent VARCHAR(255),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    INDEX idx_endpoint (endpoint),
    INDEX idx_timestamp (timestamp),
    INDEX idx_status (status_code)
);

-- Create views for common queries

-- Latest calibration for each symbol and model
CREATE VIEW latest_calibrations AS
SELECT DISTINCT ON (ms.underlying_symbol, mc.model_type)
    mc.calibration_id,
    ms.underlying_symbol,
    mc.model_type,
    mc.v0, mc.kappa, mc.theta, mc.sigma, mc.rho,
    mc.lambda, mc.mu_j, mc.sigma_j,
    mc.rmse,
    mc.calibrated_at
FROM model_calibrations mc
JOIN market_snapshots ms ON mc.snapshot_id = ms.snapshot_id
ORDER BY ms.underlying_symbol, mc.model_type, mc.calibrated_at DESC;

-- Calibration performance metrics
CREATE VIEW calibration_performance AS
SELECT 
    model_type,
    DATE(calibrated_at) as date,
    COUNT(*) as num_calibrations,
    AVG(rmse) as avg_rmse,
    AVG(calibration_time_ms) as avg_time_ms,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
FROM model_calibrations
GROUP BY model_type, DATE(calibrated_at)
ORDER BY date DESC, model_type;

-- API performance metrics
CREATE VIEW api_performance AS
SELECT 
    endpoint,
    DATE(timestamp) as date,
    COUNT(*) as request_count,
    AVG(duration_ms) as avg_duration_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as median_duration_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration_ms,
    SUM(CASE WHEN status_code < 400 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
FROM api_requests
GROUP BY endpoint, DATE(timestamp)
ORDER BY date DESC, endpoint;

-- Create functions for common operations

-- Function to clean old cache entries
CREATE OR REPLACE FUNCTION clean_old_cache_entries(days_old INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM pricing_cache
    WHERE created_at < NOW() - INTERVAL '1 day' * days_old
    AND access_count < 5;  -- Only delete rarely accessed entries
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get calibration statistics
CREATE OR REPLACE FUNCTION get_calibration_stats(
    symbol_param VARCHAR(10),
    model_param VARCHAR(20),
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    avg_v0 DECIMAL,
    avg_kappa DECIMAL,
    avg_theta DECIMAL,
    avg_sigma DECIMAL,
    avg_rho DECIMAL,
    avg_rmse DECIMAL,
    num_calibrations BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(mc.v0),
        AVG(mc.kappa),
        AVG(mc.theta),
        AVG(mc.sigma),
        AVG(mc.rho),
        AVG(mc.rmse),
        COUNT(*)
    FROM model_calibrations mc
    JOIN market_snapshots ms ON mc.snapshot_id = ms.snapshot_id
    WHERE ms.underlying_symbol = symbol_param
    AND mc.model_type = model_param
    AND mc.calibrated_at > NOW() - INTERVAL '1 day' * days_back;
END;
$$ LANGUAGE plpgsql;

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_mid_price ON options_market_data(mid_price);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_options_iv ON options_market_data(implied_volatility);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_calibrations_rmse ON model_calibrations(rmse);

-- Add comments for documentation
COMMENT ON TABLE market_snapshots IS 'Stores market snapshot data at specific timestamps';
COMMENT ON TABLE options_market_data IS 'Stores options prices and implied volatilities';
COMMENT ON TABLE model_calibrations IS 'Stores calibrated model parameters and quality metrics';
COMMENT ON TABLE pricing_cache IS 'Audit trail for pricing requests (actual cache is in Redis)';
COMMENT ON TABLE greeks_cache IS 'Stores computed Greeks for options';
COMMENT ON TABLE api_requests IS 'Logs all API requests for monitoring and debugging';

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO pricing_service;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pricing_service;
