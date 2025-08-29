"""
SQLite-Compatible Database Schema for Comprehensive Backtesting System
"""

import sqlite3
from typing import Dict, Any, List
from datetime import datetime
import json
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

class SQLiteBacktestingSchema:
    """Manages SQLite database schema for backtesting system"""
    
    def __init__(self, db_path: str = "data/historical_stocks.db"):
        self.db_path = db_path
        self.schema_version = "1.0.0"
        
    def get_connection(self):
        """Get SQLite database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
        
    def create_backtesting_tables(self):
        """Create all backtesting-related tables for SQLite"""
        
        tables_sql = {
            # Market Regime Classification
            "market_regimes": """
            CREATE TABLE IF NOT EXISTS market_regimes (
                regime_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_date DATE NOT NULL,
                end_date DATE,
                regime_name VARCHAR(50) NOT NULL,
                regime_type VARCHAR(20) NOT NULL, -- 'bull', 'bear', 'correction', 'crash', 'recovery'
                volatility_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'extreme'
                vix_range VARCHAR(20),
                avg_vix DECIMAL(5,2),
                sp500_return DECIMAL(8,4), -- Return during this regime
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_market_regimes_dates ON market_regimes(start_date, end_date);
            CREATE INDEX IF NOT EXISTS idx_market_regimes_type ON market_regimes(regime_type, volatility_level);
            """,
            
            # Backtest Configurations
            "backtest_configs": """
            CREATE TABLE IF NOT EXISTS backtest_configs (
                config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_name VARCHAR(100) NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                initial_capital DECIMAL(15,2) DEFAULT 100000,
                strategy_type VARCHAR(50) NOT NULL, -- 'equal_weight', 'concentration', 'regime_adaptive', 'sector_neutral'
                position_sizing_method VARCHAR(50) NOT NULL, -- 'equal', 'volatility_adjusted', 'kelly', 'fixed_dollar'
                max_position_size DECIMAL(5,4) DEFAULT 0.10, -- Maximum position as % of portfolio
                max_positions INTEGER DEFAULT 10,
                rebalance_frequency VARCHAR(20) DEFAULT 'daily', -- 'daily', 'weekly', 'monthly'
                transaction_costs DECIMAL(8,6) DEFAULT 0.001, -- 0.1% default
                slippage_model VARCHAR(20) DEFAULT 'fixed', -- 'fixed', 'volume_based', 'volatility_based'
                slippage_rate DECIMAL(8,6) DEFAULT 0.0005, -- 0.05% default slippage
                commission_per_trade DECIMAL(6,2) DEFAULT 1.0, -- $1 per trade
                min_trade_size DECIMAL(10,2) DEFAULT 1000, -- Minimum $1000 trade
                universe_filter TEXT DEFAULT '{"top_n_stocks": 100}', -- Stock universe constraints (JSON)
                signal_threshold DECIMAL(4,3) DEFAULT 0.5, -- Minimum signal strength
                stop_loss_pct DECIMAL(5,4), -- Stop loss percentage
                take_profit_pct DECIMAL(5,4), -- Take profit percentage
                max_holding_days INTEGER, -- Maximum holding period
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT -- Additional strategy-specific parameters (JSON)
            );
            
            CREATE INDEX IF NOT EXISTS idx_backtest_configs_dates ON backtest_configs(start_date, end_date);
            CREATE INDEX IF NOT EXISTS idx_backtest_configs_strategy ON backtest_configs(strategy_type);
            """,
            
            # Backtest Results Summary
            "backtest_results": """
            CREATE TABLE IF NOT EXISTS backtest_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id INTEGER REFERENCES backtest_configs(config_id) ON DELETE CASCADE,
                total_return DECIMAL(10,6), -- Total portfolio return
                annualized_return DECIMAL(10,6), -- Annualized return
                volatility DECIMAL(10,6), -- Annual volatility
                sharpe_ratio DECIMAL(8,4), -- Risk-adjusted return
                calmar_ratio DECIMAL(8,4), -- Return/Max Drawdown
                sortino_ratio DECIMAL(8,4), -- Downside risk adjusted
                max_drawdown DECIMAL(8,4), -- Maximum portfolio drawdown
                max_drawdown_duration INTEGER, -- Days to recover from max DD
                avg_drawdown DECIMAL(8,4), -- Average drawdown
                win_rate DECIMAL(5,4), -- Percentage of winning trades
                avg_win DECIMAL(8,4), -- Average winning trade %
                avg_loss DECIMAL(8,4), -- Average losing trade %
                avg_win_loss_ratio DECIMAL(8,4), -- Avg win / Avg loss
                profit_factor DECIMAL(8,4), -- Gross profit / Gross loss
                expectancy DECIMAL(8,6), -- Expected return per trade
                total_trades INTEGER, -- Total number of trades
                profitable_trades INTEGER, -- Number of winning trades
                losing_trades INTEGER, -- Number of losing trades
                avg_holding_days DECIMAL(6,2), -- Average holding period
                turnover_rate DECIMAL(8,4), -- Portfolio turnover
                information_ratio DECIMAL(8,4), -- Active return / Tracking error
                tracking_error DECIMAL(8,4), -- Std dev of excess returns
                beta DECIMAL(6,4), -- Portfolio beta vs benchmark
                alpha DECIMAL(8,4), -- Jensen's alpha
                var_95 DECIMAL(8,4), -- 95% Value at Risk
                cvar_95 DECIMAL(8,4), -- 95% Conditional VaR
                benchmark_return DECIMAL(10,6), -- Benchmark total return
                benchmark_volatility DECIMAL(10,6), -- Benchmark volatility
                excess_return DECIMAL(10,6), -- Portfolio - Benchmark return
                active_return DECIMAL(10,6), -- Annualized active return
                correlation_with_benchmark DECIMAL(6,4), -- Correlation coefficient
                upside_capture DECIMAL(6,4), -- Upside capture ratio
                downside_capture DECIMAL(6,4), -- Downside capture ratio
                up_months INTEGER, -- Number of positive months
                down_months INTEGER, -- Number of negative months
                best_month DECIMAL(8,4), -- Best monthly return
                worst_month DECIMAL(8,4), -- Worst monthly return
                final_portfolio_value DECIMAL(15,2), -- Final portfolio value
                total_fees DECIMAL(10,2), -- Total transaction costs
                execution_time_seconds DECIMAL(8,3), -- Backtest execution time
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT -- Additional metrics and metadata (JSON)
            );
            
            CREATE INDEX IF NOT EXISTS idx_backtest_results_config ON backtest_results(config_id);
            CREATE INDEX IF NOT EXISTS idx_backtest_results_performance ON backtest_results(sharpe_ratio DESC, total_return DESC);
            """,
            
            # Individual Trade Records
            "backtest_trades": """
            CREATE TABLE IF NOT EXISTS backtest_trades (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER REFERENCES backtest_results(result_id) ON DELETE CASCADE,
                symbol VARCHAR(10) NOT NULL,
                entry_date DATE NOT NULL,
                exit_date DATE,
                entry_price DECIMAL(12,4) NOT NULL,
                exit_price DECIMAL(12,4),
                quantity INTEGER NOT NULL,
                trade_direction VARCHAR(5) NOT NULL CHECK (trade_direction IN ('LONG', 'SHORT')),
                entry_signal_strength DECIMAL(5,3), -- Signal strength at entry
                exit_signal_strength DECIMAL(5,3), -- Signal strength at exit
                signal_components TEXT, -- Individual indicator contributions (JSON)
                entry_regime VARCHAR(50), -- Market regime at entry
                exit_regime VARCHAR(50), -- Market regime at exit
                market_conditions TEXT, -- VIX, sector performance, etc. (JSON)
                sector VARCHAR(30), -- Stock sector
                market_cap VARCHAR(20), -- Large/Mid/Small cap
                entry_reason VARCHAR(50), -- 'signal', 'rebalance', 'forced'
                exit_reason VARCHAR(50), -- 'signal', 'stop_loss', 'take_profit', 'time_limit', 'rebalance'
                gross_pnl DECIMAL(15,4), -- P&L before costs
                net_pnl DECIMAL(15,4), -- P&L after costs
                pnl_percent DECIMAL(8,4), -- Return percentage
                commission DECIMAL(8,4), -- Commission paid
                slippage DECIMAL(8,4), -- Slippage cost
                total_costs DECIMAL(8,4), -- Total transaction costs
                holding_days INTEGER, -- Days held
                portfolio_weight DECIMAL(6,4), -- Position size as % of portfolio
                position_value DECIMAL(15,4), -- Total position value
                is_open INTEGER DEFAULT 0, -- Still open position (0/1 for SQLite boolean)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_backtest_trades_result ON backtest_trades(result_id);
            CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol ON backtest_trades(symbol, entry_date);
            CREATE INDEX IF NOT EXISTS idx_backtest_trades_performance ON backtest_trades(pnl_percent DESC, holding_days);
            CREATE INDEX IF NOT EXISTS idx_backtest_trades_dates ON backtest_trades(entry_date, exit_date);
            """,
            
            # Daily Portfolio Values
            "backtest_portfolio_values": """
            CREATE TABLE IF NOT EXISTS backtest_portfolio_values (
                value_id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER REFERENCES backtest_results(result_id) ON DELETE CASCADE,
                date DATE NOT NULL,
                portfolio_value DECIMAL(15,4) NOT NULL, -- Total portfolio value
                cash_balance DECIMAL(15,4) NOT NULL, -- Available cash
                invested_amount DECIMAL(15,4) NOT NULL, -- Amount invested in positions
                daily_return DECIMAL(8,6), -- Daily return percentage
                cumulative_return DECIMAL(10,6), -- Cumulative return since start
                drawdown DECIMAL(8,4), -- Current drawdown from peak
                active_positions INTEGER, -- Number of open positions
                portfolio_beta DECIMAL(6,4), -- Rolling beta vs benchmark
                portfolio_volatility DECIMAL(8,4), -- Rolling volatility
                benchmark_value DECIMAL(15,4), -- Benchmark value for comparison
                benchmark_return DECIMAL(8,6), -- Benchmark daily return
                excess_return DECIMAL(8,6), -- Portfolio - Benchmark return
                vix_level DECIMAL(5,2), -- VIX level on this date
                market_regime VARCHAR(50), -- Market regime classification
                sector_exposures TEXT, -- Sector allocation percentages (JSON)
                top_holdings TEXT, -- Top 10 holdings with weights (JSON)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(result_id, date)
            );
            
            CREATE INDEX IF NOT EXISTS idx_portfolio_values_result_date ON backtest_portfolio_values(result_id, date);
            CREATE INDEX IF NOT EXISTS idx_portfolio_values_performance ON backtest_portfolio_values(cumulative_return DESC);
            """,
            
            # Benchmark Performance Data
            "benchmark_performance": """
            CREATE TABLE IF NOT EXISTS benchmark_performance (
                benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(10) NOT NULL, -- SPY, QQQ, IWM, RSP, VTI
                date DATE NOT NULL,
                open_price DECIMAL(12,4) NOT NULL,
                high_price DECIMAL(12,4) NOT NULL,
                low_price DECIMAL(12,4) NOT NULL,
                close_price DECIMAL(12,4) NOT NULL,
                adj_close_price DECIMAL(12,4) NOT NULL,
                volume INTEGER,
                daily_return DECIMAL(8,6), -- Daily return percentage
                cumulative_return DECIMAL(10,6), -- Cumulative return since start
                volatility_20d DECIMAL(8,4), -- 20-day rolling volatility
                max_drawdown DECIMAL(8,4), -- Maximum drawdown to date
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(symbol, date)
            );
            
            CREATE INDEX IF NOT EXISTS idx_benchmark_symbol_date ON benchmark_performance(symbol, date);
            CREATE INDEX IF NOT EXISTS idx_benchmark_date ON benchmark_performance(date);
            """,
            
            # Signal Performance Analysis
            "signal_performance_analysis": """
            CREATE TABLE IF NOT EXISTS signal_performance_analysis (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER REFERENCES backtest_results(result_id) ON DELETE CASCADE,
                signal_component VARCHAR(50) NOT NULL, -- 'RSI', 'MACD', 'Volume', 'Bollinger', etc.
                regime_type VARCHAR(20), -- Market regime
                timeframe VARCHAR(20), -- '1D', '5D', '1M', etc.
                total_trades INTEGER, -- Number of trades using this signal
                win_rate DECIMAL(5,4), -- Win rate for this component
                avg_return DECIMAL(8,4), -- Average return per trade
                volatility DECIMAL(8,4), -- Return volatility
                sharpe_ratio DECIMAL(8,4), -- Risk-adjusted performance
                max_drawdown DECIMAL(8,4), -- Max drawdown
                profit_factor DECIMAL(8,4), -- Profit factor
                success_rate DECIMAL(5,4), -- Percentage above threshold
                contribution_to_alpha DECIMAL(8,4), -- Alpha contribution
                signal_strength_avg DECIMAL(5,3), -- Average signal strength
                false_positive_rate DECIMAL(5,4), -- Rate of false signals
                signal_decay_days DECIMAL(6,2), -- Days until signal loses effectiveness
                correlation_with_returns DECIMAL(6,4), -- Correlation with future returns
                information_coefficient DECIMAL(6,4), -- Rank correlation with returns
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_signal_analysis_result ON signal_performance_analysis(result_id);
            CREATE INDEX IF NOT EXISTS idx_signal_analysis_component ON signal_performance_analysis(signal_component, regime_type);
            """,
            
            # Regime-Based Performance
            "regime_performance": """
            CREATE TABLE IF NOT EXISTS regime_performance (
                regime_perf_id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER REFERENCES backtest_results(result_id) ON DELETE CASCADE,
                regime_id INTEGER REFERENCES market_regimes(regime_id),
                regime_name VARCHAR(50) NOT NULL,
                regime_start_date DATE NOT NULL,
                regime_end_date DATE,
                strategy_return DECIMAL(10,6), -- Strategy return during regime
                benchmark_return DECIMAL(10,6), -- Benchmark return during regime
                excess_return DECIMAL(10,6), -- Outperformance
                volatility DECIMAL(8,4), -- Strategy volatility during regime
                sharpe_ratio DECIMAL(8,4), -- Regime-specific Sharpe
                max_drawdown DECIMAL(8,4), -- Max drawdown during regime
                win_rate DECIMAL(5,4), -- Win rate during regime
                avg_position_size DECIMAL(6,4), -- Average position sizing
                total_trades INTEGER, -- Number of trades during regime
                profitable_trades INTEGER, -- Profitable trades during regime
                avg_holding_days DECIMAL(6,2), -- Average holding period
                turnover_rate DECIMAL(8,4), -- Portfolio turnover
                beta DECIMAL(6,4), -- Beta during this regime
                correlation DECIMAL(6,4), -- Correlation with benchmark
                up_capture DECIMAL(6,4), -- Upside capture ratio
                down_capture DECIMAL(6,4), -- Downside capture ratio
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_regime_performance_result ON regime_performance(result_id);
            CREATE INDEX IF NOT EXISTS idx_regime_performance_regime ON regime_performance(regime_id);
            """,
            
            # Benchmark Reference Data
            "benchmark_reference": """
            CREATE TABLE IF NOT EXISTS benchmark_reference (
                ref_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol VARCHAR(10) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                asset_class VARCHAR(50),
                market_cap_focus VARCHAR(50),
                weighting VARCHAR(50),
                inception_date DATE,
                expense_ratio DECIMAL(6,4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        }
        
        logger.info("Creating backtesting database tables in SQLite...")
        
        with self.get_connection() as conn:
            for table_name, sql in tables_sql.items():
                try:
                    # Split and execute each statement separately
                    statements = sql.strip().split(';')
                    for statement in statements:
                        if statement.strip():
                            conn.execute(statement)
                    
                    conn.commit()
                    logger.info(f"Created/updated table: {table_name}")
                except Exception as e:
                    logger.error(f"Failed to create table {table_name}: {str(e)}")
                    raise
        
        logger.info("All backtesting tables created successfully")
    
    def populate_market_regimes(self):
        """Populate market regime data for 2019-2024"""
        
        market_regimes_data = [
            {
                'regime_name': 'Pre-COVID Bull Market',
                'start_date': '2019-01-01',
                'end_date': '2020-02-19',
                'regime_type': 'bull',
                'volatility_level': 'low',
                'vix_range': '12-20',
                'avg_vix': 16.5,
                'sp500_return': 0.285, # ~28.5% during this period
                'description': 'Strong bull market with low volatility, driven by economic growth and low interest rates'
            },
            {
                'regime_name': 'COVID Crash',
                'start_date': '2020-02-20',
                'end_date': '2020-03-23',
                'regime_type': 'crash',
                'volatility_level': 'extreme',
                'vix_range': '30-82',
                'avg_vix': 57.0,
                'sp500_return': -0.338, # -33.8% crash
                'description': 'Fastest bear market in history due to COVID-19 pandemic and economic shutdown'
            },
            {
                'regime_name': 'COVID Recovery Bull',
                'start_date': '2020-03-24',
                'end_date': '2020-11-30',
                'regime_type': 'recovery',
                'volatility_level': 'high',
                'vix_range': '20-40',
                'avg_vix': 28.5,
                'sp500_return': 0.675, # 67.5% recovery rally
                'description': 'Fastest recovery in market history, driven by massive fiscal and monetary stimulus'
            },
            {
                'regime_name': 'Low Volatility Bull',
                'start_date': '2020-12-01',
                'end_date': '2021-12-31',
                'regime_type': 'bull',
                'volatility_level': 'low',
                'vix_range': '15-25',
                'avg_vix': 19.5,
                'sp500_return': 0.215, # 21.5% gain in 2021
                'description': 'Continued bull market with rotation from growth to value, meme stock phenomena'
            },
            {
                'regime_name': 'Inflation Bear Market',
                'start_date': '2022-01-01',
                'end_date': '2022-10-31',
                'regime_type': 'bear',
                'volatility_level': 'high',
                'vix_range': '20-35',
                'avg_vix': 28.0,
                'sp500_return': -0.251, # -25.1% decline
                'description': 'Bear market driven by inflation concerns, Fed tightening, and recession fears'
            },
            {
                'regime_name': 'AI/Tech Rally',
                'start_date': '2022-11-01',
                'end_date': '2024-07-31',
                'regime_type': 'bull',
                'volatility_level': 'medium',
                'vix_range': '16-30',
                'avg_vix': 22.0,
                'sp500_return': 0.445, # 44.5% rally
                'description': 'Bull market driven by AI enthusiasm, Magnificent Seven stocks, and soft landing expectations'
            },
            {
                'regime_name': 'Current Period',
                'start_date': '2024-08-01',
                'end_date': None,
                'regime_type': 'bull',
                'volatility_level': 'medium',
                'vix_range': '18-25',
                'avg_vix': 21.0,
                'sp500_return': 0.125, # Year-to-date performance
                'description': 'Ongoing bull market with rotation between sectors, election uncertainty'
            }
        ]
        
        logger.info("Populating market regime data...")
        
        with self.get_connection() as conn:
            # Check if data already exists
            existing_count = conn.execute("SELECT COUNT(*) as count FROM market_regimes").fetchone()['count']
            
            if existing_count > 0:
                logger.info(f"Market regimes already populated ({existing_count} records)")
                return
            
            # Insert data
            for regime in market_regimes_data:
                conn.execute("""
                    INSERT INTO market_regimes 
                    (regime_name, start_date, end_date, regime_type, volatility_level, 
                     vix_range, avg_vix, sp500_return, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    regime['regime_name'], regime['start_date'], regime['end_date'],
                    regime['regime_type'], regime['volatility_level'], regime['vix_range'],
                    regime['avg_vix'], regime['sp500_return'], regime['description']
                ))
            
            conn.commit()
            logger.info(f"Successfully populated {len(market_regimes_data)} market regimes")
    
    def create_benchmark_symbols_reference(self):
        """Create reference data for benchmark symbols"""
        
        benchmark_symbols = [
            {
                'symbol': 'SPY',
                'name': 'SPDR S&P 500 ETF Trust',
                'description': 'Large-cap US market benchmark',
                'asset_class': 'US Equity',
                'market_cap_focus': 'Large Cap',
                'weighting': 'Market Cap Weighted',
                'inception_date': '1993-01-22',
                'expense_ratio': 0.0945
            },
            {
                'symbol': 'QQQ',
                'name': 'Invesco QQQ Trust',
                'description': 'NASDAQ 100 technology-focused benchmark',
                'asset_class': 'US Equity',
                'market_cap_focus': 'Large Cap Tech',
                'weighting': 'Market Cap Weighted',
                'inception_date': '1999-03-10',
                'expense_ratio': 0.20
            },
            {
                'symbol': 'IWM',
                'name': 'iShares Russell 2000 ETF',
                'description': 'Small-cap US market benchmark',
                'asset_class': 'US Equity',
                'market_cap_focus': 'Small Cap',
                'weighting': 'Market Cap Weighted',
                'inception_date': '2000-05-22',
                'expense_ratio': 0.19
            },
            {
                'symbol': 'RSP',
                'name': 'Invesco S&P 500 Equal Weight ETF',
                'description': 'Equal-weighted S&P 500 benchmark',
                'asset_class': 'US Equity',
                'market_cap_focus': 'Large Cap',
                'weighting': 'Equal Weighted',
                'inception_date': '2003-04-24',
                'expense_ratio': 0.20
            },
            {
                'symbol': 'VTI',
                'name': 'Vanguard Total Stock Market ETF',
                'description': 'Total US stock market benchmark',
                'asset_class': 'US Equity',
                'market_cap_focus': 'All Cap',
                'weighting': 'Market Cap Weighted',
                'inception_date': '2001-05-24',
                'expense_ratio': 0.03
            }
        ]
        
        logger.info("Creating benchmark reference data...")
        
        with self.get_connection() as conn:
            # Check if data already exists
            existing_count = conn.execute("SELECT COUNT(*) as count FROM benchmark_reference").fetchone()['count']
            
            if existing_count > 0:
                logger.info(f"Benchmark reference already populated ({existing_count} records)")
                return
            
            # Insert data
            for benchmark in benchmark_symbols:
                conn.execute("""
                    INSERT INTO benchmark_reference 
                    (symbol, name, description, asset_class, market_cap_focus, 
                     weighting, inception_date, expense_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    benchmark['symbol'], benchmark['name'], benchmark['description'],
                    benchmark['asset_class'], benchmark['market_cap_focus'], benchmark['weighting'],
                    benchmark['inception_date'], benchmark['expense_ratio']
                ))
            
            conn.commit()
            logger.info(f"Successfully populated {len(benchmark_symbols)} benchmark symbols")
    
    def initialize_backtesting_schema(self):
        """Initialize complete backtesting schema"""
        logger.info("Initializing comprehensive backtesting schema for SQLite...")
        
        try:
            # Create all tables
            self.create_backtesting_tables()
            
            # Populate reference data
            self.populate_market_regimes()
            self.create_benchmark_symbols_reference()
            
            # Log schema information
            schema_info = self.get_schema_info()
            logger.info("Backtesting schema initialized successfully", extra=schema_info)
            
        except Exception as e:
            logger.error(f"Failed to initialize backtesting schema: {str(e)}")
            raise
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the backtesting schema"""
        
        try:
            with self.get_connection() as conn:
                # Get all tables
                tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%backtest%' OR name IN ('market_regimes', 'benchmark_reference', 'benchmark_performance')"
                tables = conn.execute(tables_query).fetchall()
                
                # Get row counts
                table_counts = {}
                for table in tables:
                    table_name = table['name']
                    count_result = conn.execute(f"SELECT COUNT(*) as count FROM {table_name}").fetchone()
                    table_counts[table_name] = count_result['count']
                
                return {
                    'schema_version': self.schema_version,
                    'db_path': self.db_path,
                    'tables_created': len(tables),
                    'table_names': [table['name'] for table in tables],
                    'table_counts': table_counts,
                    'initialization_date': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get schema info: {str(e)}")
            return {
                'schema_version': self.schema_version,
                'db_path': self.db_path,
                'error': str(e),
                'initialization_date': datetime.utcnow().isoformat()
            }

# Global SQLite schema manager
sqlite_backtesting_schema = SQLiteBacktestingSchema()

# Convenience function for initialization
def initialize_sqlite_backtesting_database():
    """Initialize the SQLite backtesting database schema"""
    return sqlite_backtesting_schema.initialize_backtesting_schema()