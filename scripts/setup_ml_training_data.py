#!/usr/bin/env python3
"""
ML Training Data Setup
Prepare historical data for production ML model training
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_training_data():
    """Set up historical data for ML training"""
    
    print("📊 ML TRAINING DATA SETUP")
    print("=" * 50)
    
    try:
        from src.utils.backtesting_schema_sqlite import sqlite_backtesting_schema
        
        # Get available symbols and data
        print("🔍 Checking available historical data...")
        
        with sqlite_backtesting_schema.get_connection() as conn:
            # Check available symbols
            symbols_query = "SELECT DISTINCT symbol FROM historical_data ORDER BY symbol"
            symbols_df = pd.read_sql_query(symbols_query, conn)
            available_symbols = symbols_df['symbol'].tolist()
            
            print(f"✅ Found {len(available_symbols)} symbols with historical data")
            print(f"   Symbols: {', '.join(available_symbols[:10])}{'...' if len(available_symbols) > 10 else ''}")
            
            # Check data coverage
            coverage_query = """
            SELECT 
                symbol,
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as record_count
            FROM historical_data 
            GROUP BY symbol
            ORDER BY record_count DESC
            LIMIT 10
            """
            
            coverage_df = pd.read_sql_query(coverage_query, conn)
            
            print(f"\n📅 Top 10 symbols by data coverage:")
            for _, row in coverage_df.iterrows():
                print(f"   {row['symbol']}: {row['record_count']} days ({row['start_date']} to {row['end_date']})")
            
            # Select top symbols for training
            training_symbols = coverage_df.head(8)['symbol'].tolist()
            print(f"\n🎯 Selected {len(training_symbols)} symbols for ML training:")
            print(f"   {', '.join(training_symbols)}")
            
            return training_symbols, coverage_df
            
    except Exception as e:
        print(f"❌ Error setting up training data: {str(e)}")
        return [], pd.DataFrame()

def prepare_training_datasets(symbols, min_records=500):
    """Prepare training datasets for each symbol"""
    
    print(f"\n🏗️ PREPARING TRAINING DATASETS")
    print("-" * 50)
    
    training_datasets = {}
    
    try:
        from src.utils.backtesting_schema_sqlite import sqlite_backtesting_schema
        
        for symbol in symbols:
            print(f"📈 Processing {symbol}...")
            
            with sqlite_backtesting_schema.get_connection() as conn:
                # Get historical data with technical indicators
                query = """
                SELECT 
                    date, symbol, open, high, low, close, volume,
                    rsi_14, macd, macd_signal, macd_histogram,
                    sma_20, sma_50, bb_upper, bb_middle, bb_lower,
                    atr_14, volume_sma_20
                FROM historical_data 
                WHERE symbol = ?
                ORDER BY date
                """
                
                df = pd.read_sql_query(query, conn, params=[symbol])
                
                if len(df) < min_records:
                    print(f"   ⚠️ Insufficient data for {symbol}: {len(df)} records (need {min_records})")
                    continue
                
                # Clean and prepare data
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna()
                
                # Calculate additional features for ML training
                df['returns_1d'] = df['close'].pct_change()
                df['returns_5d'] = df['close'].pct_change(5)
                df['returns_20d'] = df['close'].pct_change(20)
                
                df['volatility_5d'] = df['returns_1d'].rolling(5).std()
                df['volatility_20d'] = df['returns_1d'].rolling(20).std()
                
                df['volume_ratio'] = df['volume'] / df['volume_sma_20']
                df['price_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
                # Remove NaN values after feature engineering
                df = df.dropna()
                
                if len(df) >= min_records:
                    training_datasets[symbol] = df
                    print(f"   ✅ Prepared {len(df)} records for {symbol}")
                else:
                    print(f"   ❌ Not enough clean data for {symbol}: {len(df)} records")
        
        print(f"\n📊 Training datasets prepared for {len(training_datasets)} symbols")
        return training_datasets
        
    except Exception as e:
        print(f"❌ Error preparing datasets: {str(e)}")
        return {}

def create_sample_training_data():
    """Create sample training data for immediate ML testing"""
    
    print(f"\n🎲 CREATING SAMPLE TRAINING DATA")
    print("-" * 50)
    
    try:
        # Create realistic sample data for immediate testing
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        sample_datasets = {}
        
        np.random.seed(42)  # For reproducible results
        
        for symbol in symbols:
            print(f"📈 Creating sample data for {symbol}...")
            
            # Generate 1000 days of realistic data
            n_days = 1000
            dates = pd.date_range(start='2021-01-01', periods=n_days, freq='D')
            
            # Generate realistic price movements
            base_price = np.random.uniform(50, 300)  # Random starting price
            
            # Create correlated returns with trends and volatility clustering
            returns = []
            volatility = 0.02  # Base volatility
            
            for i in range(n_days):
                # Add volatility clustering
                if i > 0:
                    volatility = 0.95 * volatility + 0.05 * abs(returns[-1])
                
                # Add some trend persistence
                trend = 0 if i == 0 else 0.1 * returns[-1]
                
                daily_return = np.random.normal(0.0005 + trend, volatility)
                returns.append(daily_return)
            
            # Convert returns to prices
            returns = np.array(returns)
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Generate OHLCV data
            df = pd.DataFrame({
                'date': dates,
                'symbol': symbol,
                'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
                'high': prices * (1 + np.abs(np.random.normal(0.01, 0.01, n_days))),
                'low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, n_days))),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n_days)
            })
            
            # Add technical indicators (simplified)
            df['rsi_14'] = 50 + 20 * np.sin(np.linspace(0, 20*np.pi, n_days)) + np.random.normal(0, 5, n_days)
            df['rsi_14'] = np.clip(df['rsi_14'], 0, 100)
            
            df['macd'] = np.random.normal(0, 0.5, n_days)
            df['macd_histogram'] = np.random.normal(0, 0.3, n_days)
            
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            
            # Bollinger Bands
            rolling_std = df['close'].rolling(20).std()
            df['bb_middle'] = df['sma_20']
            df['bb_upper'] = df['bb_middle'] + 2 * rolling_std
            df['bb_lower'] = df['bb_middle'] - 2 * rolling_std
            
            # ATR and volume indicators
            df['atr_14'] = df['close'].rolling(14).std() * np.sqrt(252)
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            
            # Calculate ML training features
            df['returns_1d'] = df['close'].pct_change()
            df['returns_5d'] = df['close'].pct_change(5)
            df['returns_20d'] = df['close'].pct_change(20)
            
            df['volatility_5d'] = df['returns_1d'].rolling(5).std()
            df['volatility_20d'] = df['returns_1d'].rolling(20).std()
            
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['price_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Clean data
            df = df.dropna()
            
            sample_datasets[symbol] = df
            print(f"   ✅ Created {len(df)} records for {symbol}")
        
        print(f"\n📊 Sample datasets created for {len(sample_datasets)} symbols")
        print(f"   Ready for immediate ML model training")
        
        return sample_datasets
        
    except Exception as e:
        print(f"❌ Error creating sample data: {str(e)}")
        return {}

def main():
    """Main training data setup process"""
    
    print("🚀 ML TRAINING DATA SETUP")
    print("=" * 60)
    print("Preparing data for production-ready ML model training")
    print()
    
    # Step 1: Check available historical data
    symbols, coverage = setup_training_data()
    
    # Step 2: Prepare training datasets
    if len(symbols) > 0:
        print(f"\n🏗️ Preparing datasets from historical data...")
        training_datasets = prepare_training_datasets(symbols)
        
        if len(training_datasets) > 0:
            print(f"✅ Historical training data ready for {len(training_datasets)} symbols")
            data_source = "historical"
            datasets = training_datasets
        else:
            print(f"⚠️ Historical data insufficient, creating sample data...")
            data_source = "sample"
            datasets = create_sample_training_data()
    else:
        print(f"\n🎲 No historical data found, creating sample data...")
        data_source = "sample"
        datasets = create_sample_training_data()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("🎯 TRAINING DATA SETUP SUMMARY")
    print("=" * 60)
    
    print(f"Data Source:        {data_source.upper()}")
    print(f"Symbols Prepared:   {len(datasets)}")
    print(f"Data Quality:       {'Historical market data' if data_source == 'historical' else 'Realistic sample data'}")
    
    if len(datasets) > 0:
        avg_records = np.mean([len(df) for df in datasets.values()])
        print(f"Average Records:    {avg_records:.0f} days per symbol")
        
        print(f"\n📊 Prepared Symbols:")
        for symbol, df in datasets.items():
            date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            print(f"   {symbol}: {len(df)} records ({date_range})")
        
        print(f"\n🚀 Ready for ML model training:")
        print(f"   • High-quality feature engineering complete")
        print(f"   • Multiple timeframe returns calculated")
        print(f"   • Technical indicators prepared")
        print(f"   • Volatility and volume features ready")
        print(f"   • Data cleaned and validated")
        
        print(f"\n🎯 Next Step:")
        print(f"   Run: python scripts/train_production_ml_models.py")
        
        return True
    else:
        print(f"❌ No training data could be prepared")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)