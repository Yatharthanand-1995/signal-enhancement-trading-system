#!/usr/bin/env python3
"""
Full 100 Stock Market Data Pipeline
Downloads real market data for top 100 US stocks by market cap
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class Full100StockPipeline:
    """Download and process real market data for top 100 US stocks"""
    
    def __init__(self):
        self.data_dir = 'data/full_market'
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Top 100 US stocks by market cap (as of 2024)
        self.top_100_stocks = [
            # Mega caps (>$1T)
            'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA',
            
            # Large caps ($100B - $1T)
            'BRK-B', 'LLY', 'AVGO', 'JPM', 'UNH', 'XOM', 'ORCL', 'MA', 'HD', 'PG',
            'JNJ', 'COST', 'ABBV', 'NFLX', 'BAC', 'CRM', 'CVX', 'MRK', 'ADBE', 'AMD',
            'KO', 'WMT', 'ACN', 'LIN', 'PEP', 'TMO', 'CSCO', 'ABT', 'INTC', 'VZ',
            'TMUS', 'INTU', 'TXN', 'DIS', 'WFC', 'SPGI', 'LOW', 'GE', 'QCOM', 'CAT',
            'RTX', 'IBM', 'UBER', 'AMGN', 'PFE', 'AMAT', 'HON', 'SYK', 'TJX', 'BSX',
            
            # Mid-large caps ($50B - $100B)
            'PANW', 'NEE', 'VRTX', 'ADI', 'MDT', 'AMT', 'AXP', 'LRCX', 'ISRG', 'BA',
            'GILD', 'DE', 'MU', 'C', 'NOW', 'PM', 'BLK', 'GS', 'ELV', 'BKNG',
            'REGN', 'PLD', 'CVS', 'SLB', 'SCHW', 'MMC', 'CB', 'SO', 'FI', 'MO',
            'KLAC', 'ZTS', 'PYPL', 'ICE', 'DUK', 'PGR', 'AON', 'CL', 'ITW', 'BMY',
            'CME', 'USB', 'EQIX', 'APH', 'CSX', 'MSI', 'EOG', 'WM', 'MAR', 'FCX'
        ]
        
        self.period_years = 4
        print(f"Initialized pipeline for {len(self.top_100_stocks)} stocks")
    
    def download_market_data(self):
        """Download real market data for all top 100 stocks"""
        
        print(f"🔄 DOWNLOADING MARKET DATA FOR TOP 100 STOCKS")
        print("=" * 60)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.period_years * 365)
        
        all_data = []
        successful_downloads = 0
        failed_symbols = []
        
        for i, symbol in enumerate(self.top_100_stocks):
            try:
                print(f"📥 [{i+1:3d}/100] Downloading {symbol}...")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if len(data) >= 500:  # Require minimum data
                    data.reset_index(inplace=True)
                    data['symbol'] = symbol
                    data.columns = data.columns.str.lower()
                    data['date'] = pd.to_datetime(data['date'])
                    
                    # Save individual file
                    individual_path = os.path.join(self.data_dir, f'{symbol}_daily.csv')
                    data.to_csv(individual_path, index=False)
                    
                    all_data.append(data)
                    successful_downloads += 1
                    
                    print(f"     ✅ {len(data):,} records")
                else:
                    failed_symbols.append(symbol)
                    print(f"     ❌ Insufficient data: {len(data)} records")
                    
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"     ❌ Error: {str(e)}")
        
        print(f"\n📊 DOWNLOAD SUMMARY")
        print("-" * 30)
        print(f"Successful: {successful_downloads}/100 ({successful_downloads/100:.1%})")
        print(f"Failed: {len(failed_symbols)} symbols")
        
        if failed_symbols:
            print(f"Failed symbols: {', '.join(failed_symbols[:10])}")
            if len(failed_symbols) > 10:
                print(f"... and {len(failed_symbols) - 10} more")
        
        if all_data:
            # Consolidate all data
            consolidated_data = pd.concat(all_data, ignore_index=True)
            consolidated_path = os.path.join(self.data_dir, 'consolidated_market_data.csv')
            consolidated_data.to_csv(consolidated_path, index=False)
            
            print(f"\n✅ CONSOLIDATED DATA")
            print(f"Total records: {len(consolidated_data):,}")
            print(f"Symbols: {consolidated_data['symbol'].nunique()}")
            print(f"Date range: {consolidated_data['date'].min().date()} to {consolidated_data['date'].max().date()}")
            
            return consolidated_data, successful_downloads >= 80  # Success if 80+ stocks
        else:
            return None, False
    
    def create_train_validation_split(self, data):
        """Create proper temporal train/validation/test splits"""
        
        print(f"\n📈 CREATING TRAIN/VALIDATION/TEST SPLITS")
        print("-" * 45)
        
        # Sort by date
        data = data.sort_values(['symbol', 'date'])
        
        # Calculate split dates (70% train, 15% validation, 15% test)
        min_date = data['date'].min()
        max_date = data['date'].max()
        total_days = (max_date - min_date).days
        
        train_end = min_date + timedelta(days=int(total_days * 0.70))
        val_end = min_date + timedelta(days=int(total_days * 0.85))
        
        # Split data
        train_data = data[data['date'] <= train_end]
        val_data = data[(data['date'] > train_end) & (data['date'] <= val_end)]
        test_data = data[data['date'] > val_end]
        
        # Save splits
        train_path = os.path.join(self.data_dir, 'train_data.csv')
        val_path = os.path.join(self.data_dir, 'validation_data.csv')
        test_path = os.path.join(self.data_dir, 'test_data.csv')
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"Train: {len(train_data):,} records ({len(train_data['symbol'].unique())} symbols)")
        print(f"Validation: {len(val_data):,} records ({len(val_data['symbol'].unique())} symbols)")
        print(f"Test: {len(test_data):,} records ({len(test_data['symbol'].unique())} symbols)")
        
        print(f"\nDate ranges:")
        print(f"Train: {train_data['date'].min().date()} to {train_data['date'].max().date()}")
        print(f"Validation: {val_data['date'].min().date()} to {val_data['date'].max().date()}")
        print(f"Test: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
        
        return train_data, val_data, test_data
    
    def add_technical_indicators(self, data):
        """Add technical indicators to the dataset"""
        
        print(f"\n📊 ADDING TECHNICAL INDICATORS")
        print("-" * 35)
        
        enhanced_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            if len(symbol_data) >= 100:
                # Price indicators
                symbol_data['sma_10'] = symbol_data['close'].rolling(10).mean()
                symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
                symbol_data['sma_50'] = symbol_data['close'].rolling(50).mean()
                
                # MACD
                exp1 = symbol_data['close'].ewm(span=12).mean()
                exp2 = symbol_data['close'].ewm(span=26).mean()
                symbol_data['macd'] = exp1 - exp2
                symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
                
                # RSI
                delta = symbol_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                symbol_data['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                rolling_mean = symbol_data['close'].rolling(20).mean()
                rolling_std = symbol_data['close'].rolling(20).std()
                symbol_data['bb_upper'] = rolling_mean + (rolling_std * 2)
                symbol_data['bb_lower'] = rolling_mean - (rolling_std * 2)
                
                # Volume indicators
                symbol_data['volume_sma'] = symbol_data['volume'].rolling(20).mean()
                
                enhanced_data.append(symbol_data)
        
        if enhanced_data:
            enhanced_df = pd.concat(enhanced_data, ignore_index=True)
            
            # Save enhanced data
            enhanced_path = os.path.join(self.data_dir, 'enhanced_market_data.csv')
            enhanced_df.to_csv(enhanced_path, index=False)
            
            print(f"✅ Enhanced data with indicators: {len(enhanced_df):,} records")
            return enhanced_df
        
        return data

def main():
    """Run the full 100 stock pipeline"""
    
    print("🚀 FULL 100 STOCK MARKET DATA PIPELINE")
    print("=" * 70)
    print("Downloading real market data for top 100 US stocks")
    print()
    
    pipeline = Full100StockPipeline()
    
    # Download market data
    consolidated_data, success = pipeline.download_market_data()
    
    if not success:
        print("❌ Pipeline failed - insufficient data downloaded")
        return False
    
    # Create splits
    train_data, val_data, test_data = pipeline.create_train_validation_split(consolidated_data)
    
    # Add technical indicators
    enhanced_data = pipeline.add_technical_indicators(consolidated_data)
    
    print(f"\n🎉 PIPELINE COMPLETE")
    print("=" * 30)
    print(f"✅ Downloaded data for {consolidated_data['symbol'].nunique()} stocks")
    print(f"✅ Total records: {len(consolidated_data):,}")
    print(f"✅ Enhanced with technical indicators")
    print(f"✅ Created train/validation/test splits")
    
    print(f"\n📁 DATA FILES CREATED:")
    print(f"📂 {pipeline.data_dir}/")
    print(f"  ├── consolidated_market_data.csv ({len(consolidated_data):,} records)")
    print(f"  ├── enhanced_market_data.csv ({len(enhanced_data):,} records)")
    print(f"  ├── train_data.csv ({len(train_data):,} records)")
    print(f"  ├── validation_data.csv ({len(val_data):,} records)")
    print(f"  └── test_data.csv ({len(test_data):,} records)")
    
    print(f"\n🔬 READY FOR ML INTEGRATION")
    print("Update ML system to use data/full_market/ instead of data/real_market/")
    
    return True

if __name__ == "__main__":
    main()