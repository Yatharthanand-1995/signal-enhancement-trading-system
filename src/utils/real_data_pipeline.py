#!/usr/bin/env python3
"""
Real Market Data Pipeline - Phase 1, Day 1
Replace synthetic data with actual market data for ML validation
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class RealMarketDataPipeline:
    """Pipeline for downloading, validating, and preparing real market data"""
    
    def __init__(self):
        self.data_dir = 'data/real_market'
        os.makedirs(self.data_dir, exist_ok=True)
        self.symbols = [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            # Other Sectors
            'JPM', 'JNJ', 'PG', 'DIS', 'V',
            # ETFs for broader market context
            'SPY', 'QQQ', 'IWM', 'VIX'
        ]
        self.period_years = 4  # 4 years for comprehensive testing
        
    def download_market_data(self):
        """Download real market data for all symbols"""
        
        print("📡 DOWNLOADING REAL MARKET DATA")
        print("=" * 50)
        print(f"Symbols: {len(self.symbols)} ({', '.join(self.symbols[:5])}...)")
        print(f"Period: {self.period_years} years")
        print(f"Data directory: {self.data_dir}")
        print()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.period_years * 365)
        
        successful_downloads = 0
        failed_downloads = []
        market_data = {}
        
        for symbol in self.symbols:
            try:
                print(f"📊 Downloading {symbol}...", end=" ")
                
                # Download data
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if len(data) < 500:  # Minimum viable dataset
                    raise ValueError(f"Insufficient data: {len(data)} days")
                
                # Clean and prepare data
                data = data.reset_index()
                data.columns = [col.lower() for col in data.columns]
                data['symbol'] = symbol
                
                # Basic validation
                if data['close'].isnull().sum() > len(data) * 0.05:  # >5% missing
                    raise ValueError("Too many missing values")
                
                # Forward fill minor gaps
                data = data.fillna(method='ffill').fillna(method='bfill')
                
                # Save to file
                file_path = os.path.join(self.data_dir, f"{symbol}_daily.csv")
                data.to_csv(file_path, index=False)
                
                market_data[symbol] = data
                successful_downloads += 1
                
                print(f"✅ {len(data)} days ({data['date'].min().date()} to {data['date'].max().date()})")
                
            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                failed_downloads.append(symbol)
        
        print(f"\n📋 DOWNLOAD SUMMARY")
        print("-" * 30)
        print(f"Successful: {successful_downloads}/{len(self.symbols)}")
        print(f"Failed: {len(failed_downloads)} {failed_downloads if failed_downloads else ''}")
        
        if successful_downloads < 5:
            raise RuntimeError("Insufficient data downloads - need at least 5 symbols")
        
        return market_data
    
    def validate_data_quality(self, market_data):
        """Validate quality of downloaded market data"""
        
        print(f"\n🔍 DATA QUALITY VALIDATION")
        print("=" * 40)
        
        quality_report = {}
        
        for symbol, data in market_data.items():
            report = {
                'days': len(data),
                'missing_values': data.isnull().sum().sum(),
                'zero_volume_days': (data['volume'] == 0).sum(),
                'price_gaps': self._detect_price_gaps(data),
                'start_date': data['date'].min(),
                'end_date': data['date'].max(),
                'avg_volume': data['volume'].mean(),
                'price_range': f"${data['low'].min():.2f} - ${data['high'].max():.2f}"
            }
            
            quality_score = self._calculate_quality_score(report)
            report['quality_score'] = quality_score
            quality_report[symbol] = report
            
            status = "✅ GOOD" if quality_score > 85 else "⚠️ FAIR" if quality_score > 70 else "❌ POOR"
            print(f"{symbol:<6}: {report['days']:>4} days, Quality: {quality_score:>3.0f}% {status}")
        
        # Overall quality assessment
        avg_quality = np.mean([r['quality_score'] for r in quality_report.values()])
        high_quality_symbols = sum(1 for r in quality_report.values() if r['quality_score'] > 85)
        
        print(f"\n📊 OVERALL DATA QUALITY")
        print(f"Average quality score: {avg_quality:.1f}%")
        print(f"High quality symbols: {high_quality_symbols}/{len(quality_report)}")
        
        if avg_quality < 80:
            print("⚠️ WARNING: Overall data quality below 80%")
        else:
            print("✅ Data quality acceptable for ML training")
        
        return quality_report
    
    def _detect_price_gaps(self, data):
        """Detect significant price gaps in the data"""
        if len(data) < 2:
            return 0
        
        # Calculate overnight gaps (open vs previous close)
        prev_close = data['close'].shift(1)
        gaps = abs(data['open'] - prev_close) / prev_close
        
        # Count gaps > 5%
        significant_gaps = (gaps > 0.05).sum()
        return significant_gaps
    
    def _calculate_quality_score(self, report):
        """Calculate data quality score (0-100)"""
        score = 100
        
        # Penalize missing values
        missing_pct = (report['missing_values'] / (report['days'] * 6)) * 100  # 6 columns (OHLCV + Date)
        score -= missing_pct * 10
        
        # Penalize zero volume days
        zero_vol_pct = (report['zero_volume_days'] / report['days']) * 100
        score -= zero_vol_pct * 5
        
        # Penalize excessive price gaps
        gap_pct = (report['price_gaps'] / report['days']) * 100
        score -= min(gap_pct * 2, 20)  # Cap at 20 point penalty
        
        # Penalize insufficient data
        if report['days'] < 800:
            score -= (800 - report['days']) / 10
        
        return max(0, min(100, score))
    
    def create_consolidated_dataset(self, market_data):
        """Create consolidated dataset for ML training"""
        
        print(f"\n🔧 CREATING CONSOLIDATED DATASET")
        print("-" * 40)
        
        # Combine all symbols into single dataset
        all_data = []
        for symbol, data in market_data.items():
            data_copy = data.copy()
            data_copy['symbol'] = symbol
            all_data.append(data_copy)
        
        consolidated = pd.concat(all_data, ignore_index=True)
        consolidated = consolidated.sort_values(['symbol', 'date'])
        
        # Add basic derived features
        print("Adding technical indicators...")
        consolidated = self._add_technical_indicators(consolidated)
        
        # Create train/validation/test splits (temporal)
        print("Creating temporal data splits...")
        splits = self._create_temporal_splits(consolidated)
        
        # Save datasets
        consolidated_path = os.path.join(self.data_dir, 'consolidated_market_data.csv')
        consolidated.to_csv(consolidated_path, index=False)
        
        for split_name, split_data in splits.items():
            split_path = os.path.join(self.data_dir, f'{split_name}_data.csv')
            split_data.to_csv(split_path, index=False)
        
        print(f"\n📁 DATASET CREATION COMPLETE")
        print(f"Total records: {len(consolidated):,}")
        print(f"Date range: {consolidated['date'].min().date()} to {consolidated['date'].max().date()}")
        print(f"Symbols: {consolidated['symbol'].nunique()}")
        print(f"Features: {len(consolidated.columns)} columns")
        
        # Split summary
        for split_name, split_data in splits.items():
            date_range = f"{split_data['date'].min().date()} to {split_data['date'].max().date()}"
            print(f"  {split_name.capitalize()}: {len(split_data):,} records ({date_range})")
        
        return consolidated, splits
    
    def _add_technical_indicators(self, data):
        """Add essential technical indicators"""
        
        result_data = []
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy().sort_values('date')
            
            # Moving averages
            symbol_data['sma_5'] = symbol_data['close'].rolling(5).mean()
            symbol_data['sma_10'] = symbol_data['close'].rolling(10).mean()
            symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
            symbol_data['sma_50'] = symbol_data['close'].rolling(50).mean()
            
            # RSI
            symbol_data['rsi_14'] = self._calculate_rsi(symbol_data['close'], 14)
            
            # MACD
            ema_12 = symbol_data['close'].ewm(span=12).mean()
            ema_26 = symbol_data['close'].ewm(span=26).mean()
            symbol_data['macd'] = ema_12 - ema_26
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
            symbol_data['macd_histogram'] = symbol_data['macd'] - symbol_data['macd_signal']
            
            # Bollinger Bands
            bb_sma = symbol_data['close'].rolling(20).mean()
            bb_std = symbol_data['close'].rolling(20).std()
            symbol_data['bb_upper'] = bb_sma + (bb_std * 2)
            symbol_data['bb_lower'] = bb_sma - (bb_std * 2)
            symbol_data['bb_position'] = (symbol_data['close'] - symbol_data['bb_lower']) / (symbol_data['bb_upper'] - symbol_data['bb_lower'])
            
            # ATR (Average True Range)
            symbol_data['atr_14'] = self._calculate_atr(symbol_data, 14)
            
            # Volume indicators
            symbol_data['volume_sma_20'] = symbol_data['volume'].rolling(20).mean()
            symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma_20']
            
            # Price changes and returns
            symbol_data['price_change'] = symbol_data['close'].pct_change(1)
            symbol_data['price_change_5d'] = symbol_data['close'].pct_change(5)
            symbol_data['price_change_20d'] = symbol_data['close'].pct_change(20)
            
            result_data.append(symbol_data)
        
        return pd.concat(result_data, ignore_index=True)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _create_temporal_splits(self, data):
        """Create proper temporal train/validation/test splits"""
        
        # Sort by date
        data = data.sort_values(['date'])
        
        # Calculate split points (60% train, 20% validation, 20% test)
        total_days = len(data['date'].unique())
        train_days = int(total_days * 0.6)
        val_days = int(total_days * 0.2)
        
        unique_dates = sorted(data['date'].unique())
        train_end_date = unique_dates[train_days - 1]
        val_end_date = unique_dates[train_days + val_days - 1]
        
        # Create splits
        train_data = data[data['date'] <= train_end_date]
        val_data = data[(data['date'] > train_end_date) & (data['date'] <= val_end_date)]
        test_data = data[data['date'] > val_end_date]
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }

def main():
    """Main execution function"""
    
    print("🚀 REAL MARKET DATA PIPELINE - PHASE 1")
    print("=" * 60)
    print("Goal: Replace synthetic data with real market data")
    print("Approach: Download, validate, and prepare 4 years of market data")
    print()
    
    try:
        # Initialize pipeline
        pipeline = RealMarketDataPipeline()
        
        # Step 1: Download market data
        market_data = pipeline.download_market_data()
        
        # Step 2: Validate data quality
        quality_report = pipeline.validate_data_quality(market_data)
        
        # Step 3: Create consolidated dataset
        consolidated, splits = pipeline.create_consolidated_dataset(market_data)
        
        print(f"\n🎯 PHASE 1 COMPLETION STATUS")
        print("=" * 40)
        
        # Success criteria check
        successful_symbols = len(market_data)
        avg_quality = np.mean([r['quality_score'] for r in quality_report.values()])
        total_records = len(consolidated)
        
        criteria_met = 0
        total_criteria = 4
        
        print(f"✅ Real data source: {'PASS' if successful_symbols >= 5 else 'FAIL'}")
        if successful_symbols >= 5: criteria_met += 1
        
        print(f"✅ Data quality: {'PASS' if avg_quality > 80 else 'FAIL'} ({avg_quality:.1f}%)")
        if avg_quality > 80: criteria_met += 1
        
        print(f"✅ Dataset size: {'PASS' if total_records > 10000 else 'FAIL'} ({total_records:,} records)")
        if total_records > 10000: criteria_met += 1
        
        print(f"✅ Time period: {'PASS' if len(consolidated['date'].unique()) > 1000 else 'FAIL'} ({len(consolidated['date'].unique())} days)")
        if len(consolidated['date'].unique()) > 1000: criteria_met += 1
        
        print(f"\n📊 SUCCESS RATE: {criteria_met}/{total_criteria} criteria met")
        
        if criteria_met >= 3:
            print(f"\n🎉 PHASE 1 SUCCESS!")
            print("Real market data pipeline operational")
            print("Ready to proceed to feature validation")
            
            print(f"\n📁 Data Files Created:")
            print(f"  • consolidated_market_data.csv ({len(consolidated):,} records)")
            print(f"  • train_data.csv ({len(splits['train']):,} records)")
            print(f"  • validation_data.csv ({len(splits['validation']):,} records)")
            print(f"  • test_data.csv ({len(splits['test']):,} records)")
            
            return True
        else:
            print(f"\n⚠️ PHASE 1 PARTIAL SUCCESS")
            print("Some criteria not met - may need data source adjustments")
            return False
            
    except Exception as e:
        print(f"\n❌ PHASE 1 FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🚀 NEXT STEP: Feature correlation analysis on real data")
        print("Run: python validate_real_features.py")
    else:
        print(f"\n🔧 REQUIRED: Fix data pipeline issues before proceeding")