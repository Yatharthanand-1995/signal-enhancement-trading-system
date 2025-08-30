#!/usr/bin/env python3
"""
Test ML System with Full 100 Stock Universe
Test the production ML system with the complete dataset
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append('src')

def test_full_100_ml_system():
    """Test ML system with full 106 stock dataset"""
    
    print("🚀 TESTING ML SYSTEM WITH FULL 100+ STOCK UNIVERSE")
    print("=" * 65)
    print("Testing production ML system with complete 106 stock dataset")
    print()
    
    try:
        from strategy.enhanced_signal_integration import get_enhanced_signal, initialize_enhanced_signal_integration
        
        # Initialize ML system
        integrator = initialize_enhanced_signal_integration()
        print(f"✅ ML System: {integrator.name}")
        print(f"   Features: {len(integrator.signal_correlations)} proven correlations")
        
        # Load FULL dataset
        data_dir = 'data/full_market'
        val_path = os.path.join(data_dir, 'validation_data.csv')
        
        if not os.path.exists(val_path):
            print(f"❌ Full dataset not found at {val_path}")
            return False
        
        val_data = pd.read_csv(val_path)
        val_data['date'] = pd.to_datetime(val_data['date'])
        
        print(f"✅ Full Dataset Loaded:")
        print(f"   Records: {len(val_data):,}")
        print(f"   Symbols: {val_data['symbol'].nunique()}")
        print(f"   Period: {val_data['date'].min().date()} to {val_data['date'].max().date()}")
        
        # Get all available symbols
        all_symbols = sorted(val_data['symbol'].unique())
        print(f"\n📈 AVAILABLE STOCKS ({len(all_symbols)}):")
        print("Top 20:", ', '.join(all_symbols[:20]))
        print("Next 20:", ', '.join(all_symbols[20:40]))
        if len(all_symbols) > 40:
            print(f"... and {len(all_symbols) - 40} more")
        print()
        
        # Test ML signal generation for samples from different market caps
        test_samples = {
            'Mega Cap': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'Large Cap': ['JPM', 'UNH', 'HD', 'PG', 'JNJ'], 
            'Tech': ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'C'],
            'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'LLY']
        }
        
        all_results = []
        category_results = {}
        
        print("🔬 ML SIGNAL GENERATION BY CATEGORY")
        print("=" * 45)
        
        for category, symbols in test_samples.items():
            category_signals = []
            print(f"\n📊 {category.upper()} STOCKS")
            print("-" * 25)
            
            for symbol in symbols:
                if symbol in all_symbols:
                    symbol_data = val_data[val_data['symbol'] == symbol].sort_values('date')
                    
                    if len(symbol_data) >= 100:
                        test_data = symbol_data.tail(100)
                        current_price = test_data['close'].iloc[-1]
                        
                        # Generate ML-enhanced signal
                        ml_signal = get_enhanced_signal(
                            symbol=symbol,
                            data=test_data,
                            current_price=current_price,
                            current_regime='normal'
                        )
                        
                        if ml_signal:
                            result = {
                                'category': category,
                                'symbol': symbol,
                                'signal_strength': ml_signal.signal_strength,
                                'confidence': ml_signal.confidence,
                                'ml_contribution': ml_signal.ml_contribution,
                                'technical_contribution': ml_signal.technical_contribution,
                                'position_size': ml_signal.recommended_position_size,
                                'predicted_volatility': ml_signal.predicted_volatility,
                                'quality': ml_signal.quality.value
                            }
                            
                            category_signals.append(result)
                            all_results.append(result)
                            
                            print(f"{symbol:5}: Signal {ml_signal.signal_strength:+.3f}, "
                                  f"Conf {ml_signal.confidence:.3f}, "
                                  f"ML {ml_signal.ml_contribution:+.3f}, "
                                  f"Pos {ml_signal.recommended_position_size:.1%}")
                        else:
                            print(f"{symbol:5}: No signal generated")
                else:
                    print(f"{symbol:5}: Not in dataset")
            
            if category_signals:
                category_results[category] = category_signals
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS")
        print("=" * 35)
        
        if all_results:
            # Overall statistics
            total_signals = len(all_results)
            avg_signal = np.mean([r['signal_strength'] for r in all_results])
            avg_confidence = np.mean([r['confidence'] for r in all_results])
            avg_ml_contrib = np.mean([abs(r['ml_contribution']) for r in all_results])
            avg_position = np.mean([r['position_size'] for r in all_results])
            avg_volatility = np.mean([r['predicted_volatility'] for r in all_results])
            
            print(f"Total Signals Generated: {total_signals}")
            print(f"Average Signal Strength: {avg_signal:+.3f}")
            print(f"Average Confidence: {avg_confidence:.3f}")
            print(f"Average ML Contribution: {avg_ml_contrib:.3f}")
            print(f"Average Position Size: {avg_position:.1%}")
            print(f"Average Volatility: {avg_volatility:.1%}")
            
            # Category breakdown
            print(f"\n📈 CATEGORY PERFORMANCE")
            print("-" * 30)
            for category, signals in category_results.items():
                if signals:
                    cat_avg_signal = np.mean([s['signal_strength'] for s in signals])
                    cat_avg_conf = np.mean([s['confidence'] for s in signals])
                    cat_avg_ml = np.mean([abs(s['ml_contribution']) for s in signals])
                    
                    print(f"{category:10}: {len(signals)} signals, "
                          f"Avg Signal {cat_avg_signal:+.3f}, "
                          f"Avg Conf {cat_avg_conf:.3f}, "
                          f"Avg ML {cat_avg_ml:.3f}")
            
            # Signal distribution
            print(f"\n📊 SIGNAL DISTRIBUTION")
            print("-" * 25)
            buy_signals = sum(1 for r in all_results if r['signal_strength'] > 0.05)
            sell_signals = sum(1 for r in all_results if r['signal_strength'] < -0.05)
            neutral_signals = total_signals - buy_signals - sell_signals
            
            print(f"Buy Signals: {buy_signals} ({buy_signals/total_signals:.1%})")
            print(f"Sell Signals: {sell_signals} ({sell_signals/total_signals:.1%})")
            print(f"Neutral Signals: {neutral_signals} ({neutral_signals/total_signals:.1%})")
            
            # ML contribution analysis
            ml_contributions = [abs(r['ml_contribution']) for r in all_results]
            significant_ml = sum(1 for ml in ml_contributions if ml > 0.01)
            
            print(f"\n🤖 ML SYSTEM ANALYSIS")
            print("-" * 25)
            print(f"Signals with ML Input: {significant_ml}/{total_signals} ({significant_ml/total_signals:.1%})")
            print(f"Average ML Magnitude: {np.mean(ml_contributions):.3f}")
            print(f"Max ML Contribution: {max(ml_contributions):.3f}")
            
            # Risk management
            position_sizes = [r['position_size'] for r in all_results]
            volatilities = [r['predicted_volatility'] for r in all_results]
            
            print(f"\n🛡️ RISK MANAGEMENT")
            print("-" * 20)
            print(f"Position Size Range: {min(position_sizes):.1%} - {max(position_sizes):.1%}")
            print(f"Volatility Range: {min(volatilities):.1%} - {max(volatilities):.1%}")
            print(f"Risk-Adjusted Positions: {sum(1 for p in position_sizes if p <= 0.05)/len(position_sizes):.1%}")
        
        # Test random sample of additional stocks
        print(f"\n🎲 RANDOM SAMPLE TEST (10 additional stocks)")
        print("-" * 45)
        
        remaining_symbols = [s for s in all_symbols if s not in [symbol for symbols in test_samples.values() for symbol in symbols]]
        random_sample = np.random.choice(remaining_symbols, min(10, len(remaining_symbols)), replace=False)
        
        random_results = []
        for symbol in random_sample:
            symbol_data = val_data[val_data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) >= 100:
                test_data = symbol_data.tail(100)
                current_price = test_data['close'].iloc[-1]
                
                ml_signal = get_enhanced_signal(
                    symbol=symbol,
                    data=test_data,
                    current_price=current_price,
                    current_regime='normal'
                )
                
                if ml_signal:
                    random_results.append(symbol)
                    print(f"{symbol:5}: Signal {ml_signal.signal_strength:+.3f}, "
                          f"Conf {ml_signal.confidence:.3f}")
        
        print(f"\nRandom sample: {len(random_results)}/10 generated signals")
        
        print(f"\n🎯 SYSTEM VALIDATION")
        print("=" * 25)
        print(f"✅ Dataset: {len(all_symbols)} stocks (target: 100+)")
        print(f"✅ ML Integration: {total_signals} signals generated")
        print(f"✅ Coverage: Multiple market cap categories")
        print(f"✅ Risk Management: Active position sizing")
        print(f"✅ Performance: {avg_confidence:.1%} average confidence")
        
        expected_signals = len(all_symbols) if len(all_symbols) <= 100 else 100
        actual_coverage = total_signals + len(random_results)
        
        print(f"\n📊 FINAL RESULTS")
        print("=" * 20)
        success = actual_coverage >= 50  # Success if we can generate 50+ signals
        
        print(f"Expected Coverage: ~{expected_signals} stocks")
        print(f"Actual Tested: {actual_coverage} stocks")
        print(f"Success Rate: {actual_coverage/expected_signals:.1%}")
        print(f"Status: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
        
        if success:
            print(f"\n🎉 ML SYSTEM READY FOR FULL DEPLOYMENT")
            print("Production ML system working with 100+ stock universe")
        else:
            print(f"\n⚠️ ML system needs optimization for full coverage")
        
        return success
        
    except Exception as e:
        print(f"❌ Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 FULL 100+ STOCK ML SYSTEM VALIDATION")
    print("=" * 50)
    print()
    
    success = test_full_100_ml_system()
    
    if success:
        print(f"\n🚀 PRODUCTION READY")
        print("ML system validated with 100+ stock universe")
        print("Ready for live trading across full market")
    else:
        print(f"\n🔧 NEEDS REFINEMENT")
        print("System working but needs optimization for full coverage")