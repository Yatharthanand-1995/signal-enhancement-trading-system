#!/usr/bin/env python3
"""
Test ML Integration with Existing System
Comprehensive testing of production ML system integration
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_signal_integration():
    """Test the new ML-enhanced signal integration"""
    
    print("🧪 TESTING ML INTEGRATION WITH EXISTING SYSTEM")
    print("=" * 60)
    print("Goal: Verify production ML system works with existing infrastructure")
    print()
    
    try:
        # Test 1: Import the enhanced signal integration
        print("📋 TEST 1: Import Enhanced Signal Integration")
        print("-" * 40)
        
        from src.strategy.enhanced_signal_integration import get_enhanced_signal, initialize_enhanced_signal_integration
        print("✅ Successfully imported enhanced signal integration")
        
        # Test 2: Initialize the system
        print(f"\n📋 TEST 2: Initialize ML Signal System")
        print("-" * 40)
        
        integrator = initialize_enhanced_signal_integration()
        print("✅ Successfully initialized Production ML Signal Integrator")
        print(f"   System: {integrator.name if hasattr(integrator, 'name') else 'ProductionMLSignalIntegrator'}")
        print(f"   ML Features: {len(integrator.signal_correlations)} proven correlations")
        
        # Test 3: Load test data
        print(f"\n📋 TEST 3: Load Real Market Data")
        print("-" * 40)
        
        data_dir = 'data/full_market'
        val_path = os.path.join(data_dir, 'validation_data.csv')
        
        if not os.path.exists(val_path):
            print("❌ Validation data not found - run real_data_pipeline.py first")
            return False
        
        val_data = pd.read_csv(val_path)
        val_data['date'] = pd.to_datetime(val_data['date'])
        
        print(f"✅ Loaded validation data: {len(val_data):,} records")
        print(f"   Symbols: {val_data['symbol'].nunique()}")
        print(f"   Date range: {val_data['date'].min().date()} to {val_data['date'].max().date()}")
        
        # Test 4: Generate ML-enhanced signals
        print(f"\n📋 TEST 4: Generate ML-Enhanced Signals")
        print("-" * 40)
        
        test_symbols = ['AAPL', 'MSFT', 'SPY']
        signal_results = []
        
        for symbol in test_symbols:
            if symbol in val_data['symbol'].values:
                symbol_data = val_data[val_data['symbol'] == symbol].sort_values('date')
                
                if len(symbol_data) >= 100:
                    # Test signal generation
                    test_data = symbol_data.tail(100)
                    current_price = test_data['close'].iloc[-1]
                    
                    enhanced_signal = get_enhanced_signal(
                        symbol=symbol,
                        data=test_data,
                        current_price=current_price,
                        current_regime='normal'
                    )
                    
                    if enhanced_signal:
                        signal_results.append({
                            'symbol': symbol,
                            'signal_strength': enhanced_signal.signal_strength,
                            'confidence': enhanced_signal.confidence,
                            'ml_contribution': enhanced_signal.ml_contribution,
                            'technical_contribution': enhanced_signal.technical_contribution,
                            'predicted_volatility': enhanced_signal.predicted_volatility,
                            'position_size': enhanced_signal.recommended_position_size,
                            'stop_loss_pct': enhanced_signal.stop_loss_pct,
                            'quality': enhanced_signal.quality.value,
                            'ml_explanation': enhanced_signal.ml_explanation
                        })
                        
                        print(f"✅ {symbol}: Signal {enhanced_signal.signal_strength:.3f}, "
                              f"Confidence {enhanced_signal.confidence:.3f}, "
                              f"ML Contribution {enhanced_signal.ml_contribution:.3f}")
                    else:
                        print(f"❌ {symbol}: No signal generated")
        
        print(f"\nSuccessfully generated {len(signal_results)} ML-enhanced signals")
        
        # Test 5: Compatibility with existing backtesting
        print(f"\n📋 TEST 5: Backtesting Engine Compatibility")
        print("-" * 40)
        
        try:
            from src.backtesting.enhanced_backtest_engine import EnhancedBacktestEngine
            
            # Create backtest engine
            engine = EnhancedBacktestEngine()
            print("✅ Successfully imported enhanced backtest engine")
            
            # Test signal calculation method
            if hasattr(engine, '_calculate_signal_strength'):
                print("✅ Signal calculation method exists")
                
                # Test with sample data
                if signal_results:
                    sample_symbol = signal_results[0]['symbol']
                    sample_data = val_data[val_data['symbol'] == sample_symbol].tail(1)
                    
                    if not sample_data.empty:
                        row = sample_data.iloc[0]
                        try:
                            signal_strength, components = engine._calculate_signal_strength(row)
                            print(f"✅ Backtesting integration working:")
                            print(f"   Signal strength: {signal_strength:.3f}")
                            print(f"   Components: {list(components.keys())}")
                        except Exception as e:
                            print(f"⚠️ Backtesting integration issue: {str(e)}")
            
        except ImportError:
            print("⚠️ Enhanced backtest engine not available")
        
        # Test 6: Signal quality and components
        print(f"\n📋 TEST 6: Signal Quality Analysis")
        print("-" * 40)
        
        if signal_results:
            avg_signal_strength = np.mean([s['signal_strength'] for s in signal_results])
            avg_confidence = np.mean([s['confidence'] for s in signal_results])
            avg_ml_contribution = np.mean([s['ml_contribution'] for s in signal_results])
            avg_position_size = np.mean([s['position_size'] for s in signal_results])
            avg_volatility = np.mean([s['predicted_volatility'] for s in signal_results])
            
            print(f"✅ Signal Quality Metrics:")
            print(f"   Average Signal Strength: {avg_signal_strength:.3f}")
            print(f"   Average Confidence: {avg_confidence:.3f}")
            print(f"   Average ML Contribution: {avg_ml_contribution:.3f}")
            print(f"   Average Position Size: {avg_position_size:.1%}")
            print(f"   Average Predicted Volatility: {avg_volatility:.1%}")
            
            # Component analysis
            print(f"\n   Component Breakdown:")
            for result in signal_results:
                print(f"   {result['symbol']}: "
                      f"Tech {result['technical_contribution']:.3f}, "
                      f"ML {result['ml_contribution']:.3f}, "
                      f"Vol {result['predicted_volatility']:.3f}")
        
        # Test 7: Performance validation
        print(f"\n📋 TEST 7: Performance Validation")
        print("-" * 40)
        
        validation_passed = True
        
        # Check signal strength range
        if signal_results:
            signal_strengths = [abs(s['signal_strength']) for s in signal_results]
            if all(s <= 1.0 for s in signal_strengths):
                print("✅ Signal strengths within valid range [-1, 1]")
            else:
                print("❌ Signal strengths outside valid range")
                validation_passed = False
            
            # Check confidence range
            confidences = [s['confidence'] for s in signal_results]
            if all(0.5 <= c <= 1.0 for c in confidences):
                print("✅ Confidence scores within valid range [0.5, 1.0]")
            else:
                print("❌ Confidence scores outside valid range")
                validation_passed = False
            
            # Check ML contributions exist
            ml_contributions = [abs(s['ml_contribution']) for s in signal_results]
            if any(ml > 0.001 for ml in ml_contributions):
                print("✅ ML contributions detected in signals")
            else:
                print("⚠️ No significant ML contributions detected")
            
            # Check risk management components
            position_sizes = [s['position_size'] for s in signal_results]
            if all(0.02 <= p <= 0.25 for p in position_sizes):
                print("✅ Position sizes within reasonable range [2%, 25%]")
            else:
                print("⚠️ Position sizes outside expected range")
            
            volatilities = [s['predicted_volatility'] for s in signal_results]
            if all(0.005 <= v <= 0.08 for v in volatilities):
                print("✅ Volatility predictions within reasonable range [0.5%, 8%]")
            else:
                print("⚠️ Volatility predictions outside expected range")
        
        return validation_passed
        
    except Exception as e:
        print(f"❌ ML integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test backward compatibility with existing system"""
    
    print(f"\n🔄 TESTING BACKWARDS COMPATIBILITY")
    print("=" * 50)
    
    try:
        from src.strategy.enhanced_signal_integration import get_enhanced_signal
        
        # Test the function signature that existing system expects
        data_dir = 'data/full_market'
        val_path = os.path.join(data_dir, 'validation_data.csv')
        
        if os.path.exists(val_path):
            val_data = pd.read_csv(val_path)
            val_data['date'] = pd.to_datetime(val_data['date'])
            
            aapl_data = val_data[val_data['symbol'] == 'AAPL'].tail(100)
            
            if len(aapl_data) > 0:
                # Test old function signature
                signal = get_enhanced_signal(
                    symbol='AAPL',
                    data=aapl_data,
                    current_price=aapl_data['close'].iloc[-1],
                    current_regime='normal'
                )
                
                if signal and hasattr(signal, 'signal_strength') and hasattr(signal, 'confidence'):
                    print("✅ Backward compatibility maintained")
                    print(f"   Signal strength: {signal.signal_strength:.3f}")
                    print(f"   Confidence: {signal.confidence:.3f}")
                    
                    # Check for new ML attributes
                    if hasattr(signal, 'ml_contribution'):
                        print(f"   ML contribution: {signal.ml_contribution:.3f}")
                    if hasattr(signal, 'recommended_position_size'):
                        print(f"   Position size: {signal.recommended_position_size:.3f}")
                    
                    return True
                else:
                    print("❌ Backward compatibility broken")
                    return False
            else:
                print("⚠️ Insufficient test data")
                return False
        else:
            print("⚠️ Test data not available")
            return False
    
    except Exception as e:
        print(f"❌ Compatibility test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 PRODUCTION ML SYSTEM INTEGRATION TESTING")
    print("=" * 70)
    print("Comprehensive testing of ML integration with existing infrastructure")
    print()
    
    # Test 1: ML Integration
    integration_success = test_enhanced_signal_integration()
    
    # Test 2: Backward Compatibility
    compatibility_success = test_backwards_compatibility()
    
    print(f"\n🎯 INTEGRATION TEST RESULTS")
    print("=" * 40)
    print(f"ML Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    print(f"Backward Compatibility: {'✅ PASS' if compatibility_success else '❌ FAIL'}")
    
    overall_success = integration_success and compatibility_success
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Production ML system successfully integrated")
        print("Ready for comprehensive backtesting")
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        print("Review failed components before proceeding")
    
    print(f"\n📊 Next Step: Run comprehensive backtest")
    print("Command: python comprehensive_ml_backtest.py")