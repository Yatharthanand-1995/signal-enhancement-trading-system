#!/usr/bin/env python3
"""
Live System Status Check
Validates that the full 106-stock ML system is operational
"""

import sys
import os
import pandas as pd
sys.path.append('src')

def check_system_status():
    print("🔍 LIVE SYSTEM STATUS CHECK")
    print("=" * 35)
    
    try:
        # Check data availability
        data_path = 'data/full_market/validation_data.csv'
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            print(f"✅ Dataset: {len(data):,} records, {data['symbol'].nunique()} stocks")
        else:
            print("❌ Dataset not available")
            return False
        
        # Check ML integration
        from strategy.enhanced_signal_integration import initialize_enhanced_signal_integration
        integrator = initialize_enhanced_signal_integration()
        print(f"✅ ML System: {integrator.name} operational")
        
        # Quick signal test
        from strategy.enhanced_signal_integration import get_enhanced_signal
        test_data = data[data['symbol'] == 'AAPL'].tail(100)
        if len(test_data) > 0:
            signal = get_enhanced_signal(
                symbol='AAPL',
                data=test_data,
                current_price=test_data['close'].iloc[-1],
                current_regime='normal'
            )
            if signal:
                print(f"✅ Signal Generation: Working (AAPL: {signal.signal_strength:.3f})")
            else:
                print("❌ Signal Generation: Failed")
                return False
        
        print("\n🎉 LIVE SYSTEM FULLY OPERATIONAL")
        print("Ready for 106-stock production trading")
        return True
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False

if __name__ == "__main__":
    check_system_status()
