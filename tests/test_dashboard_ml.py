#!/usr/bin/env python3
"""
Test ML functionality through dashboard imports
"""
import sys
import os

# Add src to path
sys.path.append('src')

def test_ml_dashboard_integration():
    """Test ML system integration through dashboard"""
    try:
        print("=== Testing Dashboard ML Integration ===")
        
        # Test the ML fallback system directly
        print("\n1. Testing ML fallback system...")
        from utils.ml_fallback import get_ml_status, initialize_ml_libraries
        
        # Get ML status
        ml_results = initialize_ml_libraries()
        ml_status = get_ml_status()
        
        print(f"ML Library Results: {ml_results}")
        print(f"ML Status: {ml_status}")
        
        # Test dashboard imports
        print("\n2. Testing dashboard ML imports...")
        from dashboard.main import ML_AVAILABLE, get_ml_status as dashboard_get_ml_status
        
        dashboard_ml_status = dashboard_get_ml_status()
        print(f"Dashboard ML Available: {ML_AVAILABLE}")
        print(f"Dashboard ML Status: {dashboard_ml_status}")
        
        # Report results
        print("\n=== ML Integration Test Results ===")
        tf_available = ml_status.get('tensorflow_available', False)
        xgb_available = ml_status.get('xgboost_available', False)
        
        print(f"TensorFlow: {'✅ Available' if tf_available else '❌ Not Available'}")
        print(f"XGBoost: {'✅ Available' if xgb_available else '❌ Not Available'}")
        print(f"Dashboard Integration: {'✅ Working' if ML_AVAILABLE else '❌ Failed'}")
        
        return tf_available and xgb_available
        
    except Exception as e:
        print(f"❌ Dashboard ML test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ml_dashboard_integration()
    if success:
        print("\n🎉 Dashboard ML integration test: SUCCESS")
    else:
        print("\n❌ Dashboard ML integration test: FAILED")