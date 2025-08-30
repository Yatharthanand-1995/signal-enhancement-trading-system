#!/usr/bin/env python3
"""
Direct ML library test - simplified approach
"""
import os
import sys

# Set environment before importing anything
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def test_tensorflow():
    """Test TensorFlow directly"""
    try:
        print("Testing TensorFlow...")
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        # Simple test
        test = tf.constant([1.0])
        print(f"✅ TensorFlow basic operation successful: {test}")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_xgboost():
    """Test XGBoost directly"""
    try:
        print("\nTesting XGBoost...")
        import xgboost as xgb
        print(f"✅ XGBoost {xgb.__version__} imported successfully")
        
        # Simple test
        import numpy as np
        data = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 1])
        dtrain = xgb.DMatrix(data, label=labels)
        print("✅ XGBoost basic operation successful")
        return True
        
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Direct ML Library Testing ===")
    tf_result = test_tensorflow()
    xgb_result = test_xgboost()
    
    print(f"\n=== Results ===")
    print(f"TensorFlow: {'✅ Working' if tf_result else '❌ Failed'}")
    print(f"XGBoost: {'✅ Working' if xgb_result else '❌ Failed'}")