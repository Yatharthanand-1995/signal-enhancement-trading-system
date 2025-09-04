#!/usr/bin/env python3
"""
Phase 2 ML Monitoring System Testing Script
Comprehensive testing of ML monitoring, ensemble optimization, retraining, alerting, explainability, and versioning
"""
import sys
import os
import time
import tempfile
import shutil
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_model_performance_monitor():
    """Test Model Performance Monitoring System"""
    print("🔍 Testing Model Performance Monitor...")
    
    try:
        from ml_monitoring.model_performance_monitor import (
            ModelPerformanceMonitor, ModelPerformanceMetrics, AlertConfig
        )
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            monitor = ModelPerformanceMonitor(db_path=tmp_db.name)
        
        # Test prediction logging
        monitor.log_prediction(
            model_name='test_lstm',
            prediction=0.75,
            features={'rsi': 60, 'macd': 0.5},
            confidence=0.8,
            actual_value=0.70
        )
        
        # Add more predictions for testing
        for i in range(20):
            monitor.log_prediction(
                model_name='test_lstm',
                prediction=0.6 + np.random.normal(0, 0.1),
                features={'rsi': 50 + np.random.normal(0, 10), 'macd': np.random.normal(0, 0.2)},
                confidence=0.7 + np.random.uniform(0, 0.2),
                actual_value=0.6 + np.random.normal(0, 0.15)
            )
        
        # Test performance calculation
        metrics = monitor.calculate_model_performance('test_lstm', window_hours=1)
        assert metrics is not None, "Should calculate performance metrics"
        assert 0 <= metrics.accuracy <= 1, f"Accuracy should be 0-1, got {metrics.accuracy}"
        assert 0 <= metrics.confidence_score <= 1, f"Confidence should be 0-1, got {metrics.confidence_score}"
        
        # Test alert checking
        alerts = monitor.check_alerts(metrics)
        assert isinstance(alerts, list), "Should return list of alerts"
        
        # Test health status
        health = monitor.get_model_health_status('test_lstm')
        assert health['status'] in ['healthy', 'warning', 'critical', 'unknown'], f"Invalid health status: {health['status']}"
        
        # Cleanup
        os.unlink(tmp_db.name)
        
        print("✅ Model Performance Monitor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Model Performance Monitor test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_ensemble_optimizer():
    """Test Ensemble Optimizer System"""
    print("🔍 Testing Ensemble Optimizer...")
    
    try:
        from ml_monitoring.ensemble_optimizer import (
            EnsembleOptimizer, ModelPrediction, EnsembleConfig
        )
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            optimizer = EnsembleOptimizer(db_path=tmp_db.name)
        
        # Register models
        optimizer.register_model('lstm_model', 'lstm', 'v1.0.0')
        optimizer.register_model('xgb_model', 'xgboost', 'v1.0.0')
        optimizer.register_model('rf_model', 'random_forest', 'v1.0.0')
        
        # Add predictions from different models
        for i in range(30):
            timestamp = datetime.now() - timedelta(minutes=i)
            
            # LSTM predictions
            optimizer.add_prediction(ModelPrediction(
                model_name='lstm_model',
                prediction=0.6 + np.random.normal(0, 0.1),
                confidence=0.8 + np.random.uniform(-0.1, 0.1),
                timestamp=timestamp,
                features={'feature_1': np.random.normal(0, 1)},
                model_version='v1.0.0'
            ))
            
            # XGBoost predictions
            optimizer.add_prediction(ModelPrediction(
                model_name='xgb_model',
                prediction=0.65 + np.random.normal(0, 0.08),
                confidence=0.75 + np.random.uniform(-0.1, 0.1),
                timestamp=timestamp,
                features={'feature_1': np.random.normal(0, 1)},
                model_version='v1.0.0'
            ))
            
            # Random Forest predictions
            optimizer.add_prediction(ModelPrediction(
                model_name='rf_model',
                prediction=0.55 + np.random.normal(0, 0.12),
                confidence=0.7 + np.random.uniform(-0.1, 0.1),
                timestamp=timestamp,
                features={'feature_1': np.random.normal(0, 1)},
                model_version='v1.0.0'
            ))
        
        # Test weight optimization
        optimized_weights = optimizer.optimize_weights()
        assert isinstance(optimized_weights, dict), "Should return weight dictionary"
        assert len(optimized_weights) > 0, "Should have optimized weights"
        assert abs(sum(optimized_weights.values()) - 1.0) < 0.01, "Weights should sum to ~1.0"
        
        # Test ensemble prediction
        individual_predictions = {
            'lstm_model': ModelPrediction(
                model_name='lstm_model', prediction=0.7, confidence=0.8,
                timestamp=datetime.now(), features={}, model_version='v1.0.0'
            ),
            'xgb_model': ModelPrediction(
                model_name='xgb_model', prediction=0.65, confidence=0.75,
                timestamp=datetime.now(), features={}, model_version='v1.0.0'
            ),
            'rf_model': ModelPrediction(
                model_name='rf_model', prediction=0.6, confidence=0.7,
                timestamp=datetime.now(), features={}, model_version='v1.0.0'
            )
        }
        
        ensemble_pred, ensemble_conf, metadata = optimizer.generate_ensemble_prediction('AAPL', individual_predictions)
        assert isinstance(ensemble_pred, float), "Should return float prediction"
        assert 0 <= ensemble_conf <= 1, f"Ensemble confidence should be 0-1, got {ensemble_conf}"
        assert 'model_count' in metadata, "Metadata should contain model count"
        
        # Test model rankings
        rankings = optimizer.get_model_rankings()
        assert isinstance(rankings, list), "Should return list of rankings"
        assert len(rankings) > 0, "Should have model rankings"
        
        # Test ensemble health
        health = optimizer.get_ensemble_health()
        assert 'status' in health, "Health should contain status"
        assert health['active_models'] >= 0, "Should report active model count"
        
        # Cleanup
        os.unlink(tmp_db.name)
        
        print("✅ Ensemble Optimizer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Ensemble Optimizer test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_model_retraining_pipeline():
    """Test Model Retraining Pipeline"""
    print("🔍 Testing Model Retraining Pipeline...")
    
    try:
        from ml_monitoring.model_retraining_pipeline import (
            ModelRetrainingPipeline, RetrainingTrigger, RetrainingConfig
        )
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            pipeline = ModelRetrainingPipeline(db_path=tmp_db.name)
        
        # Mock data provider
        def mock_data_provider():
            return pd.DataFrame({
                'feature_1': np.random.randn(1000),
                'feature_2': np.random.randn(1000),
                'target': np.random.randn(1000)
            })
        
        # Mock retraining function
        def mock_retrain_function(training_data, config):
            return {
                'status': 'success',
                'model': 'mock_model_object',
                'metrics': {
                    'accuracy': 0.85,
                    'loss': 0.15
                }
            }
        
        # Register model for retraining
        pipeline.register_model(
            model_name='test_model',
            retrain_function=mock_retrain_function,
            data_provider=mock_data_provider,
            model_config={'param1': 'value1'}
        )
        
        # Test trigger creation
        trigger = RetrainingTrigger(
            trigger_type='performance',
            model_name='test_model',
            trigger_value=0.65,
            threshold=0.7,
            timestamp=datetime.now(),
            metadata={'metric': 'accuracy'}
        )
        
        pipeline.add_retraining_trigger(trigger)
        
        # Test retraining status
        status = pipeline.get_retraining_status()
        assert 'monitoring_active' in status, "Status should contain monitoring flag"
        assert 'registered_models' in status, "Status should contain model count"
        assert status['registered_models'] > 0, "Should have registered models"
        
        # Cleanup
        os.unlink(tmp_db.name)
        
        print("✅ Model Retraining Pipeline tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Model Retraining Pipeline test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_advanced_alerting_system():
    """Test Advanced Alerting System"""
    print("🔍 Testing Advanced Alerting System...")
    
    try:
        from ml_monitoring.advanced_alerting_system import (
            AdvancedAlertingSystem, AlertRule, AlertSeverity, AlertStatus
        )
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            alerting = AdvancedAlertingSystem(db_path=tmp_db.name)
        
        # Setup default ML rules
        alerting.setup_default_ml_rules()
        assert len(alerting.alert_rules) > 0, "Should have default alert rules"
        
        # Test custom alert rule
        custom_rule = AlertRule(
            rule_id='test_accuracy_rule',
            name='Test Accuracy Rule',
            description='Test rule for accuracy monitoring',
            metric_name='accuracy',
            condition='lt',
            threshold=0.8,
            severity=AlertSeverity.HIGH,
            notification_channels=['email']
        )
        
        alerting.add_alert_rule(custom_rule)
        assert 'test_accuracy_rule' in alerting.alert_rules, "Should add custom rule"
        
        # Test metric checking (should trigger alert)
        triggered_alerts = alerting.check_metric(
            model_name='test_model',
            metric_name='accuracy', 
            value=0.75,  # Below threshold
            metadata={'test': True}
        )
        
        assert len(triggered_alerts) > 0, "Should trigger alerts for low accuracy"
        
        # Test no alert trigger
        no_alerts = alerting.check_metric(
            model_name='test_model',
            metric_name='accuracy',
            value=0.85  # Above threshold
        )
        
        # Should not trigger due to cooldown or value above threshold
        
        # Test active alerts
        active_alerts = alerting.get_active_alerts()
        assert isinstance(active_alerts, list), "Should return list of active alerts"
        
        # Test alert acknowledgment
        if active_alerts:
            alert_id = active_alerts[0].alert_id
            success = alerting.acknowledge_alert(alert_id, 'test_user')
            assert success, "Should acknowledge alert"
        
        # Test alert summary
        summary = alerting.get_alert_summary()
        assert 'total_active_alerts' in summary, "Summary should contain alert count"
        assert 'severity_distribution' in summary, "Summary should contain severity breakdown"
        
        # Cleanup
        os.unlink(tmp_db.name)
        
        print("✅ Advanced Alerting System tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Advanced Alerting System test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_model_explainability():
    """Test Model Explainability System"""
    print("🔍 Testing Model Explainability System...")
    
    try:
        from ml_monitoring.model_explainability import (
            ModelExplainabilitySystem, FeatureImportance, PredictionExplanation
        )
        from sklearn.ensemble import RandomForestRegressor
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            explainer = ModelExplainabilitySystem(db_path=tmp_db.name)
        
        # Create mock model and data
        np.random.seed(42)
        X_train = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100)
        })
        y_train = X_train['feature_1'] * 2 + X_train['feature_2'] - X_train['feature_3'] + np.random.randn(100) * 0.1
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test surrogate model creation
        success = explainer.create_surrogate_model('test_rf_model', model, X_train, y_train)
        assert success, "Should create surrogate model"
        
        # Test global feature importance
        feature_importance = explainer.get_global_feature_importance('test_rf_model', model, X_train)
        assert len(feature_importance) > 0, "Should return feature importance"
        assert all(isinstance(fi, FeatureImportance) for fi in feature_importance), "Should return FeatureImportance objects"
        
        # Test surrogate explanation
        test_instance = X_train.iloc[0]
        surrogate_explanation = explainer.explain_prediction_surrogate('test_rf_model', test_instance)
        assert surrogate_explanation is not None, "Should generate surrogate explanation"
        assert isinstance(surrogate_explanation.prediction_value, float), "Should have prediction value"
        assert len(surrogate_explanation.feature_contributions) > 0, "Should have feature contributions"
        
        # Test comprehensive explanation
        explanations = explainer.explain_prediction(
            model_name='test_rf_model', 
            model=model, 
            instance=test_instance, 
            methods=['surrogate']
        )
        
        assert 'surrogate' in explanations, "Should contain surrogate explanation"
        
        # Test explanation report
        report = explainer.generate_explanation_report('test_rf_model')
        assert 'model_name' in report, "Report should contain model name"
        assert 'top_features' in report, "Report should contain top features"
        
        # Test feature impact summary
        if feature_importance:
            feature_name = feature_importance[0].feature_name
            impact_summary = explainer.get_feature_impact_summary('test_rf_model', feature_name)
            assert 'feature_name' in impact_summary, "Impact summary should contain feature name"
        
        # Cleanup
        os.unlink(tmp_db.name)
        
        print("✅ Model Explainability System tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Model Explainability System test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_model_versioning_system():
    """Test Model Versioning System"""
    print("🔍 Testing Model Versioning System...")
    
    try:
        from ml_monitoring.model_versioning_system import (
            ModelVersioningSystem, ModelStatus, DeploymentConfig, DeploymentStrategy
        )
        from sklearn.ensemble import RandomForestRegressor
        
        # Create temporary directories and database
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage_path = os.path.join(tmp_dir, 'model_storage')
            db_path = os.path.join(tmp_dir, 'versioning.db')
            
            versioning = ModelVersioningSystem(storage_path=storage_path, db_path=db_path)
            
            # Create mock model
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            X_dummy = np.random.randn(50, 3)
            y_dummy = np.random.randn(50)
            model.fit(X_dummy, y_dummy)
            
            # Test model registration
            model_id = versioning.register_model(
                model_obj=model,
                model_name='test_model',
                model_type='random_forest',
                performance_metrics={'accuracy': 0.85, 'mse': 0.15},
                training_config={'n_estimators': 10, 'random_state': 42},
                created_by='test_user',
                tags=['test', 'phase2']
            )
            
            assert model_id is not None, "Should return model ID"
            assert model_id in versioning.model_registry, "Model should be in registry"
            
            # Test model status update
            success = versioning.update_model_status(model_id, ModelStatus.STAGING)
            assert success, "Should update model status"
            assert versioning.model_registry[model_id].status == ModelStatus.STAGING, "Status should be updated"
            
            # Test model loading
            loaded_model = versioning.load_model(model_id)
            assert loaded_model is not None, "Should load model"
            
            # Make a prediction to verify model works
            test_pred = loaded_model.predict(X_dummy[:1])
            assert len(test_pred) == 1, "Should make prediction"
            
            # Test model deployment
            deploy_config = DeploymentConfig(
                strategy=DeploymentStrategy.REPLACE,
                success_criteria={'accuracy': 0.8},
                rollback_conditions={'accuracy': 0.6}
            )
            
            deployment_id = versioning.deploy_model(
                model_id=model_id,
                environment='test',
                config=deploy_config,
                deployed_by='test_user'
            )
            
            assert deployment_id is not None, "Should return deployment ID"
            
            # Test getting model versions
            versions = versioning.get_model_versions(model_name='test_model')
            assert len(versions) > 0, "Should return model versions"
            assert versions[0].model_name == 'test_model', "Should filter by model name"
            
            # Test model lineage
            lineage = versioning.get_model_lineage(model_id)
            assert 'model_id' in lineage, "Lineage should contain model ID"
            assert 'parents' in lineage, "Lineage should contain parents"
            assert 'children' in lineage, "Lineage should contain children"
            
            # Test deployment status
            status = versioning.get_deployment_status()
            assert 'deployments_by_environment' in status, "Status should contain deployments"
            
            print("✅ Model Versioning System tests passed")
            return True
        
    except Exception as e:
        print(f"❌ Model Versioning System test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between Phase 2 components"""
    print("🔍 Testing Phase 2 integration...")
    
    try:
        # Test that components can work together
        from ml_monitoring.model_performance_monitor import ModelPerformanceMonitor
        from ml_monitoring.advanced_alerting_system import AdvancedAlertingSystem
        
        # Create temporary databases
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as monitor_db:
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as alert_db:
                # Initialize systems
                monitor = ModelPerformanceMonitor(db_path=monitor_db.name)
                alerting = AdvancedAlertingSystem(db_path=alert_db.name)
                
                # Setup alerts
                alerting.setup_default_ml_rules()
                
                # Simulate poor model performance
                for i in range(10):
                    monitor.log_prediction(
                        model_name='integrated_test_model',
                        prediction=0.4 + np.random.normal(0, 0.1),  # Poor predictions
                        features={'feature_1': np.random.randn()},
                        confidence=0.5,
                        actual_value=0.8  # Much higher than predictions (poor performance)
                    )
                
                # Calculate performance
                metrics = monitor.calculate_model_performance('integrated_test_model', window_hours=1)
                assert metrics is not None, "Should calculate metrics"
                
                # Check if this triggers alerts
                alerts = alerting.check_metric(
                    model_name='integrated_test_model',
                    metric_name='accuracy',
                    value=metrics.accuracy
                )
                
                # Get system health
                monitor_health = monitor.get_model_health_status('integrated_test_model')
                alert_summary = alerting.get_alert_summary()
                
                assert 'status' in monitor_health, "Monitor should provide health status"
                assert 'total_active_alerts' in alert_summary, "Alerting should provide summary"
                
                # Cleanup
                os.unlink(monitor_db.name)
                os.unlink(alert_db.name)
        
        print("✅ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_performance():
    """Test performance of Phase 2 components"""
    print("🔍 Testing Phase 2 performance...")
    
    try:
        from ml_monitoring.model_performance_monitor import ModelPerformanceMonitor
        
        # Performance test: batch prediction logging
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            monitor = ModelPerformanceMonitor(db_path=tmp_db.name)
            
            start_time = time.time()
            
            # Log many predictions
            for i in range(100):
                monitor.log_prediction(
                    model_name='perf_test_model',
                    prediction=np.random.uniform(0.3, 0.9),
                    features={f'feature_{j}': np.random.randn() for j in range(10)},
                    confidence=np.random.uniform(0.5, 0.95),
                    actual_value=np.random.uniform(0.3, 0.9)
                )
            
            logging_time = time.time() - start_time
            
            # Calculate performance
            start_time = time.time()
            metrics = monitor.calculate_model_performance('perf_test_model', window_hours=1)
            calculation_time = time.time() - start_time
            
            # Performance assertions
            assert logging_time < 5.0, f"Prediction logging too slow: {logging_time:.3f}s for 100 predictions"
            assert calculation_time < 2.0, f"Metrics calculation too slow: {calculation_time:.3f}s"
            assert metrics is not None, "Should calculate metrics within time limit"
            
            print(f"  ✅ Prediction logging: {logging_time:.3f}s for 100 predictions")
            print(f"  ✅ Metrics calculation: {calculation_time:.3f}s")
            
            # Cleanup
            os.unlink(tmp_db.name)
        
        print("✅ Performance tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all Phase 2 ML Monitoring tests"""
    print("🚀 Phase 2 ML Monitoring System Testing")
    print("=" * 50)
    
    test_results = {
        'Model Performance Monitor': test_model_performance_monitor(),
        'Ensemble Optimizer': test_ensemble_optimizer(),
        'Model Retraining Pipeline': test_model_retraining_pipeline(),
        'Advanced Alerting System': test_advanced_alerting_system(),
        'Model Explainability': test_model_explainability(),
        'Model Versioning System': test_model_versioning_system(),
        'Integration Testing': test_integration(),
        'Performance Testing': test_performance()
    }
    
    print("\n" + "=" * 50)
    print("📊 Phase 2 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:30} {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All Phase 2 ML Monitoring tests PASSED! System is ready for deployment!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)