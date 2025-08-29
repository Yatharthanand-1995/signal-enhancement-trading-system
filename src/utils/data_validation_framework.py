#!/usr/bin/env python3
"""
Comprehensive Data Validation Framework
Prevents data quality issues through multi-layer validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from dataclasses import dataclass
import logging

@dataclass
class ValidationResult:
    """Data validation result"""
    is_valid: bool
    symbol: str
    issues: List[str]
    warnings: List[str]
    data_quality_score: float
    metadata: Dict

class DataValidator:
    """Multi-layer data validation system"""
    
    def __init__(self):
        self.validation_rules = {
            'price': {'min': 0.01, 'max': 50000, 'required': True},
            'volume': {'min': 0, 'max': 1e12, 'required': True},
            'rsi': {'min': 0, 'max': 100, 'required': True},
            'macd': {'min': -1000, 'max': 1000, 'required': True},
            'change_1d': {'min': -50, 'max': 50, 'required': False},  # Allow extreme moves
            'volatility': {'min': 0, 'max': 200, 'required': True}
        }
        
    def validate_symbol_data(self, symbol: str, data: Dict) -> ValidationResult:
        """Comprehensive validation of single symbol data"""
        issues = []
        warnings = []
        
        # 1. Essential data presence check
        required_fields = ['current_price', 'volume', 'rsi_14', 'company_name']
        for field in required_fields:
            if field not in data or pd.isna(data[field]):
                issues.append(f"Missing required field: {field}")
        
        # 2. Data type validation
        numeric_fields = ['current_price', 'volume', 'rsi_14', 'change_1d', 'volatility_20d']
        for field in numeric_fields:
            if field in data and not isinstance(data[field], (int, float, np.number)):
                try:
                    data[field] = float(data[field])
                except:
                    issues.append(f"Invalid data type for {field}: {type(data[field])}")
        
        # 3. Range validation
        if 'current_price' in data:
            price = data['current_price']
            if price <= 0:
                issues.append(f"Invalid price: ${price}")
            elif price > 10000:
                warnings.append(f"Extremely high price: ${price:,.2f}")
                
        if 'rsi_14' in data:
            rsi = data['rsi_14']
            if not (0 <= rsi <= 100):
                issues.append(f"RSI out of range: {rsi}")
                
        if 'volume' in data:
            volume = data['volume']
            if volume < 0:
                issues.append(f"Negative volume: {volume}")
            elif volume == 0:
                warnings.append("Zero trading volume")
        
        # 4. Data freshness check
        if 'last_updated' in data:
            try:
                last_update = pd.to_datetime(data['last_updated'])
                hours_old = (datetime.now() - last_update).total_seconds() / 3600
                if hours_old > 24:
                    warnings.append(f"Stale data: {hours_old:.1f} hours old")
            except:
                warnings.append("Invalid timestamp format")
        
        # 5. Calculate data quality score
        quality_score = 100.0
        quality_score -= len(issues) * 25  # Major penalty for issues
        quality_score -= len(warnings) * 5  # Minor penalty for warnings
        quality_score = max(0, quality_score)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            symbol=symbol,
            issues=issues,
            warnings=warnings,
            data_quality_score=quality_score,
            metadata={
                'validation_timestamp': datetime.now().isoformat(),
                'total_checks': 10,
                'passed_checks': 10 - len(issues) - len(warnings)
            }
        )
    
    def validate_batch(self, stock_data: pd.DataFrame) -> Dict[str, ValidationResult]:
        """Validate entire batch of stock data"""
        results = {}
        
        for _, row in stock_data.iterrows():
            symbol = row['symbol']
            data = row.to_dict()
            results[symbol] = self.validate_symbol_data(symbol, data)
            
        return results
    
    def generate_validation_report(self, validation_results: Dict[str, ValidationResult]) -> Dict:
        """Generate comprehensive validation report"""
        total_symbols = len(validation_results)
        valid_symbols = sum(1 for r in validation_results.values() if r.is_valid)
        
        # Categorize issues
        issue_categories = {}
        warning_categories = {}
        
        for result in validation_results.values():
            for issue in result.issues:
                issue_type = issue.split(':')[0]
                issue_categories[issue_type] = issue_categories.get(issue_type, 0) + 1
                
            for warning in result.warnings:
                warning_type = warning.split(':')[0]
                warning_categories[warning_type] = warning_categories.get(warning_type, 0) + 1
        
        # Calculate quality metrics
        avg_quality_score = np.mean([r.data_quality_score for r in validation_results.values()])
        
        return {
            'summary': {
                'total_symbols': total_symbols,
                'valid_symbols': valid_symbols,
                'invalid_symbols': total_symbols - valid_symbols,
                'success_rate': valid_symbols / total_symbols * 100,
                'avg_quality_score': avg_quality_score
            },
            'issue_categories': issue_categories,
            'warning_categories': warning_categories,
            'problematic_symbols': [
                r.symbol for r in validation_results.values() 
                if not r.is_valid or r.data_quality_score < 70
            ],
            'validation_timestamp': datetime.now().isoformat()
        }


class APIHealthChecker:
    """Monitor API health and performance"""
    
    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Always-reliable test symbols
        
    def check_api_health(self) -> Dict:
        """Check Yahoo Finance API health"""
        results = {
            'api_status': 'unknown',
            'response_times': [],
            'success_rate': 0,
            'errors': []
        }
        
        success_count = 0
        
        for symbol in self.test_symbols:
            start_time = datetime.now()
            
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1d')
                
                if not data.empty:
                    success_count += 1
                    response_time = (datetime.now() - start_time).total_seconds()
                    results['response_times'].append(response_time)
                else:
                    results['errors'].append(f"{symbol}: Empty data returned")
                    
            except Exception as e:
                results['errors'].append(f"{symbol}: {str(e)[:50]}")
        
        results['success_rate'] = success_count / len(self.test_symbols) * 100
        
        # Determine API status
        if results['success_rate'] >= 100:
            results['api_status'] = 'excellent'
        elif results['success_rate'] >= 80:
            results['api_status'] = 'good'
        elif results['success_rate'] >= 50:
            results['api_status'] = 'degraded'
        else:
            results['api_status'] = 'poor'
            
        results['avg_response_time'] = np.mean(results['response_times']) if results['response_times'] else None
        
        return results


class DataQualityMonitor:
    """Comprehensive data quality monitoring system"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.api_checker = APIHealthChecker()
        self.quality_history = []
        
    def run_full_assessment(self, stock_data: pd.DataFrame) -> Dict:
        """Run complete data quality assessment"""
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'data_validation': None,
            'api_health': None,
            'recommendations': []
        }
        
        # 1. Validate all stock data
        validation_results = self.validator.validate_batch(stock_data)
        assessment['data_validation'] = self.validator.generate_validation_report(validation_results)
        
        # 2. Check API health
        assessment['api_health'] = self.api_checker.check_api_health()
        
        # 3. Generate recommendations
        recommendations = []
        
        if assessment['data_validation']['summary']['success_rate'] < 90:
            recommendations.append("🔴 Data quality below 90% - investigate failing symbols")
            
        if assessment['api_health']['api_status'] in ['poor', 'degraded']:
            recommendations.append("🟡 API performance degraded - consider fallback sources")
            
        if assessment['api_health']['avg_response_time'] and assessment['api_health']['avg_response_time'] > 5:
            recommendations.append("⚠️ Slow API response times - optimize request patterns")
            
        if assessment['data_validation']['summary']['avg_quality_score'] < 80:
            recommendations.append("📊 Low average quality score - review data sources")
        
        assessment['recommendations'] = recommendations
        
        # Store in history
        self.quality_history.append(assessment)
        
        return assessment
    
    def get_quality_trends(self, days: int = 7) -> Dict:
        """Analyze quality trends over time"""
        if len(self.quality_history) < 2:
            return {'trend': 'insufficient_data', 'message': 'Need more data points for trend analysis'}
        
        recent_assessments = self.quality_history[-days:] if len(self.quality_history) >= days else self.quality_history
        
        success_rates = [a['data_validation']['summary']['success_rate'] for a in recent_assessments]
        quality_scores = [a['data_validation']['summary']['avg_quality_score'] for a in recent_assessments]
        
        return {
            'trend': 'improving' if success_rates[-1] > success_rates[0] else 'declining' if success_rates[-1] < success_rates[0] else 'stable',
            'current_success_rate': success_rates[-1],
            'avg_success_rate': np.mean(success_rates),
            'current_quality_score': quality_scores[-1],
            'avg_quality_score': np.mean(quality_scores),
            'total_assessments': len(recent_assessments)
        }


if __name__ == "__main__":
    # Example usage
    monitor = DataQualityMonitor()
    
    # This would be called with actual stock data
    print("Data Quality Monitoring Framework initialized successfully!")
    print("Use monitor.run_full_assessment(stock_data) to validate your data")