"""
Portfolio Optimization Engine
Advanced portfolio optimization using modern portfolio theory and risk models
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sqlite3
import json
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    max_weight: float = 0.10  # Maximum weight per asset
    min_weight: float = 0.0   # Minimum weight per asset
    max_sector_weight: float = 0.30  # Maximum sector concentration
    max_turnover: float = 0.20  # Maximum turnover per rebalance
    target_volatility: float = None  # Target portfolio volatility
    min_expected_return: float = None  # Minimum expected return
    beta_range: Tuple[float, float] = None  # Beta constraints (min, max)
    
@dataclass  
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    diversification_ratio: float
    turnover: float
    objective_value: float
    success: bool
    message: str
    constraints_satisfied: bool

class PortfolioOptimizer:
    """
    Advanced portfolio optimization engine with multiple optimization methods
    and risk models
    """
    
    def __init__(self, db_path: str = "portfolio_optimization.db"):
        self.db_path = db_path
        self.risk_models = {}
        self.factor_models = {}
        self.optimization_history = []
        
        # Initialize database
        self._init_database()
        
        logger.info("Portfolio optimizer initialized")
    
    def _init_database(self):
        """Initialize optimization database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id TEXT NOT NULL,
                        optimization_date DATE NOT NULL,
                        optimization_method TEXT NOT NULL,
                        objective_function TEXT NOT NULL,
                        weights TEXT NOT NULL,
                        expected_return REAL NOT NULL,
                        expected_volatility REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        turnover REAL,
                        constraints TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS risk_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_date DATE NOT NULL,
                        covariance_matrix TEXT NOT NULL,
                        expected_returns TEXT NOT NULL,
                        factor_loadings TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS rebalancing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        portfolio_id TEXT NOT NULL,
                        rebalance_date DATE NOT NULL,
                        old_weights TEXT NOT NULL,
                        new_weights TEXT NOT NULL,
                        turnover REAL NOT NULL,
                        transaction_costs REAL,
                        reason TEXT,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
        except Exception as e:
            logger.error(f"Error initializing optimization database: {e}")
            raise
    
    def estimate_risk_model(self, returns_data: pd.DataFrame, 
                          method: str = 'historical', window: int = 252) -> Dict[str, Any]:
        """
        Estimate risk model (covariance matrix and expected returns)
        
        Methods:
        - 'historical': Sample covariance and mean returns
        - 'shrinkage': Ledoit-Wolf shrinkage estimator
        - 'exponential': Exponentially weighted covariance
        - 'factor_model': Multi-factor risk model
        """
        try:
            if len(returns_data) < 50:
                raise ValueError(f"Insufficient data for risk model: {len(returns_data)} observations")
            
            # Use most recent data within window
            recent_data = returns_data.tail(window)
            
            if method == 'historical':
                # Simple historical estimators
                expected_returns = recent_data.mean() * 252  # Annualized
                covariance_matrix = recent_data.cov() * 252  # Annualized
                
            elif method == 'shrinkage':
                # Ledoit-Wolf shrinkage estimator
                expected_returns = recent_data.mean() * 252
                
                # Shrinkage towards identity matrix
                sample_cov = recent_data.cov() * 252
                n_assets = len(sample_cov)
                
                # Shrinkage target (identity matrix scaled by average variance)
                avg_var = np.trace(sample_cov) / n_assets
                target = np.eye(n_assets) * avg_var
                
                # Shrinkage intensity (simplified)
                shrinkage_intensity = 0.2  # Can be optimized
                covariance_matrix = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target
                
            elif method == 'exponential':
                # Exponentially weighted covariance
                expected_returns = recent_data.mean() * 252
                
                decay_factor = 0.94  # Common choice for daily data
                weights = np.array([(1 - decay_factor) * (decay_factor ** i) for i in range(len(recent_data))][::-1])
                weights = weights / weights.sum()
                
                # Weighted covariance calculation
                centered_returns = recent_data - recent_data.mean()
                covariance_matrix = np.zeros((len(recent_data.columns), len(recent_data.columns)))
                
                for i, weight in enumerate(weights):
                    ret = centered_returns.iloc[i].values
                    covariance_matrix += weight * np.outer(ret, ret)
                
                covariance_matrix = pd.DataFrame(covariance_matrix, 
                                               index=recent_data.columns, 
                                               columns=recent_data.columns) * 252
                
            elif method == 'factor_model':
                # Multi-factor model (simplified Fama-French)
                expected_returns, covariance_matrix = self._estimate_factor_model(recent_data)
                
            else:
                raise ValueError(f"Unknown risk model method: {method}")
            
            # Validate results
            if not self._validate_risk_model(expected_returns, covariance_matrix):
                logger.warning(f"Risk model validation failed for method: {method}")
            
            risk_model = {
                'method': method,
                'expected_returns': expected_returns,
                'covariance_matrix': covariance_matrix,
                'estimation_date': datetime.now(),
                'data_window': len(recent_data),
                'assets': list(returns_data.columns)
            }
            
            # Store risk model
            self._store_risk_model(risk_model)
            
            return risk_model
            
        except Exception as e:
            logger.error(f"Error estimating risk model: {e}")
            raise
    
    def _estimate_factor_model(self, returns_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Estimate multi-factor risk model"""
        try:
            # Simplified factor model using market factor
            # In practice, this would use Fama-French factors or custom factors
            
            # Market factor (equal-weighted portfolio)
            market_returns = returns_data.mean(axis=1)
            
            # Estimate betas for each asset
            betas = {}
            alphas = {}
            residual_vars = {}
            
            for asset in returns_data.columns:
                asset_returns = returns_data[asset].dropna()
                common_dates = asset_returns.index.intersection(market_returns.index)
                
                if len(common_dates) > 20:
                    y = asset_returns.loc[common_dates]
                    x = market_returns.loc[common_dates]
                    
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    betas[asset] = slope
                    alphas[asset] = intercept
                    
                    # Residual variance
                    predicted = intercept + slope * x
                    residuals = y - predicted
                    residual_vars[asset] = residuals.var()
                else:
                    betas[asset] = 1.0
                    alphas[asset] = 0.0
                    residual_vars[asset] = asset_returns.var()
            
            # Expected returns from factor model
            market_risk_premium = market_returns.mean() * 252
            expected_returns = pd.Series({
                asset: alphas[asset] * 252 + betas[asset] * market_risk_premium
                for asset in returns_data.columns
            })
            
            # Covariance matrix from factor model
            market_var = market_returns.var() * 252
            n_assets = len(returns_data.columns)
            
            covariance_matrix = np.zeros((n_assets, n_assets))
            assets = list(returns_data.columns)
            
            for i, asset_i in enumerate(assets):
                for j, asset_j in enumerate(assets):
                    if i == j:
                        # Diagonal: systematic + idiosyncratic risk
                        covariance_matrix[i, j] = (betas[asset_i] ** 2 * market_var + 
                                                 residual_vars[asset_i] * 252)
                    else:
                        # Off-diagonal: systematic risk only
                        covariance_matrix[i, j] = betas[asset_i] * betas[asset_j] * market_var
            
            covariance_matrix = pd.DataFrame(covariance_matrix, index=assets, columns=assets)
            
            return expected_returns, covariance_matrix
            
        except Exception as e:
            logger.error(f"Error estimating factor model: {e}")
            # Fallback to historical model
            expected_returns = returns_data.mean() * 252
            covariance_matrix = returns_data.cov() * 252
            return expected_returns, covariance_matrix
    
    def _validate_risk_model(self, expected_returns: pd.Series, 
                           covariance_matrix: pd.DataFrame) -> bool:
        """Validate risk model estimates"""
        try:
            # Check for NaN values
            if expected_returns.isna().any() or covariance_matrix.isna().any().any():
                return False
            
            # Check if covariance matrix is positive semi-definite
            eigenvalues = np.linalg.eigvals(covariance_matrix.values)
            if np.any(eigenvalues < -1e-8):  # Allow small numerical errors
                return False
            
            # Check reasonable ranges
            if (expected_returns.abs() > 2.0).any():  # >200% annual return seems unrealistic
                return False
                
            if (np.diag(covariance_matrix) < 0).any():  # Negative variances
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating risk model: {e}")
            return False
    
    def optimize_portfolio(self, expected_returns: pd.Series, 
                         covariance_matrix: pd.DataFrame,
                         constraints: OptimizationConstraints = None,
                         objective: str = 'max_sharpe',
                         risk_free_rate: float = 0.02,
                         current_weights: Dict[str, float] = None) -> OptimizationResult:
        """
        Optimize portfolio using specified objective and constraints
        
        Objectives:
        - 'max_sharpe': Maximize Sharpe ratio
        - 'min_variance': Minimize portfolio variance
        - 'max_return': Maximize expected return
        - 'risk_parity': Equal risk contribution
        - 'mean_reversion': Mean reversion strategy
        """
        try:
            constraints = constraints or OptimizationConstraints()
            current_weights = current_weights or {}
            
            assets = list(expected_returns.index)
            n_assets = len(assets)
            
            if n_assets == 0:
                raise ValueError("No assets provided for optimization")
            
            # Initial guess (equal weights or current weights)
            if current_weights:
                x0 = np.array([current_weights.get(asset, 1/n_assets) for asset in assets])
                x0 = x0 / x0.sum()  # Normalize
            else:
                x0 = np.ones(n_assets) / n_assets
            
            # Set up constraints
            scipy_constraints = []
            
            # Weights sum to 1
            scipy_constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
            
            # Weight bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Turnover constraint
            if current_weights and constraints.max_turnover:
                current_w = np.array([current_weights.get(asset, 0) for asset in assets])
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': lambda w: constraints.max_turnover - np.sum(np.abs(w - current_w))
                })
            
            # Target volatility constraint
            if constraints.target_volatility:
                scipy_constraints.append({
                    'type': 'eq',
                    'fun': lambda w: np.sqrt(np.dot(w, np.dot(covariance_matrix.values, w))) - constraints.target_volatility
                })
            
            # Minimum expected return constraint
            if constraints.min_expected_return:
                scipy_constraints.append({
                    'type': 'ineq',
                    'fun': lambda w: np.dot(w, expected_returns.values) - constraints.min_expected_return
                })
            
            # Define objective function
            def objective_function(weights):
                portfolio_return = np.dot(weights, expected_returns.values)
                portfolio_var = np.dot(weights, np.dot(covariance_matrix.values, weights))
                portfolio_vol = np.sqrt(portfolio_var)
                
                if objective == 'max_sharpe':
                    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
                    return -sharpe  # Minimize negative Sharpe
                    
                elif objective == 'min_variance':
                    return portfolio_var
                    
                elif objective == 'max_return':
                    return -portfolio_return
                    
                elif objective == 'risk_parity':
                    # Risk parity: minimize sum of squared risk contributions
                    risk_contributions = weights * np.dot(covariance_matrix.values, weights)
                    target_risk_contrib = portfolio_var / n_assets
                    return np.sum((risk_contributions - target_risk_contrib) ** 2)
                    
                else:
                    raise ValueError(f"Unknown objective: {objective}")
            
            # Optimize
            try:
                result = minimize(
                    objective_function,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=scipy_constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
                
                if not result.success:
                    # Try with differential evolution as fallback
                    result = differential_evolution(
                        objective_function,
                        bounds,
                        maxiter=100,
                        seed=42
                    )
                    
                    # Apply constraints manually for differential evolution
                    if scipy_constraints:
                        weights = result.x
                        weights = weights / weights.sum()  # Normalize
                        result.x = weights
                
                optimized_weights = result.x
                
            except Exception as e:
                logger.warning(f"Optimization failed: {e}. Using equal weights.")
                optimized_weights = np.ones(n_assets) / n_assets
                result = type('Result', (), {'success': False, 'message': str(e)})()
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimized_weights, expected_returns.values)
            portfolio_var = np.dot(optimized_weights, np.dot(covariance_matrix.values, optimized_weights))
            portfolio_vol = np.sqrt(portfolio_var)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(covariance_matrix.values))
            weighted_avg_vol = np.dot(optimized_weights, individual_vols)
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
            
            # Turnover calculation
            if current_weights:
                current_w = np.array([current_weights.get(asset, 0) for asset in assets])
                turnover = np.sum(np.abs(optimized_weights - current_w))
            else:
                turnover = 1.0  # Full turnover if no current weights
            
            # Max drawdown estimate (simplified)
            max_drawdown_estimate = -2 * portfolio_vol  # Rule of thumb
            
            # Check constraint satisfaction
            constraints_satisfied = self._check_constraints_satisfied(
                optimized_weights, constraints, assets, current_weights
            )
            
            # Convert weights to dictionary
            weights_dict = dict(zip(assets, optimized_weights))
            
            optimization_result = OptimizationResult(
                weights=weights_dict,
                expected_return=float(portfolio_return),
                expected_volatility=float(portfolio_vol),
                sharpe_ratio=float(sharpe_ratio),
                max_drawdown_estimate=float(max_drawdown_estimate),
                diversification_ratio=float(diversification_ratio),
                turnover=float(turnover),
                objective_value=float(result.fun) if hasattr(result, 'fun') else 0.0,
                success=bool(getattr(result, 'success', True)),
                message=getattr(result, 'message', 'Optimization completed'),
                constraints_satisfied=constraints_satisfied
            )
            
            # Store optimization result
            self._store_optimization_result(optimization_result, objective, constraints)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return OptimizationResult(
                weights={}, expected_return=0.0, expected_volatility=0.0,
                sharpe_ratio=0.0, max_drawdown_estimate=0.0, diversification_ratio=0.0,
                turnover=0.0, objective_value=0.0, success=False, message=str(e),
                constraints_satisfied=False
            )
    
    def _check_constraints_satisfied(self, weights: np.ndarray, 
                                   constraints: OptimizationConstraints,
                                   assets: List[str],
                                   current_weights: Dict[str, float] = None) -> bool:
        """Check if optimization result satisfies all constraints"""
        try:
            # Weight bounds
            if np.any(weights < constraints.min_weight - 1e-6) or np.any(weights > constraints.max_weight + 1e-6):
                return False
            
            # Sum to 1
            if abs(weights.sum() - 1.0) > 1e-6:
                return False
            
            # Turnover constraint
            if current_weights and constraints.max_turnover:
                current_w = np.array([current_weights.get(asset, 0) for asset in assets])
                turnover = np.sum(np.abs(weights - current_w))
                if turnover > constraints.max_turnover + 1e-6:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking constraints: {e}")
            return False
    
    def efficient_frontier(self, expected_returns: pd.Series, 
                         covariance_matrix: pd.DataFrame,
                         n_points: int = 50,
                         constraints: OptimizationConstraints = None) -> Dict[str, List]:
        """Generate efficient frontier points"""
        try:
            constraints = constraints or OptimizationConstraints()
            
            # Range of target returns
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            target_returns = np.linspace(min_return, max_return, n_points)
            
            frontier_points = {
                'returns': [],
                'volatilities': [],
                'sharpe_ratios': [],
                'weights': []
            }
            
            for target_return in target_returns:
                # Add minimum return constraint
                target_constraints = OptimizationConstraints(
                    max_weight=constraints.max_weight,
                    min_weight=constraints.min_weight,
                    max_sector_weight=constraints.max_sector_weight,
                    max_turnover=constraints.max_turnover,
                    target_volatility=constraints.target_volatility,
                    min_expected_return=target_return,
                    beta_range=constraints.beta_range
                )
                
                # Optimize for minimum variance at target return
                result = self.optimize_portfolio(
                    expected_returns, covariance_matrix,
                    constraints=target_constraints,
                    objective='min_variance'
                )
                
                if result.success:
                    frontier_points['returns'].append(result.expected_return)
                    frontier_points['volatilities'].append(result.expected_volatility)
                    frontier_points['sharpe_ratios'].append(result.sharpe_ratio)
                    frontier_points['weights'].append(result.weights)
                    
            return frontier_points
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            return {'returns': [], 'volatilities': [], 'sharpe_ratios': [], 'weights': []}
    
    def _store_risk_model(self, risk_model: Dict[str, Any]):
        """Store risk model in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO risk_models 
                    (model_name, model_date, covariance_matrix, expected_returns, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    risk_model['method'],
                    datetime.now().date(),
                    json.dumps(risk_model['covariance_matrix'].to_dict()),
                    json.dumps(risk_model['expected_returns'].to_dict()),
                    json.dumps({
                        'estimation_date': risk_model['estimation_date'].isoformat(),
                        'data_window': risk_model['data_window'],
                        'assets': risk_model['assets']
                    })
                ))
                
        except Exception as e:
            logger.error(f"Error storing risk model: {e}")
    
    def _store_optimization_result(self, result: OptimizationResult, 
                                 objective: str, constraints: OptimizationConstraints):
        """Store optimization result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO optimization_results 
                    (portfolio_id, optimization_date, optimization_method, objective_function,
                     weights, expected_return, expected_volatility, sharpe_ratio, turnover,
                     constraints, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    'default_portfolio',
                    datetime.now().date(),
                    'scipy_optimize',
                    objective,
                    json.dumps(result.weights),
                    result.expected_return,
                    result.expected_volatility,
                    result.sharpe_ratio,
                    result.turnover,
                    json.dumps({
                        'max_weight': constraints.max_weight,
                        'min_weight': constraints.min_weight,
                        'max_turnover': constraints.max_turnover
                    }),
                    json.dumps({
                        'success': result.success,
                        'message': result.message,
                        'constraints_satisfied': result.constraints_satisfied,
                        'diversification_ratio': result.diversification_ratio
                    })
                ))
                
        except Exception as e:
            logger.error(f"Error storing optimization result: {e}")
    
    def get_optimization_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get optimization history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                history = pd.read_sql_query('''
                    SELECT * FROM optimization_results 
                    WHERE optimization_date >= ?
                    ORDER BY optimization_date DESC
                ''', conn, params=(cutoff_date,))
            
            return history.to_dict('records') if not history.empty else []
            
        except Exception as e:
            logger.error(f"Error getting optimization history: {e}")
            return []