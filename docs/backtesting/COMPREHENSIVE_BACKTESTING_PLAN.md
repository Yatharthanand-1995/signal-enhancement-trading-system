# 🎯 Comprehensive Backtesting Plan - Signal Logic Validation

**Objective:** Validate our signal logic performance against Top 100 US stocks across different market conditions (2020-2025)

**Date:** August 30, 2025  
**Status:** PLANNING PHASE  
**Expected Duration:** 5-7 days implementation + analysis

---

## 📊 **BACKTESTING SCOPE & OBJECTIVES**

### **Primary Objectives:**
1. **Validate signal accuracy** across different market regimes
2. **Measure risk-adjusted returns** vs benchmarks (S&P 500, Buy & Hold)
3. **Analyze performance** by market condition (bull, bear, volatile, trending)
4. **Optimize signal parameters** based on historical performance
5. **Stress test** our ML component and environment filters

### **Success Metrics:**
- **Target Sharpe Ratio:** >1.5 (vs S&P 500 ~1.2)
- **Target Win Rate:** >65% (our research suggests 64-72%)
- **Target Max Drawdown:** <25% (vs market drawdowns)
- **Target Alpha:** >5% annual vs S&P 500
- **Target Signal Accuracy:** >70% directional accuracy

---

## 🕐 **MARKET REGIMES ANALYSIS (2020-2025)**

### **Regime 1: COVID CRASH (Feb-Mar 2020)**
- **Duration:** 6 weeks
- **Market Condition:** Extreme volatility, VIX >75, -35% drawdown
- **Test Focus:** How our volatility filters and dynamic thresholds performed
- **Expected Behavior:** Minimal buy signals, strong sell signal accuracy

### **Regime 2: COVID RECOVERY (Apr 2020 - Dec 2021)**
- **Duration:** 20 months
- **Market Condition:** Bull market, low VIX, Fed stimulus, +80% gains
- **Test Focus:** ML component performance in trending markets
- **Expected Behavior:** Strong buy signal performance, momentum capture

### **Regime 3: INFLATION PERIOD (Jan 2021 - Nov 2021)**
- **Duration:** 11 months
- **Market Condition:** Rising rates, sector rotation, growth→value shift
- **Test Focus:** Sector multiplier effectiveness, breadth filters
- **Expected Behavior:** Mixed signals, sector-dependent performance

### **Regime 4: TAPER TANTRUM & BEAR MARKET (Dec 2021 - Oct 2022)**
- **Duration:** 10 months
- **Market Condition:** Rising rates, -25% drawdown, poor breadth
- **Test Focus:** Bear market signal accuracy, drawdown control
- **Expected Behavior:** Defensive positioning, sell signal accuracy

### **Regime 5: FED PIVOT RECOVERY (Nov 2022 - Dec 2023)**
- **Duration:** 13 months
- **Market Condition:** Recovery rally, narrow leadership (Tech), +25% gains
- **Test Focus:** Breadth filters during narrow rallies
- **Expected Behavior:** Selective buy signals, avoid false breakouts

### **Regime 6: AI BOOM & CURRENT (Jan 2024 - Present)**
- **Duration:** 20 months
- **Market Condition:** AI-driven rally, concentrated gains, high valuations
- **Test Focus:** Valuation filters, momentum vs mean reversion balance
- **Expected Behavior:** Cautious signals in overvalued names

---

## 🏗️ **BACKTESTING FRAMEWORK ARCHITECTURE**

### **1. Data Infrastructure**
```python
# Historical Data Requirements
data_sources = {
    "price_data": "Yahoo Finance API (OHLCV + splits/dividends)",
    "fundamental_data": "Calculated market cap, P/E ratios",
    "technical_indicators": "RSI, MACD, Bollinger Bands, Volume ratios",
    "market_environment": "VIX, AAII sentiment, Put/Call ratios, yield curves",
    "sector_data": "GICS sector classifications",
    "benchmark_data": "SPY, QQQ, sector ETFs performance"
}

# Data Frequency
frequency = "Daily" # Signal calculations
rebalance_frequency = "Weekly" # Portfolio updates
```

### **2. Signal Calculation Engine**
```python
# Core Components (from our current dashboard logic)
components = {
    "individual_signals": ["RSI", "MACD", "Volume", "BB", "MA", "Momentum", "Volatility"],
    "ml_component": "Meta-signal combining all indicators (20% weight)",
    "market_environment": ["VIX_filter", "breadth_filter", "sentiment_filter"],
    "regime_adjustments": ["sector_multipliers", "market_cap_adjustments"],
    "dynamic_thresholds": "VIX and breadth-based threshold adjustments"
}
```

### **3. Portfolio Simulation Framework**
```python
# Portfolio Parameters
portfolio_config = {
    "initial_capital": 1000000,  # $1M starting capital
    "max_positions": 20,         # Maximum 20 positions
    "position_sizing": "Equal weight with volatility adjustment",
    "transaction_costs": 0.001,  # 10 bps per trade
    "rebalancing": "Weekly",
    "cash_management": "Hold cash when no signals"
}
```

---

## 🔢 **DETAILED TESTING METHODOLOGY**

### **Phase 1: Historical Data Preparation (Day 1-2)**

#### **1.1 Stock Universe Definition**
```python
# Top 100 US Stocks by Market Cap (as of each historical period)
stock_selection_criteria = {
    "market_cap": ">$10B (large cap focus)",
    "liquidity": "Average daily volume >1M shares", 
    "sector_distribution": "Balanced across major sectors",
    "survivorship_bias": "Include delisted stocks that were Top 100"
}
```

#### **1.2 Data Collection Strategy**
- **Price Data:** OHLCV data from yfinance for all 100+ stocks
- **Market Environment:** VIX, AAII sentiment, advance/decline ratios
- **Fundamental Data:** Market cap, sector classifications
- **Quality Checks:** Handle splits, dividends, missing data

#### **1.3 Technical Indicator Calculation**
- Replicate exact dashboard logic for all indicators
- Calculate market environment metrics historically
- Validate against known historical values

### **Phase 2: Market Regime Classification (Day 2)**

#### **2.1 Regime Detection Algorithm**
```python
def classify_market_regime(date, vix, market_return, breadth):
    if vix > 30 and market_return < -15:
        return "CRASH"
    elif vix < 15 and market_return > 15:
        return "BULL_MARKET"
    elif vix > 25:
        return "HIGH_VOLATILITY"
    elif breadth < 0.4:
        return "POOR_BREADTH"
    else:
        return "NORMAL"
```

#### **2.2 Regime Validation**
- Map historical periods to regime classifications
- Validate against known market events
- Ensure comprehensive coverage of all market conditions

### **Phase 3: Signal Generation Engine (Day 3)**

#### **3.1 Historical Signal Calculation**
```python
# For each stock, each day:
def generate_historical_signals(stock_data, market_env, date):
    # 1. Calculate individual indicators (exact dashboard logic)
    individual_signals = calculate_individual_signals(stock_data, market_env)
    
    # 2. Calculate ML component
    ml_signal = calculate_ml_component(individual_signals)
    
    # 3. Apply market environment filters
    filtered_score = apply_environment_filters(raw_score, market_env)
    
    # 4. Apply dynamic thresholds
    final_signal = apply_dynamic_thresholds(filtered_score, market_env)
    
    return final_signal
```

#### **3.2 Signal Validation**
- Compare historical signals against known market events
- Verify signal distribution matches expected patterns
- Check for look-ahead bias and data leakage

### **Phase 4: Portfolio Simulation (Day 3-4)**

#### **4.1 Trading Logic Implementation**
```python
def execute_trading_logic(signals, portfolio, date):
    # Entry Logic
    buy_signals = signals[signals['signal'].isin(['BUY', 'STRONG_BUY'])]
    buy_signals = buy_signals.sort_values('confidence', ascending=False)
    
    # Position Sizing
    for signal in buy_signals:
        position_size = calculate_position_size(
            capital=available_cash,
            volatility=signal['volatility'],
            confidence=signal['confidence'],
            max_position_pct=0.05  # 5% max per position
        )
    
    # Exit Logic
    exit_positions = check_exit_conditions(portfolio, current_signals)
    
    return updated_portfolio
```

#### **4.2 Risk Management Rules**
- **Stop Losses:** 10% initial, trailing stops for profits
- **Position Limits:** Max 5% per position, 25% per sector
- **Cash Management:** Hold cash when signals < threshold
- **Rebalancing:** Weekly position size adjustments

### **Phase 5: Performance Analysis (Day 4-5)**

#### **5.1 Core Performance Metrics**
```python
performance_metrics = {
    # Return Metrics
    "total_return": "Portfolio total return vs benchmarks",
    "annualized_return": "CAGR over full period",
    "excess_return": "Alpha vs S&P 500",
    
    # Risk Metrics  
    "sharpe_ratio": "Risk-adjusted return measure",
    "max_drawdown": "Maximum peak-to-trough decline",
    "volatility": "Annualized standard deviation",
    "beta": "Market sensitivity",
    
    # Trading Metrics
    "win_rate": "% of profitable trades",
    "average_holding_period": "Days per position",
    "turnover": "Portfolio turnover rate",
    "transaction_costs": "Total trading costs impact"
}
```

#### **5.2 Regime-Specific Analysis**
```python
def analyze_by_regime(portfolio_returns, market_regimes):
    regime_analysis = {}
    
    for regime in ["CRASH", "BULL_MARKET", "HIGH_VOLATILITY", "POOR_BREADTH", "NORMAL"]:
        regime_periods = market_regimes[market_regimes == regime]
        regime_returns = portfolio_returns[regime_periods.index]
        
        regime_analysis[regime] = {
            "return": regime_returns.sum(),
            "sharpe": calculate_sharpe(regime_returns),
            "max_drawdown": calculate_max_drawdown(regime_returns),
            "win_rate": calculate_win_rate(regime_returns)
        }
    
    return regime_analysis
```

---

## 📈 **EXPECTED RESULTS & HYPOTHESES**

### **Hypothesis 1: Signal Accuracy by Regime**
```python
expected_performance = {
    "COVID_CRASH": {
        "signal_accuracy": 0.80,  # Strong defensive signals
        "max_drawdown": -15,      # Better than market (-35%)
        "notes": "Volatility filters should prevent false buy signals"
    },
    "BULL_MARKET": {
        "signal_accuracy": 0.70,  # Good trend following
        "excess_return": 8,       # Should outperform via momentum
        "notes": "ML component should capture sustained trends"
    },
    "BEAR_MARKET": {
        "signal_accuracy": 0.75,  # Good sell signal accuracy  
        "max_drawdown": -18,      # Better than market (-25%)
        "notes": "Dynamic thresholds should reduce long exposure"
    },
    "HIGH_VOLATILITY": {
        "signal_accuracy": 0.65,  # More challenging environment
        "sharpe_ratio": 1.2,      # Risk-adjusted outperformance
        "notes": "VIX filters should reduce position sizes"
    }
}
```

### **Hypothesis 2: Component Effectiveness**
- **RSI Component (15%):** Most effective in volatile markets
- **ML Component (20%):** Best in trending markets
- **Volume Analysis (12%):** Critical for breakout confirmation
- **Environment Filters:** Should reduce drawdowns by 30-40%

### **Hypothesis 3: Benchmark Comparison**
```python
benchmark_expectations = {
    "vs_SPY": {
        "excess_return": "5-8% annually",
        "sharpe_improvement": "0.3-0.5 points",
        "max_drawdown_reduction": "20-30%"
    },
    "vs_Equal_Weight": {
        "excess_return": "3-5% annually", 
        "reduced_volatility": "10-15%"
    },
    "vs_Buy_Hold": {
        "risk_adjusted_return": "Significantly better Sharpe ratio",
        "downside_protection": "Better performance in bear markets"
    }
}
```

---

## 🛠️ **IMPLEMENTATION PLAN**

### **Day 1: Data Infrastructure Setup**
- [ ] Set up yfinance data collection pipeline
- [ ] Identify Top 100 stocks for each historical year
- [ ] Download and clean 5 years of OHLCV data
- [ ] Calculate all technical indicators historically
- [ ] Gather market environment data (VIX, sentiment, etc.)

### **Day 2: Signal Engine Development**  
- [ ] Port dashboard signal logic to backtesting framework
- [ ] Implement historical market environment calculation
- [ ] Create regime classification system
- [ ] Validate signal generation against known periods

### **Day 3: Portfolio Simulation**
- [ ] Build portfolio management system
- [ ] Implement position sizing and risk management
- [ ] Create transaction cost modeling
- [ ] Test trading logic with sample data

### **Day 4: Backtesting Execution**
- [ ] Run full 5-year backtest across all regimes
- [ ] Generate trade-by-trade analysis
- [ ] Calculate all performance metrics
- [ ] Create regime-specific performance breakdowns

### **Day 5: Analysis & Reporting**
- [ ] Generate comprehensive performance report
- [ ] Create visualizations for key metrics
- [ ] Perform sensitivity analysis on parameters
- [ ] Document findings and recommendations

### **Days 6-7: Optimization & Validation**
- [ ] Optimize signal parameters based on results
- [ ] Run walk-forward analysis for robustness
- [ ] Validate results against out-of-sample data
- [ ] Final report and recommendations

---

## 📊 **DELIVERABLES**

### **1. Comprehensive Performance Report**
- Executive summary with key findings
- Detailed performance metrics by regime
- Benchmark comparisons with statistical significance
- Risk analysis and drawdown periods

### **2. Interactive Dashboard**
- Performance visualization by time period
- Signal accuracy heatmaps by market condition
- Portfolio composition evolution over time
- Risk metrics and attribution analysis

### **3. Optimization Recommendations**
- Parameter tuning suggestions based on results
- Regime-specific adjustments to improve performance
- Risk management enhancements
- Future research directions

### **4. Implementation Code**
- Complete backtesting framework (reusable)
- Signal generation engine (production-ready)
- Portfolio simulation system
- Performance analysis tools

---

## 🎯 **SUCCESS CRITERIA**

### **Minimum Acceptable Performance:**
- **Sharpe Ratio:** >1.3 (vs SPY ~1.2)
- **Max Drawdown:** <30% (vs SPY worst ~35%)
- **Win Rate:** >60% of trades profitable
- **Annual Alpha:** >3% vs SPY

### **Target Performance:**
- **Sharpe Ratio:** >1.5
- **Max Drawdown:** <25%
- **Win Rate:** >65%
- **Annual Alpha:** >5% vs SPY
- **Downside Capture:** <80% of market declines

### **Exceptional Performance:**
- **Sharpe Ratio:** >1.7
- **Max Drawdown:** <20%
- **Win Rate:** >70%
- **Annual Alpha:** >8% vs SPY

---

## ⚠️ **RISK FACTORS & LIMITATIONS**

### **Data Quality Risks:**
- Survivorship bias in stock selection
- Data quality issues (splits, dividends)
- Look-ahead bias in indicator calculations

### **Methodology Risks:**
- Overfitting to historical data
- Transaction cost assumptions
- Liquidity constraints not modeled

### **Market Structure Changes:**
- Algorithm trading evolution
- Market microstructure changes
- Regulatory environment shifts

### **Mitigation Strategies:**
- Out-of-sample validation periods
- Walk-forward analysis approach
- Conservative transaction cost assumptions
- Regime-specific parameter testing

---

**Next Steps:** Begin implementation with Day 1 data infrastructure setup.

---

*Plan Created: August 30, 2025*  
*Expected Completion: September 6, 2025*  
*Status: READY FOR IMPLEMENTATION* 🚀