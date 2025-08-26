# 🚀 Advanced US Stock Trading System

A comprehensive 2-15 day trading system for the top 100 US stocks, combining machine learning, technical analysis, and adaptive risk management.

## 📋 System Overview

This trading system implements a sophisticated approach combining:

- **LSTM-XGBoost Ensemble ML Models** for price prediction (targeting 93%+ accuracy)
- **Technical Indicators** optimized for medium-term trading (RSI, MACD, Bollinger Bands)
- **Hidden Markov Model Regime Detection** for adaptive strategy parameters
- **Dynamic Risk Management** with Kelly Criterion position sizing
- **Comprehensive Backtesting** with walk-forward optimization
- **Real-time Dashboard** for monitoring and signal generation

## 🏗️ Architecture

```
├── src/
│   ├── data_management/         # Data fetching and technical indicators
│   ├── models/                  # ML models and regime detection
│   ├── risk_management/         # Portfolio and risk management
│   ├── backtesting/            # Backtesting framework
│   ├── strategy/               # Trading strategies
│   └── dashboard/              # Streamlit dashboard
├── database/                   # PostgreSQL schemas and scripts
├── config/                     # Configuration files
├── scripts/                    # Utility and setup scripts
├── tests/                      # Testing framework
└── logs/                       # System logs
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- 8GB+ RAM recommended

### 1. Clone and Setup Environment

```bash
git clone <repository>
cd "SIgnal - US"

# Create virtual environment
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Database Services

```bash
# Start PostgreSQL and Redis
docker-compose up -d

# Wait for services to be ready
docker exec trading_postgres pg_isready
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys (optional for basic functionality)
# ALPHA_VANTAGE_KEY=your_key_here
# POLYGON_KEY=your_key_here
```

### 4. Initialize System

```bash
# Run system initialization
python scripts/initialize_system.py
```

This will:
- Create and populate database tables
- Fetch historical data for top 10 stocks (sample)
- Calculate technical indicators
- Validate system functionality

### 5. Start Trading System

```bash
# Start the dashboard
streamlit run src/dashboard/main.py

# Or run individual components
python src/models/ml_ensemble.py  # Train ML models
python src/models/regime_detection.py  # Test regime detection
```

## 📊 Key Features

### 1. Data Management
- **Top 100 US stocks** by market cap
- **Real-time data updates** from yfinance
- **Technical indicators** calculated and stored
- **Data quality monitoring** and validation

### 2. Machine Learning Models
- **LSTM Neural Network** for sequential pattern recognition
- **XGBoost** for feature-based prediction
- **Ensemble combination** for improved accuracy
- **Feature engineering** with 50+ technical features

### 3. Market Regime Detection
- **2-state HMM** for volatility regimes (Low/High)
- **3-state HMM** for market conditions (Bull/Bear/Sideways)
- **Adaptive parameters** based on detected regime
- **Risk adjustment** for different market conditions

### 4. Risk Management
- **Kelly Criterion** position sizing
- **Dynamic stop losses** based on ATR
- **Portfolio heat** monitoring
- **Sector concentration** limits
- **Maximum drawdown** protection

### 5. Performance Monitoring
- **Real-time dashboard** with signal tracking
- **Performance metrics** (Sharpe, Calmar ratios)
- **Risk analytics** (VaR, Expected Shortfall)
- **Trade execution** monitoring

## 🔧 Configuration

### Trading Parameters

Edit `config/config.py` to adjust:

```python
# Position sizing
max_position_size: float = 0.25  # 25% max per position
min_position_size: float = 0.02  # 2% minimum

# Risk management
max_drawdown: float = 0.15       # 15% max drawdown
volatility_target: float = 0.12  # 12% target volatility
stop_loss_atr: float = 2.0       # 2 ATR stop loss

# Signal thresholds
min_signal_strength: float = 0.6  # Minimum signal to trade
min_confidence: float = 0.65      # Minimum confidence to trade
```

### ML Model Parameters

```python
# LSTM configuration
lstm_units: list = [250, 200, 150, 50]
dropout_rate: float = 0.2
batch_size: int = 32
epochs: int = 50

# XGBoost configuration
n_estimators: int = 1000
max_depth: int = 10
learning_rate: float = 0.06
```

## 📈 Usage Examples

### 1. Generate Trading Signals

```python
from src.data_management.technical_indicators import TechnicalIndicatorCalculator
from src.models.ml_ensemble import LSTMXGBoostEnsemble

# Calculate indicators
calculator = TechnicalIndicatorCalculator()
features = calculator.generate_signal_features('AAPL')

# Get ML prediction
ensemble = LSTMXGBoostEnsemble()
prediction, confidence, explanation = ensemble.predict_single(stock_data)
```

### 2. Risk Management

```python
from src.risk_management.risk_manager import AdaptiveRiskManager

# Initialize risk manager
risk_manager = AdaptiveRiskManager(initial_capital=100000)

# Calculate position size
position_info = risk_manager.calculate_position_size(
    symbol='AAPL',
    signal_strength=0.8,
    confidence=0.75,
    current_price=150.0,
    atr=3.0,
    regime='Low_Volatility'
)
```

### 3. Market Regime Detection

```python
from src.models.regime_detection import MarketRegimeDetector

# Detect current regime
detector = MarketRegimeDetector(n_regimes=2)
detector.fit(market_data)
current_regime = detector.predict_regime(recent_data)

# Get trading adjustments
adjustments = detector.get_trading_adjustments()
```

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **Real-time signal monitoring** for all stocks
- **Performance tracking** with equity curves
- **Risk metrics** visualization
- **Regime detection** status
- **Position management** interface
- **Backtesting results** analysis

Access at: `http://localhost:8501`

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_data_management.py -v
pytest tests/test_models.py -v
pytest tests/test_risk_management.py -v
```

## 🔄 Maintenance

### Daily Operations

```bash
# Update stock data (run daily)
python scripts/daily_update.py

# Monitor system health
python scripts/system_health_check.py
```

### Weekly Operations

```bash
# Update top 100 stocks list
python scripts/update_stock_universe.py

# Retrain ML models
python scripts/retrain_models.py
```

### Setup Cron Jobs

```bash
# Daily data update at 7 AM
0 7 * * * /path/to/scripts/daily_update.sh

# Weekly model retraining on Sundays at 2 AM
0 2 * * 0 /path/to/scripts/weekly_maintenance.sh
```

## 📝 Performance Expectations

Based on academic research and backtesting:

- **Expected Annual Return**: 20-30%
- **Target Sharpe Ratio**: 2.0-3.0
- **Maximum Drawdown**: <15%
- **Win Rate**: 65-75%
- **Holding Period**: 2-15 days average

## ⚠️ Risk Disclaimers

- **Educational Purpose**: This system is for educational and research purposes
- **No Financial Advice**: Not intended as financial or investment advice
- **Risk of Loss**: Trading involves substantial risk of loss
- **Paper Trading**: Test thoroughly with paper trading before live deployment
- **Regulatory Compliance**: Ensure compliance with local trading regulations

## 🛟 Support and Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check if PostgreSQL is running
   docker ps
   # Restart if needed
   docker-compose restart postgres
   ```

2. **Insufficient Data**
   ```bash
   # Re-run data initialization
   python scripts/initialize_system.py
   ```

3. **ML Model Training Failed**
   ```bash
   # Check system resources (8GB+ RAM recommended)
   # Reduce batch size in config if needed
   ```

### Logs

System logs are stored in:
- `logs/system_init.log` - System initialization
- `logs/daily_update.log` - Daily data updates
- `logs/trading_system.log` - General system logs

## 🔮 Future Enhancements

- **Alternative data sources** (news sentiment, options flow)
- **Portfolio optimization** algorithms
- **Multi-timeframe analysis**
- **Reinforcement learning** models
- **Real-time broker integration**
- **Cloud deployment** options

## 📚 References

This system is built on academic research including:

- LSTM for Financial Time Series Prediction
- Hidden Markov Models for Regime Detection
- Kelly Criterion for Optimal Position Sizing
- Technical Analysis for Short-term Trading
- Risk Parity and Volatility Targeting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Disclaimer**: This software is for educational purposes only. The authors are not responsible for any financial losses incurred from using this system. Always conduct thorough testing and risk assessment before deploying any trading system.