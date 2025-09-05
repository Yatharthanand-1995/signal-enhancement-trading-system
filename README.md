# 🎯 Signal-US Trading Intelligence System

## 📊 Overview

A comprehensive, production-ready trading signal generation and analysis system for US equity markets. Features real-time data processing, advanced technical analysis, and an intuitive Streamlit dashboard for 100+ stocks.

## ✨ Key Features

- 🔴 **Live Data Feeds**: Real-time price data for 100 stocks
- 📈 **Technical Analysis**: 29 technical indicators (RSI, MACD, Bollinger Bands, SMA, etc.)
- 🌍 **Market Environment**: VIX, Fear & Greed Index, Market Breadth analysis
- 🎯 **Signal Generation**: Advanced signal weighting and confidence scoring
- 📊 **Interactive Dashboard**: Streamlit-based real-time visualization
- 🔄 **Backtesting Framework**: Historical performance validation
- ⚡ **High Performance**: Parallel data fetching with 15-25 workers

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run src/dashboard/main.py --server.port 8504
```

### 3. Access Dashboard
Open your browser to: **http://localhost:8504**

## 🏗️ System Architecture

```
src/
├── dashboard/main.py           # Main Streamlit dashboard (3,609 lines)
├── strategies/                 # Signal generation algorithms
├── utils/                      # Data management & utilities
├── backtesting/               # Performance validation
└── signals/                   # Signal processing engine

docs/                          # 📚 Organized documentation
├── project/                   # Project summaries & status
├── dashboard/                 # Dashboard documentation
├── backtesting/              # Backtesting guides
├── phases/                   # Development phase history
└── research/                 # Research findings & analysis

archive/                       # 🗃️ Archived files
├── old_files/                # Deprecated logs & configs
└── old_scripts/              # Legacy test scripts
```

## 💻 Current System Status

- ✅ **Dashboard**: Fully operational Streamlit interface with optimized performance
- ✅ **Data Pipeline**: 100 stocks with 1,256 days historical + live feeds
- ✅ **Technical Indicators**: 29 indicators calculated per stock
- ✅ **Market Analysis**: Real-time market environment assessment
- ✅ **Performance**: Optimized with parallel processing and session caching
- ✅ **Clean Architecture**: Dash removed, Streamlit-only implementation
- ✅ **Error Resolution**: All critical dashboard errors fixed (Sept 2025)
- ✅ **Redis Integration**: Proper caching with error-free authentication

## 🔧 Configuration

System configuration is managed through:
- `config/config.py` - Core system settings
- `src/utils/` - Data management configuration
- `requirements.txt` - Python dependencies

## 📚 Documentation

All documentation is organized in the `docs/` folder:
- **Getting Started**: See `docs/dashboard/FINAL_DASHBOARD_DOCUMENTATION.md`
- **Architecture**: See `docs/dashboard/DASHBOARD_ARCHITECTURE_ANALYSIS.md`  
- **Project Status**: See `docs/project/PROJECT_COMPLETION_SUMMARY.md`
- **Research Findings**: See `docs/research/` folder

## 🎯 Key Scripts

- `start_dashboard.py` - Dashboard launcher
- `run_comprehensive_backtest.py` - Full backtesting suite
- `run_quick_backtest.py` - Quick performance validation

## 🏆 Performance Highlights

- **Data Loading**: ~30-40 seconds for 100 stocks with full technical analysis
- **Dashboard Response**: Real-time updates with 5-minute refresh intervals
- **Memory Efficiency**: Optimized data structures and caching
- **Scalability**: Parallel processing architecture

## 🔧 Troubleshooting

### Common Issues & Solutions

**Import Path Errors**
```bash
# If you see "ModuleNotFoundError", ensure proper project structure:
cd /path/to/SIgnal-US
source trading_env/bin/activate
cd src/dashboard
streamlit run main.py --server.port 8504
```

**Redis Authentication Issues**
```bash
# Redis authentication errors are resolved in latest version
# If still experiencing issues, check config/enhanced_config.py
```

**Dashboard Won't Start**
```bash
# Check if port is already in use:
lsof -i :8504
# Kill existing process if needed:
kill -9 <PID>
# Then restart dashboard
```

**Performance Issues**
- Dashboard now uses session caching for heavy objects
- EnsembleSignalScorer and HistoricalDataManager are cached automatically
- Clear browser cache if experiencing slow loads

## 📈 Recent Improvements

- **September 2025**: Critical dashboard error fixes and performance optimization
  - Fixed import path issues in main dashboard
  - Resolved Redis authentication errors
  - Added session state caching for heavy objects
  - Project structure cleanup and organization
- **September 2025**: Complete Dash migration reversal to Streamlit-only
- **August 2025**: Project reorganization and documentation cleanup  
- **August 2025**: Enhanced technical indicator suite and live data integration
- **August 2025**: Advanced backtesting framework implementation

---

**🚀 Ready to analyze 100+ stocks with comprehensive technical analysis and real-time market insights!**

*Last updated: September 2025*