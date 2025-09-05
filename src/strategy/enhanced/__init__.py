"""
Enhanced Signal Strategy Components

This package contains academic research-backed enhancements to the signal scoring system:
- Enhanced ensemble signal scoring with regime awareness
- Market regime detection and classification
- Macro-economic signal integration
- Dynamic position sizing with Kelly criterion
- Factor timing and attribution analysis
- Transaction cost optimization
"""

from .enhanced_ensemble_signal_scoring import EnhancedEnsembleSignalScoring
from .regime_detection import AdvancedRegimeDetector, MarketRegime
from .macro_integration import MacroIntegrationEngine, MacroRegime, FedPolicyStance
from .position_sizing import DynamicPositionSizer, PositionSizingMethod

__version__ = "1.0.0"
__author__ = "Signal Enhancement Team"

__all__ = [
    "EnhancedEnsembleSignalScoring",
    "AdvancedRegimeDetector", 
    "MarketRegime",
    "MacroIntegrationEngine",
    "MacroRegime", 
    "FedPolicyStance",
    "DynamicPositionSizer",
    "PositionSizingMethod"
]