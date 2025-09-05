"""
Automated Reporting and Visualization Package
Comprehensive trading system reporting with performance metrics and visualizations
"""

from .report_generator import (
    ReportGenerator,
    ReportMetrics,
    ReportConfig,
    generate_report,
    generate_daily_summary
)

__all__ = [
    'ReportGenerator',
    'ReportMetrics', 
    'ReportConfig',
    'generate_report',
    'generate_daily_summary'
]