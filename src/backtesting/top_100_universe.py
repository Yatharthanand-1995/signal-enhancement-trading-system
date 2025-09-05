"""
Top 100 US Stocks Universe Selection
Selects liquid, diversified universe for comprehensive backtesting
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import requests
import json
import logging

logger = logging.getLogger(__name__)

class Top100Universe:
    """Manages selection and maintenance of Top 100 US stocks universe"""
    
    def __init__(self):
        self.universe = []
        self.sector_allocations = {
            'Technology': 25,      # 25%
            'Health Care': 15,     # 15%
            'Financials': 15,      # 15%
            'Consumer Discretionary': 12,  # 12%
            'Communication Services': 8,   # 8%
            'Industrials': 8,      # 8%
            'Consumer Staples': 5, # 5%
            'Energy': 4,           # 4%
            'Utilities': 4,        # 4%
            'Materials': 2,        # 2%
            'Real Estate': 2       # 2%
        }
        
        self.selection_criteria = {
            'market_cap_min': 10_000_000_000,    # $10B minimum
            'avg_volume_min': 1_000_000,         # 1M shares/day minimum
            'price_min': 5.0,                    # $5 minimum (no penny stocks)
            'listing_years_min': 3,              # 3+ years listed
            'max_missing_data': 0.05             # <5% missing data allowed
        }
    
    def get_sp500_components(self) -> List[str]:
        """Get current S&P 500 components from reliable source"""
        try:
            # Use SSL unverified context as fallback (temporary fix)
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Wikipedia table with S&P 500 components
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_df = tables[0]
            
            # Clean up symbols (remove dots, handle special cases)
            symbols = sp500_df['Symbol'].tolist()
            cleaned_symbols = []
            
            for symbol in symbols:
                # Handle special cases
                cleaned_symbol = symbol.replace('.', '-')  # BRK.B -> BRK-B
                cleaned_symbols.append(cleaned_symbol)
            
            logger.info(f"Retrieved {len(cleaned_symbols)} S&P 500 components")
            return cleaned_symbols
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 components: {e}")
            # Fallback to manually curated list of top stocks
            return self._get_fallback_universe()
    
    def _get_fallback_universe(self) -> List[str]:
        """Fallback universe of major US stocks by sector"""
        fallback_universe = {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 
                'ADBE', 'CRM', 'INTC', 'AMD', 'AVGO', 'ORCL', 'CSCO', 'IBM',
                'NOW', 'INTU', 'TXN', 'QCOM', 'MU', 'AMAT', 'ADI', 'LRCX', 'MRVL'
            ],
            'Health Care': [
                'UNH', 'JNJ', 'PFE', 'ABT', 'TMO', 'AZN', 'MRK', 'DHR', 'BMY',
                'AMGN', 'MDT', 'ISRG', 'GILD', 'CVS', 'CI'
            ],
            'Financials': [
                'BRK-B', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 
                'SPGI', 'CB', 'MMC', 'ICE', 'CME', 'V'
            ],
            'Consumer Discretionary': [
                'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'GM', 'F',
                'EBAY', 'MAR', 'HLT', 'YUM'
            ],
            'Communication Services': [
                'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'NFLX', 'GOOGL'
            ],
            'Industrials': [
                'MMM', 'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'DE'
            ],
            'Consumer Staples': [
                'WMT', 'PG', 'KO', 'PEP', 'COST'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'SLB'
            ],
            'Utilities': [
                'NEE', 'SO', 'DUK', 'D'
            ],
            'Materials': ['LIN', 'APD'],
            'Real Estate': ['AMT', 'PLD']
        }
        
        all_symbols = []
        for sector_symbols in fallback_universe.values():
            all_symbols.extend(sector_symbols)
        
        return all_symbols
    
    def screen_stocks(self, symbols: List[str], 
                     reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """Screen stocks based on selection criteria"""
        
        if reference_date is None:
            reference_date = datetime.now()
        
        screened_data = []
        
        logger.info(f"Screening {len(symbols)} stocks...")
        
        # Process in batches to avoid rate limits
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            batch_data = self._process_batch(batch, reference_date)
            screened_data.extend(batch_data)
            
            # Progress logging
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(symbols)} stocks")
        
        # Convert to DataFrame
        df = pd.DataFrame(screened_data)
        
        if df.empty:
            logger.warning("No stocks passed screening criteria")
            return df
        
        # Apply screening filters
        df = self._apply_screening_filters(df)
        
        logger.info(f"Screening complete: {len(df)} stocks passed criteria")
        return df
    
    def _process_batch(self, symbols: List[str], 
                      reference_date: datetime) -> List[Dict]:
        """Process a batch of symbols for screening"""
        batch_data = []
        
        try:
            # Use yfinance to get batch data
            tickers = yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    
                    # Get basic info
                    info = ticker.info
                    
                    # Get historical data for volume and price analysis
                    end_date = reference_date
                    start_date = end_date - timedelta(days=365)  # 1 year of data
                    
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if hist.empty:
                        continue
                    
                    # Calculate metrics
                    avg_volume = hist['Volume'].mean()
                    avg_price = hist['Close'].mean()
                    current_price = hist['Close'].iloc[-1] if not hist.empty else 0
                    
                    # Get market cap (from info or calculate)
                    market_cap = info.get('marketCap')
                    if market_cap is None:
                        shares_outstanding = info.get('sharesOutstanding')
                        if shares_outstanding:
                            market_cap = shares_outstanding * current_price
                    
                    # Get sector
                    sector = info.get('sector', 'Unknown')
                    
                    # Check data completeness
                    data_completeness = 1 - (hist.isnull().sum().sum() / (len(hist) * len(hist.columns)))
                    
                    stock_data = {
                        'symbol': symbol,
                        'market_cap': market_cap or 0,
                        'avg_volume': avg_volume,
                        'avg_price': avg_price,
                        'current_price': current_price,
                        'sector': sector,
                        'data_completeness': data_completeness,
                        'trading_days': len(hist),
                        'company_name': info.get('longName', symbol)
                    }
                    
                    batch_data.append(stock_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
        
        return batch_data
    
    def _apply_screening_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply screening criteria to filter stocks"""
        
        initial_count = len(df)
        
        # Market cap filter
        df = df[df['market_cap'] >= self.selection_criteria['market_cap_min']]
        logger.info(f"Market cap filter: {initial_count} -> {len(df)} stocks")
        
        # Volume filter
        df = df[df['avg_volume'] >= self.selection_criteria['avg_volume_min']]
        logger.info(f"Volume filter: {initial_count} -> {len(df)} stocks")
        
        # Price filter (no penny stocks)
        df = df[df['avg_price'] >= self.selection_criteria['price_min']]
        logger.info(f"Price filter: {initial_count} -> {len(df)} stocks")
        
        # Data completeness filter
        df = df[df['data_completeness'] >= (1 - self.selection_criteria['max_missing_data'])]
        logger.info(f"Data completeness filter: {initial_count} -> {len(df)} stocks")
        
        # Minimum trading days - simplified and more flexible for backtesting
        min_days = 180  # Start with 6 months minimum (reduced from 3 years)
        
        df_filtered = df[df['trading_days'] >= min_days]
        if len(df_filtered) < 30:  # If less than 30 stocks, reduce further
            min_days = 90  # 3 months minimum
            df_filtered = df[df['trading_days'] >= min_days]
        
        if len(df_filtered) < 20:  # If still too few, take what we have with any data
            min_days = 60  # 2 months minimum
            df_filtered = df[df['trading_days'] >= min_days]
        
        df = df_filtered
        logger.info(f"Trading days filter: {initial_count} -> {len(df)} stocks (min_days: {min_days})")
        
        return df.reset_index(drop=True)
    
    def select_diversified_universe(self, screened_df: pd.DataFrame, 
                                  target_count: int = 100) -> pd.DataFrame:
        """Select diversified universe based on sector allocations"""
        
        if screened_df.empty:
            return screened_df
        
        selected_stocks = []
        
        # Sort by market cap within each sector
        screened_df = screened_df.sort_values('market_cap', ascending=False)
        
        for sector, allocation_pct in self.sector_allocations.items():
            # Calculate target count for this sector
            target_sector_count = int((allocation_pct / 100) * target_count)
            
            # Get stocks in this sector
            sector_stocks = screened_df[screened_df['sector'] == sector]
            
            if len(sector_stocks) == 0:
                logger.warning(f"No stocks found in sector: {sector}")
                continue
            
            # Select top stocks by market cap
            selected_sector_stocks = sector_stocks.head(target_sector_count)
            selected_stocks.append(selected_sector_stocks)
            
            logger.info(f"Selected {len(selected_sector_stocks)} stocks from {sector} sector")
        
        # Combine all selected stocks
        if selected_stocks:
            final_universe = pd.concat(selected_stocks, ignore_index=True)
        else:
            final_universe = pd.DataFrame()
        
        # If we haven't reached target count, add more from largest remaining stocks
        remaining_count = target_count - len(final_universe)
        if remaining_count > 0:
            selected_symbols = set(final_universe['symbol'].tolist())
            remaining_stocks = screened_df[~screened_df['symbol'].isin(selected_symbols)]
            additional_stocks = remaining_stocks.head(remaining_count)
            
            if not additional_stocks.empty:
                final_universe = pd.concat([final_universe, additional_stocks], ignore_index=True)
        
        # Final sort by market cap
        final_universe = final_universe.sort_values('market_cap', ascending=False)
        final_universe = final_universe.head(target_count).reset_index(drop=True)
        
        logger.info(f"Final universe: {len(final_universe)} stocks selected")
        return final_universe
    
    def generate_top_100_universe(self, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """Main method to generate Top 100 universe"""
        
        logger.info("Starting Top 100 Universe generation...")
        
        # Get candidate symbols
        symbols = self.get_sp500_components()
        
        # Screen stocks
        screened_df = self.screen_stocks(symbols, reference_date)
        
        if screened_df.empty:
            raise ValueError("No stocks passed screening criteria")
        
        # Select diversified universe
        final_universe = self.select_diversified_universe(screened_df, target_count=100)
        
        # Add additional metadata
        final_universe['selection_date'] = reference_date or datetime.now()
        final_universe['rank'] = range(1, len(final_universe) + 1)
        
        # Calculate sector distribution
        sector_distribution = final_universe['sector'].value_counts()
        logger.info("Final sector distribution:")
        for sector, count in sector_distribution.items():
            logger.info(f"  {sector}: {count} stocks ({count/len(final_universe)*100:.1f}%)")
        
        self.universe = final_universe
        return final_universe
    
    def save_universe(self, filepath: str):
        """Save universe to CSV file"""
        if self.universe is not None and not self.universe.empty:
            self.universe.to_csv(filepath, index=False)
            logger.info(f"Universe saved to {filepath}")
        else:
            logger.warning("No universe to save")
    
    def load_universe(self, filepath: str) -> pd.DataFrame:
        """Load universe from CSV file"""
        try:
            self.universe = pd.read_csv(filepath)
            logger.info(f"Universe loaded from {filepath}: {len(self.universe)} stocks")
            return self.universe
        except Exception as e:
            logger.error(f"Error loading universe: {e}")
            return pd.DataFrame()
    
    def get_universe_symbols(self) -> List[str]:
        """Get list of symbols in current universe"""
        if self.universe is not None and not self.universe.empty:
            return self.universe['symbol'].tolist()
        return []
    
    def get_sector_weights(self) -> Dict[str, float]:
        """Get actual sector weights in current universe"""
        if self.universe is not None and not self.universe.empty:
            sector_counts = self.universe['sector'].value_counts()
            total = len(self.universe)
            return {sector: count/total for sector, count in sector_counts.items()}
        return {}


def main():
    """Example usage"""
    
    # Create universe selector
    selector = Top100Universe()
    
    # Generate universe (can specify reference date for historical universe)
    reference_date = datetime(2024, 1, 1)  # Or None for current
    
    try:
        universe = selector.generate_top_100_universe(reference_date)
        
        print(f"\\nGenerated Top 100 Universe:")
        print(f"Total stocks: {len(universe)}")
        print(f"\\nTop 10 by market cap:")
        print(universe[['symbol', 'company_name', 'sector', 'market_cap']].head(10))
        
        print(f"\\nSector distribution:")
        sector_weights = selector.get_sector_weights()
        for sector, weight in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sector}: {weight:.1%}")
        
        # Save universe
        selector.save_universe('top_100_universe.csv')
        
        return universe
        
    except Exception as e:
        logger.error(f"Error generating universe: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()