"""
Hot Data Manager - Redis-based hot data storage
Manages real-time data with short TTL for live signals and prices
Part of the tiered data storage architecture
"""
import redis
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from config.config import config

logger = logging.getLogger(__name__)

@dataclass
class HotDataConfig:
    """Configuration for hot data management"""
    live_signal_ttl: int = 300      # 5 minutes
    current_price_ttl: int = 60     # 1 minute  
    market_data_ttl: int = 180      # 3 minutes
    indicator_ttl: int = 300        # 5 minutes
    max_retries: int = 3
    retry_delay: float = 0.1

class HotDataManager:
    """
    Manages hot data in Redis for high-frequency access
    Implements the 'Hot' tier of the data storage architecture
    
    Data Types:
    - Live signals: 5-minute TTL
    - Current prices: 1-minute TTL  
    - Market environment: 3-minute TTL
    - Real-time indicators: 5-minute TTL
    """
    
    def __init__(self, redis_config: Optional[Dict] = None, hot_config: Optional[HotDataConfig] = None):
        self.redis_config = redis_config or {
            'host': config.redis.host,
            'port': config.redis.port,
            'db': config.redis.db,
            'password': config.redis.password if config.redis.password else None,
            'decode_responses': False,  # We'll handle encoding manually
            'socket_connect_timeout': 5,
            'socket_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        
        self.config = hot_config or HotDataConfig()
        self.redis_client = None
        self._connection_pool = None
        self._initialize_connection()
        
        # Key prefixes for different data types
        self.key_prefixes = {
            'live_signal': 'signal:live',
            'current_price': 'price:current', 
            'market_env': 'market:environment',
            'indicator': 'indicator:realtime',
            'portfolio': 'portfolio:snapshot',
            'alert': 'alert:active'
        }
    
    def _initialize_connection(self) -> None:
        """Initialize Redis connection with connection pooling"""
        try:
            # Create connection pool for better performance
            self._connection_pool = redis.ConnectionPool(**self.redis_config)
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis hot data connection established successfully")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None
        except Exception as e:
            logger.error(f"Unexpected error initializing Redis connection: {str(e)}")
            self.redis_client = None
    
    def is_connected(self) -> bool:
        """Check if Redis connection is active"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    @contextmanager
    def _get_connection(self):
        """Context manager for Redis connection with automatic retry"""
        if not self.is_connected():
            self._initialize_connection()
        
        if not self.redis_client:
            raise redis.ConnectionError("Unable to establish Redis connection")
        
        yield self.redis_client
    
    def store_live_signal(self, symbol: str, signal_data: Dict[str, Any]) -> bool:
        """
        Store live signal data with 5-minute TTL
        
        Args:
            symbol: Stock symbol
            signal_data: Signal data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as client:
                key = f"{self.key_prefixes['live_signal']}:{symbol}"
                
                # Add timestamp to signal data
                signal_data['timestamp'] = datetime.now().isoformat()
                signal_data['symbol'] = symbol
                
                # Serialize data
                serialized_data = json.dumps(signal_data, default=str)
                
                # Store with TTL
                success = client.setex(
                    key, 
                    self.config.live_signal_ttl, 
                    serialized_data
                )
                
                if success:
                    logger.debug(f"Stored live signal for {symbol}")
                    
                    # Also update the signals list
                    self._update_active_signals_list(symbol)
                    
                return bool(success)
                
        except Exception as e:
            logger.error(f"Error storing live signal for {symbol}: {str(e)}")
            return False
    
    def get_live_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve live signal for a specific symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with signal data or None if not found/expired
        """
        try:
            with self._get_connection() as client:
                key = f"{self.key_prefixes['live_signal']}:{symbol}"
                data = client.get(key)
                
                if data:
                    signal_data = json.loads(data.decode('utf-8'))
                    logger.debug(f"Retrieved live signal for {symbol}")
                    return signal_data
                    
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving live signal for {symbol}: {str(e)}")
            return None
    
    def get_live_signals(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve live signals for multiple symbols efficiently using pipeline
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbols to their signal data
        """
        try:
            with self._get_connection() as client:
                # Use pipeline for efficient batch retrieval
                pipeline = client.pipeline()
                keys = [f"{self.key_prefixes['live_signal']}:{symbol}" for symbol in symbols]
                
                for key in keys:
                    pipeline.get(key)
                
                results = pipeline.execute()
                
                # Process results
                signals = {}
                for i, (symbol, data) in enumerate(zip(symbols, results)):
                    if data:
                        try:
                            signal_data = json.loads(data.decode('utf-8'))
                            signals[symbol] = signal_data
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON data for signal {symbol}")
                
                logger.debug(f"Retrieved {len(signals)} live signals from {len(symbols)} requested")
                return signals
                
        except Exception as e:
            logger.error(f"Error retrieving live signals: {str(e)}")
            return {}
    
    def store_current_price(self, symbol: str, price_data: Dict[str, Any]) -> bool:
        """
        Store current price data with 1-minute TTL
        
        Args:
            symbol: Stock symbol  
            price_data: Price data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as client:
                key = f"{self.key_prefixes['current_price']}:{symbol}"
                
                # Add timestamp
                price_data['timestamp'] = datetime.now().isoformat()
                price_data['symbol'] = symbol
                
                # Serialize data
                serialized_data = json.dumps(price_data, default=str)
                
                # Store with TTL
                success = client.setex(
                    key,
                    self.config.current_price_ttl,
                    serialized_data
                )
                
                if success:
                    logger.debug(f"Stored current price for {symbol}: ${price_data.get('price', 'N/A')}")
                
                return bool(success)
                
        except Exception as e:
            logger.error(f"Error storing current price for {symbol}: {str(e)}")
            return False
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve current prices for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbols to their price data
        """
        try:
            with self._get_connection() as client:
                pipeline = client.pipeline()
                keys = [f"{self.key_prefixes['current_price']}:{symbol}" for symbol in symbols]
                
                for key in keys:
                    pipeline.get(key)
                
                results = pipeline.execute()
                
                prices = {}
                for symbol, data in zip(symbols, results):
                    if data:
                        try:
                            price_data = json.loads(data.decode('utf-8'))
                            prices[symbol] = price_data
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON data for price {symbol}")
                
                return prices
                
        except Exception as e:
            logger.error(f"Error retrieving current prices: {str(e)}")
            return {}
    
    def store_market_environment(self, market_data: Dict[str, Any]) -> bool:
        """
        Store market environment data (VIX, Fear & Greed, etc.)
        
        Args:
            market_data: Market environment data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as client:
                key = f"{self.key_prefixes['market_env']}:current"
                
                # Add timestamp
                market_data['timestamp'] = datetime.now().isoformat()
                
                # Serialize data
                serialized_data = json.dumps(market_data, default=str)
                
                # Store with TTL
                success = client.setex(
                    key,
                    self.config.market_data_ttl,
                    serialized_data
                )
                
                if success:
                    logger.debug("Stored market environment data")
                
                return bool(success)
                
        except Exception as e:
            logger.error(f"Error storing market environment: {str(e)}")
            return False
    
    def get_market_environment(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve current market environment data
        
        Returns:
            Dict with market environment data or None
        """
        try:
            with self._get_connection() as client:
                key = f"{self.key_prefixes['market_env']}:current"
                data = client.get(key)
                
                if data:
                    market_data = json.loads(data.decode('utf-8'))
                    logger.debug("Retrieved market environment data")
                    return market_data
                    
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving market environment: {str(e)}")
            return None
    
    def store_realtime_indicator(self, symbol: str, indicator_name: str, 
                                indicator_data: Dict[str, Any]) -> bool:
        """
        Store real-time technical indicator data
        
        Args:
            symbol: Stock symbol
            indicator_name: Name of the indicator (rsi, macd, etc.)
            indicator_data: Indicator data dictionary
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as client:
                key = f"{self.key_prefixes['indicator']}:{symbol}:{indicator_name}"
                
                # Add timestamp
                indicator_data['timestamp'] = datetime.now().isoformat()
                indicator_data['symbol'] = symbol
                indicator_data['indicator'] = indicator_name
                
                # Serialize data
                serialized_data = json.dumps(indicator_data, default=str)
                
                # Store with TTL
                success = client.setex(
                    key,
                    self.config.indicator_ttl,
                    serialized_data
                )
                
                if success:
                    logger.debug(f"Stored {indicator_name} indicator for {symbol}")
                
                return bool(success)
                
        except Exception as e:
            logger.error(f"Error storing {indicator_name} indicator for {symbol}: {str(e)}")
            return False
    
    def get_realtime_indicators(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all real-time indicators for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict mapping indicator names to their data
        """
        try:
            with self._get_connection() as client:
                # Get all indicator keys for this symbol
                pattern = f"{self.key_prefixes['indicator']}:{symbol}:*"
                keys = client.keys(pattern)
                
                if not keys:
                    return {}
                
                # Get all indicator data using pipeline
                pipeline = client.pipeline()
                for key in keys:
                    pipeline.get(key)
                
                results = pipeline.execute()
                
                indicators = {}
                for key, data in zip(keys, results):
                    if data:
                        try:
                            # Extract indicator name from key
                            indicator_name = key.decode('utf-8').split(':')[-1]
                            indicator_data = json.loads(data.decode('utf-8'))
                            indicators[indicator_name] = indicator_data
                        except (json.JSONDecodeError, IndexError):
                            logger.warning(f"Invalid data for key {key}")
                
                return indicators
                
        except Exception as e:
            logger.error(f"Error retrieving indicators for {symbol}: {str(e)}")
            return {}
    
    def _update_active_signals_list(self, symbol: str) -> None:
        """Update list of symbols with active signals"""
        try:
            with self._get_connection() as client:
                list_key = "active_signals_list"
                # Add symbol to set (automatically handles duplicates)
                client.sadd(list_key, symbol)
                # Set expiration for the list itself
                client.expire(list_key, self.config.live_signal_ttl)
                
        except Exception as e:
            logger.error(f"Error updating active signals list: {str(e)}")
    
    def get_active_signals_symbols(self) -> List[str]:
        """Get list of symbols with active signals"""
        try:
            with self._get_connection() as client:
                list_key = "active_signals_list" 
                symbols = client.smembers(list_key)
                return [symbol.decode('utf-8') for symbol in symbols]
                
        except Exception as e:
            logger.error(f"Error retrieving active signals list: {str(e)}")
            return []
    
    def clear_expired_data(self) -> Dict[str, int]:
        """
        Manually clear expired data (Redis handles this automatically, 
        but this can be used for cleanup)
        
        Returns:
            Dict with count of cleared items by data type
        """
        cleared_counts = {
            'signals': 0,
            'prices': 0, 
            'indicators': 0,
            'market_data': 0
        }
        
        try:
            with self._get_connection() as client:
                current_time = datetime.now()
                
                # Check signal keys
                for pattern, data_type in [
                    (f"{self.key_prefixes['live_signal']}:*", 'signals'),
                    (f"{self.key_prefixes['current_price']}:*", 'prices'),
                    (f"{self.key_prefixes['indicator']}:*", 'indicators'),
                    (f"{self.key_prefixes['market_env']}:*", 'market_data')
                ]:
                    keys = client.keys(pattern)
                    
                    for key in keys:
                        ttl = client.ttl(key)
                        if ttl == -2:  # Key doesn't exist (expired)
                            cleared_counts[data_type] += 1
                            
                logger.info(f"Cleared expired data: {cleared_counts}")
                return cleared_counts
                
        except Exception as e:
            logger.error(f"Error clearing expired data: {str(e)}")
            return cleared_counts
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get Redis storage statistics"""
        try:
            with self._get_connection() as client:
                info = client.info()
                
                # Count keys by type
                key_counts = {}
                for prefix_name, prefix in self.key_prefixes.items():
                    pattern = f"{prefix}:*"
                    keys = client.keys(pattern)
                    key_counts[prefix_name] = len(keys)
                
                stats = {
                    'total_keys': info.get('db0', {}).get('keys', 0),
                    'memory_used': info.get('used_memory_human', 'Unknown'),
                    'memory_used_bytes': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'key_counts': key_counts,
                    'hit_rate': self._calculate_hit_rate(info),
                    'uptime_seconds': info.get('uptime_in_seconds', 0)
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {}
    
    def _calculate_hit_rate(self, redis_info: Dict) -> float:
        """Calculate cache hit rate"""
        try:
            hits = redis_info.get('keyspace_hits', 0)
            misses = redis_info.get('keyspace_misses', 0)
            
            if hits + misses == 0:
                return 0.0
            
            hit_rate = hits / (hits + misses) * 100
            return round(hit_rate, 2)
            
        except:
            return 0.0
    
    def flush_all_hot_data(self) -> bool:
        """
        Flush all hot data (USE WITH CAUTION)
        
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as client:
                # Only flush keys with our prefixes to be safe
                for prefix in self.key_prefixes.values():
                    pattern = f"{prefix}:*"
                    keys = client.keys(pattern)
                    
                    if keys:
                        client.delete(*keys)
                
                logger.warning("Flushed all hot data from Redis")
                return True
                
        except Exception as e:
            logger.error(f"Error flushing hot data: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on hot data system"""
        health_status = {
            'status': 'unknown',
            'connection': False,
            'latency_ms': None,
            'memory_usage': None,
            'error': None
        }
        
        try:
            start_time = datetime.now()
            
            # Test connection
            with self._get_connection() as client:
                client.ping()
                health_status['connection'] = True
                
                # Measure latency
                latency = (datetime.now() - start_time).total_seconds() * 1000
                health_status['latency_ms'] = round(latency, 2)
                
                # Get memory usage
                info = client.info()
                health_status['memory_usage'] = info.get('used_memory_human', 'Unknown')
                
                # Overall status
                if latency < 100:  # Less than 100ms is good
                    health_status['status'] = 'healthy'
                elif latency < 500:  # Less than 500ms is acceptable
                    health_status['status'] = 'warning'
                else:
                    health_status['status'] = 'slow'
                    
        except Exception as e:
            health_status['status'] = 'error'
            health_status['error'] = str(e)
            logger.error(f"Hot data health check failed: {str(e)}")
        
        return health_status