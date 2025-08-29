"""
Real-time Data Processing System
WebSocket connections, event-driven architecture, and high-throughput stream processing.
"""
import asyncio
import websockets
import json
import time
import queue
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import aiohttp
import concurrent.futures

from src.utils.caching import cache_manager, cached
from src.utils.error_handling import NetworkError, ErrorSeverity, handle_errors
from src.utils.logging_setup import get_logger, perf_logger
from config.enhanced_config import enhanced_config

logger = get_logger(__name__)

class DataSourceType(Enum):
    """Types of real-time data sources"""
    MARKET_DATA = "market_data"
    NEWS_FEED = "news_feed"
    SOCIAL_SENTIMENT = "social_sentiment"
    TRADING_SIGNALS = "trading_signals"
    SYSTEM_METRICS = "system_metrics"

@dataclass
class StreamConfig:
    """Configuration for data streams"""
    source_type: DataSourceType
    symbols: List[str] = field(default_factory=list)
    update_interval: float = 1.0
    buffer_size: int = 1000
    batch_size: int = 100
    enable_caching: bool = True
    cache_ttl: int = 60
    retry_attempts: int = 3
    timeout: float = 30.0

@dataclass
class DataEvent:
    """Real-time data event"""
    event_type: str
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class EventProcessor:
    """Process and route real-time data events"""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.stats = {
            'events_processed': 0,
            'events_per_second': 0,
            'processing_errors': 0,
            'last_reset': time.time()
        }
        self._stats_lock = threading.Lock()
        self._start_stats_updater()
    
    def _start_stats_updater(self):
        """Start background thread to update processing statistics"""
        def update_stats():
            while True:
                try:
                    with self._stats_lock:
                        current_time = time.time()
                        elapsed = current_time - self.stats['last_reset']
                        if elapsed > 0:
                            self.stats['events_per_second'] = self.stats['events_processed'] / elapsed
                        
                        # Reset counters every minute
                        if elapsed > 60:
                            self.stats['events_processed'] = 0
                            self.stats['last_reset'] = current_time
                    
                    time.sleep(10)  # Update every 10 seconds
                except Exception as e:
                    logger.error("Stats updater error", exception=e, component='realtime')
        
        stats_thread = threading.Thread(target=update_stats, daemon=True)
        stats_thread.start()
    
    def register_handler(self, event_type: str, handler: Callable[[DataEvent], None]):
        """Register event handler for specific event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type}", component='realtime')
    
    async def process_event(self, event: DataEvent):
        """Process a single data event"""
        start_time = time.time()
        
        try:
            # Update statistics
            with self._stats_lock:
                self.stats['events_processed'] += 1
            
            # Cache the event if caching is enabled
            if event.source in ['market_data', 'trading_signals']:
                cache_key = f"{event.symbol}:{event.event_type}"
                cache_manager.set('realtime_events', cache_key, event.data, ttl_override=60)
            
            # Route to registered handlers
            handlers = self.handlers.get(event.event_type, [])
            if handlers:
                tasks = [asyncio.create_task(self._safe_handler_call(handler, event)) 
                        for handler in handlers]
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log processing time
            processing_time = time.time() - start_time
            perf_logger.log_execution_time('event_processing', processing_time)
            
            logger.debug(
                f"Processed event: {event.event_type} for {event.symbol}",
                component='realtime',
                processing_time=processing_time,
                handlers_count=len(handlers)
            )
            
        except Exception as e:
            with self._stats_lock:
                self.stats['processing_errors'] += 1
            logger.error(
                f"Event processing failed: {e}",
                exception=e,
                component='realtime',
                event_type=event.event_type,
                symbol=event.symbol
            )
    
    async def _safe_handler_call(self, handler: Callable, event: DataEvent):
        """Safely call event handler with error handling"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                # Run synchronous handler in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler, event)
        except Exception as e:
            logger.error(
                f"Handler error: {e}",
                exception=e,
                component='realtime',
                handler=handler.__name__
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event processing statistics"""
        with self._stats_lock:
            return {
                **self.stats,
                'active_handlers': sum(len(handlers) for handlers in self.handlers.values()),
                'handler_types': list(self.handlers.keys())
            }

class WebSocketManager:
    """Manage WebSocket connections for real-time data"""
    
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'connection_errors': 0
        }
        self._stats_lock = threading.Lock()
    
    async def start_server(self, host: str = 'localhost', port: int = 8765):
        """Start WebSocket server for real-time data distribution"""
        try:
            async def handle_connection(websocket, path):
                await self._handle_client_connection(websocket, path)
            
            server = await websockets.serve(handle_connection, host, port)
            logger.info(f"WebSocket server started on {host}:{port}", component='websocket')
            return server
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}", exception=e, component='websocket')
            raise NetworkError(f"WebSocket server startup failed: {e}", severity=ErrorSeverity.HIGH, original_error=e)
    
    async def _handle_client_connection(self, websocket, path):
        """Handle individual client WebSocket connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}:{time.time()}"
        
        try:
            with self._stats_lock:
                self.connections[client_id] = websocket
                self.connection_stats['total_connections'] += 1
                self.connection_stats['active_connections'] += 1
            
            logger.info(f"Client connected: {client_id}", component='websocket')
            
            # Send welcome message
            welcome_msg = {
                'type': 'connection_established',
                'client_id': client_id,
                'timestamp': datetime.utcnow().isoformat(),
                'available_streams': ['market_data', 'trading_signals', 'system_metrics']
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Listen for client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, client_id, data)
                    
                    with self._stats_lock:
                        self.connection_stats['messages_received'] += 1
                        
                except json.JSONDecodeError:
                    error_msg = {'type': 'error', 'message': 'Invalid JSON format'}
                    await websocket.send(json.dumps(error_msg))
                except Exception as e:
                    logger.error(f"Message handling error for {client_id}: {e}", exception=e, component='websocket')
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}", component='websocket')
        except Exception as e:
            logger.error(f"WebSocket connection error for {client_id}: {e}", exception=e, component='websocket')
            with self._stats_lock:
                self.connection_stats['connection_errors'] += 1
        
        finally:
            # Cleanup
            with self._stats_lock:
                if client_id in self.connections:
                    del self.connections[client_id]
                    self.connection_stats['active_connections'] -= 1
    
    async def _handle_client_message(self, websocket, client_id: str, data: Dict[str, Any]):
        """Handle messages from WebSocket clients"""
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            # Handle subscription requests
            streams = data.get('streams', [])
            symbols = data.get('symbols', [])
            
            response = {
                'type': 'subscription_confirmed',
                'streams': streams,
                'symbols': symbols,
                'client_id': client_id
            }
            await websocket.send(json.dumps(response))
            
            logger.info(
                f"Client {client_id} subscribed to streams: {streams}",
                component='websocket',
                streams=streams,
                symbols=symbols
            )
        
        elif message_type == 'ping':
            # Handle ping requests
            pong_response = {'type': 'pong', 'timestamp': datetime.utcnow().isoformat()}
            await websocket.send(json.dumps(pong_response))
        
        else:
            # Unknown message type
            error_msg = {'type': 'error', 'message': f'Unknown message type: {message_type}'}
            await websocket.send(json.dumps(error_msg))
    
    async def broadcast_data(self, data: Dict[str, Any], stream_type: str = None):
        """Broadcast data to all connected WebSocket clients"""
        if not self.connections:
            return
        
        message = {
            'type': 'data_update',
            'stream_type': stream_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        message_json = json.dumps(message)
        disconnected_clients = []
        
        for client_id, websocket in self.connections.items():
            try:
                await websocket.send(message_json)
                with self._stats_lock:
                    self.connection_stats['messages_sent'] += 1
                    
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Broadcast error to {client_id}: {e}", exception=e, component='websocket')
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            with self._stats_lock:
                if client_id in self.connections:
                    del self.connections[client_id]
                    self.connection_stats['active_connections'] -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        with self._stats_lock:
            return {
                **self.connection_stats,
                'timestamp': datetime.utcnow().isoformat()
            }

class StreamProcessor:
    """High-throughput stream processing system"""
    
    def __init__(self):
        self.streams: Dict[str, StreamConfig] = {}
        self.event_processor = EventProcessor()
        self.websocket_manager = WebSocketManager()
        self.data_buffers: Dict[str, queue.Queue] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.stats = {
            'active_streams': 0,
            'total_messages_processed': 0,
            'messages_per_second': 0,
            'buffer_utilization': 0.0,
            'last_reset': time.time()
        }
        self._running = False
    
    def add_stream(self, stream_id: str, config: StreamConfig):
        """Add a new data stream"""
        self.streams[stream_id] = config
        self.data_buffers[stream_id] = queue.Queue(maxsize=config.buffer_size)
        
        logger.info(
            f"Added stream: {stream_id}",
            component='stream_processor',
            source_type=config.source_type.value,
            symbols=config.symbols,
            buffer_size=config.buffer_size
        )
    
    async def start_processing(self):
        """Start stream processing"""
        self._running = True
        self.stats['active_streams'] = len(self.streams)
        
        # Start WebSocket server
        try:
            await self.websocket_manager.start_server()
        except Exception as e:
            logger.warning(f"WebSocket server not started: {e}", component='stream_processor')
        
        # Start processing tasks for each stream
        for stream_id, config in self.streams.items():
            task = asyncio.create_task(self._process_stream(stream_id, config))
            self.processing_tasks[stream_id] = task
        
        logger.info(f"Started processing {len(self.streams)} streams", component='stream_processor')
        
        # Start statistics updater
        asyncio.create_task(self._update_stats())
    
    async def _process_stream(self, stream_id: str, config: StreamConfig):
        """Process individual data stream"""
        buffer = self.data_buffers[stream_id]
        batch = []
        last_process_time = time.time()
        
        while self._running:
            try:
                # Collect batch of messages
                while len(batch) < config.batch_size:
                    try:
                        # Non-blocking get with timeout
                        message = buffer.get(timeout=0.1)
                        batch.append(message)
                        buffer.task_done()
                    except queue.Empty:
                        break
                
                # Process batch if we have messages or timeout reached
                current_time = time.time()
                if batch and (len(batch) >= config.batch_size or 
                             current_time - last_process_time >= config.update_interval):
                    
                    await self._process_batch(stream_id, batch, config)
                    batch = []
                    last_process_time = current_time
                
                # Short sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Stream processing error for {stream_id}: {e}", exception=e, component='stream_processor')
                await asyncio.sleep(1)  # Back off on error
    
    async def _process_batch(self, stream_id: str, batch: List[Dict[str, Any]], config: StreamConfig):
        """Process a batch of messages"""
        start_time = time.time()
        
        try:
            # Convert messages to events
            events = []
            for message in batch:
                event = DataEvent(
                    event_type=message.get('type', 'unknown'),
                    symbol=message.get('symbol', ''),
                    data=message.get('data', {}),
                    timestamp=datetime.fromisoformat(message.get('timestamp', datetime.utcnow().isoformat())),
                    source=config.source_type.value,
                    metadata={'stream_id': stream_id}
                )
                events.append(event)
            
            # Process events
            processing_tasks = [self.event_processor.process_event(event) for event in events]
            await asyncio.gather(*processing_tasks, return_exceptions=True)
            
            # Broadcast to WebSocket clients
            if events:
                broadcast_data = {
                    'stream_id': stream_id,
                    'batch_size': len(batch),
                    'events': [{'type': e.event_type, 'symbol': e.symbol, 'data': e.data} for e in events[:10]]  # Limit size
                }
                await self.websocket_manager.broadcast_data(broadcast_data, config.source_type.value)
            
            # Update statistics
            self.stats['total_messages_processed'] += len(batch)
            
            # Log batch processing
            processing_time = time.time() - start_time
            perf_logger.log_execution_time('batch_processing', processing_time)
            
            logger.debug(
                f"Processed batch: {stream_id}",
                component='stream_processor',
                batch_size=len(batch),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Batch processing error for {stream_id}: {e}", exception=e, component='stream_processor')
    
    async def _update_stats(self):
        """Update processing statistics"""
        while self._running:
            try:
                current_time = time.time()
                elapsed = current_time - self.stats['last_reset']
                
                if elapsed > 0:
                    self.stats['messages_per_second'] = self.stats['total_messages_processed'] / elapsed
                
                # Calculate buffer utilization
                total_capacity = sum(config.buffer_size for config in self.streams.values())
                total_used = sum(buffer.qsize() for buffer in self.data_buffers.values())
                self.stats['buffer_utilization'] = (total_used / total_capacity * 100) if total_capacity > 0 else 0
                
                # Reset counters every minute
                if elapsed > 60:
                    self.stats['total_messages_processed'] = 0
                    self.stats['last_reset'] = current_time
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error("Stats updater error", exception=e, component='stream_processor')
                await asyncio.sleep(10)
    
    def add_message(self, stream_id: str, message: Dict[str, Any]) -> bool:
        """Add message to stream buffer"""
        if stream_id not in self.data_buffers:
            logger.warning(f"Stream not found: {stream_id}", component='stream_processor')
            return False
        
        buffer = self.data_buffers[stream_id]
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = datetime.utcnow().isoformat()
            
            buffer.put_nowait(message)
            return True
            
        except queue.Full:
            logger.warning(f"Buffer full for stream: {stream_id}", component='stream_processor')
            return False
        except Exception as e:
            logger.error(f"Failed to add message to {stream_id}: {e}", exception=e, component='stream_processor')
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            'stream_processor': self.stats,
            'event_processor': self.event_processor.get_stats(),
            'websocket_manager': self.websocket_manager.get_stats(),
            'buffer_status': {
                stream_id: {
                    'size': buffer.qsize(),
                    'max_size': config.buffer_size,
                    'utilization': (buffer.qsize() / config.buffer_size * 100) if config.buffer_size > 0 else 0
                }
                for stream_id, (buffer, config) in zip(self.data_buffers.keys(), 
                                                     zip(self.data_buffers.values(), self.streams.values()))
            }
        }
    
    async def stop_processing(self):
        """Stop stream processing"""
        self._running = False
        
        # Cancel processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        logger.info("Stream processing stopped", component='stream_processor')

# Global stream processor instance
stream_processor = StreamProcessor()

# Convenience functions
def add_event_handler(event_type: str, handler: Callable[[DataEvent], None]):
    """Register event handler"""
    stream_processor.event_processor.register_handler(event_type, handler)

def add_data_stream(stream_id: str, config: StreamConfig):
    """Add new data stream"""
    stream_processor.add_stream(stream_id, config)

def send_realtime_data(stream_id: str, message: Dict[str, Any]) -> bool:
    """Send message to stream"""
    return stream_processor.add_message(stream_id, message)

async def start_realtime_processing():
    """Start real-time processing system"""
    await stream_processor.start_processing()

def get_realtime_stats() -> Dict[str, Any]:
    """Get real-time processing statistics"""
    return stream_processor.get_stats()