"""
Service Registry Implementation
Service registration and discovery using Redis or Consul backends
"""

import json
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import redis
import aioredis

logger = logging.getLogger(__name__)

@dataclass
class ServiceInfo:
    """Service information"""
    name: str
    url: str
    health_check_url: str
    version: str = "1.0.0"
    metadata: Dict[str, Any] = None
    registered_at: str = None
    last_heartbeat: str = None
    status: str = "unknown"  # unknown, healthy, unhealthy
    weight: int = 1  # For load balancing
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.registered_at is None:
            self.registered_at = datetime.utcnow().isoformat()

class ServiceRegistry(ABC):
    """Abstract base class for service registry"""
    
    @abstractmethod
    async def register(self, service_info: ServiceInfo) -> bool:
        """Register a service"""
        pass
    
    @abstractmethod
    async def deregister(self, service_name: str) -> bool:
        """Deregister a service"""
        pass
    
    @abstractmethod
    async def discover(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover a service by name"""
        pass
    
    @abstractmethod
    async def list_services(self) -> Dict[str, ServiceInfo]:
        """List all registered services"""
        pass
    
    @abstractmethod
    async def update_health(self, service_name: str, status: str):
        """Update service health status"""
        pass
    
    @abstractmethod
    async def heartbeat(self, service_name: str):
        """Send heartbeat for service"""
        pass

class RedisServiceRegistry(ServiceRegistry):
    """Redis-based service registry"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.ttl_seconds = 30  # Service registration TTL
        self.heartbeat_interval = 10  # Heartbeat interval in seconds
    
    async def _get_client(self):
        """Get or create Redis client"""
        if not self.redis_client:
            self.redis_client = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                health_check_interval=30
            )
        return self.redis_client
    
    async def register(self, service_info: ServiceInfo) -> bool:
        """Register a service in Redis"""
        try:
            client = await self._get_client()
            key = f"services:{service_info.name}"
            
            service_data = asdict(service_info)
            service_data['last_heartbeat'] = datetime.utcnow().isoformat()
            
            await client.setex(
                key, 
                self.ttl_seconds,
                json.dumps(service_data)
            )
            
            logger.info(f"Registered service: {service_info.name} at {service_info.url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service_info.name}: {e}")
            return False
    
    async def deregister(self, service_name: str) -> bool:
        """Deregister a service from Redis"""
        try:
            client = await self._get_client()
            key = f"services:{service_name}"
            
            result = await client.delete(key)
            
            if result:
                logger.info(f"Deregistered service: {service_name}")
            else:
                logger.warning(f"Service {service_name} was not registered")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_name}: {e}")
            return False
    
    async def discover(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover a service by name"""
        try:
            client = await self._get_client()
            key = f"services:{service_name}"
            
            service_data = await client.get(key)
            
            if service_data:
                data = json.loads(service_data)
                return ServiceInfo(**data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to discover service {service_name}: {e}")
            return None
    
    async def list_services(self) -> Dict[str, ServiceInfo]:
        """List all registered services"""
        try:
            client = await self._get_client()
            keys = await client.keys("services:*")
            
            services = {}
            for key in keys:
                service_name = key.split(":", 1)[1]
                service_data = await client.get(key)
                
                if service_data:
                    data = json.loads(service_data)
                    services[service_name] = ServiceInfo(**data)
            
            return services
            
        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            return {}
    
    async def update_health(self, service_name: str, status: str):
        """Update service health status"""
        try:
            client = await self._get_client()
            key = f"services:{service_name}"
            
            service_data = await client.get(key)
            if service_data:
                data = json.loads(service_data)
                data['status'] = status
                data['last_heartbeat'] = datetime.utcnow().isoformat()
                
                await client.setex(
                    key,
                    self.ttl_seconds,
                    json.dumps(data)
                )
                
                logger.debug(f"Updated health for {service_name}: {status}")
            
        except Exception as e:
            logger.error(f"Failed to update health for {service_name}: {e}")
    
    async def heartbeat(self, service_name: str):
        """Send heartbeat for service"""
        try:
            client = await self._get_client()
            key = f"services:{service_name}"
            
            # Extend TTL and update last heartbeat
            service_data = await client.get(key)
            if service_data:
                data = json.loads(service_data)
                data['last_heartbeat'] = datetime.utcnow().isoformat()
                
                await client.setex(
                    key,
                    self.ttl_seconds,
                    json.dumps(data)
                )
                
                logger.debug(f"Heartbeat sent for {service_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat for {service_name}: {e}")
            return False
    
    async def cleanup_expired_services(self):
        """Remove expired services (called periodically)"""
        try:
            services = await self.list_services()
            now = datetime.utcnow()
            
            for service_name, service_info in services.items():
                if service_info.last_heartbeat:
                    last_heartbeat = datetime.fromisoformat(service_info.last_heartbeat)
                    if now - last_heartbeat > timedelta(seconds=self.ttl_seconds):
                        await self.deregister(service_name)
                        logger.info(f"Cleaned up expired service: {service_name}")
        
        except Exception as e:
            logger.error(f"Failed to cleanup expired services: {e}")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

class ConsulServiceRegistry(ServiceRegistry):
    """Consul-based service registry (placeholder for Consul integration)"""
    
    def __init__(self, consul_url: str = "http://localhost:8500"):
        self.consul_url = consul_url
        self.services = {}  # In-memory fallback
        logger.warning("Consul integration not implemented, using in-memory registry")
    
    async def register(self, service_info: ServiceInfo) -> bool:
        """Register service (in-memory fallback)"""
        self.services[service_info.name] = service_info
        logger.info(f"Registered service: {service_info.name} (in-memory)")
        return True
    
    async def deregister(self, service_name: str) -> bool:
        """Deregister service"""
        if service_name in self.services:
            del self.services[service_name]
            logger.info(f"Deregistered service: {service_name}")
            return True
        return False
    
    async def discover(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover service"""
        return self.services.get(service_name)
    
    async def list_services(self) -> Dict[str, ServiceInfo]:
        """List all services"""
        return self.services.copy()
    
    async def update_health(self, service_name: str, status: str):
        """Update service health"""
        if service_name in self.services:
            self.services[service_name].status = status
            self.services[service_name].last_heartbeat = datetime.utcnow().isoformat()
    
    async def heartbeat(self, service_name: str):
        """Send heartbeat"""
        if service_name in self.services:
            self.services[service_name].last_heartbeat = datetime.utcnow().isoformat()
            return True
        return False

class ServiceRegistryManager:
    """Manager for service registry with auto-heartbeats"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.registered_services = set()
        self._heartbeat_task = None
        self._cleanup_task = None
        self._running = False
    
    async def start(self):
        """Start the registry manager"""
        if self._running:
            return
        
        self._running = True
        
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start cleanup task (for Redis registry)
        if hasattr(self.registry, 'cleanup_expired_services'):
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Service registry manager started")
    
    async def stop(self):
        """Stop the registry manager"""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Deregister all services
        for service_name in list(self.registered_services):
            await self.registry.deregister(service_name)
            self.registered_services.discard(service_name)
        
        if hasattr(self.registry, 'close'):
            await self.registry.close()
        
        logger.info("Service registry manager stopped")
    
    async def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a service and track it"""
        result = await self.registry.register(service_info)
        if result:
            self.registered_services.add(service_info.name)
        return result
    
    async def deregister_service(self, service_name: str) -> bool:
        """Deregister a service and stop tracking it"""
        result = await self.registry.deregister(service_name)
        if result:
            self.registered_services.discard(service_name)
        return result
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats for registered services"""
        while self._running:
            try:
                for service_name in list(self.registered_services):
                    await self.registry.heartbeat(service_name)
                
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired services"""
        while self._running:
            try:
                await self.registry.cleanup_expired_services()
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10)

# Factory function
def create_service_registry(registry_type: str = "redis", **kwargs) -> ServiceRegistry:
    """Create a service registry instance"""
    
    if registry_type.lower() == "redis":
        redis_url = kwargs.get('redis_url', 'redis://localhost:6379/0')
        return RedisServiceRegistry(redis_url)
    elif registry_type.lower() == "consul":
        consul_url = kwargs.get('consul_url', 'http://localhost:8500')
        return ConsulServiceRegistry(consul_url)
    else:
        raise ValueError(f"Unsupported registry type: {registry_type}")

# Usage example
async def example_usage():
    """Example usage of service registry"""
    
    # Create registry
    registry = create_service_registry("redis")
    
    # Create service info
    service = ServiceInfo(
        name="data-service",
        url="http://localhost:8001",
        health_check_url="http://localhost:8001/health",
        version="1.0.0",
        metadata={"description": "Data management service"}
    )
    
    # Register service
    await registry.register(service)
    
    # Discover service
    discovered = await registry.discover("data-service")
    print(f"Discovered: {discovered}")
    
    # List all services
    services = await registry.list_services()
    print(f"All services: {list(services.keys())}")
    
    # Clean up
    if hasattr(registry, 'close'):
        await registry.close()

if __name__ == "__main__":
    asyncio.run(example_usage())