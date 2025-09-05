"""
Authentication and Authorization
Common authentication utilities for microservices
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import wraps
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import json

security = HTTPBearer()

class AuthManager:
    """Authentication manager for microservices"""
    
    def __init__(self, 
                 secret_key: Optional[str] = None,
                 token_expiry_hours: int = 24,
                 redis_url: Optional[str] = None):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "your-secret-key")
        self.token_expiry_hours = token_expiry_hours
        self.algorithm = "HS256"
        
        # Redis for token blacklisting and session management
        self.redis_client = None
        if redis_url or os.getenv("REDIS_URL"):
            try:
                import redis
                self.redis_client = redis.from_url(
                    redis_url or os.getenv("REDIS_URL"), 
                    decode_responses=True
                )
            except ImportError:
                pass
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=30)  # Longer expiry for refresh
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            # Check if token is blacklisted
            if self.redis_client and self.redis_client.get(f"blacklist:{token}"):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def revoke_token(self, token: str):
        """Revoke token by adding to blacklist"""
        if self.redis_client:
            try:
                # Get token expiry to set appropriate TTL
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
                exp_timestamp = payload.get('exp')
                
                if exp_timestamp:
                    ttl = int(exp_timestamp - datetime.utcnow().timestamp())
                    if ttl > 0:
                        self.redis_client.setex(f"blacklist:{token}", ttl, "revoked")
            except jwt.InvalidTokenError:
                pass
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token"""
        payload = self.verify_token(refresh_token)
        
        if payload.get('type') != 'refresh':
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Create new access token with same user data (excluding exp and type)
        user_data = {k: v for k, v in payload.items() if k not in ['exp', 'type', 'iat']}
        return self.create_access_token(user_data)

class TokenValidator:
    """Token validation dependency for FastAPI"""
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
    
    async def __call__(self, credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
        """Validate token and return user info"""
        if not credentials:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        
        return self.auth_manager.verify_token(credentials.credentials)

class RoleChecker:
    """Role-based access control"""
    
    def __init__(self, required_roles: List[str]):
        self.required_roles = required_roles
    
    def __call__(self, user_info: Dict[str, Any] = Depends()) -> Dict[str, Any]:
        """Check if user has required roles"""
        user_roles = user_info.get('roles', [])
        
        if not any(role in user_roles for role in self.required_roles):
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required roles: {self.required_roles}"
            )
        
        return user_info

class PermissionChecker:
    """Permission-based access control"""
    
    def __init__(self, required_permissions: List[str]):
        self.required_permissions = required_permissions
    
    def __call__(self, user_info: Dict[str, Any] = Depends()) -> Dict[str, Any]:
        """Check if user has required permissions"""
        user_permissions = user_info.get('permissions', [])
        
        if not all(perm in user_permissions for perm in self.required_permissions):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {self.required_permissions}"
            )
        
        return user_info

class ServiceAuth:
    """Service-to-service authentication"""
    
    def __init__(self, service_secrets: Dict[str, str]):
        self.service_secrets = service_secrets
    
    def create_service_token(self, service_name: str) -> str:
        """Create token for service-to-service communication"""
        if service_name not in self.service_secrets:
            raise ValueError(f"Unknown service: {service_name}")
        
        payload = {
            'service_name': service_name,
            'type': 'service',
            'exp': datetime.utcnow() + timedelta(hours=1)
        }
        
        return jwt.encode(payload, self.service_secrets[service_name], algorithm="HS256")
    
    def verify_service_token(self, token: str, expected_service: str) -> bool:
        """Verify service token"""
        try:
            if expected_service not in self.service_secrets:
                return False
            
            payload = jwt.decode(
                token, 
                self.service_secrets[expected_service], 
                algorithms=["HS256"]
            )
            
            return (
                payload.get('service_name') == expected_service and
                payload.get('type') == 'service'
            )
        except jwt.InvalidTokenError:
            return False

class APIKeyManager:
    """API key management for external access"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    def generate_api_key(self, user_id: str, name: str = "default") -> str:
        """Generate API key for user"""
        import secrets
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        
        if self.redis_client:
            key_data = {
                'user_id': user_id,
                'name': name,
                'created_at': datetime.utcnow().isoformat(),
                'last_used': None,
                'usage_count': 0
            }
            self.redis_client.setex(
                f"api_key:{api_key}", 
                30 * 24 * 3600,  # 30 days TTL
                json.dumps(key_data)
            )
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return user info"""
        if not self.redis_client:
            return None
        
        key_data = self.redis_client.get(f"api_key:{api_key}")
        if not key_data:
            return None
        
        try:
            data = json.loads(key_data)
            
            # Update last used and usage count
            data['last_used'] = datetime.utcnow().isoformat()
            data['usage_count'] = data.get('usage_count', 0) + 1
            
            self.redis_client.setex(
                f"api_key:{api_key}",
                30 * 24 * 3600,  # Reset TTL
                json.dumps(data)
            )
            
            return data
        except json.JSONDecodeError:
            return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if self.redis_client:
            return bool(self.redis_client.delete(f"api_key:{api_key}"))
        return False

# Decorators for route protection
def require_auth(auth_manager: AuthManager):
    """Decorator to require authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would be implemented differently in each framework
            # For FastAPI, use Depends() in route parameters
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_roles(roles: List[str]):
    """Decorator to require specific roles"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Role checking logic would be implemented here
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_permissions(permissions: List[str]):
    """Decorator to require specific permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Permission checking logic would be implemented here
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Session management
class SessionManager:
    """Session management for user sessions"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    def create_session(self, user_id: str, session_data: Dict[str, Any] = None) -> str:
        """Create user session"""
        import secrets
        session_id = secrets.token_urlsafe(32)
        
        if self.redis_client:
            data = {
                'user_id': user_id,
                'created_at': datetime.utcnow().isoformat(),
                'last_accessed': datetime.utcnow().isoformat(),
                **((session_data or {}))
            }
            
            self.redis_client.setex(
                f"session:{session_id}",
                24 * 3600,  # 24 hours TTL
                json.dumps(data, default=str)
            )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if not self.redis_client:
            return None
        
        session_data = self.redis_client.get(f"session:{session_id}")
        if not session_data:
            return None
        
        try:
            data = json.loads(session_data)
            
            # Update last accessed
            data['last_accessed'] = datetime.utcnow().isoformat()
            self.redis_client.setex(
                f"session:{session_id}",
                24 * 3600,  # Reset TTL
                json.dumps(data, default=str)
            )
            
            return data
        except json.JSONDecodeError:
            return None
    
    def update_session(self, session_id: str, data: Dict[str, Any]):
        """Update session data"""
        if self.redis_client:
            existing = self.get_session(session_id)
            if existing:
                existing.update(data)
                self.redis_client.setex(
                    f"session:{session_id}",
                    24 * 3600,
                    json.dumps(existing, default=str)
                )
    
    def destroy_session(self, session_id: str):
        """Destroy session"""
        if self.redis_client:
            self.redis_client.delete(f"session:{session_id}")