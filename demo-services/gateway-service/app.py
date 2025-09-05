from fastapi import FastAPI
from datetime import datetime
try:
    import httpx
except ImportError:
    httpx = None

app = FastAPI(title="Gateway Service", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "gateway-service",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {
        "message": "Signal Trading System Gateway",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/data/market")
async def proxy_market_data():
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://data-service:8001/api/v1/data/market")
                return response.json()
        except:
            pass
    return {
        "symbol": "DEMO",
        "price": 100.00,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "gateway-service-fallback"
    }