from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Data Service", version="1.0.0")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "data-service",
        "version": "1.0.0"
    }

@app.get("/api/v1/data/market")
async def get_market_data():
    return {
        "symbol": "AAPL",
        "price": 150.25,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "data-service"
    }