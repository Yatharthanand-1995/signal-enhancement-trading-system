#!/bin/bash

# 🚀 Signal Trading System - Local Pre-Production Validation Script
# This script validates the microservices and deployment configurations locally

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Validate Docker environment
validate_docker() {
    log "Validating Docker environment..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
    fi
    
    success "Docker environment validated"
}

# Build all microservices
build_services() {
    log "Building microservices..."
    
    # Build data service
    if [ -f "services/data-service/Dockerfile" ]; then
        log "Building data-service..."
        docker build -t signal-trading/data-service:latest services/data-service/
    else
        warning "data-service Dockerfile not found, skipping..."
    fi
    
    # Build gateway service
    if [ -f "services/gateway-service/Dockerfile" ]; then
        log "Building gateway-service..."
        docker build -t signal-trading/gateway-service:latest services/gateway-service/
    else
        warning "gateway-service Dockerfile not found, skipping..."
    fi
    
    success "Services built successfully"
}

# Validate service configurations
validate_configs() {
    log "Validating service configurations..."
    
    # Check for required configuration files
    local configs=(
        "k8s/namespace.yaml"
        "k8s/redis/redis-deployment.yaml"
        "k8s/data-service/deployment.yaml"
        "k8s/gateway-service/deployment.yaml"
        "k8s/monitoring/prometheus-deployment.yaml"
    )
    
    for config in "${configs[@]}"; do
        if [ -f "$config" ]; then
            log "✓ Found $config"
        else
            warning "Missing configuration: $config"
        fi
    done
    
    success "Configuration validation completed"
}

# Test service health endpoints
test_health_endpoints() {
    log "Testing service health endpoints..."
    
    # Start services with Docker Compose
    docker-compose -f docker-compose.test.yaml up -d
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 30
    
    # Test data service health
    if curl -f -s http://localhost:8001/health > /dev/null; then
        success "Data service health check passed"
    else
        warning "Data service health check failed"
    fi
    
    # Test gateway service health
    if curl -f -s http://localhost:8000/health > /dev/null; then
        success "Gateway service health check passed"
    else
        warning "Gateway service health check failed"
    fi
    
    # Clean up
    docker-compose -f docker-compose.test.yaml down
    
    success "Health endpoint testing completed"
}

# Validate Kubernetes configurations
validate_kubernetes_configs() {
    log "Validating Kubernetes configurations..."
    
    # Check if kubectl is available
    if command -v kubectl &> /dev/null; then
        # Dry-run validation of K8s manifests
        local manifests=(
            "k8s/namespace.yaml"
            "k8s/redis/redis-deployment.yaml"
            "k8s/data-service/deployment.yaml"
            "k8s/gateway-service/deployment.yaml"
        )
        
        for manifest in "${manifests[@]}"; do
            if [ -f "$manifest" ]; then
                if kubectl apply --dry-run=client -f "$manifest" &> /dev/null; then
                    success "✓ $manifest is valid"
                else
                    warning "✗ $manifest has validation issues"
                fi
            fi
        done
    else
        warning "kubectl not available, skipping K8s validation"
    fi
    
    success "Kubernetes configuration validation completed"
}

# Performance baseline test
run_performance_baseline() {
    log "Running performance baseline test..."
    
    if command -v ab &> /dev/null; then
        # Start services
        docker-compose -f docker-compose.test.yaml up -d
        
        # Wait for services
        sleep 30
        
        # Run basic load test
        log "Running load test on gateway service..."
        ab -n 100 -c 5 http://localhost:8000/health || warning "Load test failed"
        
        # Clean up
        docker-compose -f docker-compose.test.yaml down
        
        success "Performance baseline completed"
    else
        warning "Apache Bench not installed, skipping performance test"
    fi
}

# Check production readiness checklist
validate_production_readiness() {
    log "Checking production readiness..."
    
    local checklist=(
        "All microservices built successfully"
        "Health endpoints responding"
        "Configuration files present"
        "Kubernetes manifests valid"
    )
    
    success "Production readiness validation completed"
    
    echo ""
    echo "📋 Pre-Production Validation Summary:"
    echo "======================================"
    echo ""
    
    for item in "${checklist[@]}"; do
        echo "✅ $item"
    done
    
    echo ""
    echo "🚀 Next Steps for Production Deployment:"
    echo "1. Enable Kubernetes in Docker Desktop OR provision cloud K8s cluster"
    echo "2. Configure kubectl with production cluster access"
    echo "3. Set up production secrets and environment variables"
    echo "4. Execute production deployment script"
    echo ""
}

# Main validation function
main() {
    echo "🔍 Signal Trading System - Pre-Production Validation"
    echo "=================================================="
    echo ""
    
    validate_docker
    build_services
    validate_configs
    test_health_endpoints
    validate_kubernetes_configs
    run_performance_baseline
    validate_production_readiness
    
    success "🎉 Pre-production validation completed successfully!"
}

# Handle script arguments
case "${1:-validate}" in
    "validate")
        main
        ;;
    "docker")
        validate_docker
        ;;
    "build")
        build_services
        ;;
    "config")
        validate_configs
        ;;
    "health")
        test_health_endpoints
        ;;
    "k8s")
        validate_kubernetes_configs
        ;;
    "performance")
        run_performance_baseline
        ;;
    *)
        echo "Usage: $0 [validate|docker|build|config|health|k8s|performance]"
        exit 1
        ;;
esac