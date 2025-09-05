#!/bin/bash

# 🚀 Signal Trading System - Production Deployment Script
# This script automates the production deployment process
# Version: 1.0
# Date: September 3, 2025

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="signal-trading-prod"
NAMESPACE_PROD="signal-trading"
NAMESPACE_MONITORING="signal-trading-monitoring"
DOMAIN="signaltrading.com"
REGION="us-west-2"

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

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
    fi
    
    # Check if we can access the correct cluster
    CURRENT_CONTEXT=$(kubectl config current-context)
    log "Current kubectl context: $CURRENT_CONTEXT"
    
    # Verify cluster version
    CLUSTER_VERSION=$(kubectl version --short 2>/dev/null | grep "Server Version" | awk '{print $3}')
    log "Kubernetes cluster version: $CLUSTER_VERSION"
    
    success "Prerequisites check passed"
}

# Create production secrets
create_secrets() {
    log "Creating production secrets..."
    
    # Generate JWT secret if not provided
    if [ -z "${JWT_SECRET:-}" ]; then
        JWT_SECRET=$(openssl rand -base64 32)
        warning "Generated new JWT secret. Store this securely: $JWT_SECRET"
    fi
    
    # Create namespace first
    kubectl create namespace $NAMESPACE_PROD --dry-run=client -o yaml | kubectl apply -f -
    
    # Create gateway secrets
    kubectl create secret generic gateway-secrets \
        --from-literal=jwt-secret="$JWT_SECRET" \
        --namespace=$NAMESPACE_PROD \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create monitoring namespace and secrets
    kubectl create namespace $NAMESPACE_MONITORING --dry-run=client -o yaml | kubectl apply -f -
    
    # Create basic auth for monitoring (default credentials)
    MONITORING_PASSWORD=${MONITORING_PASSWORD:-"signal-trading-2025"}
    MONITORING_AUTH=$(echo -n "admin:$MONITORING_PASSWORD" | base64)
    
    kubectl create secret generic monitoring-auth \
        --from-literal=auth="admin:$(openssl passwd -apr1 $MONITORING_PASSWORD)" \
        --namespace=$NAMESPACE_MONITORING \
        --dry-run=client -o yaml | kubectl apply -f -
    
    success "Production secrets created"
}

# Deploy infrastructure components
deploy_infrastructure() {
    log "Deploying infrastructure components..."
    
    # Apply namespaces
    kubectl apply -f k8s/namespace.yaml
    
    # Apply RBAC and secrets
    kubectl apply -f k8s/secrets/secrets.yaml
    
    # Deploy Redis (Service Discovery)
    kubectl apply -f k8s/redis/redis-deployment.yaml
    
    # Wait for Redis to be ready
    log "Waiting for Redis to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n $NAMESPACE_PROD
    
    success "Infrastructure components deployed"
}

# Deploy microservices
deploy_microservices() {
    log "Deploying microservices..."
    
    # Deploy Data Service
    log "Deploying Data Service..."
    kubectl apply -f k8s/data-service/deployment.yaml
    kubectl rollout status deployment/data-service -n $NAMESPACE_PROD --timeout=600s
    
    # Deploy Gateway Service
    log "Deploying Gateway Service..."
    kubectl apply -f k8s/gateway-service/deployment.yaml
    kubectl rollout status deployment/gateway-service -n $NAMESPACE_PROD --timeout=600s
    
    # Deploy additional services if configurations exist
    for service in signal-service ml-service risk-service; do
        if [ -f "k8s/$service/deployment.yaml" ]; then
            log "Deploying $service..."
            kubectl apply -f "k8s/$service/deployment.yaml"
            kubectl rollout status "deployment/$service" -n $NAMESPACE_PROD --timeout=600s
        else
            warning "$service deployment file not found, skipping..."
        fi
    done
    
    success "Microservices deployed"
}

# Deploy networking
deploy_networking() {
    log "Deploying networking and ingress..."
    
    # Check if NGINX Ingress Controller is installed
    if ! kubectl get pods -n ingress-nginx 2>/dev/null | grep -q nginx-controller; then
        warning "NGINX Ingress Controller not found. Installing..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
        
        # Wait for ingress controller to be ready
        kubectl wait --namespace ingress-nginx \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/component=controller \
            --timeout=300s
    fi
    
    # Apply ingress configuration
    kubectl apply -f k8s/ingress/ingress.yaml
    
    # Get the external IP
    log "Waiting for external IP assignment..."
    EXTERNAL_IP=""
    while [ -z "$EXTERNAL_IP" ]; do
        sleep 10
        EXTERNAL_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        if [ -z "$EXTERNAL_IP" ]; then
            EXTERNAL_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        fi
        log "Waiting for external IP/hostname..."
    done
    
    success "Networking deployed. External IP/Hostname: $EXTERNAL_IP"
    warning "Please update your DNS records to point api.$DOMAIN to $EXTERNAL_IP"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Deploy Prometheus
    kubectl apply -f k8s/monitoring/prometheus-config.yaml
    kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
    
    # Deploy Grafana
    kubectl apply -f k8s/monitoring/grafana-deployment.yaml
    
    # Wait for monitoring components
    log "Waiting for monitoring components to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/prometheus -n $NAMESPACE_MONITORING
    kubectl wait --for=condition=available --timeout=600s deployment/grafana -n $NAMESPACE_MONITORING
    
    success "Monitoring stack deployed"
}

# Deploy auto-scaling
deploy_autoscaling() {
    log "Deploying auto-scaling configuration..."
    
    # Check if metrics server is installed
    if ! kubectl get deployment metrics-server -n kube-system &>/dev/null; then
        warning "Metrics server not found. Installing..."
        kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
        
        # Wait for metrics server
        kubectl wait --for=condition=available --timeout=300s deployment/metrics-server -n kube-system
    fi
    
    # Deploy HPA configurations
    kubectl apply -f k8s/autoscaling/hpa.yaml
    
    # Deploy VPA configurations if VPA is installed
    if kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &>/dev/null; then
        kubectl apply -f k8s/autoscaling/vpa.yaml
        success "HPA and VPA deployed"
    else
        warning "VPA not installed, skipping VPA configuration"
        success "HPA deployed"
    fi
}

# Deploy backup system
deploy_backup() {
    log "Deploying backup system..."
    
    # Check if Velero is installed
    if kubectl get namespace velero &>/dev/null; then
        kubectl apply -f k8s/backup/velero-backup.yaml
        success "Backup schedules deployed"
    else
        warning "Velero not installed. Please install Velero for automated backups."
        warning "See: https://velero.io/docs/main/basic-install/"
    fi
    
    # Deploy disaster recovery plans
    kubectl apply -f k8s/disaster-recovery/disaster-recovery-plan.yaml
    success "Disaster recovery plans deployed"
}

# Health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check pod status
    log "Checking pod status..."
    kubectl get pods -n $NAMESPACE_PROD
    kubectl get pods -n $NAMESPACE_MONITORING
    
    # Check service status
    log "Checking service status..."
    kubectl get svc -n $NAMESPACE_PROD
    kubectl get svc -n $NAMESPACE_MONITORING
    
    # Test health endpoints
    log "Testing health endpoints..."
    
    # Wait a bit for services to be fully ready
    sleep 30
    
    # Get service URLs for testing
    GATEWAY_IP=$(kubectl get svc gateway-service -n $NAMESPACE_PROD -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$GATEWAY_IP" ]; then
        GATEWAY_IP=$(kubectl get svc gateway-service -n $NAMESPACE_PROD -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [ ! -z "$GATEWAY_IP" ]; then
        if curl -f -s "http://$GATEWAY_IP:8000/health" > /dev/null; then
            success "Gateway service health check passed"
        else
            warning "Gateway service health check failed"
        fi
    else
        warning "Could not determine Gateway service external IP"
    fi
    
    # Check internal service connectivity
    kubectl exec -n $NAMESPACE_PROD deployment/gateway-service -- curl -f http://data-service:8001/health || warning "Data service connectivity check failed"
    
    success "Health checks completed"
}

# Performance test
run_performance_test() {
    log "Running basic performance test..."
    
    # Get external IP for testing
    EXTERNAL_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$EXTERNAL_IP" ]; then
        EXTERNAL_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    if [ ! -z "$EXTERNAL_IP" ]; then
        # Test using IP address (since DNS might not be configured yet)
        if command -v ab &> /dev/null; then
            log "Running load test with Apache Bench..."
            ab -n 100 -c 5 -H "Host: api.$DOMAIN" "http://$EXTERNAL_IP/health" || warning "Load test failed"
        else
            warning "Apache Bench not installed, skipping load test"
        fi
    else
        warning "Could not determine external IP for performance testing"
    fi
    
    success "Performance test completed"
}

# Display deployment summary
show_deployment_summary() {
    log "Deployment Summary"
    echo "=================="
    
    # Get resource status
    echo ""
    echo "📦 Pod Status:"
    kubectl get pods -n $NAMESPACE_PROD -o wide
    echo ""
    kubectl get pods -n $NAMESPACE_MONITORING -o wide
    
    echo ""
    echo "🌐 Service Status:"
    kubectl get svc -n $NAMESPACE_PROD
    echo ""
    kubectl get svc -n $NAMESPACE_MONITORING
    
    echo ""
    echo "🔗 Ingress Status:"
    kubectl get ingress -n $NAMESPACE_PROD
    
    # Get external access information
    EXTERNAL_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -z "$EXTERNAL_IP" ]; then
        EXTERNAL_IP=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    fi
    
    echo ""
    echo "🚀 Access Information:"
    echo "  External IP/Hostname: $EXTERNAL_IP"
    echo "  API URL (after DNS): https://api.$DOMAIN"
    echo "  Health Check: https://api.$DOMAIN/health"
    echo ""
    echo "📊 Monitoring Access:"
    echo "  Grafana: http://$EXTERNAL_IP:3000 (admin/signal-trading-2025)"
    echo "  Prometheus: http://$EXTERNAL_IP:9090"
    
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Update DNS records to point api.$DOMAIN to $EXTERNAL_IP"
    echo "  2. Configure SSL certificates (Let's Encrypt will auto-provision)"
    echo "  3. Monitor system performance and logs"
    echo "  4. Set up monitoring alerts and dashboards"
    echo "  5. Execute user acceptance testing"
    
    success "Production deployment completed successfully!"
}

# Main deployment function
main() {
    echo "🚀 Signal Trading System - Production Deployment"
    echo "================================================"
    echo ""
    
    # Run deployment steps
    check_prerequisites
    create_secrets
    deploy_infrastructure
    deploy_microservices
    deploy_networking
    deploy_monitoring
    deploy_autoscaling
    deploy_backup
    run_health_checks
    run_performance_test
    show_deployment_summary
    
    echo ""
    success "🎉 Production deployment completed successfully!"
    echo ""
    warning "⚠️  Important: Update your DNS records to point api.$DOMAIN to the external IP shown above"
    warning "⚠️  Important: Verify SSL certificate provisioning after DNS propagation"
    warning "⚠️  Important: Set up monitoring alerts and configure on-call procedures"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "check")
        check_prerequisites
        ;;
    "secrets")
        create_secrets
        ;;
    "infrastructure")
        deploy_infrastructure
        ;;
    "services")
        deploy_microservices
        ;;
    "networking")
        deploy_networking
        ;;
    "monitoring")
        deploy_monitoring
        ;;
    "autoscaling")
        deploy_autoscaling
        ;;
    "backup")
        deploy_backup
        ;;
    "health")
        run_health_checks
        ;;
    "test")
        run_performance_test
        ;;
    "summary")
        show_deployment_summary
        ;;
    *)
        echo "Usage: $0 [deploy|check|secrets|infrastructure|services|networking|monitoring|autoscaling|backup|health|test|summary]"
        exit 1
        ;;
esac