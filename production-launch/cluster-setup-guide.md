# 🚀 Production Kubernetes Cluster Setup Guide

**Project**: Signal Trading System Production Deployment  
**Phase**: Production Cluster Provisioning  
**Date**: September 3, 2025  

---

## 📋 Cluster Options

### Option 1: Docker Desktop Kubernetes (Development/Testing)

#### Enable Kubernetes in Docker Desktop:
1. Open Docker Desktop
2. Go to Settings → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"
5. Wait for Kubernetes to start (green status)

#### Verify Installation:
```bash
kubectl cluster-info
kubectl get nodes
```

### Option 2: Cloud Provider Kubernetes (Production Recommended)

#### AWS EKS Setup:
```bash
# Install eksctl (if not installed)
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create EKS cluster
eksctl create cluster \
  --name signal-trading-prod \
  --region us-west-2 \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed
```

#### Google GKE Setup:
```bash
# Install gcloud CLI and authenticate
gcloud auth login

# Create GKE cluster
gcloud container clusters create signal-trading-prod \
  --zone us-west1-a \
  --machine-type e2-medium \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10
```

#### Azure AKS Setup:
```bash
# Install Azure CLI and login
az login

# Create resource group
az group create --name signal-trading --location westus2

# Create AKS cluster
az aks create \
  --resource-group signal-trading \
  --name signal-trading-prod \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

### Option 3: Local Minikube (Development)

#### Install and Start Minikube:
```bash
# Install minikube (macOS)
brew install minikube

# Start cluster
minikube start --memory=8192 --cpus=4 --disk-size=50g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server
```

---

## 🔧 Cluster Configuration

### Required Cluster Resources:
- **Minimum**: 8 vCPU, 16GB RAM, 100GB storage
- **Recommended**: 16 vCPU, 32GB RAM, 200GB storage
- **Auto-scaling**: Enabled with 2-10 node range

### Essential Add-ons:
- **Ingress Controller**: NGINX Ingress Controller
- **Metrics Server**: For HPA/VPA functionality
- **CSI Storage**: For persistent volumes

### Network Requirements:
- **LoadBalancer** support for external access
- **NodePort** range: 30000-32767
- **Pod CIDR**: 10.244.0.0/16 (configurable)
- **Service CIDR**: 10.96.0.0/12 (configurable)

---

## 🔒 Security Configuration

### RBAC Setup:
```yaml
# Create admin service account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: signal-trading-admin
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: signal-trading-admin
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: ServiceAccount
  name: signal-trading-admin
  namespace: kube-system
```

### Network Policies:
```yaml
# Default deny all ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: signal-trading
spec:
  podSelector: {}
  policyTypes:
  - Ingress
```

---

## ✅ Cluster Validation Script

Create and run this validation script:

```bash
#!/bin/bash
# cluster-validation.sh

echo "🔍 Validating Kubernetes cluster..."

# Check cluster info
kubectl cluster-info

# Check node status
echo "📊 Node Status:"
kubectl get nodes -o wide

# Check system pods
echo "🏗️  System Pods:"
kubectl get pods -n kube-system

# Check available resources
echo "💾 Resource Availability:"
kubectl top nodes 2>/dev/null || echo "Metrics server not available"

# Check storage classes
echo "💽 Storage Classes:"
kubectl get storageclass

# Test service creation
echo "🧪 Testing Service Creation:"
kubectl create namespace test-validation
kubectl run test-pod --image=nginx --port=80 -n test-validation
kubectl wait --for=condition=ready pod/test-pod -n test-validation --timeout=60s
kubectl delete namespace test-validation

echo "✅ Cluster validation completed!"
```

---

## 📊 Monitoring Setup

### Install Metrics Server:
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Install Ingress Controller:
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
```

### Verify Installation:
```bash
# Check metrics server
kubectl get deployment metrics-server -n kube-system

# Check ingress controller
kubectl get pods -n ingress-nginx
```

---

## 🚀 Ready for Deployment

Once your cluster is set up and validated, you're ready to proceed with the production deployment:

1. **Update kubectl context** to point to your production cluster
2. **Verify cluster access** with `kubectl cluster-info`
3. **Run cluster validation** script above
4. **Execute production deployment** using `./production-launch/deploy-production.sh`

---

## 📞 Troubleshooting

### Common Issues:

#### Cluster Not Accessible:
```bash
# Check kubeconfig
kubectl config view

# Switch context if needed
kubectl config use-context <cluster-context>
```

#### Insufficient Resources:
```bash
# Check resource usage
kubectl top nodes
kubectl describe nodes
```

#### Network Issues:
```bash
# Check cluster networking
kubectl get pods -n kube-system
kubectl logs -n kube-system <network-pod>
```

### Quick Fixes:
- **Docker Desktop**: Restart Docker Desktop and re-enable Kubernetes
- **Cloud Clusters**: Check cloud provider console for cluster status
- **Network**: Verify firewall rules and security groups

---

**Next Step**: Once your cluster is ready, execute `./production-launch/deploy-production.sh` to begin the production deployment process.