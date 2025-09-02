# SAM2 API - RunPod Deployment Guide

A FastAPI server providing SAM2 segmentation, EVF-SAM2 text-prompted segmentation, and BiRefNet background removal capabilities.

## 🚀 Quick Deploy on RunPod

### Option 1: Use Pre-built Docker Image (Recommended)

1. Go to [RunPod](https://runpod.io)
2. Create a new pod with GPU support
3. Use this Docker image: `ghcr.io/claudivanfilho/sam2-api:latest`
4. Set container port: `8000`
5. Expose HTTP port: `8000`

### Option 2: Docker Command

```bash
docker run -d -p 8000:8000 --gpus all ghcr.io/claudivanfilho/sam2-api:latest
```

## 📋 API Endpoints

### Health Check

```
GET /health
```

### SAM2 Segmentation

```
POST /segment
```

### EVF-SAM2 Text-Prompted Segmentation

```
POST /segment/evf
```

### Background Removal

```
POST /remove-background
```

## 🔄 Automatic Updates

Every push to the `main` branch automatically:

1. Runs tests
2. Builds a new Docker image
3. Pushes to GitHub Container Registry
4. Tags with `latest` and commit SHA

To update your RunPod deployment:

1. Stop your current pod
2. Create a new pod with the same image (it will pull the latest version)
3. Or restart the container to pull the latest image

## 🛠️ Local Development

```bash
# Clone the repository
git clone https://github.com/claudivanfilho/sam2-api.git
cd sam2-api

# Build and run locally
docker build -t sam2-api .
docker run -p 8000:8000 --gpus all sam2-api
```

## 📊 Monitoring

- Health endpoint: `http://your-pod-url:8000/health`
- The container includes health checks that ping the `/health` endpoint every 30 seconds

## 🏗️ Architecture

- **Base Image**: PyTorch 2.1.0 with CUDA 12.1 support
- **Models**: SAM2.1 Large, EVF-SAM2, BiRefNet
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Deployment**: GitHub Actions → GitHub Container Registry → RunPod
