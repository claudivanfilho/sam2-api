#!/bin/bash

# SAM2 API Setup and Run Script
# This script installs all dependencies and runs the main.py application

set -e  # Exit on any error

echo "🚀 Starting SAM2 API setup..."

# Update system packages and install dependencies
echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

echo "✅ System dependencies installed"

# Create directories for repositories
echo "📁 Creating repository directories..."
mkdir -p /root/segment-anything-2
mkdir -p /root/EVF-SAM
mkdir -p /root/BiRefNet

# Clone required repositories to fixed locations
echo "📥 Cloning repositories..."

echo "  - Cloning segment-anything-2..."
if [ -d "/root/segment-anything-2" ]; then
    echo "    Directory already exists, removing and re-cloning..."
    rm -rf /root/segment-anything-2
fi
git clone https://github.com/facebookresearch/segment-anything-2.git /root/segment-anything-2

echo "  - Cloning EVF-SAM..."
if [ -d "/root/EVF-SAM" ]; then
    echo "    Directory already exists, removing and re-cloning..."
    rm -rf /root/EVF-SAM
fi
git clone https://github.com/hustvl/EVF-SAM.git /root/EVF-SAM

echo "  - Cloning BiRefNet..."
if [ -d "/root/BiRefNet" ]; then
    echo "    Directory already exists, removing and re-cloning..."
    rm -rf /root/BiRefNet
fi
git clone https://github.com/ZhengPeng7/BiRefNet.git /root/BiRefNet

echo "✅ All repositories cloned"

# Get current directory (where this script is located)
CURRENT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Install Python dependencies from requirements.txt
echo "🐍 Installing Python dependencies..."
if [ -f "$CURRENT_DIR/requirements.txt" ]; then
    pip install --no-cache-dir -r "$CURRENT_DIR/requirements.txt"
    echo "✅ Python dependencies installed"
else
    echo "⚠️  requirements.txt not found in $CURRENT_DIR"
fi

# Install SAM2 dependencies
echo "🔧 Installing SAM2 dependencies..."
cd /root/segment-anything-2
pip install -e .
echo "✅ SAM2 dependencies installed"

# Install EVF-SAM dependencies
echo "🔧 Installing EVF-SAM dependencies..."
cd /root/EVF-SAM
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    # Build extension for video prediction if needed
    if [ -d "model/segment_anything_2" ]; then
        cd model/segment_anything_2
        python setup.py build_ext --inplace
        cd /root/EVF-SAM
    fi
    echo "✅ EVF-SAM dependencies installed"
else
    echo "⚠️  requirements.txt not found in EVF-SAM directory"
fi

# Install BiRefNet dependencies
echo "🔧 Installing BiRefNet dependencies..."
cd /root/BiRefNet
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ BiRefNet dependencies installed"
else
    echo "⚠️  requirements.txt not found in BiRefNet directory"
fi

# Fix timm version compatibility issue
echo "🔧 Upgrading timm for compatibility..."
pip install --upgrade timm
echo "✅ timm upgraded"

# Download required model checkpoints
echo "⬇️  Downloading model checkpoints..."
cd /root/segment-anything-2
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
echo "✅ Model checkpoints downloaded"

# Return to the original directory
cd "$CURRENT_DIR"

# Run the main.py application
echo "🎯 Starting the application..."
if [ -f "$CURRENT_DIR/main.py" ]; then
    python3 main.py
else
    echo "❌ main.py not found in $CURRENT_DIR"
    exit 1
fi
