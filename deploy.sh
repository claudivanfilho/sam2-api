#!/bin/bash

# RunPod deployment script
# This script sets up the environment and starts the server

echo "🚀 Starting RunPod deployment..."

# Update system packages
apt-get update

# Clone or update the repository
if [ ! -d "/workspace/sam2-api" ]; then
    echo "📥 Cloning repository..."
    cd /workspace
    git clone https://github.com/claudivanfilho/sam2-api.git
    cd sam2-api
else
    echo "🔄 Updating repository..."
    cd /workspace/sam2-api
    git pull origin main
fi

# Install dependencies if requirements.txt has changed
pip install -r requirements.txt

# Start the server
echo "🎯 Starting FastAPI server..."
python main.py
