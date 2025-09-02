# Use RunPod's PyTorch base image with CUDA support
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Build argument for GitHub token (optional)
ARG GITHUB_TOKEN

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Clone required repositories to fixed locations
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /root/segment-anything-2
RUN git clone https://github.com/YxZhang/EVF-SAM.git /root/EVF-SAM
RUN git clone https://github.com/ZhengPeng7/BiRefNet.git /root/BiRefNet

# Clone the application repository to get requirements.txt
RUN git clone https://$GITHUB_TOKEN@github.com/claudivanfilho/sam2-api.git /app/src
WORKDIR /app/src

# Configure git and set up authenticated remote
RUN git config --global user.name "SAM2 API Bot" && \
    git config --global user.email "bot@sam2-api.com" && \
    git config --global init.defaultBranch main && \
    git remote set-url origin https://$GITHUB_TOKEN@github.com/claudivanfilho/sam2-api.git

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM2 dependencies
WORKDIR /root/segment-anything-2
RUN pip install -e .

# Install EVF-SAM dependencies
WORKDIR /root/EVF-SAM
RUN pip install -e .

# Install BiRefNet dependencies
WORKDIR /root/BiRefNet
RUN pip install -e .

# Download required model checkpoints
WORKDIR /root/segment-anything-2
RUN mkdir -p checkpoints && \
    wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Switch back to application directory
WORKDIR /app/src

# Create startup script that pulls latest changes
RUN echo '#!/bin/bash\n\
echo "🚀 Starting SAM2 API..."\n\
cd /app/src\n\
\n\
# Pull latest changes\n\
echo "📥 Pulling latest changes..."\n\
git pull origin main\n\
echo "✅ Code updated successfully"\n\
\n\
# Start the FastAPI server\n\
echo "🎯 Starting FastAPI server..."\n\
python main.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application with update check
CMD ["/app/start.sh"]
