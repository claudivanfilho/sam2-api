"""
Utility functions for SAM2 API.
Handles image loading, S3 uploads, and common operations.
"""

import base64
import io
import requests
import boto3
from botocore.exceptions import ClientError
import uuid
from datetime import datetime
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "growling-thunder-gold")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize S3 client
s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    print(f"✅ S3 client initialized")
else:
    print("⚠️  AWS credentials not found. S3 upload will be disabled.")

def load_image_from_source(image_b64=None, image_url=None):
    """
    Load image from either base64 string or URL.
    
    Args:
        image_b64: Base64 encoded image
        image_url: URL to image
        
    Returns:
        PIL.Image: Loaded image in RGB format
        
    Raises:
        ValueError: If neither source is provided or if loading fails
    """
    if image_b64:
        # Decode base64 -> PIL image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif image_url:
        # Download image from URL
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to download image from URL: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to process image from URL: {str(e)}")
    else:
        raise ValueError("Either image_b64 or imageUrl must be provided")
    
    return image

def upload_mask_to_s3(mask_array, original_image_size=None, crop_box=None):
    """
    Upload segmentation mask to S3 as a 1-bit PNG image for optimal binary image compression.
    
    Args:
        mask_array: numpy array of the segmentation mask
        original_image_size: tuple of (width, height) for resizing mask
        crop_box: tuple of (x1, y1, x2, y2) to crop the mask before uploading
    
    Returns:
        str: S3 URL of the uploaded mask, or None if upload failed
    """
    if not s3_client:
        print("⚠️  S3 client not initialized. Skipping upload.")
        return None
    
    try:
        s3_upload_start_time = datetime.now()
        
        # Optimize mask processing pipeline for performance
        # Direct binary conversion - ensure mask is boolean/binary
        mask_binary = (mask_array > 0).astype(np.uint8) * 255
        
        # Handle cropping on numpy array first (faster than PIL operations)
        if crop_box:
            x1, y1, x2, y2 = crop_box
            h, w = mask_binary.shape
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1, min(x2, w))
            y2 = max(y1, min(y2, h))
            mask_binary = mask_binary[y1:y2, x1:x2]
        
        # Convert directly to 1-bit PIL Image
        mask_image = Image.fromarray(mask_binary).convert('1')
        
        # Resize if needed (after cropping for better performance)
        if original_image_size and crop_box is None:  # Only resize if not cropped
            mask_image = mask_image.resize(original_image_size, Image.Resampling.NEAREST)
        
        # Convert to PNG bytes with optimized compression settings
        buffer = io.BytesIO()
        # Use faster compression settings - disable optimization for speed
        mask_image.save(buffer, format='PNG', optimize=False, compress_level=1)
        buffer.seek(0)
        
        # Generate unique filename with sam2-api-masks path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"sam2-api-masks/{timestamp}_{unique_id}.png"
        
        # Upload to S3 with optimized settings
        s3_client.upload_fileobj(
            buffer,
            S3_BUCKET_NAME,
            filename,
            ExtraArgs={
                'ContentType': 'image/png',
                'ACL': 'public-read',  # Make the file publicly accessible
                'StorageClass': 'STANDARD'  # Use standard storage for faster access
            }
        )
        
        s3_upload_end_time = datetime.now()
        s3_upload_duration = (s3_upload_end_time - s3_upload_start_time).total_seconds()
        
        # Generate public URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
        print(f"✅ Mask uploaded to S3 (optimized 1-bit PNG): {s3_url}")
        print(f"⏱️  S3 upload took {s3_upload_duration:.3f} seconds")
        return s3_url
        
    except ClientError as e:
        print(f"❌ Failed to upload mask to S3: {e}")
        return None
    except Exception as e:
        print(f"❌ Error uploading mask to S3: {e}")
        return None

def encode_image_to_base64(image):
    """
    Encode PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        
    Returns:
        str: Base64 encoded image
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
