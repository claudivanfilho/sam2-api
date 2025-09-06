"""
SAM2 API - Refactored main application file.
Now imports functionality from separate modules for better organization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
import time

# Import our modular components from src package
from src.background_removal import background_remover
from src.sam2_segmentation import sam2_segmenter
from src.object_detection import object_detector
from src.evf_sam_operations import evf_sam_segmenter
from src.mediapipe_operations import mediapipe_processor
from src.model_manager import get_model_manager

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    """Preload all models on application startup."""
    print("🚀 FastAPI startup: Preloading all AI models...")
    model_manager = get_model_manager()  # This will trigger preloading
    print("✅ All models preloaded successfully!")

def process_background_removal_and_upload(image_pil):
    """
    Process background removal and upload both mask and resized image to S3 in parallel.
    
    Args:
        image_pil: PIL Image object
        
    Returns:
        tuple: (background_removed_url, mask_url) - S3 URLs for both uploads
    """
    # Run background removal and get the mask for S3 upload
    background_start_time = time.time()
    background_result = background_remover.remove_background(
        image_pil=image_pil,
        return_mask=True  # We need the mask for S3 upload
    )
    background_end_time = time.time()
    background_duration = background_end_time - background_start_time
    print(f"⏱️  Background removal took {background_duration:.3f} seconds")
    
    # Prepare both images for parallel upload
    from src.utils import upload_image_to_s3
    from PIL import Image
    import asyncio
    import concurrent.futures
    from functools import partial
    
    # Get both images
    background_mask_pil = background_result['mask_pil']
    background_removed_pil = background_result['image_pil']
    
    # Resize the background-removed image (max size 1024)
    w, h = background_removed_pil.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        background_removed_resized = background_removed_pil.resize((new_w, new_h), Image.LANCZOS)
    else:
        background_removed_resized = background_removed_pil
    
    # Define async upload functions
    async def upload_mask():
        loop = asyncio.get_event_loop()
        upload_func = partial(upload_image_to_s3, background_mask_pil, s3_path="background-removal-masks")
        return await loop.run_in_executor(None, upload_func)
    
    async def upload_background_image():
        loop = asyncio.get_event_loop()
        upload_func = partial(upload_image_to_s3, background_removed_resized, s3_path="background-removed-images")
        return await loop.run_in_executor(None, upload_func)
    
    # Run both uploads in parallel
    async def run_parallel_uploads():
        s3_upload_start_time = time.time()
        mask_url, background_removed_url = await asyncio.gather(
            upload_mask(),
            upload_background_image()
        )
        s3_upload_end_time = time.time()
        s3_upload_duration = s3_upload_end_time - s3_upload_start_time
        print(f"⏱️  Parallel S3 uploads (mask + resized image) took {s3_upload_duration:.3f} seconds")
        return background_removed_url, mask_url
    
    # Execute the parallel uploads
    return asyncio.run(run_parallel_uploads())

# Pydantic models for API requests and responses
class Point(BaseModel):
    x: float
    y: float
    label: int = 1  # 1 for foreground, 0 for background

class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class SegmentRequest(BaseModel):
    image_b64: str = None    # Optional base64 encoded image
    imageUrl: str = None     # Optional image URL
    point: Point = None      # Optional point prompt
    box: Box = None          # Optional box prompt
    points: list[Point] = None  # Optional multiple points
    boxes: list[Box] = None     # Optional multiple boxes

class SegmentResponse(BaseModel):
    mask_url: Optional[str] = None  # S3 URL of the uploaded mask image

class EVFSegmentRequest(BaseModel):
    image_b64: str = None      # Optional base64 encoded image
    imageUrl: str = None       # Optional image URL  
    prompt: str                # Text prompt for segmentation
    semantic: bool = False     # Whether to use semantic segmentation

class EVFSegmentResponse(BaseModel):
    mask: list

class RemoveBackgroundRequest(BaseModel):
    image_b64: str = None      # Optional base64 encoded image
    imageUrl: str = None       # Optional image URL
    return_mask: bool = False  # Whether to return just the mask or the image with transparent background

class RemoveBackgroundResponse(BaseModel):
    image_b64: str = None      # Base64 encoded result image (with transparent background)
    mask_b64: str = None       # Base64 encoded mask (if requested)

class ObjectDetectionRequest(BaseModel):
    image_b64: str = None      # Optional base64 encoded image
    imageUrl: str = None       # Optional image URL
    task: str = "<OD>"         # Task prompt (default: object detection)
    confidence_threshold: float = 0.3  # Minimum confidence for detections
    text_input: str = None     # Text input for referring expression segmentation
    return_polygon_points: bool = False  # If True, return simplified polygon points instead of S3 URLs for masks
    douglas_peucker_epsilon: float = 0.002  # Douglas-Peucker simplification epsilon ratio (default: 0.002 = 0.2%)

class DetectedObject(BaseModel):
    label: str
    bbox: list[float]  # [x1, y1, x2, y2]
    multiclasses: Optional[dict] = None  # Selfie segmentation result (URLs or polygon points) for person objects
    segment_mask: Optional[Union[str, list[list[float]]]] = None  # S3 URL or polygon points of SAM2 segmentation mask

class ObjectDetectionResponse(BaseModel):
    objects: list[DetectedObject]
    task_used: str
    segmentation_mask: Optional[list] = None  # For RES tasks
    background_removed_url: Optional[str] = None  # S3 URL of resized background-removed image (max 1024px)
    background_mask_url: Optional[str] = None  # S3 URL of background removal mask

# API Endpoints

@app.post("/segment", response_model=SegmentResponse)
async def segment_runpod_style(request: SegmentRequest):
    """
    RunPod-style endpoint that accepts base64 image or image URL with optional prompts.
    
    Input formats:
    
    1. Basic with base64 (auto-segment center):
    {
      "image_b64": "<base64 encoded image>"
    }
    
    2. Basic with URL (auto-segment center):
    {
      "imageUrl": "https://example.com/image.jpg"
    }
    
    3. With single point:
    {
      "image_b64": "<base64 encoded image>",
      "point": {"x": 100, "y": 150, "label": 1}
    }
    
    4. With single box and URL:
    {
      "imageUrl": "https://example.com/image.jpg",
      "box": {"x1": 50, "y1": 50, "x2": 200, "y2": 200}
    }
    
    5. With multiple points:
    {
      "image_b64": "<base64 encoded image>",
      "points": [
        {"x": 100, "y": 150, "label": 1},
        {"x": 200, "y": 250, "label": 0}
      ]
    }
    
    6. With multiple boxes and URL:
    {
      "imageUrl": "https://example.com/image.jpg",
      "boxes": [
        {"x1": 50, "y1": 50, "x2": 200, "y2": 200},
        {"x1": 300, "y1": 300, "x2": 400, "y2": 400}
      ]
    }
    
    Note: Either image_b64 OR imageUrl must be provided (not both).
    
    Returns:
    {
      "mask_url": "https://your-bucket.s3.region.amazonaws.com/masks/20240902_143022_abc12345.png"
    }
    
    The segmentation mask is automatically uploaded to S3 and only the URL is returned.
    """
    try:
        # Validate input
        if not request.image_b64 and not request.imageUrl:
            raise HTTPException(status_code=400, detail="Either image_b64 or imageUrl must be provided")
        
        if request.image_b64 and request.imageUrl:
            raise HTTPException(status_code=400, detail="Provide either image_b64 OR imageUrl, not both")
        
        mask_url = sam2_segmenter.segment_image(
            image_b64=request.image_b64,
            image_url=request.imageUrl,
            point=request.point,
            box=request.box,
            points=request.points,
            boxes=request.boxes
        )
        return SegmentResponse(mask_url=mask_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

@app.post("/segment/evf", response_model=EVFSegmentResponse)
async def segment_evf(request: EVFSegmentRequest):
    """
    EVF-SAM2 text-prompted segmentation endpoint.
    Supports text prompts for referring expression segmentation and semantic segmentation.
    """
    try:
        mask_list = evf_sam_segmenter.segment_with_text(
            image_b64=request.image_b64,
            image_url=request.imageUrl,
            prompt=request.prompt,
            semantic=request.semantic
        )
        
        return EVFSegmentResponse(mask=mask_list)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EVF segmentation failed: {str(e)}")

@app.post("/detect-objects-and-bg-removal", response_model=ObjectDetectionResponse)
async def detect_objects_endpoint(request: ObjectDetectionRequest):
    """
    Object detection endpoint using Florence-2-large model.
    
    Florence-2 is a vision foundation model that can perform various vision tasks through text prompts.
    Supports object detection, image captioning, and referring expression segmentation.
    
    Example usage:
    
    1. Basic object detection with base64:
    {
      "image_b64": "<base64 encoded image>"
    }
    
    2. Object detection with URL:
    {
      "imageUrl": "https://example.com/image.jpg"
    }
    
    3. Referring expression segmentation (text-prompted segmentation):
    {
      "image_b64": "<base64 encoded image>",
      "task": "<REFERRING_EXPRESSION_SEGMENTATION>",
      "text_input": "hair"
    }
    
    4. Custom task prompt:
    {
      "image_b64": "<base64 encoded image>",
      "task": "<DENSE_REGION_CAPTION>"
    }
    
    5. Return polygon points instead of S3 URLs:
    {
      "image_b64": "<base64 encoded image>",
      "return_polygon_points": true,
      "douglas_peucker_epsilon": 0.001
    }
    
    Parameters:
    - return_polygon_points: If true, returns simplified polygon points instead of S3 URLs for masks
    - douglas_peucker_epsilon: Controls polygon simplification (lower = more detailed, higher = simpler)
    
    Supported tasks:
    - "<OD>": Object detection (default)
    - "<REFERRING_EXPRESSION_SEGMENTATION>": Segment objects by text description
    - "<DENSE_REGION_CAPTION>": Dense region captioning
    - "<REGION_PROPOSAL>": Region proposal
    - "<CAPTION>": Image captioning
    - "<DETAILED_CAPTION>": Detailed image captioning
    - "<MORE_DETAILED_CAPTION>": More detailed image captioning
    
    Returns detected objects with labels and bounding boxes.
    Objects smaller than 300px in width or height are automatically filtered out.
    For RES tasks, also returns segmentation polygons.
    Note: Florence-2 doesn't provide confidence scores, so confidence will be None.
    """
    # Start timing the entire endpoint processing
    endpoint_start_time = time.time()
    
    try:
        # Validate input
        if not request.image_b64 and not request.imageUrl:
            raise HTTPException(status_code=400, detail="Either image_b64 or imageUrl must be provided")
        
        if request.image_b64 and request.imageUrl:
            raise HTTPException(status_code=400, detail="Provide either image_b64 OR imageUrl, not both")
        
        # Load image from source - moved from object_detection to main.py
        from src.utils import load_image_from_source
        image_pil = load_image_from_source(image_b64=request.image_b64, image_url=request.imageUrl)
        
        # Run background removal and object detection in parallel
        import asyncio
        import concurrent.futures
        from functools import partial
        
        async def run_background_processing():
            """Run background removal and upload in thread pool."""
            loop = asyncio.get_event_loop()
            background_func = partial(process_background_removal_and_upload, image_pil)
            return await loop.run_in_executor(None, background_func)
        
        async def run_object_detection():
            """Run object detection in thread pool."""
            loop = asyncio.get_event_loop()
            detection_func = partial(
                object_detector.detect_objects,
                image_pil=image_pil,
                task=request.task,
                confidence_threshold=request.confidence_threshold,
                text_input=request.text_input,
                return_polygon_points=request.return_polygon_points,
                douglas_peucker_epsilon=request.douglas_peucker_epsilon
            )
            return await loop.run_in_executor(None, detection_func)
        
        # Execute both operations concurrently
        (background_removed_url, mask_url), result = await asyncio.gather(
            run_background_processing(),
            run_object_detection()
        )
        
        # Convert to response format
        objects = [
            DetectedObject(
                label=obj['label'],
                bbox=obj['bbox'],
                multiclasses=obj.get('multiclasses'),
                segment_mask=obj.get('segment_mask')
            )
            for obj in result['objects']
        ]
        
        # Calculate total endpoint processing time
        endpoint_end_time = time.time()
        endpoint_duration = endpoint_end_time - endpoint_start_time
        print(f"🏁 Total endpoint processing took {endpoint_duration:.3f} seconds")
        
        return ObjectDetectionResponse(
            objects=objects,
            task_used=result['task_used'],
            segmentation_mask=result.get('segmentation_mask'),
            background_removed_url=background_removed_url,
            background_mask_url=mask_url
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")

@app.post("/remove-background", response_model=RemoveBackgroundResponse)
async def remove_background_endpoint(request: RemoveBackgroundRequest):
    """
    Background removal endpoint using BiRefNet (official implementation).
    BiRefNet achieves SOTA performance on dichotomous image segmentation.
    """
    try:
        result = background_remover.remove_background(
            image_b64=request.image_b64,
            image_url=request.imageUrl,
            return_mask=request.return_mask
        )
        
        # Convert PIL images to base64 for API response
        import io
        import base64
        
        # Convert image_pil to base64
        image_buffer = io.BytesIO()
        result['image_pil'].save(image_buffer, format="PNG")
        image_b64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        
        response_data = {"image_b64": image_b64}
        
        # Convert mask_pil to base64 if present
        if 'mask_pil' in result:
            mask_buffer = io.BytesIO()
            result['mask_pil'].save(mask_buffer, format="PNG")
            mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
            response_data["mask_b64"] = mask_b64
        
        return RemoveBackgroundResponse(**response_data)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_manager = get_model_manager()
    return {
        "status": "healthy",
        "device": model_manager.get_device(),
        "model_loaded": True,
        "evf_model_loaded": True,
        "background_removal_loaded": True,
        "object_detection_loaded": True,
        "face_landmarks_loaded": True,
        "flux_model_loaded": False,
        "models": {
            "sam2": "SAM2.1 Large (Segmentation)",
            "evf_sam2": "EVF-SAM2 (Text-prompted Segmentation)",
            "birefnet": "BiRefNet (Background Removal)",
            "florence2": "Florence-2-large (Object Detection + Integrated Selfie Segmentation + Face Landmarks)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Preload all models at startup
    print("🚀 Starting FastAPI server...")
    print("📦 Initializing and preloading all AI models...")
    model_manager = get_model_manager()  # This will trigger preloading
    print(f"✅ Server ready on {model_manager.get_device()}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
