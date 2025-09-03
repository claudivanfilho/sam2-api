from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from pydantic import BaseModel
from typing import Optional
import torch
from PIL import Image
import io, base64
import os
import numpy as np
import requests
import sys
import boto3
from botocore.exceptions import ClientError
import uuid
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append('/root/EVF-SAM')
sys.path.append('/root/BiRefNet')
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from models.birefnet import BiRefNet
from dotenv import load_dotenv
import mediapipe as mp

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for RunPod-style API
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

# EVF-SAM request model
class EVFSegmentRequest(BaseModel):
    image_b64: str = None      # Optional base64 encoded image
    imageUrl: str = None       # Optional image URL  
    prompt: str                # Text prompt for segmentation
    semantic: bool = False     # Whether to use semantic segmentation

class EVFSegmentResponse(BaseModel):
    mask: list

# Background removal request model
class RemoveBackgroundRequest(BaseModel):
    image_b64: str = None      # Optional base64 encoded image
    imageUrl: str = None       # Optional image URL
    return_mask: bool = False  # Whether to return just the mask or the image with transparent background

class RemoveBackgroundResponse(BaseModel):
    image_b64: str = None      # Base64 encoded result image (with transparent background)
    mask_b64: str = None       # Base64 encoded mask (if requested)

# Object detection request model
class ObjectDetectionRequest(BaseModel):
    image_b64: str = None      # Optional base64 encoded image
    imageUrl: str = None       # Optional image URL
    task: str = "<OD>"         # Task prompt (default: object detection)
    confidence_threshold: float = 0.3  # Minimum confidence for detections
    text_input: str = None     # Text input for referring expression segmentation

class DetectedObject(BaseModel):
    label: str
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: Optional[float] = None
    multiclasses: Optional[dict] = None  # Selfie segmentation result for person objects
    segment_mask: Optional[str] = None  # S3 URL of SAM2 segmentation mask for the cropped object

class ObjectDetectionResponse(BaseModel):
    objects: list[DetectedObject]
    task_used: str
    segmentation_mask: Optional[list] = None  # For RES tasks


# Choose model (use relative paths like in SAM2 examples)
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Change to SAM2 directory for config loading
original_cwd = os.getcwd()
os.chdir("/root/segment-anything-2")

# Build model using local checkpoint
sam2_model = build_sam2(MODEL_CONFIG, CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

# Change back to original directory
os.chdir(original_cwd)

# Initialize EVF-SAM2 model
print("Loading EVF-SAM2 model...")

# Clear Hydra instance to avoid conflicts
from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()

evf_tokenizer = AutoTokenizer.from_pretrained(
    "YxZhang/evf-sam2-multitask",
    padding_side="right",
    use_fast=False,
)

# Import EVF-SAM2 model
from model.evf_sam2 import EvfSam2Model
evf_model = EvfSam2Model.from_pretrained(
    "YxZhang/evf-sam2-multitask",
    low_cpu_mem_usage=True,
    torch_dtype=torch.half,
).cuda().eval()

# Remove memory components for image-only inference (like in demo.py)
del evf_model.visual_model.memory_encoder
del evf_model.visual_model.memory_attention

print("EVF-SAM2 model loaded successfully!")

# Initialize BiRefNet model for background removal
print("Loading BiRefNet model...")
birefnet = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to(DEVICE)
birefnet.eval()
if DEVICE == 'cuda':
    birefnet.half()

# BiRefNet preprocessing transform
birefnet_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("BiRefNet model loaded successfully!")

# Initialize Florence-2 model for object detection
print("Loading Florence-2-large (no flash-attn) model...")
florence_device = "cuda" if torch.cuda.is_available() else "cpu"
florence_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

florence_model = AutoModelForCausalLM.from_pretrained(
    "multimodalart/Florence-2-large-no-flash-attn", 
    torch_dtype=florence_torch_dtype, 
    trust_remote_code=True
).to(florence_device)

florence_processor = AutoProcessor.from_pretrained(
    "multimodalart/Florence-2-large-no-flash-attn", 
    trust_remote_code=True
)

print("Florence-2-large (no flash-attn) model loaded successfully!")


# EVF-SAM preprocessing functions
pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

def sam_preprocess(x: np.ndarray, img_size=1024, model_type="sam2"):
    """Preprocess for SAM model."""
    x = torch.from_numpy(x).permute(2,0,1).contiguous()
    x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    x = (x - pixel_mean) / pixel_std
    return x, None

def beit3_preprocess(x: np.ndarray, img_size=224):
    """Preprocess for BEIT-3 model."""
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

def load_image_from_source(image_b64=None, image_url=None):
    """
    Load image from either base64 string or URL.
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

def process_detected_object(bbox, label, image_array, allowed_labels, person_labels, original_image):
    """
    Process a single detected object (bbox + label) with segmentation.
    This function is designed to be run in parallel for multiple objects.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        label: Object label string
        image_array: Preprocessed numpy array (already set in predictor)
        allowed_labels: List of allowed object labels
        person_labels: List of person-related labels for selfie segmentation
        original_image: PIL Image object for cropping
        
    Returns:
        dict: Processed object with segmentation results, or None if filtered out
    """
    # Filter out objects that are not in the allowed list
    if not any(allowed_word == label.lower() for allowed_word in allowed_labels):
        return None  # Skip this object
    
    obj = {
        'label': label,
        'bbox': bbox,  # [x1, y1, x2, y2]
        'confidence': None,  # Florence-2 doesn't return confidence scores
        'multiclasses': None,
        'segment_mask': None
    }
    
    # Crop the detected object from the original image for segmentation
    try:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_image.width, x2)
        y2 = min(original_image.height, y2)
        
        image_processing_start_time = time.time()
        # Crop the object from the image
        cropped_object = original_image.crop((x1, y1, x2, y2))
        
        image_processing_end_time = time.time()
        image_processing_duration = image_processing_end_time - image_processing_start_time
        print(f"⏱️  Image processing (crop) took {image_processing_duration:.3f} seconds for {label}")
        
        # Run SAM2 segmentation using the already preprocessed image
        sam2_start_time = time.time()
        
        # Prepare box input for SAM2
        input_boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        
        # Run inference directly without set_image (image already preprocessed)
        with torch.no_grad():
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False
            )
        
        # Process mask and upload to S3
        mask_array = masks[0]
        if hasattr(mask_array, 'cpu'):
            mask_array = mask_array.cpu().numpy()
        elif not isinstance(mask_array, np.ndarray):
            mask_array = np.asarray(mask_array)
        
        # Crop mask to bounding box for upload
        crop_box = (x1, y1, x2, y2)
        segment_mask_url = upload_mask_to_s3(mask_array, original_image.size, crop_box=crop_box)
        
        sam2_end_time = time.time()
        sam2_duration = sam2_end_time - sam2_start_time
        print(f"⏱️  SAM2 segmentation took {sam2_duration:.3f} seconds for {label}")
        obj['segment_mask'] = segment_mask_url
        
        # Check if detected object is a person and run selfie segmentation
        if any(person_word in label.lower() for person_word in person_labels):
            # Run selfie segmentation on the cropped person
            selfie_start_time = time.time()
            selfie_result = segment_selfie_multiclass(image_pil=cropped_object)
            selfie_end_time = time.time()
            selfie_duration = selfie_end_time - selfie_start_time
            print(f"⏱️  MediaPipe selfie segmentation took {selfie_duration:.3f} seconds for {label}")
            # obj['multiclasses'] = selfie_result
            
    except Exception as e:
        print(f"⚠️  Failed to process object {label}: {e}")
        # Continue without segmentation if it fails
        pass
    
    return obj

def detect_objects_florence2(image_b64=None, image_url=None, task="<OD>", confidence_threshold=0.3, text_input=None):
    """
    Detect objects in an image using Florence-2-large model.
    
    Args:
        image_b64: Base64 encoded image
        image_url: URL to image
        task: Task prompt (default: "<OD>" for object detection)
        confidence_threshold: Minimum confidence for detections
        text_input: Text input for referring expression segmentation
        
    Returns:
        dict: Parsed detection results with objects and their bounding boxes
    """
    # Load image from either source
    image = load_image_from_source(image_b64, image_url)
    
    # Handle referring expression segmentation
    if task == "<REFERRING_EXPRESSION_SEGMENTATION>" and text_input:
        prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{text_input}"
    else:
        prompt = task
    
    # Prepare inputs
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(florence_device, florence_torch_dtype)
    
    # Run Florence-2 detection and SAM2 preprocessing in parallel for maximum efficiency
    florence_start_time = time.time()
    
    def run_florence_detection():
        """Run Florence-2 object detection"""
        with torch.no_grad():
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                use_cache=True
            )
        return generated_ids
    
    def run_sam2_preprocessing():
        """Preprocess image for SAM2"""
        image_array = np.array(image)
        predictor.set_image(image_array)
        return image_array
    
    # Execute both operations in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks simultaneously
        florence_future = executor.submit(run_florence_detection)
        sam2_future = executor.submit(run_sam2_preprocessing)
        
        # Get results as they complete
        generated_ids = florence_future.result()
        image_array = sam2_future.result()
    
    florence_end_time = time.time()
    florence_duration = florence_end_time - florence_start_time
    print(f"⏱️  Florence-2 detection + SAM2 preprocessing (parallel) took {florence_duration:.3f} seconds")
    
    # Decode and parse results
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text, 
        task=task, 
        image_size=(image.width, image.height)
    )
    
    # Format results
    objects = []
    segmentation_mask = None
    
    if task in parsed_answer:
        detections = parsed_answer[task]
        
        # Handle different output formats
        if 'bboxes' in detections and 'labels' in detections:
            bboxes = detections['bboxes']
            labels = detections['labels']
            
            # Define allowed object labels to keep
            allowed_labels = [
                'man', 'woman', 'boy', 'girl', 'person', 'people', 'human',
                'dog', 'cat', 'horse',
                'car', 'truck', 'bicycle', 'bus', 'vehicle', 'motorcycle'
            ]
            
            # Define person-related labels to trigger selfie segmentation
            person_labels = ['human', 'man', 'woman', 'boy', 'girl', 'person', 'people']
            
            # SAM2 preprocessing already completed in parallel above - no need to repeat
            print(f"🔄 SAM2 image preprocessing completed in parallel with Florence-2")
            
            # Process objects in parallel for better performance
            objects = []
            parallel_start_time = time.time()
            
            # Use ThreadPoolExecutor to process multiple objects simultaneously
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all object processing tasks with preprocessed image
                future_to_data = {
                    executor.submit(process_detected_object, bbox, label, image_array, allowed_labels, person_labels, image): (bbox, label)
                    for bbox, label in zip(bboxes, labels)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_data):
                    bbox, label = future_to_data[future]
                    try:
                        result = future.result()
                        if result is not None:  # Only add non-filtered objects
                            objects.append(result)
                    except Exception as e:
                        print(f"⚠️  Parallel processing failed for object {label}: {e}")
            
            parallel_end_time = time.time()
            parallel_duration = parallel_end_time - parallel_start_time
            print(f"🚀 Parallel object processing took {parallel_duration:.3f} seconds for {len(objects)} objects")
        
        # Handle segmentation masks for RES tasks
        if 'polygons' in detections:
            segmentation_mask = detections['polygons']
    
    return {
        'objects': objects,
        'task_used': task,
        'segmentation_mask': segmentation_mask,
        'raw_output': parsed_answer
    }

def upload_mask_to_s3(mask_array, original_image_size=None, crop_box=None):
    """
    Upload segmentation mask to S3 as a 1-bit PNG image for optimal binary image compression with universal browser support.
    Optimized for performance with reduced PIL operations and faster compression.
    
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
        s3_upload_start_time = time.time()
        
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
        
        # Convert directly to 1-bit PIL Image (fix deprecation warning)
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
        s3_upload_end_time = time.time()
        s3_upload_duration = s3_upload_end_time - s3_upload_start_time
        
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

def upload_mask_to_s3_async(mask_array, original_image_size=None, crop_box=None):
    """
    Asynchronous version of upload_mask_to_s3 for use in ThreadPoolExecutor.
    Returns a future that can be used to get the result when ready.
    
    Args:
        mask_array: numpy array of the segmentation mask
        original_image_size: tuple of (width, height) for resizing mask
        crop_box: tuple of (x1, y1, x2, y2) to crop the mask before uploading
    
    Returns:
        str: S3 URL of the uploaded mask, or None if upload failed
    """
    # This is the same implementation as upload_mask_to_s3 but designed for async execution
    return upload_mask_to_s3(mask_array, original_image_size, crop_box)

def segment_selfie_multiclass(image_b64=None, image_url=None, image_pil=None):
    """
    Segment a selfie image using MediaPipe's multiclass selfie segmenter.
    
    Args:
        image_b64: Base64 encoded image
        image_url: URL to image
        image_pil: PIL Image object (takes precedence over image_b64 and image_url)
        
    Returns:
        dict: Contains the segmentation mask with multiple classes:
            - 0: background
            - 1: person (hair, body-skin, face-skin, clothes)
            - 2: hair
            - 3: body-skin
            - 4: face-skin  
            - 5: clothes
    """
    # Load image from either source
    if image_pil is not None:
        image = image_pil
    else:
        image = load_image_from_source(image_b64, image_url)
    
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Initialize MediaPipe selfie segmenter
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    # Create the segmenter with multiclass model
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        # Process the image
        results = selfie_segmentation.process(image_np)
        
        # Get segmentation mask (values 0-5 for different classes)
        segmentation_mask = results.segmentation_mask
        
        # Convert to proper data type and scale
        if segmentation_mask is not None:
            # MediaPipe returns float values, convert to integer classes
            segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
            
            return {
                "mask": segmentation_mask.tolist(),
                "width": segmentation_mask.shape[1], 
                "height": segmentation_mask.shape[0],
                "classes": {
                    0: "background",
                    1: "person",
                    2: "hair", 
                    3: "body-skin",
                    4: "face-skin",
                    5: "clothes"
                }
            }
        else:
            raise ValueError("Failed to generate segmentation mask")

def segment_image(image_b64=None, image_url=None, image_pil=None, point=None, box=None, points=None, boxes=None, crop_to_box=False):
    """
    Segment an image from base64 input or URL with optional prompts.
    Similar to the RunPod example function but with prompt support.
    
    Args:
        image_pil: PIL Image object (takes precedence over image_b64 and image_url)
        crop_to_box: If True and a box is provided, only upload the cropped mask region to S3
    """
    # Load image from either source
    if image_pil is not None:
        image = image_pil
    else:
        image = load_image_from_source(image_b64, image_url)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Set image in predictor
    predictor.set_image(image_array)
    
    # Prepare prompts
    input_points = None
    input_labels = None
    input_boxes = None
    
    # Handle single point
    if point:
        input_points = np.array([[point.x, point.y]], dtype=np.float32)
        input_labels = np.array([point.label], dtype=np.int32)
    
    # Handle multiple points (takes precedence over single point)
    if points:
        input_points = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        input_labels = np.array([p.label for p in points], dtype=np.int32)

    # Handle single box
    if box:
        input_boxes = np.array([[box.x1, box.y1, box.x2, box.y2]], dtype=np.float32)

    # Handle multiple boxes (takes precedence over single box)
    if boxes:
        input_boxes = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=np.float32)

    # If no prompts provided, use center point as default
    if input_points is None and input_boxes is None:
        height, width = image_array.shape[:2]
        input_points = np.array([[width // 2, height // 2]], dtype=np.float32)
        input_labels = np.array([1])  # foreground
    
    # Run inference
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxes,
            multimask_output=False
        )
    
    # Convert mask to list (similar to RunPod format)
    # Handle both PyTorch tensors and numpy arrays
    mask = masks[0]
    if hasattr(mask, 'cpu'):
        # It's a PyTorch tensor
        mask_array = mask.cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        # Convert to numpy array if it's not already
        mask_array = np.array(mask)
    else:
        mask_array = mask
    
    # Upload mask to S3
    crop_box = None
    if crop_to_box and box:
        crop_box = (int(box.x1), int(box.y1), int(box.x2), int(box.y2))
    
    mask_url = upload_mask_to_s3(mask_array, image.size, crop_box=crop_box)
    
    return mask_url

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
        
        mask_url = segment_image(
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
        # Load image from source
        image = load_image_from_source(request.image_b64, request.imageUrl)
        image_np = np.array(image)
        
        # Preprocess for EVF-SAM2
        original_size_list = [image_np.shape[:2]]
        
        # BEIT-3 preprocessing
        image_beit = beit3_preprocess(image_np, 224).to(dtype=evf_model.dtype, device=evf_model.device)
        
        # SAM preprocessing  
        image_sam, resize_shape = sam_preprocess(image_np, model_type="sam2")
        image_sam = image_sam.to(dtype=evf_model.dtype, device=evf_model.device)
        
        # Prepare text prompt
        prompt = request.prompt
        if request.semantic:
            prompt = f"[semantic] {prompt}"
            
        # Tokenize prompt
        input_ids = evf_tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=evf_model.device)
        
        # Run inference
        with torch.no_grad():
            pred_mask = evf_model.inference(
                image_sam.unsqueeze(0),
                image_beit.unsqueeze(0), 
                input_ids,
                resize_list=[resize_shape],
                original_size_list=original_size_list,
            )
            
        # Process output
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0
        
        # Convert mask to list format
        mask_list = pred_mask.astype(int).tolist()
        
        return EVFSegmentResponse(mask=mask_list)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EVF segmentation failed: {str(e)}")

@app.post("/detect-objects", response_model=ObjectDetectionResponse)
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
    
    Supported tasks:
    - "<OD>": Object detection (default)
    - "<REFERRING_EXPRESSION_SEGMENTATION>": Segment objects by text description
    - "<DENSE_REGION_CAPTION>": Dense region captioning
    - "<REGION_PROPOSAL>": Region proposal
    - "<CAPTION>": Image captioning
    - "<DETAILED_CAPTION>": Detailed image captioning
    - "<MORE_DETAILED_CAPTION>": More detailed image captioning
    
    Returns detected objects with labels and bounding boxes.
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
        
        # Run object detection
        result = detect_objects_florence2(
            image_b64=request.image_b64,
            image_url=request.imageUrl,
            task=request.task,
            confidence_threshold=request.confidence_threshold,
            text_input=request.text_input
        )
        
        # Convert to response format
        objects = [
            DetectedObject(
                label=obj['label'],
                bbox=obj['bbox'],
                confidence=obj['confidence'],
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
            segmentation_mask=result.get('segmentation_mask')
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
        # Load image from source
        image = load_image_from_source(request.image_b64, request.imageUrl)
        original_size = image.size
        
        # Preprocess image for BiRefNet
        input_images = birefnet_transform(image).unsqueeze(0).to(DEVICE)
        if DEVICE == 'cuda':
            input_images = input_images.half()
        
        # BiRefNet inference
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        
        # Process prediction
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(original_size)
        
        # Create RGBA image with transparent background
        image_rgba = image.copy()
        image_rgba.putalpha(mask)
        
        # Convert result to base64
        buffer = io.BytesIO()
        image_rgba.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response_data = {"image_b64": image_b64}
        
        # If mask is requested, return the mask
        if request.return_mask:
            mask_buffer = io.BytesIO()
            mask.save(mask_buffer, format="PNG")
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
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": True,
        "evf_model_loaded": True,
        "background_removal_loaded": True,
        "object_detection_loaded": True,
        "flux_model_loaded": False,
        "models": {
            "sam2": "SAM2.1 Large (Segmentation)",
            "evf_sam2": "EVF-SAM2 (Text-prompted Segmentation)",
            "birefnet": "BiRefNet (Background Removal)",
            "florence2": "Florence-2-large (Object Detection + Integrated Selfie Segmentation)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting FastAPI server with SAM2 + EVF-SAM2 on {DEVICE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)