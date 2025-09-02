from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from pydantic import BaseModel
import torch
from PIL import Image
import io, base64
import os
import numpy as np
import requests
import sys
sys.path.append('/root/EVF-SAM')
sys.path.append('/root/BiRefNet')
from transformers import AutoTokenizer
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from models.birefnet import BiRefNet

app = FastAPI()

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
    mask: list

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

def segment_image(image_b64=None, image_url=None, point=None, box=None, points=None, boxes=None):
    """
    Segment an image from base64 input or URL with optional prompts.
    Similar to the RunPod example function but with prompt support.
    """
    # Load image from either source
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
        input_points = np.array([[point.x, point.y]])
        input_labels = np.array([point.label])
    
    # Handle multiple points (takes precedence over single point)
    if points:
        input_points = np.array([[p.x, p.y] for p in points])
        input_labels = np.array([p.label for p in points])
    
    # Handle single box
    if box:
        input_boxes = np.array([[box.x1, box.y1, box.x2, box.y2]])
    
    # Handle multiple boxes (takes precedence over single box)
    if boxes:
        input_boxes = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes])
    
    # If no prompts provided, use center point as default
    if input_points is None and input_boxes is None:
        height, width = image_array.shape[:2]
        input_points = np.array([[width // 2, height // 2]])
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
        mask = mask.cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        # Convert to numpy array if it's not already
        mask = np.array(mask)
    
    return mask.tolist()

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
    """
    try:
        # Validate input
        if not request.image_b64 and not request.imageUrl:
            raise HTTPException(status_code=400, detail="Either image_b64 or imageUrl must be provided")
        
        if request.image_b64 and request.imageUrl:
            raise HTTPException(status_code=400, detail="Provide either image_b64 OR imageUrl, not both")
        
        mask = segment_image(
            image_b64=request.image_b64,
            image_url=request.imageUrl,
            point=request.point,
            box=request.box,
            points=request.points,
            boxes=request.boxes
        )
        return SegmentResponse(mask=mask)
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
        "flux_model_loaded": False,
        "models": {
            "sam2": "SAM2.1 Large (Segmentation)",
            "evf_sam2": "EVF-SAM2 (Text-prompted Segmentation)",
            "birefnet": "BiRefNet (Background Removal)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting FastAPI server with SAM2 + EVF-SAM2 on {DEVICE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)