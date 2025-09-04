"""
EVF-SAM text-prompted segmentation functionality.
Handles semantic and referring expression segmentation using EVF-SAM2.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .model_manager import get_model_manager
from .utils import load_image_from_source

# EVF-SAM preprocessing constants
pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

class EVFSAMSegmenter:
    """Handles EVF-SAM2 text-prompted segmentation."""
    
    def __init__(self):
        self.model_manager = get_model_manager()
    
    def sam_preprocess(self, x: np.ndarray, img_size=1024, model_type="sam2"):
        """Preprocess for SAM model."""
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
        
        # Get model to match dtype
        model = self.model_manager.get_model('evf_sam2')
        device = model.device
        dtype = model.dtype
        
        # Ensure pixel_mean and pixel_std have the same dtype and device as the model
        pixel_mean_tensor = pixel_mean.to(dtype=dtype, device=device)
        pixel_std_tensor = pixel_std.to(dtype=dtype, device=device)
        
        x = x.to(dtype=dtype, device=device)
        x = (x - pixel_mean_tensor) / pixel_std_tensor
        return x, None

    def beit3_preprocess(self, x: np.ndarray, img_size=224):
        """Preprocess for BEIT-3 model."""
        beit_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=None), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return beit_preprocess(x)
    
    def segment_with_text(self, image_b64=None, image_url=None, image_pil=None, 
                         prompt=None, semantic=False):
        """
        Segment image using text prompt with EVF-SAM2.
        
        Args:
            image_b64: Base64 encoded image
            image_url: URL to image
            image_pil: PIL Image object (takes precedence)
            prompt: Text prompt for segmentation
            semantic: Whether to use semantic segmentation
            
        Returns:
            list: Segmentation mask as nested list
        """
        # Get model and tokenizer (will load if needed)
        model = self.model_manager.get_model('evf_sam2')
        tokenizer = self.model_manager.get_model('evf_tokenizer')
        
        # Ensure model is in correct dtype state before each inference
        # This prevents dtype corruption between calls
        model = model.half()
        
        # Load image from source
        if image_pil is not None:
            image = image_pil
        else:
            image = load_image_from_source(image_b64, image_url)
        
        image_np = np.array(image)
        
        # Preprocess for EVF-SAM2
        original_size_list = [image_np.shape[:2]]
        
        # BEIT-3 preprocessing - ensure proper dtype
        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=model.dtype, device=model.device)
        
        # SAM preprocessing - ensure proper dtype
        image_sam, resize_shape = self.sam_preprocess(image_np, model_type="sam2")
        # image_sam is already converted to proper dtype and device in sam_preprocess
        
        # Prepare text prompt
        if semantic:
            prompt = f"[semantic] {prompt}"
            
        # Tokenize prompt and ensure proper device placement
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=model.device)
        
        # Run inference with proper dtype handling and error recovery
        try:
            with torch.no_grad():
                pred_mask = model.inference(
                    image_sam.unsqueeze(0),
                    image_beit.unsqueeze(0),
                    input_ids,
                    resize_list=[resize_shape],
                    original_size_list=original_size_list,
                )
        except RuntimeError as e:
            if "dtype" in str(e).lower():
                print(f"⚠️  Dtype error detected, attempting model reset: {e}")
                # Force model back to half precision
                model = model.half()
                # Retry with fresh tensor conversions
                image_beit = self.beit3_preprocess(image_np, 224).to(dtype=torch.half, device=model.device)
                image_sam, resize_shape = self.sam_preprocess(image_np, model_type="sam2")
                
                with torch.no_grad():
                    pred_mask = model.inference(
                        image_sam.unsqueeze(0),
                        image_beit.unsqueeze(0),
                        input_ids,
                        resize_list=[resize_shape],
                        original_size_list=original_size_list,
                    )
            else:
                raise e
            
        # Process output
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0
        
        # Convert mask to list format
        mask_list = pred_mask.astype(int).tolist()
        
        return mask_list

# Create global instance without initializing models
evf_sam_segmenter = EVFSAMSegmenter()
