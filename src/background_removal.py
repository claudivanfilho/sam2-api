"""
Background removal functionality using BiRefNet.
Provides high-quality background removal with transparent output.
"""

import torch
import numpy as np
from torchvision import transforms
import io
import base64
from PIL import Image
from .model_manager import get_model_manager
from .utils import load_image_from_source

class BackgroundRemover:
    """Handles background removal using BiRefNet model."""
    
    def __init__(self):
        self.model_manager = None
    
    def _get_model_manager(self):
        """Lazy initialization of model manager."""
        if self.model_manager is None:
            self.model_manager = get_model_manager(preload_models=False)  # Lazy loading
        return self.model_manager
    
    def remove_background(self, image_b64=None, image_url=None, image_pil=None, return_mask=False):
        """
        Remove background from image using BiRefNet.
        
        Args:
            image_b64: Base64 encoded image
            image_url: URL to image
            image_pil: PIL Image object (takes precedence)
            return_mask: Whether to also return the segmentation mask
            
        Returns:
            dict: Contains 'image_pil' with transparent background, 
                 optionally 'mask_pil' if return_mask=True
        """
        # Get model and transform (will load if needed)
        model_manager = self._get_model_manager()
        model = model_manager.get_model('birefnet')
        transform = model_manager.get_transform('birefnet')
        device = model_manager.get_device()
        
        # Load image from source
        if image_pil is not None:
            image = image_pil
        else:
            image = load_image_from_source(image_b64, image_url)
        
        original_size = image.size
        
        # Preprocess image for BiRefNet
        input_images = transform(image).unsqueeze(0).to(device)
        if device == 'cuda':
            input_images = input_images.half()
        
        # BiRefNet inference
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()
        
        # Process prediction
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(original_size)
        
        # Create RGBA image with transparent background
        image_rgba = image.copy()
        image_rgba.putalpha(mask)
        
        # Prepare result with PIL image
        result = {"image_pil": image_rgba}
        
        # If mask is requested, return the mask
        if return_mask:
            result["mask_pil"] = mask
        
        return result

# Create global instance without initializing models
background_remover = BackgroundRemover()
