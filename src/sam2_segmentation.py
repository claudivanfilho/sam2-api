"""
SAM2 segmentation functionality.
Handles point and box-prompted segmentation using SAM2 model.
"""

import torch
import numpy as np
from .model_manager import get_model_manager
from .utils import load_image_from_source, upload_mask_to_s3

class SAM2Segmenter:
    """Handles SAM2 segmentation with various prompts."""
    
    def __init__(self):
        self.model_manager = None
    
    def _get_model_manager(self):
        """Lazy initialization of model manager."""
        if self.model_manager is None:
            self.model_manager = get_model_manager(preload_models=False)  # Lazy loading
        return self.model_manager
    
    def segment_image(self, image_b64=None, image_url=None, image_pil=None, 
                     point=None, box=None, points=None, boxes=None, crop_to_box=False, env="staging"):
        """
        Segment an image with optional prompts.
        
        Args:
            image_b64: Base64 encoded image
            image_url: URL to image
            image_pil: PIL Image object (takes precedence)
            point: Single point prompt {x, y, label}
            box: Single box prompt {x1, y1, x2, y2}
            points: Multiple point prompts
            boxes: Multiple box prompts
            crop_to_box: If True and a box is provided, only upload the cropped mask region to S3
            env: Environment for S3 upload (staging or production)
            
        Returns:
            str: S3 URL of the uploaded mask
        """
        # Get predictor (will load if needed)
        model_manager = self._get_model_manager()
        predictor = model_manager.get_model('sam2_predictor')
        
        # Load image from source
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
        
        # Process mask
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
        
        mask_url = upload_mask_to_s3(mask_array, image.size, crop_box=crop_box, env=env)
        
        return mask_url
    
    def segment_with_box(self, image_array, bbox, point_prompt=None):
        """
        Segment using a bounding box (used by object detection).
        
        Args:
            image_array: Preprocessed numpy array (already set in predictor)
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            point_prompt: Optional point coordinates [x, y] to add as additional prompt
            
        Returns:
            numpy.ndarray: mask_array 
        """
        # Get predictor (will load if needed)
        model_manager = self._get_model_manager()
        predictor = model_manager.get_model('sam2_predictor')
        
        # Prepare box input for SAM2
        input_boxes = np.array([bbox], dtype=np.float32)
        
        # Prepare point input if provided
        input_points = None
        input_labels = None
        if point_prompt:
            input_points = np.array([point_prompt], dtype=np.float32)
            input_labels = np.array([1], dtype=np.int32)  # foreground point
        
        # Run inference directly without set_image (image already preprocessed)
        with torch.no_grad():
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_boxes,
                multimask_output=False
            )
        
        # Process mask
        mask_array = masks[0]
        if hasattr(mask_array, 'cpu'):
            mask_array = mask_array.cpu().numpy()
        elif not isinstance(mask_array, np.ndarray):
            mask_array = np.asarray(mask_array)
        
        return mask_array

# Create global instance without initializing models
sam2_segmenter = SAM2Segmenter()
