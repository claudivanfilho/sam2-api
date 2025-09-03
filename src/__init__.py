"""
SAM2 API Source Package
Contains all the modular components for the SAM2 API application.
"""

__version__ = "1.0.0"

# Import all main classes for easy access
from .model_manager import get_model_manager
from .background_removal import background_remover
from .sam2_segmentation import sam2_segmenter
from .evf_sam_operations import evf_sam_segmenter
from .object_detection import object_detector
from .mediapipe_operations import mediapipe_processor

__all__ = [
    'get_model_manager',
    'background_remover',
    'sam2_segmenter', 
    'evf_sam_segmenter',
    'object_detector',
    'mediapipe_processor'
]
