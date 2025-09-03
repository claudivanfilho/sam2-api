"""
MediaPipe operations for selfie segmentation and face landmarks detection.
Provides multiclass person segmentation and detailed face landmark detection.
"""

import numpy as np
import mediapipe as mp
from .utils import load_image_from_source

class MediaPipeProcessor:
    """Handles MediaPipe operations for face and body analysis."""
    
    def __init__(self):
        # MediaPipe solutions
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_face_mesh = mp.solutions.face_mesh
    
    def segment_selfie_multiclass(self, image_b64=None, image_url=None, image_pil=None):
        """
        Segment a selfie image using MediaPipe's multiclass selfie segmenter.
        
        Args:
            image_b64: Base64 encoded image
            image_url: URL to image
            image_pil: PIL Image object (takes precedence)
            
        Returns:
            dict: Contains the segmentation mask with multiple classes:
                - 0: background
                - 1: person (hair, body-skin, face-skin, clothes)
                - 2: hair
                - 3: body-skin
                - 4: face-skin  
                - 5: clothes
        """
        # Load image from source
        if image_pil is not None:
            image = image_pil
        else:
            image = load_image_from_source(image_b64, image_url)
        
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Create the segmenter with multiclass model
        with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
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
    
    def detect_face_landmarks(self, image_b64=None, image_url=None, image_pil=None):
        """
        Detect face landmarks using MediaPipe Face Mesh.
        
        Args:
            image_b64: Base64 encoded image
            image_url: URL to image
            image_pil: PIL Image object (takes precedence)
            
        Returns:
            dict: Contains face landmarks information with normalized coordinates:
                - landmarks: List of 468 3D face landmarks (x, y, z)
                - face_count: Number of faces detected
                - image_dimensions: Width and height of the processed image
        """
        # Load image from source
        if image_pil is not None:
            image = image_pil
        else:
            image = load_image_from_source(image_b64, image_url)
        
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Create the face mesh detector
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,  # Detect up to 5 faces
            refine_landmarks=True,  # Include iris landmarks (468 total landmarks)
            min_detection_confidence=0.5
        ) as face_mesh:
            # Process the image
            results = face_mesh.process(image_np)
            
            if results.multi_face_landmarks:
                faces_landmarks = []
                for face_landmarks in results.multi_face_landmarks:
                    # Extract landmarks as list of [x, y, z] coordinates
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        landmarks.append([
                            landmark.x,  # Normalized x coordinate (0.0 to 1.0)
                            landmark.y,  # Normalized y coordinate (0.0 to 1.0) 
                            landmark.z   # Normalized z coordinate (relative depth)
                        ])
                    faces_landmarks.append(landmarks)
                
                return {
                    "landmarks": faces_landmarks,
                    "face_count": len(faces_landmarks),
                    "image_dimensions": {
                        "width": image.width,
                        "height": image.height
                    },
                    "landmark_count": 468,  # Total landmarks per face with iris refinement
                    "coordinate_format": "normalized",  # Values are between 0.0 and 1.0
                    "description": "468 3D face landmarks including facial contours, eyes, eyebrows, nose, mouth, and iris"
                }
            else:
                return {
                    "landmarks": [],
                    "face_count": 0,
                    "image_dimensions": {
                        "width": image.width,
                        "height": image.height
                    },
                    "landmark_count": 0,
                    "coordinate_format": "normalized",
                    "description": "No faces detected in the image"
                }

# Global MediaPipe processor instance
mediapipe_processor = MediaPipeProcessor()
