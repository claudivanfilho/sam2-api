"""
MediaPipe operations for selfie segmentation and face landmarks detection.
Provides multiclass person segmentation and detailed face landmark detection.
"""

import numpy as np
import mediapipe as mp
import requests
import tempfile
import os

# Handle imports for both module and standalone usage
try:
    from .utils import load_image_from_source
except ImportError:
    from utils import load_image_from_source

class MediaPipeProcessor:
    """Handles MediaPipe operations for face and body analysis."""
    
    def __init__(self):
        # MediaPipe solutions
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Download and cache the multiclass model
        self.multiclass_model_path = self._download_multiclass_model()
    
    def _download_multiclass_model(self):
        """Download the multiclass selfie segmentation model if not already cached."""
        model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite"
        
        # Create a cache directory
        cache_dir = os.path.expanduser("~/.mediapipe_models")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = os.path.join(cache_dir, "selfie_multiclass_256x256.tflite")
        
        # Download if not exists
        if not os.path.exists(model_path):
            print(f"📥 Downloading multiclass selfie segmentation model...")
            try:
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✅ Model downloaded successfully to {model_path}")
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
                return None
        else:
            print(f"✅ Using cached multiclass model: {model_path}")
            
        return model_path
    
    def segment_selfie_multiclass(self, image_b64=None, image_url=None, image_pil=None):
        """
        Segment a selfie image using MediaPipe's true multiclass selfie segmenter.
        
        Args:
            image_b64: Base64 encoded image
            image_url: URL to image
            image_pil: PIL Image object (takes precedence)
            
        Returns:
            dict: Contains the segmentation mask with multiple classes:
                - 0: background
                - 1: hair
                - 2: body-skin
                - 3: face-skin  
                - 4: clothes
                - 5: others (accessories, etc.)
        """
        if not self.multiclass_model_path:
            raise ValueError("Multiclass model not available. Please check model download.")
            
        # Load image from source
        if image_pil is not None:
            image = image_pil
        else:
            image = load_image_from_source(image_b64, image_url)
        
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Create ImageSegmenter with the multiclass model
        base_options = mp.tasks.BaseOptions(model_asset_path=self.multiclass_model_path)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False
        )
        
        with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            # Perform segmentation
            segmentation_result = segmenter.segment(mp_image)
            
            if segmentation_result.category_mask is not None:
                # Get the category mask (contains class indices 0-5)
                category_mask = segmentation_result.category_mask.numpy_view()
                
                return {
                    "mask": category_mask.tolist(),
                    "width": category_mask.shape[1], 
                    "height": category_mask.shape[0],
                    "classes": {
                        0: "background",
                        1: "hair",
                        2: "body-skin", 
                        3: "face-skin",
                        4: "clothes",
                        5: "others"
                    },
                    "model": "selfie_multiclass_256x256"
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
