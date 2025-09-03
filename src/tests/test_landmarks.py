#!/usr/bin/env python3

import numpy as np
import requests
from PIL import Image
import io
import mediapipe as mp

def load_image_from_url(image_url):
    """Load image from URL"""
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image

def detect_face_landmarks(image_pil):
    """
    Detect face landmarks using MediaPipe Face Mesh.
    
    Args:
        image_pil: PIL Image object
        
    Returns:
        dict: Contains face landmarks information with normalized coordinates
    """
    # Convert PIL image to numpy array
    image_np = np.array(image_pil)
    
    # Initialize MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Create the face mesh detector
    with mp_face_mesh.FaceMesh(
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
                    "width": image_pil.width,
                    "height": image_pil.height
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
                    "width": image_pil.width,
                    "height": image_pil.height
                },
                "landmark_count": 0,
                "coordinate_format": "normalized",
                "description": "No faces detected in the image"
            }

if __name__ == "__main__":
    print('🧪 Testing face landmarks detection...')
    
    # Test with a face image
    test_image_url = 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face'
    
    try:
        print(f'📥 Loading image from: {test_image_url}')
        test_image = load_image_from_url(test_image_url)
        print(f'✅ Image loaded: {test_image.size}')
        
        print('🔍 Detecting face landmarks...')
        landmarks_result = detect_face_landmarks(test_image)
        
        print(f'✅ Face landmarks detected: {landmarks_result["face_count"]} faces')
        print(f'   📊 Landmark count per face: {landmarks_result["landmark_count"]}')
        print(f'   📐 Image dimensions: {landmarks_result["image_dimensions"]}')
        print(f'   📝 Description: {landmarks_result["description"]}')
        
        if landmarks_result['face_count'] > 0:
            print(f'   🎯 First face landmarks sample (first 3 points):')
            for i, point in enumerate(landmarks_result["landmarks"][0][:3]):
                print(f'      Point {i}: x={point[0]:.4f}, y={point[1]:.4f}, z={point[2]:.4f}')
            
            # Show some key landmark indices
            landmarks = landmarks_result["landmarks"][0]
            print(f'   👁️  Left eye center (landmark 159): x={landmarks[159][0]:.4f}, y={landmarks[159][1]:.4f}')
            print(f'   👁️  Right eye center (landmark 386): x={landmarks[386][0]:.4f}, y={landmarks[386][1]:.4f}')
            print(f'   👃 Nose tip (landmark 1): x={landmarks[1][0]:.4f}, y={landmarks[1][1]:.4f}')
            print(f'   👄 Mouth center (landmark 13): x={landmarks[13][0]:.4f}, y={landmarks[13][1]:.4f}')
        
    except Exception as e:
        print(f'❌ Error testing face landmarks: {e}')
        import traceback
        traceback.print_exc()
