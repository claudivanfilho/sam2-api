#!/usr/bin/env python3
"""
Test script to verify the polygon points feature in object detection.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from object_detection import object_detector
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def create_test_image():
    """Create a simple test image with a person-like shape."""
    # Create a 500x500 white image
    img = Image.new('RGB', (500, 500), color='white')
    pixels = np.array(img)
    
    # Draw a simple person-like shape (head and body)
    # Head (circle)
    center_x, center_y = 250, 150
    radius = 50
    for y in range(max(0, center_y - radius), min(500, center_y + radius)):
        for x in range(max(0, center_x - radius), min(500, center_x + radius)):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                pixels[y, x] = [0, 0, 0]  # Black
    
    # Body (rectangle)
    for y in range(200, 400):
        for x in range(200, 300):
            pixels[y, x] = [0, 0, 0]  # Black
    
    return Image.fromarray(pixels)

def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_polygon_feature():
    """Test the polygon points feature."""
    print("🔧 Creating test image...")
    test_img = create_test_image()
    test_img_b64 = image_to_base64(test_img)
    
    print("🔧 Testing with return_polygon_points=False (default - URLs)...")
    try:
        result_urls = object_detector.detect_objects(
            image_b64=test_img_b64,
            return_polygon_points=False
        )
        print(f"✅ URL mode result structure: {type(result_urls)}")
        if 'objects' in result_urls:
            print(f"   Found {len(result_urls['objects'])} objects")
            for i, obj in enumerate(result_urls['objects']):
                print(f"   Object {i}: {obj.get('label', 'Unknown')}")
                if 'segment_mask' in obj:
                    print(f"     segment_mask type: {type(obj['segment_mask'])}")
    except Exception as e:
        print(f"❌ URL mode failed: {e}")
    
    print("\n🔧 Testing with return_polygon_points=True (polygon points)...")
    try:
        result_polygons = object_detector.detect_objects(
            image_b64=test_img_b64,
            return_polygon_points=True
        )
        print(f"✅ Polygon mode result structure: {type(result_polygons)}")
        if 'objects' in result_polygons:
            print(f"   Found {len(result_polygons['objects'])} objects")
            for i, obj in enumerate(result_polygons['objects']):
                print(f"   Object {i}: {obj.get('label', 'Unknown')}")
                if 'segment_mask' in obj:
                    print(f"     segment_mask type: {type(obj['segment_mask'])}")
                    if isinstance(obj['segment_mask'], list):
                        print(f"     segment_mask points count: {len(obj['segment_mask'])}")
    except Exception as e:
        print(f"❌ Polygon mode failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing polygon points feature in object detection...")
    test_polygon_feature()
    print("✅ Test completed!")
