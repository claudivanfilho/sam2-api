#!/usr/bin/env python3

import requests
import json
import base64
from PIL import Image, ImageDraw
import io

def create_test_image_with_objects():
    """Create a test image with objects of different sizes"""
    # Create a 800x600 image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a large rectangle (should pass the 300px filter)
    draw.rectangle([50, 50, 400, 300], fill='blue', outline='black', width=3)
    draw.text((150, 150), "Large Object\n350x250px", fill='white')
    
    # Draw a small rectangle (should be filtered out)
    draw.rectangle([500, 50, 650, 150], fill='red', outline='black', width=3)
    draw.text((520, 90), "Small\n150x100px", fill='white')
    
    # Draw a medium rectangle (should be filtered out)
    draw.rectangle([50, 400, 300, 550], fill='green', outline='black', width=3)
    draw.text((120, 465), "Medium\n250x150px", fill='white')
    
    # Draw another large rectangle (should pass)
    draw.rectangle([400, 300, 750, 550], fill='purple', outline='black', width=3)
    draw.text((520, 410), "Large Object 2\n350x250px", fill='white')
    
    return img

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_size_filtering():
    """Test the object detection endpoint with size filtering"""
    print("🧪 Testing object detection with size filtering...")
    
    # Create test image
    test_image = create_test_image_with_objects()
    test_image.save("/root/sam2-api/test_image.png")
    print("✅ Test image created and saved as test_image.png")
    
    # Convert to base64
    image_b64 = image_to_base64(test_image)
    
    # Test the API endpoint
    api_url = "http://localhost:8000/detect-objects"
    
    payload = {
        "image_b64": image_b64,
        "task": "<OD>",
        "confidence_threshold": 0.1
    }
    
    try:
        print("📡 Sending request to API...")
        response = requests.post(api_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ API Response successful!")
            print(f"📊 Objects detected: {len(result['objects'])}")
            print(f"🎯 Task used: {result['task_used']}")
            
            for i, obj in enumerate(result['objects']):
                bbox = obj['bbox']
                width = int(bbox[2] - bbox[0])
                height = int(bbox[3] - bbox[1])
                print(f"   Object {i+1}: {obj['label']} - {width}x{height}px - bbox: {bbox}")
                
                if obj['multiclasses']:
                    print(f"      🎭 Multiclass segmentation: ✅")
                if obj['face_landmarks']:
                    face_count = obj['face_landmarks']['face_count']
                    print(f"      👤 Face landmarks: {face_count} faces detected")
                if obj['segment_mask']:
                    print(f"      🎨 Segment mask: {obj['segment_mask']}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_size_filtering()
