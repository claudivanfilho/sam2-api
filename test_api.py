#!/usr/bin/env python3

"""
Test script for the Florence-2 object detection API endpoint
"""

import requests
import json
import base64
from PIL import Image
import io

def test_object_detection_api():
    print("🧪 Testing Florence-2 object detection API endpoint...")
    
    # API endpoint
    url = "http://localhost:8000/detect-objects"
    
    # Test with image URL
    test_data = {
        "imageUrl": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    }
    
    print("🚗 Testing with car image from HuggingFace...")
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API call successful!")
            print(f"🔍 Task used: {result['task_used']}")
            print(f"🎯 Found {len(result['objects'])} objects:")
            
            for i, obj in enumerate(result['objects']):
                print(f"  {i+1}. {obj['label']} - bbox: {obj['bbox']}")
            
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_health_endpoint():
    print("\n🏥 Testing health endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"📊 Status: {result['status']}")
            print(f"💻 Device: {result['device']}")
            print("🤖 Models loaded:")
            for model_name, model_desc in result['models'].items():
                print(f"  - {model_name}: {model_desc}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    test_health_endpoint()
    test_object_detection_api()
