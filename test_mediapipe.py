#!/usr/bin/env python3
"""
Test script for MediaPipe multiclass selfie segmentation.
Tests the new implementation with the correct multiclass model.
"""

import sys
import os
import numpy as np
from PIL import Image
import requests
import io

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_mediapipe_multiclass():
    """Test MediaPipe multiclass segmentation with a sample image."""
    print("🧪 Testing MediaPipe multiclass selfie segmentation...")
    
    try:
        # Import the MediaPipe processor with absolute imports
        import mediapipe_operations
        mediapipe_processor = mediapipe_operations.mediapipe_processor
        
        # Use a sample person image from the web for testing
        test_image_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face"
        
        print(f"📥 Downloading test image from: {test_image_url}")
        response = requests.get(test_image_url, timeout=30)
        response.raise_for_status()
        
        # Convert to PIL Image
        test_image = Image.open(io.BytesIO(response.content)).convert("RGB")
        print(f"✅ Test image loaded: {test_image.size}")
        
        # Run multiclass segmentation
        print("🔄 Running multiclass segmentation...")
        result = mediapipe_processor.segment_selfie_multiclass(image_pil=test_image)
        
        if result and 'mask' in result:
            mask_array = np.array(result['mask'], dtype=np.uint8)
            print(f"✅ Segmentation successful!")
            print(f"   Mask shape: {mask_array.shape}")
            print(f"   Image dimensions: {result['width']}x{result['height']}")
            
            # Analyze class distribution
            unique_classes, counts = np.unique(mask_array, return_counts=True)
            print(f"   Classes found: {unique_classes}")
            
            total_pixels = mask_array.size
            for class_id, count in zip(unique_classes, counts):
                class_name = result['classes'].get(class_id, f"unknown_{class_id}")
                percentage = (count / total_pixels) * 100
                print(f"   - Class {class_id} ({class_name}): {count:,} pixels ({percentage:.1f}%)")
            
            # Check if we have actual person classes (not just background)
            person_classes = [c for c in unique_classes if c > 0]
            if len(person_classes) > 0:
                print(f"✅ Found {len(person_classes)} person-related classes: {person_classes}")
                
                # Save sample masks for each class
                print("💾 Saving sample class masks...")
                for class_id in person_classes:
                    class_mask = (mask_array == class_id).astype(np.uint8) * 255
                    class_name = result['classes'].get(class_id, f"class_{class_id}")
                    
                    # Create PIL image from mask
                    mask_image = Image.fromarray(class_mask, mode='L')
                    output_path = f"/tmp/test_mask_{class_name}_{class_id}.png"
                    mask_image.save(output_path)
                    print(f"   Saved {class_name} mask: {output_path}")
                
                return True
            else:
                print("⚠️  No person classes detected - only background found")
                return False
        else:
            print("❌ Segmentation failed - no mask returned")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure MediaPipe is installed: pip install mediapipe")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_download():
    """Test if the multiclass model can be downloaded successfully."""
    print("🧪 Testing model download...")
    
    try:
        import mediapipe_operations
        
        # Create processor instance (should trigger model download)
        processor = mediapipe_operations.MediaPipeProcessor()
        
        if processor.multiclass_model_path and os.path.exists(processor.multiclass_model_path):
            file_size = os.path.getsize(processor.multiclass_model_path)
            print(f"✅ Model downloaded successfully!")
            print(f"   Path: {processor.multiclass_model_path}")
            print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            return True
        else:
            print("❌ Model download failed")
            return False
            
    except Exception as e:
        print(f"❌ Model download test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting MediaPipe multiclass segmentation tests\n")
    
    # Test 1: Model download
    print("=" * 60)
    model_test = test_model_download()
    print()
    
    # Test 2: Segmentation functionality
    print("=" * 60)
    segmentation_test = test_mediapipe_multiclass()
    print()
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary:")
    print(f"   Model Download: {'✅ PASS' if model_test else '❌ FAIL'}")
    print(f"   Segmentation:   {'✅ PASS' if segmentation_test else '❌ FAIL'}")
    
    if model_test and segmentation_test:
        print("\n🎉 All tests passed! MediaPipe multiclass segmentation is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
