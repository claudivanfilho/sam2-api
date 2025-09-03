#!/usr/bin/env python3
"""
Test script to verify all modules can be imported correctly.
"""

import sys
import os

# Add root directory to path so we can import src
sys.path.insert(0, '/root/sam2-api')

def test_import(module_name):
    """Test importing a module."""
    try:
        exec(f"import {module_name}")
        print(f"✅ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} failed to import: {e}")
        return False

def main():
    """Run all import tests."""
    print("🚀 Testing module imports...")
    
    modules_to_test = [
        'src.utils',
        'src.mediapipe_operations',
        'src.sam2_segmentation',
        'src.background_removal',
        'src.evf_sam_operations',
        'src.object_detection',
        'src.model_manager'
    ]
    
    success_count = 0
    for module in modules_to_test:
        if test_import(module):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{len(modules_to_test)} modules imported successfully")
    
    if success_count == len(modules_to_test):
        print("🎉 All modules imported successfully!")
        # Try importing main
        if test_import('main'):
            print("🎉 Main application ready!")
            return True
    
    return False

if __name__ == "__main__":
    main()
