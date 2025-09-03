#!/usr/bin/env python3
"""
Simple test script to verify module imports work without initializing models.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '/root/sam2-api')

def test_simple_imports():
    """Test importing modules without model initialization."""
    try:
        print("Testing basic imports...")
        
        # Test src package
        import src
        print("✅ src package imported")
        
        # Test utils
        from src import utils
        print("✅ src.utils imported")
        
        # Test mediapipe operations
        from src import mediapipe_operations
        print("✅ src.mediapipe_operations imported")
        
        # Test model manager class definition (without initialization)
        from src.model_manager import ModelManager, get_model_manager
        print("✅ src.model_manager classes imported")
        
        # Test other modules (they should import without loading models now)
        from src import background_removal
        print("✅ src.background_removal imported")
        
        from src import sam2_segmentation  
        print("✅ src.sam2_segmentation imported")
        
        from src import evf_sam_operations
        print("✅ src.evf_sam_operations imported")
        
        from src import object_detection
        print("✅ src.object_detection imported")
        
        print("\n🎉 All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_imports()
