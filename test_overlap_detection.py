#!/usr/bin/env python3
"""
Test script to verify the containment detection logic works correctly.
Tests if a box has 80% of another box inside it OR if a box is 80% inside another.
"""

from src.object_detection import ObjectDetector

def test_containment_detection():
    """Test the _check_box_overlap method for containment scenarios."""
    detector = ObjectDetector()
    
    # Test case 1: Small box 90% inside larger box
    bboxes1 = [
        [10, 10, 100, 100],  # Large outer box (90x90 = 8100 area)
        [15, 15, 85, 85],    # Small box (70x70 = 4900 area) - 100% inside large box
        [200, 200, 300, 300] # Separate box
    ]
    
    print("Test 1: Small box 100% inside larger box")
    for i, bbox in enumerate(bboxes1):
        has_overlap = detector._check_box_overlap(bbox, bboxes1, i)
        print(f"  Box {i} {bbox}: has_overlap = {has_overlap}")
    
    # Test case 2: Large box contains 90% of smaller box
    bboxes2 = [
        [50, 50, 80, 80],     # Small box (30x30 = 900 area)
        [10, 10, 100, 100],   # Large box - contains 100% of small box
        [200, 200, 300, 300]  # Separate box
    ]
    
    print("\nTest 2: Large box contains 100% of smaller box")
    for i, bbox in enumerate(bboxes2):
        has_overlap = detector._check_box_overlap(bbox, bboxes2, i)
        print(f"  Box {i} {bbox}: has_overlap = {has_overlap}")
    
    # Test case 3: Boxes with only 50% containment (should be False)
    bboxes3 = [
        [10, 10, 60, 60],     # Box 1 (50x50 = 2500 area)
        [35, 35, 85, 85],     # Box 2 (50x50 = 2500 area), intersection: 25x25 = 625 (25% of each)
    ]
    
    print("\nTest 3: Boxes with only 25% containment each")
    for i, bbox in enumerate(bboxes3):
        has_overlap = detector._check_box_overlap(bbox, bboxes3, i)
        print(f"  Box {i} {bbox}: has_overlap = {has_overlap}")
    
    # Test case 4: No overlaps at all
    bboxes4 = [
        [10, 10, 100, 100],   # Box 1
        [150, 150, 250, 250], # Box 2 (no overlap)
        [300, 300, 400, 400]  # Box 3 (no overlap)
    ]
    
    print("\nTest 4: No overlaps")
    for i, bbox in enumerate(bboxes4):
        has_overlap = detector._check_box_overlap(bbox, bboxes4, i)
        print(f"  Box {i} {bbox}: has_overlap = {has_overlap}")
    
    # Test case 5: Exactly 80% containment (should be True)
    bboxes5 = [
        [0, 0, 100, 100],     # Large box (100x100 = 10000 area)
        [0, 0, 80, 100]       # Box with 80x100 = 8000 area (80% of large box area)
                              # This box is 100% contained in the large box
    ]
    
    print("\nTest 5: 80% area box completely inside larger box")
    for i, bbox in enumerate(bboxes5):
        has_overlap = detector._check_box_overlap(bbox, bboxes5, i)
        print(f"  Box {i} {bbox}: has_overlap = {has_overlap}")
    
    # Test case 6: Edge case - exactly 80% of small box inside large box
    bboxes6 = [
        [0, 0, 100, 100],     # Large box
        [20, 0, 120, 100]     # Box that extends outside: intersection is 80x100 = 8000
                              # This box has area 100x100 = 10000, so 80% is inside large box
    ]
    
    print("\nTest 6: Exactly 80% of box inside another")
    for i, bbox in enumerate(bboxes6):
        has_overlap = detector._check_box_overlap(bbox, bboxes6, i)
        print(f"  Box {i} {bbox}: has_overlap = {has_overlap}")

if __name__ == "__main__":
    test_containment_detection()
