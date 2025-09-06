#!/usr/bin/env python3
"""
Test script to verify the filtering and convex hull logic works correctly.
"""

def test_filtering_logic():
    """Test the early filtering logic."""
    
    # Mock detections data
    mock_bboxes = [
        [10, 10, 300, 300],    # Large person - should pass
        [20, 20, 50, 50],      # Small person - should fail size filter  
        [100, 100, 400, 400],  # Large car - should pass
        [200, 200, 250, 250],  # Medium elephant - should fail label filter
        [300, 300, 600, 600]   # Large dog - should pass
    ]
    
    mock_labels = ['person', 'person', 'car', 'elephant', 'dog']
    
    # Define allowed labels (same as in the code)
    allowed_labels = [
        'man', 'woman', 'boy', 'girl', 'person', 'people', 'human',
        'dog', 'cat', 'horse',
        'car', 'truck', 'bicycle', 'bus', 'vehicle', 'motorcycle'
    ]
    
    print("Testing filtering logic:")
    print("Original objects:", len(mock_bboxes))
    
    # Apply the same filtering logic as in the code
    filtered_bboxes = []
    filtered_labels = []
    
    for bbox, label in zip(mock_bboxes, mock_labels):
        # Check if label is in allowed list
        if any(allowed_word == label.lower() for allowed_word in allowed_labels):
            # Calculate bounding box dimensions for size filtering
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Filter out objects smaller than 250px in width or height
            if bbox_width >= 250 or bbox_height >= 250:
                filtered_bboxes.append(bbox)
                filtered_labels.append(label)
                print(f"✅ Kept {label}: size {bbox_width}x{bbox_height}px")
            else:
                print(f"⚠️  Filtering out {label}: size {bbox_width}x{bbox_height}px (< 250)")
        else:
            print(f"⚠️  Filtering out {label}: not in allowed labels")
    
    print(f"\nFiltered from {len(mock_bboxes)} to {len(filtered_bboxes)} objects")
    print("Remaining objects:", filtered_labels)
    
    # Test overlap detection
    print("\nTesting overlap detection:")
    
    # Add a nested box scenario
    test_bboxes = [
        [10, 10, 300, 300],    # Large outer box
        [50, 50, 200, 200],    # Inner box (should have overlap)
        [400, 400, 600, 600]   # Separate box (no overlap)
    ]
    
    # Simulate the overlap check logic
    def check_box_overlap(current_bbox, all_bboxes, current_index):
        x1, y1, x2, y2 = current_bbox
        
        for i, other_bbox in enumerate(all_bboxes):
            if i == current_index:  # Skip self
                continue
                
            ox1, oy1, ox2, oy2 = other_bbox
            
            # Check if current box is inside other box
            if ox1 <= x1 and oy1 <= y1 and ox2 >= x2 and oy2 >= y2:
                return True
                
            # Check if other box is inside current box
            if x1 <= ox1 and y1 <= oy1 and x2 >= ox2 and y2 >= oy2:
                return True
                
        return False
    
    for i, bbox in enumerate(test_bboxes):
        has_overlap = check_box_overlap(bbox, test_bboxes, i)
        print(f"Box {i} {bbox}: has_overlap = {has_overlap}")

if __name__ == "__main__":
    test_filtering_logic()
