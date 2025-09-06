#!/usr/bin/env python3
"""
Visual examples of containment detection scenarios using ASCII art.
Shows when segment_mask_convexhull will be generated (80%+ containment).
"""

from src.object_detection import ObjectDetector

def draw_boxes(bboxes, labels, title, description=""):
    """Draw bounding boxes using ASCII art with dashes and pipes."""
    print(f"\n{title}")
    print("=" * len(title))
    if description:
        print(f"{description}\n")
    
    # Create a grid to draw on (simplified coordinates)
    grid_size = 50
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Draw each box with different characters
    box_chars = ['#', '*', '@', '%', '&', '+']
    
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = bbox
        # Scale down coordinates to fit in grid
        x1, y1, x2, y2 = x1//8, y1//8, x2//8, y2//8
        char = box_chars[i % len(box_chars)]
        
        # Draw box borders
        for x in range(max(0, x1), min(grid_size, x2+1)):
            if y1 >= 0 and y1 < grid_size:
                grid[y1][x] = '-' if x == x1 or x == x2 else char
            if y2 >= 0 and y2 < grid_size:
                grid[y2][x] = '-' if x == x1 or x == x2 else char
        
        for y in range(max(0, y1), min(grid_size, y2+1)):
            if x1 >= 0 and x1 < grid_size:
                grid[y][x1] = '|' if y == y1 or y == y2 else char
            if x2 >= 0 and x2 < grid_size:
                grid[y][x2] = '|' if y == y1 or y == y2 else char
        
        # Fill corners
        if x1 >= 0 and x1 < grid_size and y1 >= 0 and y1 < grid_size:
            grid[y1][x1] = '+'
        if x2 >= 0 and x2 < grid_size and y1 >= 0 and y1 < grid_size:
            grid[y1][x2] = '+'
        if x1 >= 0 and x1 < grid_size and y2 >= 0 and y2 < grid_size:
            grid[y1][x2] = '+'
        if x2 >= 0 and x2 < grid_size and y2 >= 0 and y2 < grid_size:
            grid[y2][x2] = '+'
    
    # Print the grid
    for row in grid:
        print(''.join(row))
    
    # Print legend
    print("\nLegend:")
    for i, label in enumerate(labels):
        char = box_chars[i % len(box_chars)]
        print(f"  Box {i}: {char} = {label}")

def visual_containment_examples():
    """Show visual examples of different containment scenarios."""
    detector = ObjectDetector()
    
    print("VISUAL CONTAINMENT DETECTION EXAMPLES")
    print("=====================================")
    print("Shows when segment_mask_convexhull will be generated (80%+ containment)")
    
    # Example 1: Complete containment (will generate convex hull)
    bboxes1 = [
        [80, 80, 320, 320],   # Large box
        [120, 120, 280, 280]  # Small box completely inside
    ]
    labels1 = ["Large Box", "Small Box (100% inside)"]
    
    draw_boxes(bboxes1, labels1, 
               "EXAMPLE 1: COMPLETE CONTAINMENT ✅ CONVEX HULL",
               "Small box is 100% inside large box → segment_mask_convexhull generated")
    
    # Test and show results
    print("\nContainment Analysis:")
    for i, bbox in enumerate(bboxes1):
        has_overlap = detector._check_box_overlap(bbox, bboxes1, i)
        print(f"  {labels1[i]}: has_overlap = {has_overlap}")
    
    # Example 2: Partial overlap - below threshold (no convex hull)
    bboxes2 = [
        [80, 80, 240, 240],   # Box 1
        [180, 180, 340, 340]  # Box 2 with partial overlap
    ]
    labels2 = ["Box 1", "Box 2 (25% overlap)"]
    
    draw_boxes(bboxes2, labels2,
               "EXAMPLE 2: LOW OVERLAP ❌ NO CONVEX HULL", 
               "Only 25% overlap between boxes → no segment_mask_convexhull")
    
    print("\nContainment Analysis:")
    for i, bbox in enumerate(bboxes2):
        has_overlap = detector._check_box_overlap(bbox, bboxes2, i)
        print(f"  {labels2[i]}: has_overlap = {has_overlap}")
    
    # Example 3: 80% containment edge case (will generate convex hull)
    bboxes3 = [
        [0, 80, 320, 240],    # Large horizontal box
        [64, 80, 320, 240]    # Box with 80% inside large box
    ]
    labels3 = ["Large Box", "Box (80% inside)"]
    
    draw_boxes(bboxes3, labels3,
               "EXAMPLE 3: 80% CONTAINMENT ✅ CONVEX HULL",
               "Box is exactly 80% inside large box → segment_mask_convexhull generated")
    
    print("\nContainment Analysis:")
    for i, bbox in enumerate(bboxes3):
        has_overlap = detector._check_box_overlap(bbox, bboxes3, i)
        print(f"  {labels3[i]}: has_overlap = {has_overlap}")
    
    # Example 4: No overlap (no convex hull)
    bboxes4 = [
        [40, 40, 160, 160],   # Box 1
        [200, 200, 320, 320], # Box 2 - separate
        [360, 40, 480, 160]   # Box 3 - separate
    ]
    labels4 = ["Box 1", "Box 2 (separate)", "Box 3 (separate)"]
    
    draw_boxes(bboxes4, labels4,
               "EXAMPLE 4: NO OVERLAP ❌ NO CONVEX HULL",
               "All boxes are separate → no segment_mask_convexhull for any")
    
    print("\nContainment Analysis:")
    for i, bbox in enumerate(bboxes4):
        has_overlap = detector._check_box_overlap(bbox, bboxes4, i)
        print(f"  {labels4[i]}: has_overlap = {has_overlap}")
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("✅ segment_mask_convexhull IS generated when:")
    print("   - Box A contains ≥80% of Box B, OR")
    print("   - Box A is ≥80% contained within Box B")
    print()
    print("❌ segment_mask_convexhull is NOT generated when:")
    print("   - Containment is <80% in both directions")
    print("   - Boxes don't overlap at all")
    print()
    print("🎯 segment_mask (Douglas-Peucker) is ALWAYS generated")
    print("="*70)

if __name__ == "__main__":
    visual_containment_examples()
