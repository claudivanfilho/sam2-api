#!/usr/bin/env python3
"""
Visual examples of containment detection scenarios using simple ASCII art.
Shows when segment_mask_convexhull will be generated (80%+ containment).
"""

from src.object_detection import ObjectDetector

def show_containment_examples():
    """Show visual examples using simple ASCII diagrams."""
    detector = ObjectDetector()
    
    print("VISUAL CONTAINMENT DETECTION EXAMPLES")
    print("=====================================")
    print("Shows when segment_mask_convexhull will be generated (80%+ containment)")
    print()
    
    # Example 1: Complete containment
    print("EXAMPLE 1: COMPLETE CONTAINMENT ✅ CONVEX HULL")
    print("=" * 50)
    print("Small box is 100% inside large box")
    print()
    print("    +------------------------+")
    print("    |                        |")
    print("    |    +-----------+       |")
    print("    |    |  Small    |       |  Large Box")
    print("    |    |   Box     |       |")
    print("    |    +-----------+       |")
    print("    |                        |")
    print("    +------------------------+")
    print()
    
    bboxes1 = [[10, 10, 100, 100], [30, 30, 70, 70]]
    for i, bbox in enumerate(bboxes1):
        has_overlap = detector._check_box_overlap(bbox, bboxes1, i)
        box_name = "Large Box" if i == 0 else "Small Box"
        print(f"  {box_name}: has_overlap = {has_overlap}")
    print("  → segment_mask_convexhull WILL be generated for both boxes")
    print()
    
    # Example 2: Partial overlap - below threshold
    print("EXAMPLE 2: LOW OVERLAP ❌ NO CONVEX HULL")
    print("=" * 50)
    print("Only 25% overlap between boxes")
    print()
    print("    +-----------+")
    print("    |   Box 1   |")
    print("    |       +---|---------+")
    print("    |       |###|         |")
    print("    +-------|###|   Box 2 |")
    print("            |###|         |")
    print("            +-------------+")
    print("            (25% overlap)")
    print()
    
    bboxes2 = [[10, 10, 60, 60], [35, 35, 85, 85]]
    for i, bbox in enumerate(bboxes2):
        has_overlap = detector._check_box_overlap(bbox, bboxes2, i)
        box_name = f"Box {i+1}"
        print(f"  {box_name}: has_overlap = {has_overlap}")
    print("  → segment_mask_convexhull will NOT be generated")
    print()
    
    # Example 3: 80% containment edge case
    print("EXAMPLE 3: 80% CONTAINMENT ✅ CONVEX HULL")
    print("=" * 50)
    print("Box is exactly 80% inside large box")
    print()
    print("    +---------------------------+")
    print("    |     +---------------+     |")
    print("    |     |   80% Box     |     |  Large Box")
    print("    |     +---------------+     |")
    print("    +---------------------------+")
    print("          (80% contained)")
    print()
    
    bboxes3 = [[0, 0, 100, 100], [20, 0, 120, 100]]
    for i, bbox in enumerate(bboxes3):
        has_overlap = detector._check_box_overlap(bbox, bboxes3, i)
        box_name = "Large Box" if i == 0 else "80% Box"
        print(f"  {box_name}: has_overlap = {has_overlap}")
    print("  → segment_mask_convexhull WILL be generated for both boxes")
    print()
    
    # Example 4: No overlap
    print("EXAMPLE 4: NO OVERLAP ❌ NO CONVEX HULL")
    print("=" * 50)
    print("All boxes are completely separate")
    print()
    print("    +-------+       +-------+       +-------+")
    print("    | Box 1 |       | Box 2 |       | Box 3 |")
    print("    +-------+       +-------+       +-------+")
    print("      (no overlap between any boxes)")
    print()
    
    bboxes4 = [[10, 10, 50, 50], [100, 100, 140, 140], [200, 200, 240, 240]]
    for i, bbox in enumerate(bboxes4):
        has_overlap = detector._check_box_overlap(bbox, bboxes4, i)
        box_name = f"Box {i+1}"
        print(f"  {box_name}: has_overlap = {has_overlap}")
    print("  → segment_mask_convexhull will NOT be generated")
    print()
    
    # Summary
    print("=" * 70)
    print("MASK GENERATION SUMMARY:")
    print("=" * 70)
    print()
    print("🎯 segment_mask (Douglas-Peucker):")
    print("   ✅ ALWAYS generated for every detected object")
    print("   📐 Simplified polygon approximation of the mask")
    print()
    print("🔷 segment_mask_convexhull (Convex Hull):")
    print("   ✅ Generated ONLY when containment ≥ 80%:")
    print("      • Box A contains ≥80% of Box B, OR")
    print("      • Box A is ≥80% contained within Box B")
    print("   🔺 Convex polygon that encompasses the mask")
    print()
    print("💡 USE CASES:")
    print("   • Douglas-Peucker: Precise object boundaries")
    print("   • Convex Hull: Simplified boundaries for overlapping objects")
    print("   • Both provide different levels of detail for same object")
    print("=" * 70)

if __name__ == "__main__":
    show_containment_examples()
