"""
Image utility functions for mask processing and contour operations.
Contains helper functions for image manipulation, mask cleaning, and polygon simplification.
"""

import cv2
import numpy as np


def get_douglas_peucker_points(mask, epsilon_ratio=0.002, return_largest_only=True):
    """
    Extract simplified contour points from a binary mask using Douglas-Peucker algorithm.
    
    Args:
        mask: Binary mask (numpy array) with values 0 or 255
        epsilon_ratio: Approximation accuracy as ratio of contour perimeter (default: 0.002 = 0.2%)
        return_largest_only: If True, return only the largest contour; if False, return all contours
        
    Returns:
        list: Simplified contour points in format [[x, y], [x, y], ...] or list of contours if return_largest_only=False
    """
    try:
        # Convert to binary if needed
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠️  No contours found in mask")
            return []
        
        if return_largest_only:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            contours_to_process = [largest_contour]
        else:
            # Process all contours
            contours_to_process = contours
        
        simplified_contours = []
        
        for contour in contours_to_process:
            # Calculate epsilon based on contour perimeter
            perimeter = cv2.arcLength(contour, True)
            epsilon = epsilon_ratio * perimeter
            
            # Apply Douglas-Peucker simplification
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert from OpenCV format [[x, y]] to simple list [[x, y], [x, y], ...]
            points = simplified.reshape(-1, 2).tolist()
            
            original_points = len(contour)
            simplified_points = len(points)
            reduction_ratio = (1 - simplified_points / original_points) * 100
            
            print(f"✅ Douglas-Peucker simplification: {original_points} → {simplified_points} points ({reduction_ratio:.1f}% reduction)")
            print(f"   Epsilon: {epsilon:.2f} (perimeter: {perimeter:.1f}, ratio: {epsilon_ratio})")
            
            simplified_contours.append(points)
        
        # Return single contour if return_largest_only=True, otherwise return all
        if return_largest_only:
            return simplified_contours[0] if simplified_contours else []
        else:
            return simplified_contours
            
    except Exception as e:
        print(f"⚠️  Failed to extract Douglas-Peucker points: {e}")
        return [] if return_largest_only else []


def remove_small_fragments(mask):
    """
    Keep only the largest connected component from a binary mask.
    
    Args:
        mask: Binary mask (numpy array) with values 0 or 255
        
    Returns:
        numpy array: Cleaned mask with only the largest component
    """
    try:
        # Convert to binary if needed
        binary_mask = (mask > 127).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # Create output mask
        cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        
        if num_labels <= 1:  # Only background, no components found
            print(f"⚠️  No connected components found")
            return cleaned_mask
        
        # Find the largest component (skip background label 0)
        largest_component_idx = 0
        largest_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_component_idx = i
        
        # Keep only the largest component
        cleaned_mask[labels == largest_component_idx] = 255
        removed_components = num_labels - 2  # Total components minus background and kept component
        print(f"✅ Kept largest component ({largest_area}px), removed {removed_components} smaller fragments")
        return cleaned_mask
            
    except Exception as e:
        print(f"⚠️  Failed to remove fragments: {e}")
        return mask
