"""
Object detection functionality using Florence-2.
Handles object detection with integrated SAM2 segmentation, face landmarks, and selfie segmentation.
"""

import torch
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .model_manager import get_model_manager
from .utils import upload_mask_to_s3
from .sam2_segmentation import sam2_segmenter
from .mediapipe_operations import mediapipe_processor
from .image_utils import get_douglas_peucker_points, remove_small_fragments
from .evf_sam_operations import evf_sam_segmenter

class ObjectDetector:
    """Handles Florence-2 object detection with integrated processing."""
    
    def __init__(self):
        self.model_manager = get_model_manager()
    
    def detect_objects(self, image_pil, task="<OD>", 
                      confidence_threshold=0.3, text_input=None, return_polygon_points=False, 
                      douglas_peucker_epsilon=0.002):
        """
        Detect objects in an image using Florence-2-large model.
        
        Args:
            image_pil: PIL Image object
            task: Task prompt (default: "<OD>" for object detection)
            confidence_threshold: Minimum confidence for detections
            text_input: Text input for referring expression segmentation
            return_polygon_points: If True, return simplified polygon points instead of S3 URLs for masks
            douglas_peucker_epsilon: Epsilon ratio for Douglas-Peucker polygon simplification (default: 0.002)
            
        Returns:
            dict: Parsed detection results with objects and their bounding boxes
        """
        # Get models (will load if needed)
        model = self.model_manager.get_model('florence2')
        processor = self.model_manager.get_model('florence2_processor')
        device = self.model_manager.get_model('florence2_device')
        dtype = self.model_manager.get_model('florence2_dtype')
        
        # Use the provided PIL image directly
        image = image_pil
        
        # Handle referring expression segmentation
        if task == "<REFERRING_EXPRESSION_SEGMENTATION>" and text_input:
            prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{text_input}"
        else:
            prompt = task
        
        # Prepare inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
        
        # Run Florence-2 detection and SAM2 preprocessing in parallel for maximum efficiency
        florence_start_time = time.time()
        
        def run_florence_detection():
            """Run Florence-2 object detection"""
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                    use_cache=True
                )
            return generated_ids
        
        def run_sam2_preprocessing():
            """Preprocess image for SAM2"""
            image_array = np.array(image)
            sam2_segmenter.model_manager.get_model('sam2_predictor').set_image(image_array)
            return image_array
        
        # Execute both operations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks simultaneously
            florence_future = executor.submit(run_florence_detection)
            sam2_future = executor.submit(run_sam2_preprocessing)
            
            # Get results as they complete
            generated_ids = florence_future.result()
            image_array = sam2_future.result()
        
        florence_end_time = time.time()
        florence_duration = florence_end_time - florence_start_time
        print(f"⏱️  Florence-2 detection + SAM2 preprocessing (parallel) took {florence_duration:.3f} seconds")
        
        # Decode and parse results
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=task, 
            image_size=(image.width, image.height)
        )
        
        # Format results
        objects = []
        segmentation_mask = None
        
        if task in parsed_answer:
            detections = parsed_answer[task]
            
            # Handle different output formats
            if 'bboxes' in detections and 'labels' in detections:
                bboxes = detections['bboxes']
                labels = detections['labels']
                
                print(detections, 'detectionss')
                
                # Define allowed object labels to keep
                allowed_labels = [
                    'man', 'woman', 'boy', 'girl', 'person', 'people', 'human',
                    'dog', 'cat', 'horse',
                    'car', 'truck', 'bicycle', 'bus', 'vehicle', 'motorcycle'
                ]
                
                # Define person-related labels to trigger selfie segmentation
                person_labels = ['human', 'man', 'woman', 'boy', 'girl', 'person', 'people']
                
                # SAM2 preprocessing already completed in parallel above - no need to repeat
                print(f"🔄 SAM2 image preprocessing completed in parallel with Florence-2")
                
                # Process objects in parallel for better performance
                objects = []
                parallel_start_time = time.time()
                
                # Use ThreadPoolExecutor to process multiple objects simultaneously
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit all object processing tasks with preprocessed image
                    future_to_data = {
                        executor.submit(self._process_detected_object, bbox, label, image_array, allowed_labels, person_labels, image, return_polygon_points, douglas_peucker_epsilon): (bbox, label)
                        for bbox, label in zip(bboxes, labels)
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_data):
                        bbox, label = future_to_data[future]
                        try:
                            result = future.result()
                            if result is not None:  # Only add non-filtered objects
                                objects.append(result)
                        except Exception as e:
                            print(f"⚠️  Parallel processing failed for object {label}: {e}")
                
                parallel_end_time = time.time()
                parallel_duration = parallel_end_time - parallel_start_time
                print(f"🚀 Parallel object processing took {parallel_duration:.3f} seconds for {len(objects)} objects")
            
            # Handle segmentation masks for RES tasks
            if 'polygons' in detections:
                segmentation_mask = detections['polygons']
        
        return {
            'objects': objects,
            'task_used': task,
            'segmentation_mask': segmentation_mask,
            'raw_output': parsed_answer
        }
    
    def _process_detected_object(self, bbox, label, image_array, allowed_labels, person_labels, original_image, return_polygon_points=False, douglas_peucker_epsilon=0.002):
        """
        Process a single detected object (bbox + label) with segmentation.
        This function is designed to be run in parallel for multiple objects.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            label: Object label string
            image_array: Preprocessed numpy array (already set in predictor)
            allowed_labels: List of allowed object labels
            person_labels: List of person-related labels for selfie segmentation
            original_image: PIL Image object for cropping
            return_polygon_points: If True, return simplified polygon points instead of S3 URLs for masks
            douglas_peucker_epsilon: Epsilon ratio for Douglas-Peucker polygon simplification
            
        Returns:
            dict: Processed object with segmentation results, or None if filtered out
        """
        # Filter out objects that are not in the allowed list
        if not any(allowed_word == label.lower() for allowed_word in allowed_labels):
            return None  # Skip this object
        
        # Calculate bounding box dimensions
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Filter out objects smaller than 250px in width or height
        if bbox_width < 250 and bbox_height < 250:
            print(f"⚠️  Filtering out {label}: size {bbox_width}x{bbox_height}px (< 250)")
            return None  # Skip this object
        
        obj = {
            'label': label,
            'bbox': bbox,  # [x1, y1, x2, y2]
            'multiclasses': None,
            'face_landmarks': None,
            'segment_mask': None
        }
        
        # Crop the detected object from the original image for segmentation
        try:
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_image.width, x2)
            y2 = min(original_image.height, y2)
            
            image_processing_start_time = time.time()
            # Crop the object from the image
            cropped_object = original_image.crop((x1, y1, x2, y2))
            
            image_processing_end_time = time.time()
            image_processing_duration = image_processing_end_time - image_processing_start_time
            print(f"⏱️  Image processing (crop) took {image_processing_duration:.3f} seconds for {label}")
            
            # Run SAM2 segmentation using the already preprocessed image
            sam2_start_time = time.time()
            
            # Calculate point prompt at middle x and 25% from top
            middle_x = (x1 + x2) // 2
            point_y = y1 + int((y2 - y1) * 0.25)  # 25% from top of bbox
            point_prompt = [middle_x, point_y]
            
            # Get mask array from SAM2 segmenter with box and point prompts
            mask_array = sam2_segmenter.segment_with_box(image_array, [x1, y1, x2, y2], point_prompt=point_prompt)
            
            # Remove small fragments from SAM2 mask, keeping only the largest component
            mask_array_cleaned = remove_small_fragments(mask_array.astype(np.uint8) * 255)
            
            # Process cleaned mask based on return_polygon_points flag
            if return_polygon_points:
                # Extract simplified polygon points from the mask
                polygon_points = get_douglas_peucker_points(mask_array_cleaned, epsilon_ratio=douglas_peucker_epsilon)
                obj['segment_mask'] = polygon_points
                print(f"✅ Extracted {len(polygon_points)} simplified polygon points for {label}")
            else:
                # Upload mask to S3 and return URL
                crop_box = (x1, y1, x2, y2)
                segment_mask_url = upload_mask_to_s3(mask_array_cleaned, original_image.size, crop_box=crop_box)
                obj['segment_mask'] = segment_mask_url
            
            # Check if detected object is a person and run selfie segmentation + face landmarks
            if any(person_word in label.lower() for person_word in person_labels):
                # Run selfie segmentation on the cropped person
                selfie_start_time = time.time()
                selfie_result = mediapipe_processor.segment_selfie_multiclass(image_pil=cropped_object)
                selfie_end_time = time.time()
                selfie_duration = selfie_end_time - selfie_start_time
                print(f"⏱️  MediaPipe selfie segmentation took {selfie_duration:.3f} seconds for {label}")
                
                # Get the refined crop based on hair and face_skin regions
                refined_crop_info = self._get_refined_crop_from_masks(
                    selfie_result, cropped_object
                )

                landmarks_image = cropped_object  # Default to original crop
                refined_selfie_result = None  # Initialize for face landmark processing
                
                if refined_crop_info and refined_crop_info['image']:
                    landmarks_image = refined_crop_info['image']
                    print(f"🎯 Using refined crop for face landmarks: {landmarks_image.size}")
                    
                    # Run MediaPipe segmentation on the refined crop early for face landmark processing
                    refined_selfie_result = mediapipe_processor.segment_selfie_multiclass(
                        image_pil=landmarks_image
                    )

                landmarks_start_time = time.time()
                # Run face landmark detection - use refined crop if available, otherwise use original crop
                landmarks_result = mediapipe_processor.detect_face_landmarks(image_pil=landmarks_image)
                landmarks_end_time = time.time()
                landmarks_duration = landmarks_end_time - landmarks_start_time
                print(f"⏱️  MediaPipe face landmarks took {landmarks_duration:.3f} seconds for {label}")
                
                # Create face landmark mask if landmarks were detected
                face_landmark_result = None
                if landmarks_result and 'landmarks' in landmarks_result:
                    face_mask_start_time = time.time()
                    
                    # Get face skin mask from refined_selfie_result (if available) or original selfie_result
                    face_skin_mask = None
                    segmentation_result = refined_selfie_result if refined_selfie_result else selfie_result
                    
                    if segmentation_result and 'mask' in segmentation_result:
                        selfie_mask = np.array(segmentation_result['mask'], dtype=np.uint8)
                        # Extract face skin mask (class 3) from selfie segmentation
                        face_skin_mask = (selfie_mask == 3).astype(np.uint8)
                        source_type = "refined selfie" if refined_selfie_result else "original selfie"
                    
                    # Expand face landmark points using the expandContourPoints function
                    original_landmarks = landmarks_result['landmarks'][0]
                    
                    expanded_landmarks = self.expandContourPoints(
                        originalPoints=original_landmarks,
                        faceSkinMask=face_skin_mask,
                        image_size=landmarks_image.size
                    )
                    
                    # Create face landmark mask using expanded points
                    face_landmark_result = self._create_face_landmark_mask(
                        expanded_landmarks, landmarks_image.size, return_polygon_points, douglas_peucker_epsilon
                    )
                    if face_landmark_result:
                        if 'multiclasses' not in obj or obj['multiclasses'] is None:
                            obj['multiclasses'] = {}
                        # Use 'url' or 'points' key based on return_polygon_points flag
                        result_key = 'points' if return_polygon_points else 'url'
                        obj['multiclasses']['face_landmark_mask'] = face_landmark_result[result_key]
                    
                    face_mask_end_time = time.time()
                    face_mask_duration = face_mask_end_time - face_mask_start_time
                    print(f"⏱️  Face landmark mask creation (with expansion) took {face_mask_duration:.3f} seconds for {label}")
                
                # Initialize processed_classes as empty
                processed_classes = {}
                refined_crop_start_time = time.time()

                if refined_crop_info and refined_crop_info['image']:
                    refined_cropped_image = refined_crop_info['image']
                    
                    # Use the already computed refined_selfie_result if available, otherwise compute it
                    if not refined_selfie_result:
                        refined_selfie_result = mediapipe_processor.segment_selfie_multiclass(
                            image_pil=refined_cropped_image
                        )
                    
                    if refined_selfie_result and 'mask' in refined_selfie_result:
                        # Reprocess the masks with the refined segmentation
                        # Still need SAM2 intersection but with adjusted coordinates for refined crop
                        processed_classes = self._process_refined_multiclass_masks(
                            refined_selfie_result, refined_cropped_image, mask_array_cleaned, x1, y1, x2, y2, original_image, face_landmark_result, return_polygon_points, douglas_peucker_epsilon
                        )
                        
                        if processed_classes:
                            refined_crop_end_time = time.time()
                            refined_crop_duration = refined_crop_end_time - refined_crop_start_time
                            print(f"🚀 Refined crop reprocessing took {refined_crop_duration:.3f} seconds")
                
                if processed_classes:
                    # Merge processed_classes with existing multiclasses to preserve face_landmark_mask
                    if 'multiclasses' not in obj or obj['multiclasses'] is None:
                        obj['multiclasses'] = {}
                    obj['multiclasses'].update(processed_classes)
            
            # Check if detected object is an animal and create face mask
            animal_labels = ['dog', 'cat', 'horse']
            if any(animal_word in label.lower() for animal_word in animal_labels):
                animal_face_start_time = time.time()
                
                # Use EVF-SAM2 to segment animal face with text prompt
                try:
                    face_mask_result = evf_sam_segmenter.segment_with_text(
                        image_pil=cropped_object,
                        prompt="animal head",
                        semantic=True
                    )
                    
                    if face_mask_result:
                        # Convert list mask to numpy array
                        animal_face_mask = np.array(face_mask_result, dtype=np.uint8) * 255
                        
                        # Remove small fragments from animal face mask
                        animal_face_mask_cleaned = remove_small_fragments(animal_face_mask)
                        
                        if np.any(animal_face_mask_cleaned):
                            # Process based on return_polygon_points flag
                            if return_polygon_points:
                                # Extract simplified polygon points from the mask
                                polygon_points = get_douglas_peucker_points(animal_face_mask_cleaned, epsilon_ratio=douglas_peucker_epsilon)
                                if polygon_points:
                                    if 'multiclasses' not in obj or obj['multiclasses'] is None:
                                        obj['multiclasses'] = {}
                                    obj['multiclasses']['head_mask'] = polygon_points
                            else:
                                # Upload mask to S3 and return URL
                                face_mask_url = upload_mask_to_s3(animal_face_mask_cleaned, cropped_object.size, s3_path="animal_faces")
                                if face_mask_url:
                                    if 'multiclasses' not in obj or obj['multiclasses'] is None:
                                        obj['multiclasses'] = {}
                                    obj['multiclasses']['head_mask'] = face_mask_url
                        else:
                            print(f"⚠️  No valid face mask found for {label} after cleaning")
                    else:
                        print(f"⚠️  EVF-SAM2 did not return a face mask for {label}")
                        
                except Exception as e:
                    print(f"⚠️  Failed to create animal face mask for {label}: {e}")
                
                animal_face_end_time = time.time()
                animal_face_duration = animal_face_end_time - animal_face_start_time
                print(f"⏱️  Animal face mask creation took {animal_face_duration:.3f} seconds for {label}")
                
                
        except Exception as e:
            print(f"⚠️  Failed to process object {label}: {e}")
            # Continue without segmentation if it fails
            pass
        
        return obj
    
    def _get_refined_crop_from_masks(self, selfie_result, cropped_image):
        """
        Create a refined crop from the original cropped image based on hair and face_skin regions.
        Crops from Y coordinate 0 (top) to the bottommost pixel of either face_skin or hair.
        
        Args:
            selfie_result: MediaPipe selfie segmentation result
            cropped_image: PIL Image of the original cropped person
            
        Returns:
            dict: Contains 'image' (PIL.Image), or None if no valid regions found
        """
        try:
            selfie_mask = np.array(selfie_result['mask'], dtype=np.uint8)
            
            # Find hair pixels (class 1) and face_skin pixels (class 3)
            hair_pixels = np.where(selfie_mask == 1)
            face_skin_pixels = np.where(selfie_mask == 3)
            
            # Check if we have either hair or face_skin regions
            if len(hair_pixels[0]) == 0 and len(face_skin_pixels[0]) == 0:
                print("⚠️  No hair or face_skin regions found for refined crop")
                return None
            
            # Get the bottommost y-coordinate from both hair and face_skin
            bottom_y_coords = []
            if len(hair_pixels[0]) > 0:
                bottom_y_coords.extend(hair_pixels[0])
            if len(face_skin_pixels[0]) > 0:
                bottom_y_coords.extend(face_skin_pixels[0])
            
            if not bottom_y_coords:
                return None
                
            # Start from top (Y=0) and go to the bottommost pixel
            refined_top = 0
            bottom_y = max(bottom_y_coords)
            
            # Add some padding (10% of the detected height range)
            detected_height = bottom_y - refined_top
            padding = max(10, int(detected_height * 0.1))
            
            # Ensure coordinates are within image bounds
            refined_bottom = min(selfie_mask.shape[0], bottom_y + padding)
            
            # Crop the image from top to bottom of detected regions (keep full width)
            refined_cropped = cropped_image.crop((0, refined_top, cropped_image.width, refined_bottom))
            
            return {
                'image': refined_cropped
            }
            
        except Exception as e:
            print(f"⚠️  Failed to create refined crop: {e}")
            return None
    
    def _create_face_landmark_mask(self, face_landmarks, image_size, return_polygon_points=False, douglas_peucker_epsilon=0.002):
        """
        Create a face mask using convex hull of face landmarks.
        
        Args:
            face_landmarks: List of landmark points in format [[x, y, z], ...]
                           where coordinates are normalized (0-1)
            image_size: Tuple of (width, height) for the image
            return_polygon_points: If True, return simplified polygon points instead of S3 URL
            douglas_peucker_epsilon: Epsilon ratio for Douglas-Peucker polygon simplification
            
        Returns:
            dict: Contains 'url'/'points' and 'mask' (numpy array), or None if failed
        """
        try:
            if not face_landmarks or len(face_landmarks) < 3:
                print("⚠️  Not enough face landmarks to create convex hull")
                return None
            
            # Convert normalized landmarks to pixel coordinates
            # landmarks format: [[x, y, z], [x, y, z], ...]
            # where x, y are normalized (0-1) and z is depth
            width, height = image_size
            points = []
            
            for landmark in face_landmarks:
                if len(landmark) >= 2:  # Ensure we have at least x, y coordinates
                    # Convert normalized coordinates to pixel coordinates
                    pixel_x = int(landmark[0] * width)
                    pixel_y = int(landmark[1] * height)
                    points.append([pixel_x, pixel_y])
            
            if len(points) < 3:
                print("⚠️  Not enough valid landmark points to create convex hull")
                return None
            
            # Convert to numpy array
            points = np.array(points, dtype=np.int32)
            
            # Create convex hull
            hull = cv2.convexHull(points)
            
            # Create mask with image dimensions (height, width)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Fill convex polygon
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Remove small fragments from the face landmark mask
            mask_cleaned = remove_small_fragments(mask)
            
            # Return either polygon points or S3 URL based on flag
            if return_polygon_points:
                # Extract simplified polygon points from the mask
                polygon_points = get_douglas_peucker_points(mask_cleaned, epsilon_ratio=douglas_peucker_epsilon)
                return {
                    'points': polygon_points,
                    'mask': mask_cleaned
                }
            else:
                # Upload cleaned mask to S3
                mask_url = upload_mask_to_s3(mask_cleaned, image_size, s3_path="face_landmarks")
                return {
                    'url': mask_url,
                    'mask': mask_cleaned
                }
            
        except Exception as e:
            print(f"⚠️  Failed to create face landmark mask: {e}")
            return None
    
    def _process_refined_multiclass_masks(self, refined_selfie_result, refined_cropped_image, mask_array, x1, y1, x2, y2, original_image, face_landmark_result=None, return_polygon_points=False, douglas_peucker_epsilon=0.002):
        """
        Process refined selfie segmentation masks with SAM2 intersection.
        Only processes hair, face_skin, and others categories from the refined crop.
        Since refined crop always starts from Y=0, SAM2 intersection is straightforward.
        
        Args:
            refined_selfie_result: MediaPipe selfie segmentation result from refined crop
            refined_cropped_image: PIL Image of the refined cropped region
            mask_array: SAM2 mask array (full image dimensions)
            x1, y1, x2, y2: Original bounding box coordinates
            original_image: PIL Image object for sizing
            face_landmark_result: Dict containing face landmark mask data {'url': str, 'mask': np.array}
            return_polygon_points: If True, return simplified polygon points instead of S3 URLs
            douglas_peucker_epsilon: Epsilon ratio for Douglas-Peucker polygon simplification
            
        Returns:
            dict: Dictionary of class names to S3 URLs or polygon points, or None if no valid masks
        """
        try:
            # Convert refined selfie mask to numpy array
            refined_selfie_mask = np.array(refined_selfie_result['mask'], dtype=np.uint8)
            
            # Get the SAM2 mask for the original crop region
            sam2_mask_cropped = mask_array[y1:y2, x1:x2]
            
            # Since refined crop always starts from Y=0, extract SAM2 region from top
            original_crop_height = y2 - y1
            refined_crop_height = refined_cropped_image.height
            
            # Extract SAM2 region from top of the original crop up to refined crop height
            sam2_refined_end = min(original_crop_height, refined_crop_height)
            sam2_mask_refined = sam2_mask_cropped[0:sam2_refined_end, :]
            
            # Resize SAM2 mask to match refined selfie mask dimensions if needed
            if sam2_mask_refined.shape != refined_selfie_mask.shape:
                from PIL import Image
                sam2_mask_pil = Image.fromarray((sam2_mask_refined > 0).astype(np.uint8) * 255)
                sam2_mask_pil = sam2_mask_pil.resize((refined_selfie_mask.shape[1], refined_selfie_mask.shape[0]), Image.Resampling.NEAREST)
                sam2_mask_refined_resized = np.array(sam2_mask_pil) > 0
            else:
                sam2_mask_refined_resized = sam2_mask_refined > 0
            
            # Process specific classes with SAM2 intersection
            masks_to_upload = []
            class_names = []
            
            # Hair class (1) - compute for head_mask but don't upload separately
            hair_mask = (refined_selfie_mask == 1) & sam2_mask_refined_resized
            
            # Face-skin class (3) - compute for head_mask but don't upload separately
            face_skin_mask = (refined_selfie_mask == 3) & sam2_mask_refined_resized
            
            # Others class (5) - but only above the bottommost face_skin pixel
            others_mask_full = (refined_selfie_mask == 5) & sam2_mask_refined_resized
            others_mask = others_mask_full.copy()
            
            # Find the bottommost face_skin pixel to limit others mask
            if np.any(face_skin_mask):
                face_skin_y_coords = np.where(face_skin_mask)[0]
                bottommost_face_skin_y = np.max(face_skin_y_coords)
                
                # Zero out all others pixels below the bottommost face_skin
                others_mask[bottommost_face_skin_y+1:, :] = False
            
            if np.any(others_mask):
                masks_to_upload.append(others_mask.astype(np.uint8) * 255)
                class_names.append('others')
            
            # Create head_mask as union of intersected hair, face_skin, limited others, and face landmark mask
            head_mask_components = []
            if np.any(hair_mask):
                head_mask_components.append(hair_mask)
            if np.any(face_skin_mask):
                head_mask_components.append(face_skin_mask)
            if np.any(others_mask):
                head_mask_components.append(others_mask)
            
            # Add face landmark mask if available
            face_landmark_mask_resized = None
            if face_landmark_result and 'mask' in face_landmark_result:
                face_landmark_mask = face_landmark_result['mask']
                # Resize face landmark mask to match refined selfie mask dimensions if needed
                if face_landmark_mask.shape != refined_selfie_mask.shape:
                    from PIL import Image
                    landmark_mask_pil = Image.fromarray(face_landmark_mask)
                    landmark_mask_pil = landmark_mask_pil.resize((refined_selfie_mask.shape[1], refined_selfie_mask.shape[0]), Image.Resampling.NEAREST)
                    face_landmark_mask_resized = np.array(landmark_mask_pil) > 0
                else:
                    face_landmark_mask_resized = face_landmark_mask > 0
                
                # Add face landmark mask as a separate uploadable mask
                if np.any(face_landmark_mask_resized):
                    masks_to_upload.append(face_landmark_mask_resized.astype(np.uint8) * 255)
                    class_names.append('face_landmark_mask')
                    head_mask_components.append(face_landmark_mask_resized)
            
            if head_mask_components:
                # Union all components to create head_mask
                head_mask = np.logical_or.reduce(head_mask_components)
                if np.any(head_mask):
                    # Remove small fragments from head_mask
                    head_mask_cleaned = remove_small_fragments(head_mask.astype(np.uint8) * 255)
                    if np.any(head_mask_cleaned):
                        masks_to_upload.append(head_mask_cleaned)
                        class_names.append('head_mask')
            
            # Process all refined masks based on return_polygon_points flag
            if masks_to_upload:
                mask_size = (refined_selfie_mask.shape[1], refined_selfie_mask.shape[0])  # (width, height)
                processed_classes = {}
                
                if return_polygon_points:
                    # Extract polygon points from all masks
                    for mask, class_name in zip(masks_to_upload, class_names):
                        polygon_points = get_douglas_peucker_points(mask, epsilon_ratio=douglas_peucker_epsilon)
                        if polygon_points:  # Only add if we got valid points
                            processed_classes[class_name] = polygon_points
                    
                else:
                    # Use ThreadPoolExecutor for parallel S3 uploads
                    with ThreadPoolExecutor(max_workers=3) as upload_executor:
                        upload_futures = {
                            upload_executor.submit(upload_mask_to_s3, mask, mask_size, s3_path="body_parts_refined"): class_name
                            for mask, class_name in zip(masks_to_upload, class_names)
                        }
                        
                        # Collect results as they complete
                        for future in as_completed(upload_futures):
                            class_name = upload_futures[future]
                            try:
                                s3_url = future.result()
                                if s3_url:  # Only add if upload was successful
                                    processed_classes[class_name] = s3_url
                            except Exception as e:
                                print(f"⚠️  Failed to upload refined {class_name} mask: {e}")
                        
                        print(f"✅ Refined masks with SAM2 intersection uploaded: {len(processed_classes)} classes")
                
                return processed_classes if processed_classes else None
            
            return None
            
        except Exception as e:
            print(f"⚠️  Failed to process refined multiclass masks: {e}")
            return None
    
    def expandContourPoints(self, originalPoints, faceSkinMask=None, image_size=None):
        """
        Expands face contour points outward to create a more inclusive face boundary,
        specifically targeting the forehead area.
        
        Args:
            originalPoints: List of face contour points from MediaPipe in format [[x, y, z], ...]
                           where coordinates are normalized (0-1)
            faceSkinMask: Face skin segmentation mask (numpy array) to constrain expansion
            image_size: Tuple of (width, height) for the image
            
        Returns:
            list: Expanded contour points in the same format as input
        """
        try:
            if not originalPoints or len(originalPoints) < 3:
                print("⚠️  Not enough original points for expansion")
                return originalPoints
            
            # Forehead indices (points to be expanded)
            FOREHEAD_INDICES = [10, 338, 297, 332, 284, 251, 389, 162, 21, 54, 103, 67, 109]
            
            # Convert normalized points to pixel coordinates if image_size is provided
            if image_size:
                width, height = image_size
                pixel_points = []
                for point in originalPoints:
                    if len(point) >= 2:
                        pixel_x = point[0] * width
                        pixel_y = point[1] * height
                        z = point[2] if len(point) > 2 else 0
                        pixel_points.append([pixel_x, pixel_y, z])
                    else:
                        pixel_points.append(point)
            else:
                pixel_points = [point[:] for point in originalPoints]  # Deep copy
            
            # Calculate face center from all contour points
            face_center_x = sum(point[0] for point in pixel_points) / len(pixel_points)
            face_center_y = sum(point[1] for point in pixel_points) / len(pixel_points)
            
            # Calculate original face landmark bounding box
            min_x = min(point[0] for point in pixel_points)
            max_x = max(point[0] for point in pixel_points)
            min_y = min(point[1] for point in pixel_points)
            max_y = max(point[1] for point in pixel_points)
            
            # Expand only forehead points
            expanded_points = pixel_points[:]  # Start with copy of original points
            
            for i, point in enumerate(pixel_points):
                # Only expand forehead points
                if i not in FOREHEAD_INDICES:
                    continue
                
                current_x, current_y = point[0], point[1]
                
                # Calculate outward direction vector from face center to current point
                direction_x = current_x - face_center_x
                direction_y = current_y - face_center_y
                
                # Normalize direction vector
                direction_length = np.sqrt(direction_x**2 + direction_y**2)
                if direction_length == 0:
                    continue  # Skip if point is at face center
                
                direction_x /= direction_length
                direction_y /= direction_length
                
                # Expand point outward in 2-pixel steps (max 80 pixels for forehead)
                max_expansion = 80
                step_size = 2
                
                new_x, new_y = current_x, current_y
                
                for step in range(0, max_expansion, step_size):
                    test_x = current_x + direction_x * step
                    test_y = current_y + direction_y * step
                    
                    # Convert to integer coordinates for mask checking
                    test_x_int = int(round(test_x))
                    test_y_int = int(round(test_y))
                    
                    # Check bounds
                    if image_size:
                        width, height = image_size
                        if test_x_int < 0 or test_x_int >= width or test_y_int < 0 or test_y_int >= height:
                            break
                    
                    # Stop expansion if X position is outside original face landmark bounding box
                    if test_x < min_x or test_x > max_x:
                        break
                    
                    # Check if expanded point goes outside the face skin mask (stop when outside face skin)
                    if faceSkinMask is not None:
                        if (test_y_int < faceSkinMask.shape[0] and 
                            test_x_int < faceSkinMask.shape[1]):
                            if faceSkinMask[test_y_int, test_x_int] == 0:
                                break
                        else:
                            # Point is outside mask dimensions, stop expansion
                            break
                    
                    # Update position if we can continue
                    new_x, new_y = test_x, test_y
                
                # Update the expanded point
                if len(expanded_points[i]) > 2:
                    expanded_points[i] = [new_x, new_y, point[2]]  # Keep original z coordinate
                else:
                    expanded_points[i] = [new_x, new_y]
                
                # Calculate expansion distance
                expansion_distance = np.sqrt((new_x - current_x)**2 + (new_y - current_y)**2)
            
            # Convert back to normalized coordinates if image_size was provided
            if image_size:
                width, height = image_size
                normalized_points = []
                for point in expanded_points:
                    if len(point) >= 2:
                        norm_x = point[0] / width
                        norm_y = point[1] / height
                        if len(point) > 2:
                            normalized_points.append([norm_x, norm_y, point[2]])
                        else:
                            normalized_points.append([norm_x, norm_y])
                    else:
                        normalized_points.append(point)
                return normalized_points
            
            return expanded_points
            
        except Exception as e:
            print(f"⚠️  Failed to expand contour points: {e}")
            return originalPoints

# Create global instance without initializing models
object_detector = ObjectDetector()
