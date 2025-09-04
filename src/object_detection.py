"""
Object detection functionality using Florence-2.
Handles object detection with integrated SAM2 segmentation, face landmarks, and selfie segmentation.
"""

import torch
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .model_manager import get_model_manager
from .utils import load_image_from_source, upload_mask_to_s3
from .sam2_segmentation import sam2_segmenter
from .mediapipe_operations import mediapipe_processor

class ObjectDetector:
    """Handles Florence-2 object detection with integrated processing."""
    
    def __init__(self):
        self.model_manager = get_model_manager()
    
    def detect_objects(self, image_b64=None, image_url=None, task="<OD>", 
                      confidence_threshold=0.3, text_input=None):
        """
        Detect objects in an image using Florence-2-large model.
        
        Args:
            image_b64: Base64 encoded image
            image_url: URL to image
            task: Task prompt (default: "<OD>" for object detection)
            confidence_threshold: Minimum confidence for detections
            text_input: Text input for referring expression segmentation
            
        Returns:
            dict: Parsed detection results with objects and their bounding boxes
        """
        # Get models (will load if needed)
        model = self.model_manager.get_model('florence2')
        processor = self.model_manager.get_model('florence2_processor')
        device = self.model_manager.get_model('florence2_device')
        dtype = self.model_manager.get_model('florence2_dtype')
        
        # Load image from either source
        image = load_image_from_source(image_b64, image_url)
        
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
                        executor.submit(self._process_detected_object, bbox, label, image_array, allowed_labels, person_labels, image): (bbox, label)
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
    
    def _process_detected_object(self, bbox, label, image_array, allowed_labels, person_labels, original_image):
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
        
        # Filter out objects smaller than 300px in width or height
        if bbox_width < 300 or bbox_height < 300:
            print(f"⚠️  Filtering out {label}: size {bbox_width}x{bbox_height}px (< 300px)")
            return None  # Skip this object
        
        obj = {
            'label': label,
            'bbox': bbox,  # [x1, y1, x2, y2]
            'confidence': None,  # Florence-2 doesn't return confidence scores
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
            
            # Process mask and upload to S3
            crop_box = (x1, y1, x2, y2)
            segment_mask_url = upload_mask_to_s3(mask_array, original_image.size, crop_box=crop_box)
            
            sam2_end_time = time.time()
            sam2_duration = sam2_end_time - sam2_start_time
            print(f"⏱️  SAM2 segmentation took {sam2_duration:.3f} seconds for {label}")
            obj['segment_mask'] = segment_mask_url
            
            # Check if detected object is a person and run selfie segmentation + face landmarks
            if any(person_word in label.lower() for person_word in person_labels):
                # Run selfie segmentation on the cropped person
                selfie_start_time = time.time()
                selfie_result = mediapipe_processor.segment_selfie_multiclass(image_pil=cropped_object)
                selfie_end_time = time.time()
                selfie_duration = selfie_end_time - selfie_start_time
                print(f"⏱️  MediaPipe selfie segmentation took {selfie_duration:.3f} seconds for {label}")
                
                # Process only hair, face-skin, and items classes, intersected with SAM2 mask
                if selfie_result and 'mask' in selfie_result:
                    # Initialize processed_classes as empty
                    processed_classes = {}
                    
                    # Get the refined crop based on hair and face_skin regions
                    refined_crop_info = self._get_refined_crop_from_masks(
                        selfie_result, cropped_object
                    )
                    
                    refined_crop_start_time = time.time()
                    if refined_crop_info and refined_crop_info['image']:
                        refined_cropped_image = refined_crop_info['image']
                        print(f"🔄 Reprocessing with refined crop: {refined_cropped_image.size}")
                        
                        # Run MediaPipe segmentation again on the refined crop
                        refined_selfie_result = mediapipe_processor.segment_selfie_multiclass(
                            image_pil=refined_cropped_image
                        )
                        
                        if refined_selfie_result and 'mask' in refined_selfie_result:
                            # Reprocess the masks with the refined segmentation
                            # Still need SAM2 intersection but with adjusted coordinates for refined crop
                            processed_classes = self._process_refined_multiclass_masks(
                                refined_selfie_result, refined_cropped_image, mask_array, x1, y1, x2, y2, original_image
                            )
                            
                            if processed_classes:
                                refined_crop_end_time = time.time()
                                refined_crop_duration = refined_crop_end_time - refined_crop_start_time
                                print(f"🚀 Refined crop reprocessing took {refined_crop_duration:.3f} seconds")
                                print(f"✅ Refined masks created: {list(processed_classes.keys())}")
                    
                    if processed_classes:
                        obj['multiclasses'] = processed_classes
                    
                    # Run face landmark detection - use refined crop if available, otherwise use original crop
                    landmarks_image = cropped_object  # Default to original crop
                    if refined_crop_info and refined_crop_info['image']:
                        landmarks_image = refined_crop_info['image']
                        print(f"🎯 Using refined crop for face landmarks: {landmarks_image.size}")
                    
                    landmarks_start_time = time.time()
                    landmarks_result = mediapipe_processor.detect_face_landmarks(image_pil=landmarks_image)
                    landmarks_end_time = time.time()
                    landmarks_duration = landmarks_end_time - landmarks_start_time
                    print(f"⏱️  MediaPipe face landmarks took {landmarks_duration:.3f} seconds for {label}")
                    obj['face_landmarks'] = landmarks_result
                else:
                    # Run face landmark detection on the cropped person (fallback when no selfie segmentation)
                    landmarks_start_time = time.time()
                    landmarks_result = mediapipe_processor.detect_face_landmarks(image_pil=cropped_object)
                    landmarks_end_time = time.time()
                    landmarks_duration = landmarks_end_time - landmarks_start_time
                    print(f"⏱️  MediaPipe face landmarks took {landmarks_duration:.3f} seconds for {label}")
                    obj['face_landmarks'] = landmarks_result
                
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
            
            print(f"📐 Refined crop: original {cropped_image.size} -> refined {refined_cropped.size}")
            print(f"   Y-range: {refined_top}-{refined_bottom} (hair bottom: {hair_pixels[0].max() if len(hair_pixels[0]) > 0 else 'N/A'}, face bottom: {face_skin_pixels[0].max() if len(face_skin_pixels[0]) > 0 else 'N/A'})")
            print(f"   Final bottom Y: {bottom_y} + padding: {padding} = {refined_bottom}")
            
            return {
                'image': refined_cropped
            }
            
        except Exception as e:
            print(f"⚠️  Failed to create refined crop: {e}")
            return None
    
    def _process_refined_multiclass_masks(self, refined_selfie_result, refined_cropped_image, mask_array, x1, y1, x2, y2, original_image):
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
            
        Returns:
            dict: Dictionary of class names to S3 URLs, or None if no valid masks
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
            
            print(f"🔍 Refined dimension check - Selfie mask: {refined_selfie_mask.shape}, SAM2 refined mask: {sam2_mask_refined.shape}")
            print(f"   SAM2 region from top: 0-{sam2_refined_end}")
            
            # Resize SAM2 mask to match refined selfie mask dimensions if needed
            if sam2_mask_refined.shape != refined_selfie_mask.shape:
                from PIL import Image
                print(f"🔄 Resizing SAM2 refined mask from {sam2_mask_refined.shape} to {refined_selfie_mask.shape}")
                sam2_mask_pil = Image.fromarray((sam2_mask_refined > 0).astype(np.uint8) * 255)
                sam2_mask_pil = sam2_mask_pil.resize((refined_selfie_mask.shape[1], refined_selfie_mask.shape[0]), Image.Resampling.NEAREST)
                sam2_mask_refined_resized = np.array(sam2_mask_pil) > 0
            else:
                print(f"✅ Refined dimensions match perfectly, no resizing needed")
                sam2_mask_refined_resized = sam2_mask_refined > 0
            
            # Process specific classes with SAM2 intersection
            masks_to_upload = []
            class_names = []
            
            # Hair class (1)
            hair_mask = (refined_selfie_mask == 1) & sam2_mask_refined_resized
            if np.any(hair_mask):
                masks_to_upload.append(hair_mask.astype(np.uint8) * 255)
                class_names.append('hair')
            
            # Face-skin class (3)  
            face_skin_mask = (refined_selfie_mask == 3) & sam2_mask_refined_resized
            if np.any(face_skin_mask):
                masks_to_upload.append(face_skin_mask.astype(np.uint8) * 255)
                class_names.append('face_skin')
            
            # Others class (5) - but only above the bottommost face_skin pixel
            others_mask_full = (refined_selfie_mask == 5) & sam2_mask_refined_resized
            others_mask = others_mask_full.copy()
            
            # Find the bottommost face_skin pixel to limit others mask
            if np.any(face_skin_mask):
                face_skin_y_coords = np.where(face_skin_mask)[0]
                bottommost_face_skin_y = np.max(face_skin_y_coords)
                print(f"🎯 Limiting others mask to above Y={bottommost_face_skin_y} (bottommost face_skin)")
                
                # Zero out all others pixels below the bottommost face_skin
                others_mask[bottommost_face_skin_y+1:, :] = False
            
            if np.any(others_mask):
                masks_to_upload.append(others_mask.astype(np.uint8) * 255)
                class_names.append('others')
            
            # Create head_mask as union of intersected hair, face_skin, and limited others
            head_mask_components = []
            if np.any(hair_mask):
                head_mask_components.append(hair_mask)
            if np.any(face_skin_mask):
                head_mask_components.append(face_skin_mask)
            if np.any(others_mask):
                head_mask_components.append(others_mask)
            
            if head_mask_components:
                # Union all components to create head_mask
                head_mask = np.logical_or.reduce(head_mask_components)
                if np.any(head_mask):
                    masks_to_upload.append(head_mask.astype(np.uint8) * 255)
                    class_names.append('head_mask')
                    print(f"✅ Created head_mask from {len(head_mask_components)} components")
            
            # Batch upload all refined masks to S3 in parallel
            if masks_to_upload:
                mask_size = (refined_selfie_mask.shape[1], refined_selfie_mask.shape[0])  # (width, height)
                
                # Use ThreadPoolExecutor for parallel S3 uploads
                with ThreadPoolExecutor(max_workers=3) as upload_executor:
                    upload_futures = {
                        upload_executor.submit(upload_mask_to_s3, mask, mask_size, s3_path="body_parts_refined"): class_name
                        for mask, class_name in zip(masks_to_upload, class_names)
                    }
                    
                    # Collect results as they complete
                    processed_classes = {}
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

# Create global instance without initializing models
object_detector = ObjectDetector()
