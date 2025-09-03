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
            
            # Get mask array from SAM2 segmenter
            mask_array = sam2_segmenter.segment_with_box(image_array, [x1, y1, x2, y2])
            
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
                # obj['multiclasses'] = selfie_result  # Commented out to reduce response size
                
                # Run face landmark detection on the cropped person
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

# Create global instance without initializing models
object_detector = ObjectDetector()
