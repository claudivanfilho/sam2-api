#!/usr/bin/env python3

"""
Simple test script to verify Florence-2 model can be loaded and used
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import io

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import io

# Workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename):
    """
    Workaround for unnecessary flash_attn requirement
    """
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

def test_florence2():
    print("🧪 Testing Florence-2-large model...")
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"📱 Using device: {device}")
    print(f"🔢 Using dtype: {torch_dtype}")
    
    try:
        # Load model and processor
        print("🔄 Loading Florence-2-large (no flash-attn) model...")
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            torch_dtype=torch_dtype, 
            trust_remote_code=True
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        
        # Test with a sample image
        print("🖼️  Testing with sample image...")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            print(f"📐 Image size: {image.size}")
        except Exception as e:
            print(f"⚠️  Failed to download test image: {e}")
            # Create a dummy image
            image = Image.new('RGB', (224, 224), color='red')
            print("🔴 Using dummy red image for testing")
        
        # Test object detection
        prompt = "<OD>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        
        print("🎯 Running object detection...")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                use_cache=True
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        print("✅ Object detection completed!")
        print(f"🔍 Results: {parsed_answer}")
        
        # Test other tasks
        for task in ["<CAPTION>", "<DETAILED_CAPTION>"]:
            print(f"🔄 Testing {task}...")
            inputs = processor(text=task, images=image, return_tensors="pt").to(device, torch_dtype)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    use_cache=True
                )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, 
                task=task, 
                image_size=(image.width, image.height)
            )
            
            print(f"✅ {task} result: {parsed_answer}")
        
        print("🎉 All tests passed! Florence-2 is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_florence2()
    exit(0 if success else 1)
