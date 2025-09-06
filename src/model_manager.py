"""
Model initialization and configuration for SAM2 API.
Handles loading and configuration of all AI models.
"""

import torch
import os
import sys
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from torchvision import transforms
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Add paths for external models
sys.path.append('/root/EVF-SAM')
sys.path.append('/root/BiRefNet')

from hydra.core.global_hydra import GlobalHydra

class ModelManager:
    """Manages all AI models used in the API."""
    
    def __init__(self, preload_models=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.transforms = {}
        self._initialized = set()
        
        if preload_models:
            print("🚀 Model manager initialized - preloading all models...")
            self._preload_all_models()
        else:
            print("🚀 Model manager initialized (lazy loading enabled)")
    
    def _preload_all_models(self):
        """Preload all models at initialization."""
        models_to_load = ['sam2', 'evf_sam2', 'birefnet', 'florence2']
        
        for model_name in models_to_load:
            print(f"🔄 Preloading {model_name}...")
            try:
                self._ensure_model_loaded(model_name)
                print(f"✅ {model_name} loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        
        print("🎉 All models preloaded successfully!")
    
    def _ensure_model_loaded(self, model_name):
        """Ensure a specific model is loaded."""
        if model_name in self._initialized:
            return
            
        if model_name == 'sam2':
            self._initialize_sam2()
        elif model_name == 'evf_sam2':
            self._initialize_evf_sam2()
        elif model_name == 'birefnet':
            self._initialize_birefnet()
        elif model_name == 'florence2':
            self._initialize_florence2()
        
        self._initialized.add(model_name)
    
    def _initialize_sam2(self):
        """Initialize SAM2 model."""
        print("Loading SAM2 model...")
        
        # SAM2 configuration
        MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
        
        # Change to SAM2 directory for config loading
        original_cwd = os.getcwd()
        os.chdir("/root/segment-anything-2")
        
        # Build model using local checkpoint
        sam2_model = build_sam2(MODEL_CONFIG, CHECKPOINT, device=self.device)
        predictor = SAM2ImagePredictor(sam2_model)
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        self.models['sam2'] = sam2_model
        self.models['sam2_predictor'] = predictor
        print("✅ SAM2 model loaded successfully!")
    
    def _initialize_evf_sam2(self):
        """Initialize EVF-SAM2 model."""
        print("Loading EVF-SAM2 model...")
        
        # Clear Hydra instance to avoid conflicts
        GlobalHydra.instance().clear()
        
        # Temporarily change directory to avoid import conflicts
        original_cwd = os.getcwd()
        os.chdir("/root/EVF-SAM")
        
        try:
            # Initialize tokenizer
            evf_tokenizer = AutoTokenizer.from_pretrained(
                "YxZhang/evf-sam2-multitask",
                padding_side="right",
                use_fast=False,
            )
            
            # Initialize model
            from model.evf_sam2 import EvfSam2Model
            evf_model = EvfSam2Model.from_pretrained(
                "YxZhang/evf-sam2-multitask",
                low_cpu_mem_usage=True,
                torch_dtype=torch.half,
            ).cuda().eval()
            
            # Remove memory components for image-only inference
            del evf_model.visual_model.memory_encoder
            del evf_model.visual_model.memory_attention
            
            # Ensure all model parameters are in half precision to avoid dtype mismatches
            evf_model = evf_model.half()
            
        finally:
            # Always change back to original directory
            os.chdir(original_cwd)
        
        self.models['evf_sam2'] = evf_model
        self.models['evf_tokenizer'] = evf_tokenizer
        print("✅ EVF-SAM2 model loaded successfully!")
    
    def _initialize_birefnet(self):
        """Initialize BiRefNet model."""
        print("Loading BiRefNet model...")
        
        # Temporarily change directory to avoid import conflicts
        original_cwd = os.getcwd()
        os.chdir("/root/BiRefNet")
        
        try:
            from models.birefnet import BiRefNet
            birefnet = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
        finally:
            # Always change back to original directory
            os.chdir(original_cwd)
        
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        birefnet.to(self.device)
        birefnet.eval()
        if self.device == 'cuda':
            birefnet.half()
        
        # BiRefNet preprocessing transform
        birefnet_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.models['birefnet'] = birefnet
        self.transforms['birefnet'] = birefnet_transform
        print("✅ BiRefNet model loaded successfully!")
    
    def _initialize_florence2(self):
        """Initialize Florence-2 model."""
        print("Loading Florence-2-large (no flash-attn) model...")
        
        florence_device = "cuda" if torch.cuda.is_available() else "cpu"
        florence_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        florence_model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            torch_dtype=florence_torch_dtype, 
            trust_remote_code=True
        ).to(florence_device)
        
        florence_processor = AutoProcessor.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            trust_remote_code=True
        )
        
        self.models['florence2'] = florence_model
        self.models['florence2_processor'] = florence_processor
        self.models['florence2_device'] = florence_device
        self.models['florence2_dtype'] = florence_torch_dtype
        print("✅ Florence-2-large (no flash-attn) model loaded successfully!")
    
    def get_model(self, model_name):
        """Get a specific model, loading it if necessary."""
        # Map model names to their initialization names
        model_mapping = {
            'sam2_predictor': 'sam2',
            'evf_sam2': 'evf_sam2',
            'evf_tokenizer': 'evf_sam2',
            'birefnet': 'birefnet',
            'florence2': 'florence2',
            'florence2_processor': 'florence2',
            'florence2_device': 'florence2',
            'florence2_dtype': 'florence2'
        }
        
        init_name = model_mapping.get(model_name, model_name)
        self._ensure_model_loaded(init_name)
        return self.models.get(model_name)
    
    def get_transform(self, transform_name):
        """Get a specific transform, loading the related model if necessary."""
        if transform_name == 'birefnet':
            self._ensure_model_loaded('birefnet')
        return self.transforms.get(transform_name)
    
    def get_device(self):
        """Get the device being used."""
        return self.device
    
    def is_model_loaded(self, model_name):
        """Check if a specific model is loaded."""
        return model_name in self._initialized
    
    def get_loaded_models(self):
        """Get list of loaded models."""
        return list(self._initialized)

# Global model manager instance - will be initialized lazily
model_manager = None

def get_model_manager(preload_models=True):
    """Get the global model manager instance, creating it if necessary."""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(preload_models=preload_models)
    return model_manager
