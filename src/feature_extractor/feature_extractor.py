import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Union
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoImageProcessor,
    SamModel, SamProcessor,
    DepthAnythingForDepthEstimation,
)


class BaseFeatureExtractor(ABC):
    """Base class for vision model feature extraction."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
        
    @abstractmethod
    def _load_model(self):
        """Load model and processor. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _forward(self, inputs: Dict) -> Dict:
        """Run forward pass and return raw outputs. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _extract_features(self, outputs: Dict) -> Dict[str, np.ndarray]:
        """Parse model outputs into standardized features. To be implemented by subclasses."""
        pass
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        return Image.open(image_path).convert("RGB")
    
    def preprocess(self, image: Union[str, Image.Image]) -> Dict:
        """Preprocess image into model inputs."""
        if isinstance(image, str):
            image = self.load_image(image)
        inputs = self.processor(images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def extract(self, image: Union[str, Image.Image]) -> Dict[str, np.ndarray]:
        """
        Extract features from an image.
        
        Returns:
            dict with keys:
                - 'cls_token': np.ndarray of shape (D,) or None
                - 'patch_tokens': np.ndarray of shape (N, D)
                - 'patch_grid': np.ndarray of shape (H, W, D)
                - 'grid_size': tuple (H, W)
        """
        inputs = self.preprocess(image)
        self.model = self.model.to(self.device).eval()
        
        with torch.no_grad():
            outputs = self._forward(inputs)
        
        features = self._extract_features(outputs)
        
        # Add grid size convenience field
        if 'patch_grid' in features:
            features['grid_size'] = features['patch_grid'].shape[:2]
        
        return features


# ==================== CLIP ====================

class CLIPFeatureExtractor(BaseFeatureExtractor):
    """Extractor for OpenAI CLIP models."""
    
    def _load_model(self):
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
    
    def _forward(self, inputs: Dict):
        return self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
        )
    
    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        last_hidden = outputs.last_hidden_state  # [B, 1+N, D]
        
        # CLS token
        cls_token = last_hidden[:, 0, :]  # [B, D]
        cls_embedding = torch.nn.functional.normalize(cls_token, p=2, dim=-1)
        
        # Patch tokens
        patch_tokens = last_hidden[:, 1:, :]  # [B, N, D]
        B, N, D = patch_tokens.shape
        grid_size = int(N ** 0.5)
        
        patch_grid = patch_tokens.reshape(B, grid_size, grid_size, D)
        
        return {
            'cls_token': cls_embedding[0].cpu().numpy(),      # (D,)
            'patch_tokens': patch_tokens[0].cpu().numpy(),    # (N, D)
            'patch_grid': patch_grid[0].cpu().numpy(),        # (H, W, D)
        }


# ==================== DINO v1 ====================

class DinoV1FeatureExtractor(BaseFeatureExtractor):
    """Extractor for DINO v1 (facebook/dino-vitb16, etc.)."""
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
    
    def _forward(self, inputs: Dict):
        return self.model(**inputs)
    
    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        last_hidden = outputs.last_hidden_state  # [B, 1+N, D]
        
        cls_token = last_hidden[:, 0, :]       # [B, D]
        patch_tokens = last_hidden[:, 1:, :]   # [B, N, D]
        B, N, D = patch_tokens.shape
        grid_size = int(N ** 0.5)
        
        patch_grid = patch_tokens.reshape(B, grid_size, grid_size, D)
        
        return {
            'cls_token': cls_token[0].cpu().numpy(),
            'patch_tokens': patch_tokens[0].cpu().numpy(),
            'patch_grid': patch_grid[0].cpu().numpy(),
        }


# ==================== DINO v2 ====================

class DinoV2FeatureExtractor(BaseFeatureExtractor):
    """Extractor for DINOv2 models."""
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
    
    def _forward(self, inputs: Dict):
        return self.model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
        )
    
    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        last_hidden = outputs.last_hidden_state  # [B, 1+N, D]
        
        cls_token = last_hidden[:, 0, :]
        patch_tokens = last_hidden[:, 1:, :]
        B, N, D = patch_tokens.shape
        grid_size = int(N ** 0.5)
        
        patch_grid = patch_tokens.reshape(B, grid_size, grid_size, D)
        
        return {
            'cls_token': cls_token[0].cpu().numpy(),
            'patch_tokens': patch_tokens[0].cpu().numpy(),
            'patch_grid': patch_grid[0].cpu().numpy(),
        }


# ==================== DINO v3 ====================

class DinoV3FeatureExtractor(BaseFeatureExtractor):
    """Extractor for DINOv3 models with register tokens."""
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
    
    def _forward(self, inputs: Dict):
        return self.model(**inputs)
    
    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        last_hidden = outputs.last_hidden_state  # [B, 1 + num_registers + N, D]
        
        cls_token = last_hidden[:, 0, :]
        
        # Skip register tokens
        num_registers = getattr(self.model.config, 'num_register_tokens', 4)
        patch_tokens = last_hidden[:, 1 + num_registers:, :]
        
        B, N, D = patch_tokens.shape
        grid_size = int(N ** 0.5)
        patch_grid = patch_tokens.reshape(B, grid_size, grid_size, D)
        
        return {
            'cls_token': cls_token[0].cpu().numpy(),
            'patch_tokens': patch_tokens[0].cpu().numpy(),
            'patch_grid': patch_grid[0].cpu().numpy(),
        }


# ==================== RADIO ====================

class RadioFeatureExtractor(BaseFeatureExtractor):
    """Extractor for RADIO models (nvidia/C-RADIOv4-H, etc.)."""
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(self.device)
        self.processor = CLIPImageProcessor.from_pretrained(self.model_name)
    
    def _forward(self, inputs: Dict):
        # RADIO returns (summary, features) directly
        summary, features = self.model(inputs["pixel_values"])
        return {'summary': summary, 'features': features, 'inputs': inputs}
    
    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        cls_token = outputs['summary']           # [B, D]
        patch_tokens = outputs['features']       # [B, N, D] — no CLS mixed in
        inputs = outputs['inputs']
        
        B, N, D = patch_tokens.shape
        
        # Compute grid from input dims and patch size
        patch_size = getattr(self.model.config, "patch_size", 16)
        _, _, H, W = inputs["pixel_values"].shape
        grid_h = H // patch_size
        grid_w = W // patch_size
        
        patch_grid = patch_tokens.reshape(B, grid_h, grid_w, D)
        
        return {
            'cls_token': cls_token[0].cpu().numpy(),
            'patch_tokens': patch_tokens[0].cpu().numpy(),
            'patch_grid': patch_grid[0].cpu().numpy(),
        }


# ==================== SigLIP ====================

class SigLIPFeatureExtractor(BaseFeatureExtractor):
    """Extractor for SigLIP models."""
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
    
    def _forward(self, inputs: Dict):
        return self.model.vision_model(**inputs)
    
    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        # Global pooled representation
        cls_token = outputs.pooler_output  # [B, D]
        
        # SigLIP last_hidden_state has NO CLS token — all are spatial patches
        patch_tokens = outputs.last_hidden_state  # [B, N, D]
        B, N, D = patch_tokens.shape
        grid_size = int(N ** 0.5)
        
        patch_grid = patch_tokens.reshape(B, grid_size, grid_size, D)
        
        return {
            'cls_token': cls_token[0].cpu().numpy(),
            'patch_tokens': patch_tokens[0].cpu().numpy(),
            'patch_grid': patch_grid[0].cpu().numpy(),
        }


# ==================== SAM ====================

class SAMFeatureExtractor(BaseFeatureExtractor):
    """Extractor for SAM vision encoder."""
    
    def _load_model(self):
        self.model = SamModel.from_pretrained(self.model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(self.model_name)
    
    def _forward(self, inputs: Dict):
        return self.model.vision_encoder(
            inputs["pixel_values"],
            output_hidden_states=True,
        )
    
    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        # Hidden states[-1] is last vision layer output (before neck)
        # Shape: [B, 64, 64, 768] — spatial, NO cls token
        features = outputs.hidden_states[-1]
        B, H, W, D = features.shape
        N = H * W
        
        patch_tokens = features.reshape(B, N, D)   # [B, 4096, 768]
        patch_grid = features                      # [B, 64, 64, 768]
        
        # NOTE: SAM has no CLS token. If you need final embeddings (after neck):
        # image_embeds = outputs.last_hidden_state  # [B, 256, 64, 64]
        
        return {
            'cls_token': None,  # SAM has no CLS token
            'patch_tokens': patch_tokens[0].cpu().numpy(),
            'patch_grid': patch_grid[0].cpu().numpy(),
        }


# ==================== Depth Anything ====================

class DepthAnythingFeatureExtractor(BaseFeatureExtractor):
    """Extractor for Depth Anything V2 backbone features."""
    
    def _load_model(self):
        self.model = DepthAnythingForDepthEstimation.from_pretrained(
            self.model_name
        ).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
    
    def _forward(self, inputs: Dict):
        backbone_outputs = self.model.backbone(inputs["pixel_values"])
        return {
            'features': backbone_outputs.feature_maps[-1],
            'inputs': inputs,
        }
    
    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        features = outputs['features']  # [B, 1+N, D] or [B, D, H, W]
        inputs = outputs['inputs']
        
        if features.dim() == 3:
            # [B, 1+N, D] — CLS token at index 0
            cls_token = features[:, 0, :]
            patch_tokens = features[:, 1:, :]
            
            patch_size = getattr(self.model.config, 'patch_size', 14)
            _, _, h, w = inputs["pixel_values"].shape
            grid_h = h // patch_size
            grid_w = w // patch_size
            
        elif features.dim() == 4:
            # [B, D, H, W] — pure spatial feature map (no CLS)
            B, D, H, W = features.shape
            cls_token = None
            patch_tokens = features.permute(0, 2, 3, 1).reshape(B, H * W, D)
            grid_h, grid_w = H, W
        
        patch_grid = patch_tokens.reshape(-1, grid_h, grid_w, patch_tokens.shape[-1])
        
        return {
            'cls_token': cls_token[0].cpu().numpy() if cls_token is not None else None,
            'patch_tokens': patch_tokens[0].cpu().numpy(),
            'patch_grid': patch_grid[0].cpu().numpy(),
        }


# ==================== Factory ====================

class FeatureExtractorFactory:
    """Factory to instantiate the correct extractor by model name."""
    
    _registry = {
        'clip': CLIPFeatureExtractor,
        'dino-v1': DinoV1FeatureExtractor,
        'dino-v2': DinoV2FeatureExtractor,
        'dino-v3': DinoV3FeatureExtractor,
        'radio': RadioFeatureExtractor,
        'siglip': SigLIPFeatureExtractor,
        'sam': SAMFeatureExtractor,
        'depth-anything': DepthAnythingFeatureExtractor,
    }
    
    @classmethod
    def create(cls, model_type: str, model_name: str, device: Optional[str] = None):
        """
        Create an extractor instance.
        
        Args:
            model_type: One of the registered keys (clip, dino-v2, sam, etc.)
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None for auto
        """
        model_type = model_type.lower()
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type '{model_type}'. "
                           f"Available: {list(cls._registry.keys())}")
        return cls._registry[model_type](model_name, device)


# ==================== Usage Example ====================

if __name__ == "__main__":
    image_path = "/content/5da360c98b9af0ad709fe18606992229.jpg"
    
    # --- Method 1: Direct instantiation ---
    extractor = CLIPFeatureExtractor("openai/clip-vit-base-patch16")
    features = extractor.extract(image_path)
    print(f"CLS shape: {features['cls_token'].shape}")
    print(f"Patch grid shape: {features['patch_grid'].shape}")
    
    # --- Method 2: Factory ---
    configs = [
        ('clip', 'openai/clip-vit-base-patch16'),
        ('dino-v2', 'facebook/dinov2-base'),
        ('sam', 'facebook/sam-vit-base'),
    ]
    
    for model_type, model_name in configs:
        print(f"\n--- {model_type}: {model_name} ---")
        ext = FeatureExtractorFactory.create(model_type, model_name)
        feats = ext.extract(image_path)
        print(f"  CLS: {feats['cls_token'].shape if feats['cls_token'] is not None else None}")
        print(f"  Patches: {feats['patch_tokens'].shape}")
        print(f"  Grid: {feats['grid_size']}")