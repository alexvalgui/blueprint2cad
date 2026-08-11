import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Union
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoImageProcessor,
    SamModel, SamProcessor,
    DepthAnythingForDepthEstimation,
    CLIPImageProcessor,
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
        
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
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

class SAM3FeatureExtractor(BaseFeatureExtractor):
    """Extractor for SAM3 vision backbone"""
    
    def __init__(self, model_name: str="sam3", device: Optional[str] = None, 
                 text_prompt: Optional[str] = None):
        self.text_prompt = text_prompt
        super().__init__(model_name, device)
        
    def _load_model(self):
        self.model = build_sam3_image_model().to(self.device).eval()
        self.processor = Sam3Processor(self.model)
        
    def preprocess(self, image: Union[str, Image.Image]) -> Dict:
        if isinstance(image, str):
            image = self.load_image(image)
        return {"image": image}
    
    def _forward(self, inputs: Dict) -> Dict:
        state = self.processor.set_image(inputs["image"])
        prompt_output = None
        if self.text_prompt is not None:
            prompt_output = self.processor.set_text_prompt(
                state=state, prompt=self.text_prompt
            )
        
        return {"state": state, "prompt_output":prompt_output}

    def _extract_features(self, outputs: Dict) -> Dict[str, np.ndarray]:
        embed = self._get_image_embedding(outputs["state"])
        C, H, W = embed.shape
        
        patch_grid = embed.permute(1, 2, 0)
        patch_tokens = patch_grid.reshape(H * W, C)
        
        features = {
            "cls_token": None,
            "patch_tokens": patch_tokens.float().cpu().numpy(),
            "patch_grid": patch_grid.float().cpu().numpy(),
        }
        
        if outputs["prompt_output"] is not None: 
            pooled = self._masked_pool(outpus["prompt_output"], embed, H, W)
            if pooled is not None:
                features["object_feature"] = pooled 
        
        return features
    
    def _get_image_embedding(self, state):
        """Locate the backbone embedding in the inference state"""
        candidates = ("image_embed", "vision_feats", "backbone_out", "image_embeddings")
        container = state if isinstance(state, dict) else vars(state)
        for key in candidates:
            if key in container and container[key] is not None:
                t = container[key]
                if isinstance(t, dict):
                    t = next(iter(t.values()))
                if isinstance(t, (list, tuple)):
                    t = t[-1]
                return t.squeeze(0) # (C,H,W)
        
    
    def _masked_pool(self, output, embed, H, W):
        masks, scores = output["masks"], output["scores"]
        if len(scores) == 0:
            print(f"Warning: no mask for prompt")
            return None
        best = torch.as_tensor(
            masks[int(torch.as_tensor(scores).argmax())],
            device=embed.device, dtype=torch.float32,
        )        
        
        mask_lr = torch.nn.functional.interpolate(
            best[None, None], size=(H,W), mode="bilinear", align_corners=False)[0,0]
        
        w = mask_lr.reshape(-1,1)
        if w.sum() < 1e-6:
            return None
        flat = embed.permute(1, 2, 0).reshape(-1, embed.shape[0])
        return ((flat * w).sum(0) / w.sum()).float().cpu().numpy()

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

# ==================== TIPSv2 ====================
class TipsV2FeatureExtractor(BaseFeatureExtractor):
    """Extractor for Google TIPSv2 models (e.g. google/tipsv2-b14).

    TIPSv2 doesn't ship a HF `Processor`/`ImageProcessor` — it just expects
    images resized to a fixed square resolution and scaled to [0, 1], with
    NO ImageNet mean/std normalization. Features are pulled via the model's
    own `encode_image` method rather than a generic forward pass, so
    `preprocess`, `_forward`, and `_extract_features` are all overridden
    relative to the CLIP-style flow.
    """

    def __init__(
        self,
        model_name: str = "google/tipsv2-b14",
        device: Optional[str] = None,
        image_size: int = 448,
        patch_size: int = 14,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        super().__init__(model_name, device)

    def _load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(self.device)

        # No HF processor for TIPSv2 — build the equivalent torchvision
        # transform ourselves. NOTE: ToTensor() only (no Normalize) —
        # TIPSv2 expects raw [0, 1] pixel values.
        self.processor = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def preprocess(self, image: Union[str, Image.Image]) -> Dict:
        """Preprocess image into model inputs (overridden: no HF processor)."""
        if isinstance(image, str):
            image = self.load_image(image)
        pixel_values = self.processor(image).unsqueeze(0)  # (1, 3, H, W)
        return {"pixel_values": pixel_values.to(self.device)}

    def _forward(self, inputs: Dict):
        # TIPSv2 exposes a dedicated encode_image method rather than a
        # generic __call__/forward taking **inputs.
        return self.model.encode_image(inputs["pixel_values"])

    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        # outputs.cls_token:    (1, 1, D)
        # outputs.patch_tokens: (1, N, D)   where N = (image_size / patch_size) ** 2
        cls_token = outputs.cls_token[:, 0, :]  # (1, D)
        cls_embedding = torch.nn.functional.normalize(cls_token, p=2, dim=-1)

        patch_tokens = outputs.patch_tokens  # (1, N, D)
        B, N, D = patch_tokens.shape
        grid_size = int(round(N ** 0.5))
        assert grid_size * grid_size == N, (
            f"Non-square patch grid: N={N} doesn't factor into a square"
        )

        patch_grid = patch_tokens.reshape(B, grid_size, grid_size, D)

        return {
            'cls_token': cls_embedding[0].cpu().numpy(),      # (D,)
            'patch_tokens': patch_tokens[0].cpu().numpy(),    # (N, D)
            'patch_grid': patch_grid[0].cpu().numpy(),        # (H, W, D)
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
        'sam3': SAM3FeatureExtractor,
        'depth-anything': DepthAnythingFeatureExtractor,
        'tips2': TipsV2FeatureExtractor,
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
