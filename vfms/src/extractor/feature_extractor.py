import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Union
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
from torchvision import transforms
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoImageProcessor, AutoProcessor,
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

    def load_images(self, image_paths):
        """Load a list of image paths (or pass through existing PIL Images)."""
        return [self.load_image(p) if isinstance(p, str) else p for p in image_paths]

    def preprocess(self, image: Union[str, Image.Image]) -> Dict:
        """Preprocess image into model inputs."""
        if isinstance(image, str):
            image = self.load_image(image)
        inputs = self.processor(images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def preprocess_batch(self, images) -> Dict:
        """Preprocess a list of images into one batched inputs dict.

        Default implementation assumes self.processor accepts a list of
        PIL images (true for CLIPProcessor/AutoImageProcessor/AutoProcessor).
        Subclasses with a custom, non-list-aware processor (e.g. TIPSv2's
        torchvision transform) should override this.
        """
        pil_images = self.load_images(images)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _slice_output(self, outputs, i: int):
        """Recursively slice batch dim 0 -> a single-sample (batch=1) view
        of a model output, so existing single-image `_extract_features`
        implementations can be reused unchanged per-item."""
        if isinstance(outputs, torch.Tensor):
            return outputs[i:i + 1]
        if outputs is None:
            return None
        if isinstance(outputs, (list, tuple)):
            sliced = [self._slice_output(o, i) for o in outputs]
            return type(outputs)(sliced) if isinstance(outputs, tuple) else sliced
        if isinstance(outputs, dict) or (hasattr(outputs, "keys") and hasattr(outputs, "items")):
            sliced = {k: self._slice_output(v, i) for k, v in outputs.items()}
            try:
                return type(outputs)(**sliced)
            except TypeError:
                return sliced
        if hasattr(outputs, "__dict__"):
            sliced_obj = outputs.__class__.__new__(outputs.__class__)
            for k, v in vars(outputs).items():
                setattr(sliced_obj, k, self._slice_output(v, i))
            return sliced_obj
        return outputs

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
        if 'patch_grid' in features and features['patch_grid'] is not None:
            features['grid_size'] = features['patch_grid'].shape[:2]

        return features

    def extract_batch(self, image_paths) -> "list[Dict[str, np.ndarray]]":
        """Extract features for a batch of images with a single forward pass."""
        if not image_paths:
            return []

        inputs = self.preprocess_batch(image_paths)
        self.model = self.model.to(self.device).eval()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outputs = self._forward(inputs)

        results = []
        for i in range(len(image_paths)):
            sliced = self._slice_output(outputs, i)
            features = self._extract_features(sliced)
            if features.get('patch_grid') is not None:
                features['grid_size'] = features['patch_grid'].shape[:2]
            results.append(features)
        return results


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
        self.processor = CLIPImageProcessor.from_pretrained(
            self.model_name,
            size={"height": 512, "width": 512},   # multiple of patch step
            do_resize=True,
            do_center_crop=False,
        )

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

# ==================== SigLIP 2 ====================

class SigLIP2FeatureExtractor(BaseFeatureExtractor):
    """Extractor for SigLIP 2 models (e.g. google/siglip2-base-patch16-224).

    Same output shape as SigLIP 1: `vision_model` returns a pure spatial
    `last_hidden_state` (no CLS token) plus a `pooler_output` global
    embedding. NOTE: this assumes a fixed-resolution SigLIP 2 checkpoint —
    the variable-resolution "NaFlex" checkpoints return ragged patch grids
    (with a separate `spatial_shapes`/`pixel_attention_mask`) and are NOT
    handled by this class.
    """

    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _forward(self, inputs: Dict):
        return self.model.vision_model(**inputs)

    def _extract_features(self, outputs) -> Dict[str, np.ndarray]:
        # Global pooled representation
        cls_token = outputs.pooler_output  # [B, D]

        # SigLIP2 last_hidden_state has NO CLS token — all are spatial patches
        patch_tokens = outputs.last_hidden_state  # [B, N, D]
        B, N, D = patch_tokens.shape
        grid_size = int(round(N ** 0.5))
        assert grid_size * grid_size == N, (
            f"Non-square patch grid: N={N} doesn't factor into a square "
            f"(are you using a NaFlex/variable-resolution checkpoint?)"
        )

        patch_grid = patch_tokens.reshape(B, grid_size, grid_size, D)

        return {
            'cls_token': cls_token[0].cpu().numpy(),
            'patch_tokens': patch_tokens[0].cpu().numpy(),
            'patch_grid': patch_grid[0].cpu().numpy(),
        }


# ==================== I-JEPA ====================

class IJEPAFeatureExtractor(BaseFeatureExtractor):
    """I-JEPA: no CLS token. Global descriptor = foreground-weighted mean of patch tokens
    (background patches would otherwise dominate the plain mean).

    The model input is exactly what the other encoders see (RGB via .convert("RGB"),
    so transparent render backgrounds are black). Only the pooling mask differs: it is
    taken from the ORIGINAL file before conversion -- the alpha channel for RGBA renders,
    non-white pixels for opaque drawings. Runs at `image_size`; position embeddings are
    interpolated if that differs from the checkpoint's training resolution."""

    WHITE_THRESH = 242      # 0.95 * 255: darker pixels in an opaque drawing count as ink
    MASK_THRESH = 0.05      # a resized-mask pixel counts if >5% covered (keeps 1-px lines)

    def __init__(self, model_name, device=None, image_size=448):
        self.image_size = image_size
        super().__init__(model_name, device)

    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, size={"height": self.image_size, "width": self.image_size},
            do_center_crop=False)
        self.patch = self.model.config.patch_size

    def _fg_mask(self, im: Image.Image) -> torch.Tensor:
        """Foreground mask [1, S, S] in {0,1}, resized like the model input (full image,
        no crop)."""
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        if has_alpha:
            fg = np.asarray(im.convert("RGBA"))[..., 3] > 0                    # opaque = part
        else:
            fg = np.asarray(im.convert("L")) < self.WHITE_THRESH               # dark = ink
        m = Image.fromarray(fg.astype(np.uint8) * 255).resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        m = np.asarray(m, dtype=np.float32) / 255.0
        return torch.from_numpy((m > self.MASK_THRESH).astype(np.float32))[None]

    def preprocess_batch(self, images) -> Dict:
        raw = [Image.open(p) if isinstance(p, str) else p for p in images]
        masks = torch.stack([self._fg_mask(im) for im in raw])               # [B,1,S,S]
        rgb = [im.convert("RGB") for im in raw]                              # same input as other encoders
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["fg_mask"] = masks.to(self.device)
        return inputs

    def preprocess(self, image) -> Dict:
        return self.preprocess_batch([image])

    def _forward(self, inputs):
        out = self.model(pixel_values=inputs["pixel_values"], interpolate_pos_encoding=True)
        return {"tokens": out.last_hidden_state, "fg_mask": inputs["fg_mask"]}

    def _extract_features(self, outputs):
        tokens, fg = outputs["tokens"], outputs["fg_mask"]                  # [1,N,D], [1,1,S,S]
        B, N, D = tokens.shape
        g = int(round(N ** 0.5)); assert g * g == N
        w = torch.nn.functional.adaptive_avg_pool2d(fg.float(), g).reshape(1, N, 1)  # fg fraction per patch
        if w.sum() < 1e-6:
            w = torch.ones_like(w)                                        # empty image: plain mean
        pooled = (tokens * w).sum(1) / w.sum(1)
        cls_embedding = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        # Binary variant: every patch touched by the part counts equally. In a line
        # drawing ink covers only a few % of a patch, so ink-fraction weights let
        # dense regions (hatching, hole clusters) dominate; binary weights give a
        # drawing the same footprint as the solid region of its render.
        wb = (w > 0.01).float()
        if wb.sum() < 1e-6:
            wb = torch.ones_like(wb)
        pooled_b = (tokens * wb).sum(1) / wb.sum(1)
        cls_binary = torch.nn.functional.normalize(pooled_b, p=2, dim=-1)
        plain = torch.nn.functional.normalize(tokens.mean(1), p=2, dim=-1)
        return {
            "cls_token": cls_embedding[0].float().cpu().numpy(),          # ink-fraction weighted
            "cls_token_fgbinary": cls_binary[0].float().cpu().numpy(),    # binary foreground mask
            # Plain mean over all patches. This equals the paper's original protocol
            # only when run with --image-size 224 on the 224-px checkpoint.
            "cls_token_plainmean": plain[0].float().cpu().numpy(),
            "patch_tokens": tokens[0].cpu().numpy(),
            "patch_grid": tokens[0].reshape(g, g, D).cpu().numpy(),
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
            pooled = self._masked_pool(outputs["prompt_output"], embed, H, W)
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

    def extract_batch(self, image_paths):
        """SAM3 drives a stateful per-image processor (set_image/set_text_prompt)
        and isn't batchable through the generic tensor-batching path — fall
        back to one-at-a-time (no speedup, but correct)."""
        return [self.extract(p) for p in image_paths]

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

    def preprocess_batch(self, images) -> Dict:
        """Batched version: self.processor is a torchvision transform that
        only accepts one PIL image at a time, so stack them manually."""
        pil_images = self.load_images(images)
        tensors = torch.stack([self.processor(img) for img in pil_images])
        return {"pixel_values": tensors.to(self.device)}

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
        'siglip2': SigLIP2FeatureExtractor,
        'i-jepa': IJEPAFeatureExtractor,
        'sam': SAMFeatureExtractor,
        'sam3': SAM3FeatureExtractor,
        'depth-anything': DepthAnythingFeatureExtractor,
        'tips-v2': TipsV2FeatureExtractor,
    }

    @classmethod
    def create(cls, model_type: str, model_name: str, device: Optional[str] = None, **kwargs):
        """
        Create an extractor instance.

        Args:
            model_type: One of the registered keys (clip, dino-v2, sam, etc.)
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None for auto
            **kwargs: extra constructor arguments, e.g. image_size for i-jepa / tips-v2
        """
        model_type = model_type.lower()
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type '{model_type}'. "
                           f"Available: {list(cls._registry.keys())}")
        return cls._registry[model_type](model_name, device, **kwargs)