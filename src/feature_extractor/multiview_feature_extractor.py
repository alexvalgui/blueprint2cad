#!/usr/bin/env python3
"""
Multi-View Feature Extractor
Iterates over object IDs, processes 3 views per object, and extracts features.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from PIL import Image

# Import the extractors from the previous module
# (Assuming they're saved as feature_extractors.py)
from feature_extractors import (
    BaseFeatureExtractor,
    FeatureExtractorFactory,
    CLIPFeatureExtractor,
    DinoV2FeatureExtractor,
    DinoV1FeatureExtractor,
    DinoV3FeatureExtractor,
    RadioFeatureExtractor,
    SigLIPFeatureExtractor,
    SAMFeatureExtractor,
    DepthAnythingFeatureExtractor,
)


@dataclass
class ObjectView:
    """Represents a single view of an object."""
    view_id: int
    image_path: Path
    features: Optional[Dict[str, np.ndarray]] = None


@dataclass
class ObjectInstance:
    """Represents an object with multiple views."""
    object_id: str
    views: List[ObjectView]
    aggregated_features: Optional[Dict[str, np.ndarray]] = None


class MultiViewFeatureExtractor:
    """
    Extracts features for multi-view object datasets.
    
    Expected directory structure:
        dataset/
        ├── object_001/
        │   ├── view_1.jpg
        │   ├── view_2.jpg
        │   └── view_3.jpg
        ├── object_002/
        │   ├── view_1.jpg
        │   ├── view_2.jpg
        │   └── view_3.jpg
        └── ...
    """
    
    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        num_views: int = 3,
        aggregate: bool = True,
        aggregation_method: str = "mean",  # "mean", "max", "concat"
    ):
        self.extractor = extractor
        self.num_views = num_views
        self.aggregate = aggregate
        self.aggregation_method = aggregation_method
        
    def discover_objects(self, dataset_root: Union[str, Path]) -> List[ObjectInstance]:
        """
        Auto-discover objects and their views from directory structure.
        
        Looks for folders containing view_1, view_2, view_3 (or 0, 1, 2) images.
        """
        dataset_root = Path(dataset_root)
        objects = []
        
        # Sort object directories for deterministic ordering
        object_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
        
        for obj_dir in object_dirs:
            object_id = obj_dir.name
            views = []
            
            # Try multiple naming conventions
            naming_patterns = [
                [f"view_{i}.jpg" for i in range(1, self.num_views + 1)],
                [f"view_{i}.png" for i in range(1, self.num_views + 1)],
                [f"{i}.jpg" for i in range(self.num_views)],
                [f"{i}.png" for i in range(self.num_views)],
                [f"view_{i:02d}.jpg" for i in range(1, self.num_views + 1)],
            ]
            
            found_views = False
            for pattern in naming_patterns:
                view_paths = [obj_dir / fname for fname in pattern]
                if all(p.exists() for p in view_paths):
                    for i, vp in enumerate(view_paths, 1):
                        views.append(ObjectView(view_id=i, image_path=vp))
                    found_views = True
                    break
            
            if not found_views:
                # Fallback: grab first N image files
                image_files = sorted([
                    f for f in obj_dir.iterdir() 
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
                ])
                if len(image_files) >= self.num_views:
                    for i, img_path in enumerate(image_files[:self.num_views], 1):
                        views.append(ObjectView(view_id=i, image_path=img_path))
                    found_views = True
            
            if found_views:
                objects.append(ObjectInstance(object_id=object_id, views=views))
            else:
                print(f"Warning: Could not find {self.num_views} views for {object_id}")
                
        return objects
    
    def load_from_manifest(
        self, manifest_path: Union[str, Path]
    ) -> List[ObjectInstance]:
        """
        Load objects from a JSON manifest file.
        
        Manifest format:
        {
            "object_001": [
                "path/to/view1.jpg",
                "path/to/view2.jpg", 
                "path/to/view3.jpg"
            ],
            ...
        }
        """
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        objects = []
        for obj_id, view_paths in manifest.items():
            views = [
                ObjectView(view_id=i+1, image_path=Path(vp))
                for i, vp in enumerate(view_paths[:self.num_views])
            ]
            objects.append(ObjectInstance(object_id=obj_id, views=views))
        
        return objects
    
    def extract_object_features(
        self, obj: ObjectInstance, save_dir: Optional[Path] = None
    ) -> ObjectInstance:
        """Extract features for all views of a single object."""
        
        for view in tqdm(obj.views, desc=f"  Views ({obj.object_id})", leave=False):
            try:
                features = self.extractor.extract(str(view.image_path))
                view.features = features
            except Exception as e:
                print(f"Error processing {view.image_path}: {e}")
                raise
        
        # Aggregate across views if requested
        if self.aggregate:
            obj.aggregated_features = self._aggregate_views(obj.views)
        
        # Save individual view features
        if save_dir is not None:
            self._save_object_features(obj, save_dir)
        
        return obj
    
    def _aggregate_views(self, views: List[ObjectView]) -> Dict[str, np.ndarray]:
        """Aggregate features across multiple views."""
        aggregated = {}
        
        # Get all feature keys from first view
        feature_keys = views[0].features.keys()
        
        for key in feature_keys:
            if key == 'grid_size':
                aggregated[key] = views[0].features[key]
                continue
                
            # Stack features from all views
            stacked = np.stack([v.features[key] for v in views if v.features[key] is not None])
            
            if self.aggregation_method == "mean":
                aggregated[key] = np.mean(stacked, axis=0)
            elif self.aggregation_method == "max":
                aggregated[key] = np.max(stacked, axis=0)
            elif self.aggregation_method == "concat":
                # Flatten and concatenate
                flat = [s.reshape(s.shape[0], -1) if s.ndim > 1 else s for s in stacked]
                aggregated[key] = np.concatenate(flat, axis=-1)
            elif self.aggregation_method == "first":
                aggregated[key] = stacked[0]
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation_method}")
        
        return aggregated
    
    def _save_object_features(self, obj: ObjectInstance, save_dir: Path):
        """Save features to disk organized by object ID."""
        obj_save_dir = save_dir / obj.object_id
        obj_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual view features
        for view in obj.views:
            view_file = obj_save_dir / f"view_{view.view_id}_features.npz"
            save_dict = {
                k: v for k, v in view.features.items() 
                if v is not None and k != 'grid_size'
            }
            np.savez_compressed(view_file, **save_dict)
        
        # Save aggregated features
        if obj.aggregated_features is not None:
            agg_file = obj_save_dir / "aggregated_features.npz"
            save_dict = {
                k: v for k, v in obj.aggregated_features.items()
                if v is not None and k != 'grid_size'
            }
            np.savez_compressed(agg_file, **save_dict)
            
            # Also save a flat JSON manifest for easy loading
            manifest = {
                "object_id": obj.object_id,
                "num_views": len(obj.views),
                "aggregation": self.aggregation_method,
                "feature_shapes": {
                    k: list(v.shape) for k, v in obj.aggregated_features.items()
                    if isinstance(v, np.ndarray)
                }
            }
            with open(obj_save_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
    
    def process_dataset(
        self,
        dataset_root: Union[str, Path],
        output_dir: Union[str, Path],
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> List[ObjectInstance]:
        """
        Process entire dataset and save features.
        
        Args:
            dataset_root: Root directory containing object folders
            output_dir: Where to save extracted features
            manifest_path: Optional JSON manifest (overrides auto-discovery)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover objects
        if manifest_path:
            print(f"Loading manifest from {manifest_path}")
            objects = self.load_from_manifest(manifest_path)
        else:
            print(f"Auto-discovering objects in {dataset_root}")
            objects = self.discover_objects(dataset_root)
        
        print(f"Found {len(objects)} objects with {self.num_views} views each")
        
        # Process each object
        processed_objects = []
        for obj in tqdm(objects, desc="Processing objects"):
            processed = self.extract_object_features(obj, save_dir=output_dir)
            processed_objects.append(processed)
        
        # Save dataset-level manifest
        dataset_manifest = {
            "model": self.extractor.model_name,
            "num_objects": len(processed_objects),
            "num_views": self.num_views,
            "aggregation": self.aggregation_method,
            "objects": [obj.object_id for obj in processed_objects],
            "feature_keys": list(processed_objects[0].aggregated_features.keys()) 
                if processed_objects and processed_objects[0].aggregated_features 
                else []
        }
        
        with open(output_dir / "dataset_manifest.json", 'w') as f:
            json.dump(dataset_manifest, f, indent=2)
        
        print(f"\nDone! Features saved to {output_dir}")
        return processed_objects


def create_extractor(model_type: str, model_name: str, device: Optional[str] = None):
    """Helper to create extractor with error handling."""
    try:
        return FeatureExtractorFactory.create(model_type, model_name, device)
    except Exception as e:
        print(f"Factory failed ({e}), trying direct import...")
        # Fallback direct imports
        extractors = {
            'clip': CLIPFeatureExtractor,
            'dino-v1': DinoV1FeatureExtractor,
            'dino-v2': DinoV2FeatureExtractor,
            'dino-v3': DinoV3FeatureExtractor,
            'radio': RadioFeatureExtractor,
            'siglip': SigLIPFeatureExtractor,
            'sam': SAMFeatureExtractor,
            'depth-anything': DepthAnythingFeatureExtractor,
        }
        if model_type.lower() not in extractors:
            raise ValueError(f"Unknown model type: {model_type}")
        return extractors[model_type.lower()](model_name, device)


# ==================== USAGE EXAMPLES ====================

def example_single_object():
    """Example: Process one object manually."""
    extractor = create_extractor('dino-v2', 'facebook/dinov2-base')
    
    # Manually define an object with 3 views
    obj = ObjectInstance(
        object_id="chair_001",
        views=[
            ObjectView(1, Path("data/chair_001/view_1.jpg")),
            ObjectView(2, Path("data/chair_001/view_2.jpg")),
            ObjectView(3, Path("data/chair_001/view_3.jpg")),
        ]
    )
    
    pipeline = MultiViewFeatureExtractor(extractor, aggregate=True)
    obj = pipeline.extract_object_features(obj)
    
    print(f"CLS token shape: {obj.aggregated_features['cls_token'].shape}")
    print(f"Patch grid shape: {obj.aggregated_features['patch_grid'].shape}")


def example_batch_processing():
    """Example: Batch process entire dataset."""
    extractor = create_extractor('clip', 'openai/clip-vit-base-patch16')
    
    pipeline = MultiViewFeatureExtractor(
        extractor=extractor,
        num_views=3,
        aggregate=True,
        aggregation_method="mean"  # mean-pool across 3 views
    )
    
    objects = pipeline.process_dataset(
        dataset_root="path/to/your/dataset",
        output_dir="path/to/output/features",
    )
    
    # Access aggregated features
    for obj in objects:
        cls = obj.aggregated_features['cls_token']  # (768,)
        print(f"{obj.object_id}: {cls.shape}")


def example_multiple_models():
    """Example: Extract with multiple models and compare."""
    configs = [
        ('clip', 'openai/clip-vit-base-patch16'),
        ('dino-v2', 'facebook/dinov2-base'),
        ('sam', 'facebook/sam-vit-base'),
    ]
    
    for model_type, model_name in configs:
        print(f"\n{'='*50}")
        print(f"Extracting with {model_type}: {model_name}")
        print(f"{'='*50}")
        
        extractor = create_extractor(model_type, model_name)
        pipeline = MultiViewFeatureExtractor(
            extractor, 
            num_views=3, 
            aggregate=True,
            aggregation_method="mean"
        )
        
        pipeline.process_dataset(
            dataset_root="path/to/dataset",
            output_dir=f"features/{model_type}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-View Feature Extractor")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model-type", type=str, default="dino-v2", 
                       choices=['clip', 'dino-v1', 'dino-v2', 'dino-v3', 
                               'radio', 'siglip', 'sam', 'depth-anything'])
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-base",
                       help="HuggingFace model identifier")
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--aggregate", type=str, default="mean",
                       choices=["mean", "max", "concat", "first", "none"],
                       help="How to aggregate multi-view features")
    parser.add_argument("--manifest", type=str, default=None,
                       help="Optional JSON manifest file")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu), auto-detect if not specified")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = create_extractor(args.model_type, args.model_name, args.device)
    
    # Configure aggregation
    do_agg = args.aggregate != "none"
    agg_method = args.aggregate if do_agg else "mean"
    
    pipeline = MultiViewFeatureExtractor(
        extractor=extractor,
        num_views=args.num_views,
        aggregate=do_agg,
        aggregation_method=agg_method,
    )
    
    # Run
    pipeline.process_dataset(
        dataset_root=args.dataset,
        output_dir=args.output,
        manifest_path=args.manifest,
    )