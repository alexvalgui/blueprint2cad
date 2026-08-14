#!/usr/bin/env python3
"""Multi-View Feature Extractor: iterates over object folders, extracts features per view."""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor

from feature_extractor import BaseFeatureExtractor, FeatureExtractorFactory


@dataclass
class ObjectView:
    view_id: int
    image_path: Path
    features: Optional[Dict[str, np.ndarray]] = None


@dataclass
class ObjectInstance:
    object_id: str
    views: List[ObjectView]
    aggregated_features: Optional[Dict[str, np.ndarray]] = None


class MultiViewFeatureExtractor:
    """Extracts features for datasets structured as dataset/<object_id>/<view images>."""

    def __init__(
        self,
        extractor: BaseFeatureExtractor,
        num_views: int = 3,
        aggregate: bool = True,
        aggregation_method: str = "mean",  # "mean", "max", "concat", "first"
    ):
        self.extractor = extractor
        self.num_views = num_views
        self.aggregate = aggregate
        self.aggregation_method = aggregation_method

    def discover_objects(self, dataset_root: Union[str, Path]) -> List[ObjectInstance]:
        dataset_root = Path(dataset_root)
        objects = []
        object_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

        for obj_dir in object_dirs:
            views = []
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
                    views = [ObjectView(view_id=i, image_path=vp)
                             for i, vp in enumerate(view_paths, 1)]
                    found_views = True
                    break

            if not found_views:
                image_files = sorted([
                    f for f in obj_dir.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
                ])
                if len(image_files) >= self.num_views:
                    views = [ObjectView(view_id=i, image_path=p)
                             for i, p in enumerate(image_files[:self.num_views], 1)]
                    found_views = True

            if found_views:
                objects.append(ObjectInstance(object_id=obj_dir.name, views=views))
            else:
                print(f"Warning: Could not find {self.num_views} views for {obj_dir.name}")

        return objects

    def load_from_manifest(self, manifest_path: Union[str, Path]) -> List[ObjectInstance]:
        """Manifest format: {"object_001": ["path/v1.jpg", "path/v2.jpg", ...], ...}"""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        return [
            ObjectInstance(
                object_id=obj_id,
                views=[ObjectView(view_id=i + 1, image_path=Path(vp))
                       for i, vp in enumerate(view_paths[:self.num_views])],
            )
            for obj_id, view_paths in manifest.items()
        ]

    def extract_object_features(
        self, obj: ObjectInstance, save_dir: Optional[Path] = None
    ) -> ObjectInstance:
        for view in tqdm(obj.views, desc=f"  Views ({obj.object_id})", leave=False):
            try:
                view.features = self.extractor.extract(str(view.image_path))
            except Exception as e:
                print(f"Error processing {view.image_path}: {e}")
                raise

        if self.aggregate:
            obj.aggregated_features = self._aggregate_views(obj.views)
        if save_dir is not None:
            self._save_object_features(obj, save_dir)
        return obj

    def _aggregate_views(self, views: List[ObjectView]) -> Dict[str, np.ndarray]:
        aggregated = {}
        for key in views[0].features.keys():
            if key == 'grid_size':
                aggregated[key] = views[0].features[key]
                continue

            values = [v.features[key] for v in views if v.features[key] is not None]
            if not values:
                aggregated[key] = None
                continue
            stacked = np.stack(values)

            if self.aggregation_method == "mean":
                aggregated[key] = np.mean(stacked, axis=0)
            elif self.aggregation_method == "max":
                aggregated[key] = np.max(stacked, axis=0)
            elif self.aggregation_method == "concat":
                flat = [s.reshape(s.shape[0], -1) if s.ndim > 1 else s for s in stacked]
                aggregated[key] = np.concatenate(flat, axis=-1)
            elif self.aggregation_method == "first":
                aggregated[key] = stacked[0]
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation_method}")
        return aggregated

    def _save_object_features(self, obj: ObjectInstance, save_dir: Path):
        obj_save_dir = save_dir / obj.object_id
        obj_save_dir.mkdir(parents=True, exist_ok=True)

        for view in obj.views:
            save_dict = {k: v.astype(np.float16) for k, v in view.features.items()
             if v is not None and k != 'grid_size'}
            np.savez(obj_save_dir / f"view_{view.view_id}_features.npz", **save_dict)

        if obj.aggregated_features is not None:
            save_dict = {k: v for k, v in obj.aggregated_features.items()
                         if v is not None and k != 'grid_size'}
            np.savez_compressed(obj_save_dir / "aggregated_features.npz", **save_dict)

            manifest = {
                "object_id": obj.object_id,
                "num_views": len(obj.views),
                "aggregation": self.aggregation_method,
                "feature_shapes": {k: list(v.shape) for k, v in obj.aggregated_features.items()
                                   if isinstance(v, np.ndarray)},
            }
            with open(obj_save_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)

    def process_dataset(
        self,
        dataset_root: Union[str, Path],
        output_dir: Union[str, Path],
        manifest_path: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if manifest_path:
            print(f"Loading manifest from {manifest_path}")
            objects = self.load_from_manifest(manifest_path)
        else:
            print(f"Auto-discovering objects in {dataset_root}")
            objects = self.discover_objects(dataset_root)

        print(f"Found {len(objects)} objects with {self.num_views} views each")

        processed_ids = []
        skipped = 0
        for obj in tqdm(objects, desc="Processing objects"):
            obj_out = output_dir / obj.object_id
            expected = [obj_out / f"view_{v.view_id}_features.npz" for v in obj.views]
            if self.aggregate:
                expected.append(obj_out / "aggregated_features.npz")
            if all(q.exists() for q in expected):
                skipped += 1
                continue
            processed = self.extract_object_features(obj, save_dir=output_dir)
            # Free feature arrays now that they're saved to disk
            for v in processed.views:
                v.features = None
            processed.aggregated_features = None
            processed_ids.append(processed.object_id)
        print(f"Skipped {skipped} already-processed objects")

        dataset_manifest = {
            "model": self.extractor.model_name,
            "num_objects": len(objects),
            "num_views": self.num_views,
            "aggregation": self.aggregation_method,
            "objects": [obj.object_id for obj in objects],
        }
        with open(output_dir / "dataset_manifest.json", 'w') as f:
            json.dump(dataset_manifest, f, indent=2)

        print(f"\nDone! Features saved to {output_dir}")
        return processed_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-View Feature Extractor")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model-type", type=str, default="dino-v2",
                        choices=['clip', 'dino-v1', 'dino-v2', 'dino-v3',
                                'radio', 'siglip', 'siglip2', 'i-jepa', 'sam', 'sam3',
                                'depth-anything', 'tips-v2'])
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

    extractor = FeatureExtractorFactory.create(args.model_type, args.model_name, args.device)

    do_agg = args.aggregate != "none"
    pipeline = MultiViewFeatureExtractor(
        extractor=extractor,
        num_views=args.num_views,
        aggregate=do_agg,
        aggregation_method=args.aggregate if do_agg else "mean",
    )
    pipeline.process_dataset(
        dataset_root=args.dataset,
        output_dir=args.output,
        manifest_path=args.manifest,
    )