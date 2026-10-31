#!/usr/bin/env python3
"""Multi-View Feature Extractor: iterates over object folders, extracts features per view."""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

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
        save_patches: bool = False,
    ):
        self.extractor = extractor
        self.save_patches = save_patches
        self.num_views = num_views
        self.aggregate = aggregate
        self.aggregation_method = aggregation_method

    def discover_objects(self, dataset_root: Union[str, Path]) -> List[ObjectInstance]:
        dataset_root = Path(dataset_root)
        objects = []
        object_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

        # Preferred subfolder to source views from, in priority order.
        subfolder_priority = ["normal", "dim", ""]

        for obj_dir in object_dirs:
            image_files = []
            for subfolder in subfolder_priority:
                candidate = obj_dir / subfolder if subfolder else obj_dir
                if candidate.is_dir():
                    found = sorted([
                        f for f in candidate.iterdir()
                        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
                    ])
                    if len(found) >= self.num_views:
                        image_files = found
                        break  # use the first subfolder that has enough views

            if image_files:
                views = [ObjectView(view_id=i, image_path=p)
                        for i, p in enumerate(image_files[:self.num_views], 1)]
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
        paths = [str(v.image_path) for v in obj.views]
        batch_features = self.extractor.extract_batch(paths)  # list of dicts, one per image

        for view, feats in zip(obj.views, batch_features):
            view.features = feats

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
            # Keep only global (1-D) descriptors. patch_tokens/patch_grid are the same
            # data stored twice and dominate disk usage (~1 MB/view); the retrieval
            # protocol never reads them. Set SAVE_PATCHES=1 to keep them.
            save_dict = {k: v.astype(np.float16) for k, v in view.features.items()
                         if v is not None and k != 'grid_size'
                         and (self.save_patches or np.ndim(v) == 1)}
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
        batch_size: int = 32,
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

        # Filter out objects whose outputs already exist on disk.
        pending = []
        for obj in objects:
            obj_out = output_dir / obj.object_id
            expected = [obj_out / f"view_{v.view_id}_features.npz" for v in obj.views]
            if self.aggregate:
                expected.append(obj_out / "aggregated_features.npz")
            if all(q.exists() for q in expected):
                continue
            pending.append(obj)

        skipped = len(objects) - len(pending)
        print(f"Skipped {skipped} already-processed objects")

        processed_ids = []
        objects_per_batch = max(1, batch_size // self.num_views)

        for i in tqdm(range(0, len(pending), objects_per_batch), desc="Batches"):
            chunk = pending[i:i + objects_per_batch]

            # Flatten (object, view) pairs into one list of paths for a single
            # batched forward pass across multiple objects at once.
            flat_views = [(obj, v) for obj in chunk for v in obj.views]
            all_paths = [str(v.image_path) for _, v in flat_views]

            try:
                all_feats = self.extractor.extract_batch(all_paths)
            except Exception as e:
                print(f"Error processing batch starting at object "
                      f"{chunk[0].object_id}: {e}")
                # Fall back to per-object processing for this chunk so one
                # bad image doesn't drop the whole batch.
                for obj in chunk:
                    try:
                        processed = self.extract_object_features(obj, save_dir=output_dir)
                        processed_ids.append(processed.object_id)
                    except Exception as inner_e:
                        print(f"Error processing {obj.object_id}, skipping: {inner_e}")
                    finally:
                        for v in obj.views:
                            v.features = None
                        obj.aggregated_features = None
                continue

            if len(all_feats) != len(flat_views):
                print(f"Warning: extract_batch returned {len(all_feats)} results "
                      f"for {len(flat_views)} inputs; skipping this batch.")
                continue

            # Distribute batched features back to their (object, view) pairs.
            for (obj, v), feats in zip(flat_views, all_feats):
                v.features = feats

            for obj in chunk:
                try:
                    if self.aggregate:
                        obj.aggregated_features = self._aggregate_views(obj.views)
                    self._save_object_features(obj, output_dir)
                    processed_ids.append(obj.object_id)
                except Exception as e:
                    print(f"Error saving {obj.object_id}, skipping: {e}")
                finally:
                    # Free feature arrays now that they're saved to disk
                    for v in obj.views:
                        v.features = None
                    obj.aggregated_features = None

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
                                'radio', 'siglip2', 'i-jepa', 'sam', 'sam3',
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
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images per GPU batch (not objects)")
    parser.add_argument("--save-patches", action="store_true",
                        help="Also store patch_tokens/patch_grid per view (large)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Input resolution for i-jepa / tips-v2 (default: 448). "
                             "Use 224 for the original I-JEPA ablation.")
    args = parser.parse_args()

    extra = {}
    if args.image_size is not None:
        if args.model_type not in ("i-jepa", "tips-v2"):
            parser.error("--image-size is only supported for i-jepa and tips-v2")
        extra["image_size"] = args.image_size
    extractor = FeatureExtractorFactory.create(args.model_type, args.model_name,
                                               args.device, **extra)

    do_agg = args.aggregate != "none"
    pipeline = MultiViewFeatureExtractor(
        extractor=extractor,
        num_views=args.num_views,
        aggregate=do_agg,
        aggregation_method=args.aggregate if do_agg else "mean",
        save_patches=args.save_patches,
    )
    pipeline.process_dataset(
        dataset_root=args.dataset,
        output_dir=args.output,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
    )