"""
2D CAD drawing <-> 3D render multi-view retrieval.

Expected folder layout:

  renders/<sample_id>/000.png ... 005.png      (3D shaded renders, N views)
  png_out/<sample_id>/front.png, back.png,
          left.png, right.png, top.png,
          bottom.png                            (2D CAD line drawings, 6 canonical views)

Both sides are multi-view, so both get the same treatment: encode each view
independently with a shared (frozen) DINOv3 backbone + a domain-specific LoRA
adapter, then aggregate views into a single object embedding with attention
pooling, then align the two object embeddings with a symmetric InfoNCE loss.

Requires: torch, torchvision, transformers, peft, pillow
    pip install torch torchvision transformers peft pillow --break-system-packages

DINOv3 checkpoints are gated on the Hugging Face Hub: accept the license on
the model page (e.g. huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
then authenticate before running this script:
    huggingface-cli login
  or
    export HF_TOKEN=hf_...
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class Config:
    renders_root: str = "renders"       # 3D views: <root>/<sample_id>/000.png ...
    cad_root: str = "png_out"           # 2D views: <root>/<sample_id>/front.png ...
    cad_view_names: List[str] = field(
        default_factory=lambda: ["front", "back", "left", "right", "top", "bottom"]
    )
    # DINOv3 ViT checkpoints (gated on HF hub -- run `huggingface-cli login`,
    # or set HF_TOKEN, with an account that has accepted the license first):
    #   facebook/dinov3-vits16-pretrain-lvd1689m   (21M,  small)
    #   facebook/dinov3-vitb16-pretrain-lvd1689m   (86M,  base)   <- good default
    #   facebook/dinov3-vitl16-pretrain-lvd1689m   (300M, large)
    # Also works unchanged with a DINOv2 checkpoint (e.g. "facebook/dinov2-base")
    # -- build_model() detects the family from config.model_type and picks the
    # right LoRA target-module names for whichever one you point it at.
    backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    embed_dim: int = 256                # shared retrieval embedding dim
    lora_r: int = 12
    lora_alpha: int = 24
    lora_dropout: float = 0.05
    # last-N transformer blocks to attach LoRA to (deeper blocks only ->
    # keeps early general-purpose features intact)
    lora_target_last_n_blocks: int = 6
    image_size: int = 224
    batch_size: int = 16                # objects per batch (not views)
    lr_head: float = 1e-3
    lr_lora: float = 5e-5
    weight_decay: float = 0.05
    epochs: int = 30
    temperature: float = 0.07           # InfoNCE temperature (learnable, init here)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    seed: int = 0

    # gradient accumulation -> larger effective batch of negatives without
    # blowing up memory (6 views x 2 domains x B objects per step is heavy)
    grad_accum_steps: int = 8

    # MoCo-style memory queue: negatives persist across steps instead of
    # being limited to the current batch. Only detached embeddings are
    # stored, so this adds negligible memory.
    queue_size: int = 4096

    # every N epochs, mine the hardest false-positive pairs from the current
    # embedding space and oversample them for the following epochs
    hard_negative_every_n_epochs: int = 3
    hard_negative_pool_size: int = 256   # how many hard pairs to log/oversample
    hard_negative_oversample_epochs: int = 2  # how many epochs to keep oversampling


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class CadToRenderDataset(Dataset):
    """
    One item = one object (sample_id): all 2D CAD views + all 3D render views.
    Only keeps samples that have both a complete CAD view set and >=1 render.
    """

    def __init__(self, cfg: Config, sample_ids: List[str], train: bool = True):
        self.cfg = cfg
        self.sample_ids = sample_ids
        self.train = train

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.1, 0.1, 0.1)], p=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
            ])

    @staticmethod
    def discover_samples(cfg: Config) -> List[str]:
        """Intersect sample_ids present in both trees with complete view sets."""
        render_root = Path(cfg.renders_root)
        cad_root = Path(cfg.cad_root)
        render_ids = {p.name for p in render_root.iterdir() if p.is_dir()}
        cad_ids = {p.name for p in cad_root.iterdir() if p.is_dir()}
        common = render_ids & cad_ids

        valid = []
        for sid in sorted(common):
            render_views = sorted((render_root / sid).glob("*.png"))
            cad_views = [cad_root / sid / f"{v}.png" for v in cfg.cad_view_names]
            if len(render_views) >= 1 and all(p.exists() for p in cad_views):
                valid.append(sid)
        return valid

    def __len__(self):
        return len(self.sample_ids)

    def _load(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.tf(img)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        cfg = self.cfg

        render_dir = Path(cfg.renders_root) / sid
        render_paths = sorted(render_dir.glob("*.png"))
        render_views = torch.stack([self._load(p) for p in render_paths])  # [N3D, 3, H, W]

        cad_dir = Path(cfg.cad_root) / sid
        cad_paths = [cad_dir / f"{v}.png" for v in cfg.cad_view_names]
        cad_views = torch.stack([self._load(p) for p in cad_paths])        # [N2D, 3, H, W]

        return {"sample_id": sid, "render_views": render_views, "cad_views": cad_views}


def collate_variable_views(batch):
    """3D view count can vary per object; keep as a list, pad handled in forward."""
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "render_views": [b["render_views"] for b in batch],   # list of [Ni, 3, H, W]
        "cad_views": torch.stack([b["cad_views"] for b in batch]),  # fixed count -> stack
    }


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Learned attention pooling over a set of view embeddings -> one vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_views, dim]
        scores = self.proj(x) @ self.query          # [N_views]
        weights = F.softmax(scores, dim=0)           # [N_views]
        pooled = (weights.unsqueeze(-1) * x).sum(0)   # [dim]
        return pooled


class DomainBranch(nn.Module):
    """Shared frozen DINOv2 backbone + LoRA adapter + projection head + view pooling."""

    def __init__(self, backbone: nn.Module, hidden_dim: int, cfg: Config):
        super().__init__()
        self.backbone = backbone  # a peft-wrapped model with THIS branch's adapter active
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.embed_dim),
        )
        self.pool = AttentionPool(cfg.embed_dim)

    def encode_views(self, views: torch.Tensor, adapter_name: str) -> torch.Tensor:
        """views: [N_views, 3, H, W] -> [N_views, embed_dim] (per-view embeddings)."""
        self.backbone.set_adapter(adapter_name)
        out = self.backbone(pixel_values=views)
        cls = out.last_hidden_state[:, 0, :]   # CLS token, [N_views, hidden_dim]
        emb = self.head(cls)                   # [N_views, embed_dim]
        return F.normalize(emb, dim=-1)

    def encode_object(self, views: torch.Tensor, adapter_name: str) -> torch.Tensor:
        per_view = self.encode_views(views, adapter_name)
        pooled = self.pool(per_view)
        return F.normalize(pooled, dim=0)


def build_model(cfg: Config):
    base = AutoModel.from_pretrained(cfg.backbone_name)
    hidden_dim = base.config.hidden_size

    # DINOv3 (DINOv3ViTModel) and DINOv2 (Dinov2Model) expose the transformer
    # blocks under different attribute paths and different attention proj
    # names, so the LoRA target_modules strings differ:
    #   DINOv3ViTModel:  self.model.layer[i].attention.{q_proj,k_proj,v_proj,o_proj}
    #   Dinov2Model:     self.encoder.layer[i].attention.attention.{query,key,value}
    model_type = base.config.model_type  # "dinov3_vit" or "dinov2"
    is_dinov3 = model_type.startswith("dinov3")

    n_layers = base.config.num_hidden_layers
    target_layers = list(range(n_layers - cfg.lora_target_last_n_blocks, n_layers))
    target_modules = []
    for layer_idx in target_layers:
        if is_dinov3:
            target_modules += [
                f"model.layer.{layer_idx}.attention.q_proj",
                f"model.layer.{layer_idx}.attention.k_proj",
                f"model.layer.{layer_idx}.attention.v_proj",
            ]
        else:
            target_modules += [
                f"encoder.layer.{layer_idx}.attention.attention.query",
                f"encoder.layer.{layer_idx}.attention.attention.key",
                f"encoder.layer.{layer_idx}.attention.attention.value",
            ]

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    # Two independent adapters ("cad2d", "render3d") on the same frozen backbone.
    peft_model = get_peft_model(base, lora_cfg, adapter_name="cad2d")
    peft_model.add_adapter("render3d", lora_cfg)

    for p in peft_model.base_model.parameters():
        p.requires_grad_(False)
    # get_peft_model already leaves only LoRA params trainable; base stays frozen.

    branch_2d = DomainBranch(peft_model, hidden_dim, cfg)
    branch_3d = DomainBranch(peft_model, hidden_dim, cfg)  # shares backbone object, diff adapter
    return peft_model, branch_2d, branch_3d


# --------------------------------------------------------------------------
# Memory queue (MoCo-style negatives)
# --------------------------------------------------------------------------

class EmbeddingQueue:
    """
    Ring buffer of detached embeddings + their sample_ids, used purely as
    extra negatives for the contrastive loss. Not a momentum encoder — the
    embeddings are just whatever the online branches produced when enqueued,
    which is fine since we only ever use them as negatives (no gradient).
    """

    def __init__(self, dim: int, size: int, device: str):
        self.size = size
        self.dim = dim
        self.device = device
        self.embs = torch.zeros(size, dim, device=device)
        self.ids = [None] * size
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, embs: torch.Tensor, ids: List[str]):
        embs = embs.detach()
        n = embs.size(0)
        for i in range(n):
            self.embs[self.ptr] = embs[i]
            self.ids[self.ptr] = ids[i]
            self.ptr = (self.ptr + 1) % self.size
            if self.ptr == 0:
                self.full = True

    def get(self):
        if self.full:
            return self.embs, self.ids
        return self.embs[: self.ptr], self.ids[: self.ptr]


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------

def symmetric_info_nce(
    emb_2d: torch.Tensor,
    emb_3d: torch.Tensor,
    log_temp: torch.Tensor,
    queue_2d: "EmbeddingQueue" = None,
    queue_3d: "EmbeddingQueue" = None,
    hard_neg_weight: torch.Tensor = None,
):
    """
    emb_2d, emb_3d: [B, D], L2-normalized, index-aligned (same object per row).
    Symmetric InfoNCE. In-batch negatives are augmented with detached queue
    negatives so the effective negative pool is much larger than B.

    hard_neg_weight: optional [B] tensor, >1.0 upweights the loss for rows
    known to contain a hard negative (from the periodic mining pass), so the
    model gets a stronger gradient signal on the pairs it currently confuses.
    """
    temp = log_temp.exp().clamp(max=100.0)
    B = emb_2d.size(0)
    device = emb_2d.device

    # --- 2D -> 3D direction ---
    neg_3d = queue_3d.get()[0] if queue_3d is not None else emb_3d.new_zeros(0, emb_3d.size(1))
    candidates_3d = torch.cat([emb_3d, neg_3d], dim=0)          # [B + Q, D]
    logits_2d_to_3d = emb_2d @ candidates_3d.t() * temp          # [B, B+Q]
    labels = torch.arange(B, device=device)
    loss_2d_to_3d = F.cross_entropy(logits_2d_to_3d, labels, reduction="none")

    # --- 3D -> 2D direction ---
    neg_2d = queue_2d.get()[0] if queue_2d is not None else emb_2d.new_zeros(0, emb_2d.size(1))
    candidates_2d = torch.cat([emb_2d, neg_2d], dim=0)
    logits_3d_to_2d = emb_3d @ candidates_2d.t() * temp
    loss_3d_to_2d = F.cross_entropy(logits_3d_to_2d, labels, reduction="none")

    per_row_loss = (loss_2d_to_3d + loss_3d_to_2d) / 2
    if hard_neg_weight is not None:
        per_row_loss = per_row_loss * hard_neg_weight
    return per_row_loss.mean()


# --------------------------------------------------------------------------
# Train loop
# --------------------------------------------------------------------------

def train(cfg: Config):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    all_ids = CadToRenderDataset.discover_samples(cfg)
    random.shuffle(all_ids)
    n_val = max(1, int(0.1 * len(all_ids)))
    # IMPORTANT: if sample_ids encode part-family/near-duplicate groups,
    # split by family here instead of randomly, to avoid train/val leakage.
    val_ids, train_ids = all_ids[:n_val], all_ids[n_val:]

    train_ds = CadToRenderDataset(cfg, train_ids, train=True)
    val_ds = CadToRenderDataset(cfg, val_ids, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate_variable_views,
                               drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate_variable_views)

    peft_model, branch_2d, branch_3d = build_model(cfg)
    peft_model.to(cfg.device)
    branch_2d.to(cfg.device)
    branch_3d.to(cfg.device)

    log_temp = torch.tensor(float(torch.log(torch.tensor(1 / cfg.temperature))),
                             device=cfg.device, requires_grad=True)

    lora_params = [p for n, p in peft_model.named_parameters() if p.requires_grad]
    head_params = list(branch_2d.head.parameters()) + list(branch_2d.pool.parameters()) \
        + list(branch_3d.head.parameters()) + list(branch_3d.pool.parameters())

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg.lr_lora},
        {"params": head_params, "lr": cfg.lr_head},
        {"params": [log_temp], "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    queue_2d = EmbeddingQueue(cfg.embed_dim, cfg.queue_size, cfg.device)
    queue_3d = EmbeddingQueue(cfg.embed_dim, cfg.queue_size, cfg.device)

    # sample_id -> epochs-remaining-to-oversample, populated by hard-negative mining
    oversample_until = {}

    for epoch in range(cfg.epochs):
        peft_model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            cad_views = batch["cad_views"].to(cfg.device)          # [B, N2D, 3, H, W]
            B = cad_views.size(0)
            ids = batch["sample_id"]

            emb_2d = torch.stack([
                branch_2d.encode_object(cad_views[i], "cad2d") for i in range(B)
            ])  # [B, D]

            emb_3d = torch.stack([
                branch_3d.encode_object(v.to(cfg.device), "render3d")
                for v in batch["render_views"]
            ])  # [B, D]  (each object's own view count, handled per-sample)

            # upweight rows flagged by the most recent hard-negative mining pass
            weight = torch.tensor(
                [3.0 if oversample_until.get(sid, 0) > 0 else 1.0 for sid in ids],
                device=cfg.device,
            )

            loss = symmetric_info_nce(
                emb_2d, emb_3d, log_temp,
                queue_2d=queue_2d, queue_3d=queue_3d,
                hard_neg_weight=weight,
            )
            (loss / cfg.grad_accum_steps).backward()

            # enqueue AFTER computing the loss so this step's own positives
            # don't count as their own negatives
            queue_2d.enqueue(emb_2d, ids)
            queue_3d.enqueue(emb_3d, ids)

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

        scheduler.step()
        for sid in list(oversample_until.keys()):
            oversample_until[sid] -= 1
            if oversample_until[sid] <= 0:
                del oversample_until[sid]

        val_r1 = evaluate_retrieval(cfg, branch_2d, branch_3d, val_loader)
        log_line = (f"epoch {epoch:03d}  train_loss {running_loss / len(train_loader):.4f}  "
                    f"val_R@1 {val_r1:.3f}  temp {log_temp.exp().item():.2f}")

        if (epoch + 1) % cfg.hard_negative_every_n_epochs == 0:
            hard_pairs, mean_margin = mine_hard_negatives(cfg, branch_2d, branch_3d, train_loader)
            for sid in hard_pairs:
                oversample_until[sid] = cfg.hard_negative_oversample_epochs
            log_line += f"  hard_neg_margin {mean_margin:.4f}  flagged {len(hard_pairs)}"

        print(log_line)


@torch.no_grad()
def mine_hard_negatives(cfg: Config, branch_2d, branch_3d, loader, top_k: int = None):
    """
    Encode the full (train) set once, find the CAD->render pairs the model
    currently confuses most (highest-scoring non-matching pair per row),
    and return a dict {sample_id: hard_partner_id} plus the mean margin
    between the true match and the hardest negative (useful to log/trend
    over training — a shrinking margin means the model is genuinely
    learning to separate near-duplicate parts, not just memorizing).
    """
    top_k = top_k or cfg.hard_negative_pool_size
    branch_2d.eval()
    branch_3d.eval()

    ids, embs_2d, embs_3d = [], [], []
    for batch in loader:
        cad_views = batch["cad_views"].to(cfg.device)
        for i in range(cad_views.size(0)):
            embs_2d.append(branch_2d.encode_object(cad_views[i], "cad2d"))
        for v in batch["render_views"]:
            embs_3d.append(branch_3d.encode_object(v.to(cfg.device), "render3d"))
        ids += batch["sample_id"]

    E2 = torch.stack(embs_2d)   # [M, D]
    E3 = torch.stack(embs_3d)   # [M, D]
    sims = E2 @ E3.t()          # [M, M]

    M = sims.size(0)
    diag = sims.diag()
    sims_masked = sims.clone()
    sims_masked[torch.arange(M), torch.arange(M)] = -1e9   # exclude true match
    hardest_val, hardest_idx = sims_masked.max(dim=1)
    margins = diag - hardest_val   # smaller/negative margin = more confused

    order = torch.argsort(margins)[:top_k]
    hard_pairs = {ids[i.item()]: ids[hardest_idx[i].item()] for i in order}
    mean_margin = margins.mean().item()
    return hard_pairs, mean_margin


@torch.no_grad()
def evaluate_retrieval(cfg: Config, branch_2d, branch_3d, loader) -> float:
    """CAD -> render retrieval, top-1 accuracy over the val set."""
    branch_2d.eval()
    branch_3d.eval()
    all_2d, all_3d = [], []
    for batch in loader:
        cad_views = batch["cad_views"].to(cfg.device)
        B = cad_views.size(0)
        for i in range(B):
            all_2d.append(branch_2d.encode_object(cad_views[i], "cad2d"))
        for v in batch["render_views"]:
            all_3d.append(branch_3d.encode_object(v.to(cfg.device), "render3d"))

    if not all_2d:
        return 0.0
    E2 = torch.stack(all_2d)   # [M, D]
    E3 = torch.stack(all_3d)   # [M, D]
    sims = E2 @ E3.t()
    preds = sims.argmax(dim=1)
    labels = torch.arange(len(preds), device=preds.device)
    return (preds == labels).float().mean().item()


if __name__ == "__main__":
    cfg = Config()
    train(cfg)