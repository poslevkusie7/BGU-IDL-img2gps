import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from .dataset import get_dataloader
from .model import MultiTaskResNet


class GPSTripletLoss(nn.Module):
    """
    Triplet-style loss based on GPS distance.
    Vectorized hardest-pos / hardest-neg mining (much faster than Python loop).
    """
    def __init__(self, margin=0.3, pos_thresh=15.0, neg_thresh=50.0):
        super().__init__()
        self.margin = margin
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

    def forward(self, embeddings, gps_coords):
        # Pairwise distances
        dist_emb = torch.cdist(embeddings, embeddings, p=2)   # [B,B]
        dist_gps = torch.cdist(gps_coords, gps_coords, p=2)   # [B,B]

        # Masks
        mask_pos = (dist_gps < self.pos_thresh) & (dist_gps > 0)
        mask_neg = (dist_gps > self.neg_thresh)

        has_pos = mask_pos.any(dim=1)
        has_neg = mask_neg.any(dim=1)
        valid = has_pos & has_neg

        if not valid.any():
            return torch.zeros((), device=embeddings.device)

        # hardest positive: max dist among positives
        pos_d = dist_emb.masked_fill(~mask_pos, float("-inf")).max(dim=1).values
        # hardest negative: min dist among negatives
        neg_d = dist_emb.masked_fill(~mask_neg, float("inf")).min(dim=1).values

        pos_d = pos_d[valid]
        neg_d = neg_d[valid]

        loss = torch.relu(pos_d.pow(2) - neg_d.pow(2) + self.margin)
        return loss.mean()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_batch_metrics(cls_logits, labels, embeddings, gps):
    preds = cls_logits.argmax(dim=1)
    correct = (preds == labels).sum().item()

    dist_emb = torch.cdist(embeddings, embeddings, p=2)
    dist_emb.fill_diagonal_(float("inf"))
    nn_idx = dist_emb.argmin(dim=1)
    gps_nn = gps[nn_idx]
    gps_nn_sum = torch.norm(gps - gps_nn, dim=1).sum().item()

    return {
        "correct": correct,
        "gps_nn_sum": gps_nn_sum,
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion_cls, criterion_triplet, lambda_weight, device, amp=False):
    model.eval()
    total = 0.0
    total_cls = 0.0
    total_trip = 0.0
    total_correct = 0
    total_gps_nn = 0.0
    n = 0

    for images, labels, gps in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        gps = gps.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            cls_out, emb_out = model(images)
            loss_cls = criterion_cls(cls_out, labels)
            loss_trip = criterion_triplet(emb_out, gps)
            loss = loss_cls + lambda_weight * loss_trip

        metrics = compute_batch_metrics(cls_out, labels, emb_out, gps)
        total_correct += metrics["correct"]
        total_gps_nn += metrics["gps_nn_sum"]

        bs = images.size(0)
        total += loss.item() * bs
        total_cls += loss_cls.item() * bs
        total_trip += loss_trip.item() * bs
        n += bs

    avg_loss = total / max(n, 1)
    avg_cls = total_cls / max(n, 1)
    avg_trip = total_trip / max(n, 1)
    avg_acc = total_correct / max(n, 1)
    avg_gps_nn = total_gps_nn / max(n, 1)
    return avg_loss, avg_cls, avg_trip, avg_acc, avg_gps_nn


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=20,
    device="cuda",
    lr=1e-4,
    weight_decay=1e-4,
    lambda_weight=1.0,
    amp=True,
    compile_model=False,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Speed knobs (GPU)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_triplet = GPSTripletLoss(margin=0.3, pos_thresh=15, neg_thresh=50)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", leave=True)
    for epoch in epoch_bar:
        model.train()
        running = 0.0
        running_cls = 0.0
        running_trip = 0.0
        running_correct = 0
        running_gps_nn = 0.0
        seen = 0

        for images, labels, gps in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            gps = gps.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
                cls_out, emb_out = model(images)
                loss_cls = criterion_cls(cls_out, labels)
                loss_trip = criterion_triplet(emb_out, gps)
                loss = loss_cls + lambda_weight * loss_trip

            metrics = compute_batch_metrics(cls_out, labels, emb_out, gps)
            running_correct += metrics["correct"]
            running_gps_nn += metrics["gps_nn_sum"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running += loss.item() * bs
            running_cls += loss_cls.item() * bs
            running_trip += loss_trip.item() * bs
            seen += bs

        scheduler.step()

        train_loss = running / max(seen, 1)
        train_cls = running_cls / max(seen, 1)
        train_trip = running_trip / max(seen, 1)
        train_acc = running_correct / max(seen, 1)
        train_gps_nn = running_gps_nn / max(seen, 1)

        if val_loader is not None:
            val_loss, val_cls, val_trip, val_acc, val_gps_nn = evaluate(
                model, val_loader, criterion_cls, criterion_triplet, lambda_weight, device, amp=amp
            )
            epoch_bar.set_postfix({
                "loss": train_loss,
                "cls": train_cls,
                "trip": train_trip,
                "acc": train_acc,
                "nn_gps_m": train_gps_nn,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_nn_gps_m": val_gps_nn,
                "lr": scheduler.get_last_lr()[0],
            })
        else:
            epoch_bar.set_postfix({
                "loss": train_loss,
                "cls": train_cls,
                "trip": train_trip,
                "acc": train_acc,
                "nn_gps_m": train_gps_nn,
                "lr": scheduler.get_last_lr()[0],
            })

    return model


def main():
    CSV_PATH = "data/metadata1.csv"
    IMG_DIR  = "data/images"
    BATCH_SIZE = 32
    EPOCHS = 20
    NUM_WORKERS = 4
    DEVICE = "cuda"
    AMP = True
    COMPILE = False
    SAVE_PATH = "model.pt"
    SEED = 42

    seed_everything(SEED)

    df = pd.read_csv(CSV_PATH)
    if not {"image_id", "sector_label", "lat", "lon"}.issubset(df.columns):
        missing = sorted({"image_id", "sector_label", "lat", "lon"} - set(df.columns))
        raise KeyError(f"metadata.csv missing required columns: {missing}")

    # Ensure labels are 0..(num_classes-1) for CrossEntropyLoss
    df["sector_label"] = pd.factorize(df["sector_label"])[0].astype(int)
    num_sectors = int(df["sector_label"].nunique())
    model = MultiTaskResNet(num_sectors=num_sectors)

    train_df = df.sample(frac=0.9, random_state=SEED)
    val_df = df.drop(train_df.index)

    train_loader = get_dataloader(
        train_df, IMG_DIR, batch_size=BATCH_SIZE, mode="train", num_workers=NUM_WORKERS
    )
    val_loader = get_dataloader(
        val_df, IMG_DIR, batch_size=BATCH_SIZE, mode="val", num_workers=NUM_WORKERS
    )

    model = train_model(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        device=DEVICE,
        amp=AMP,
        compile_model=COMPILE,
    )

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
