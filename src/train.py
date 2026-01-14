import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from dataset import get_dataloader
from model import MultiTaskResNet

class GPSTripletLoss(nn.Module):
    def __init__(self, margin=0.3, pos_thresh=15.0, neg_thresh=50.0):
        super().__init__()
        self.margin = margin
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

    def forward(self, embeddings, gps_coords):
        dist_emb = torch.cdist(embeddings, embeddings, p=2)
        dist_gps = torch.cdist(gps_coords, gps_coords, p=2)

        mask_pos = (dist_gps < self.pos_thresh) & (dist_gps > 0)
        mask_neg = (dist_gps > self.neg_thresh)

        valid = mask_pos.any(dim=1) & mask_neg.any(dim=1)
        if not valid.any():
            return torch.zeros((), device=embeddings.device)

        pos_d = dist_emb.masked_fill(~mask_pos, float("-inf")).max(dim=1).values
        neg_d = dist_emb.masked_fill(~mask_neg, float("inf")).min(dim=1).values

        loss = torch.relu(pos_d[valid].pow(2) - neg_d[valid].pow(2) + self.margin)
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
    return {"correct": correct, "gps_nn_sum": gps_nn_sum}

@torch.no_grad()
def evaluate(model, dataloader, criterion_cls, criterion_triplet, lambda_weight, device, amp=False):
    model.eval()
    total, total_cls, total_trip, total_correct, total_gps_nn, n = 0.0, 0.0, 0.0, 0, 0.0, 0
    for images, labels, gps in dataloader:
        images, labels, gps = images.to(device), labels.to(device), gps.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            cls_out, emb_out = model(images)
            loss_cls = criterion_cls(cls_out, labels)
            loss_trip = criterion_triplet(emb_out, gps)
            loss = loss_cls + lambda_weight * loss_trip
        
        m = compute_batch_metrics(cls_out, labels, emb_out, gps)
        total_correct += m["correct"]
        total_gps_nn += m["gps_nn_sum"]
        bs = images.size(0)
        total += loss.item() * bs
        total_cls += loss_cls.item() * bs
        total_trip += loss_trip.item() * bs
        n += bs
    return total/n, total_cls/n, total_trip/n, total_correct/n, total_gps_nn/n

def train_model(model, train_loader, val_loader, epochs=20, device="cuda", lr=1e-2, lambda_weight=0.0, amp=True):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_triplet = GPSTripletLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Global TQDM bar over all epochs and batches
    total_steps = epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training Progress")

    for epoch in range(1, epochs + 1):
        model.train()
        running_correct, seen = 0, 0
        
        for images, labels, gps in train_loader:
            images, labels, gps = images.to(device), labels.to(device), gps.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                cls_out, emb_out = model(images)

            l_cls = criterion_cls(cls_out, labels)
            l_trip = criterion_triplet(emb_out, gps)
            loss = l_cls + (lambda_weight * l_trip)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            seen += images.size(0)
            running_correct += (cls_out.argmax(1) == labels).sum().item()
            
            # Update global bar per batch
            pbar.update(1)
            pbar.set_postfix({
                "ep": epoch,
                "loss": f"{loss.item():.4f}",
                "acc": f"{running_correct/seen:.4f}"
            })
        
        scheduler.step()
        
        # Validation update at end of epoch
        if val_loader:
            v_loss, _, _, v_acc, _ = evaluate(model, val_loader, criterion_cls, criterion_triplet, lambda_weight, device, amp)
            # Log validation to the bar's description or postfix
            pbar.write(f"[Epoch {epoch}] Val Acc: {v_acc:.4f}, Val Loss: {v_loss:.4f}")

    pbar.close()
    return model

def main():
    CSV_PATH = "data/metadata1.csv"

    seed_everything()

    df = pd.read_csv(CSV_PATH)
    if not {"image_id", "sector_label", "lat", "lon"}.issubset(df.columns):
        missing = sorted({"image_id", "sector_label", "lat", "lon"} - set(df.columns))
        raise KeyError(f"metadata.csv missing required columns: {missing}")

    # Ensure labels are 0..(num_classes-1) for CrossEntropyLoss
    df["sector_label"] = pd.factorize(df["sector_label"])[0].astype(int)
    
    train_df = df.sample(frac=0.9, random_state=42)
    val_df = df.drop(train_df.index)

    train_loader = get_dataloader(train_df, "data/images", mode="train")
    val_loader = get_dataloader(val_df, "data/images", mode="val")

    num_sectors = df["sector_label"].nunique()
    model = MultiTaskResNet(num_sectors=num_sectors)
    
    model = train_model(model, train_loader, val_loader, epochs=20)
    
    torch.save(model.state_dict(), "model.pt")
    print("Training complete. Model saved to model.pt")

if __name__ == "__main__":
    main()