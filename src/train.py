import os, random, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from src.dataset import get_dataloader
from src.model import EmbedNet, RefineHead

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GPSTripletLoss(nn.Module):
    def __init__(self, margin=0.3, pos_thresh=30.0, neg_thresh=80.0):
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

@torch.no_grad()
def build_db(embed_model, loader, device, amp=False):
    embed_model.eval()
    E_list, XY_list = [], []
    for images, _labels, gps in tqdm(loader, desc="Building DB", leave=False):
        images = images.to(device)
        gps = gps.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type=="cuda"):
            e = embed_model(images)
        E_list.append(e.detach().float().cpu())
        XY_list.append(gps.detach().float().cpu())
    E = torch.cat(E_list, dim=0)   # [N,D]
    XY = torch.cat(XY_list, dim=0) # [N,2]
    # normalize for dot-product retrieval
    E = torch.nn.functional.normalize(E, p=2, dim=1)
    return E, XY

@torch.no_grad()
def retrieve_xy0(E_query, E_db, XY_db, K=10, tau=0.07, device="cpu", chunk=4096):
    """
    Chunked top-K over database to avoid huge memory.
    E_query: [B,D] (device)
    E_db: [N,D] (cpu)
    XY_db: [N,2] (cpu)
    Returns xy0: [B,2] (device)
    """
    B, D = E_query.shape
    # keep best across chunks
    best_vals = torch.full((B, K), -1e9, device=device)
    best_idx  = torch.full((B, K), -1, dtype=torch.long, device=device)

    E_db = E_db.to(device)
    # chunk over db rows
    for start in range(0, E_db.size(0), chunk):
        end = min(start + chunk, E_db.size(0))
        sims = E_query @ E_db[start:end].T  # [B,chunk]
        vals, idx = sims.topk(K, dim=1)
        idx = idx + start

        # merge with current best
        merged_vals = torch.cat([best_vals, vals], dim=1)  # [B,2K]
        merged_idx  = torch.cat([best_idx,  idx], dim=1)

        best_vals, pos = merged_vals.topk(K, dim=1)
        best_idx = torch.gather(merged_idx, 1, pos)

    # gather XY and weight
    XY_db_dev = XY_db.to(device)
    nn_xy = XY_db_dev[best_idx]  # [B,K,2]

    w = torch.softmax(best_vals / tau, dim=1)  # [B,K]
    xy0 = (w.unsqueeze(-1) * nn_xy).sum(dim=1) # [B,2]
    return xy0

@torch.no_grad()
def eval_retrieval(embed_model, loader, E_db, XY_db, device, K=10, tau=0.07, amp=False):
    embed_model.eval()
    errs = []
    for images, _labels, gps in tqdm(loader, desc="Eval retrieval", leave=False):
        images = images.to(device)
        gps = gps.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type=="cuda"):
            e = embed_model(images)
        e = torch.nn.functional.normalize(e, p=2, dim=1)
        xy0 = retrieve_xy0(e, E_db, XY_db, K=K, tau=tau, device=device)
        err = torch.norm(xy0 - gps, dim=1)
        errs.append(err.detach().cpu())
    errs = torch.cat(errs)
    return float(errs.mean()), float(errs.median())

@torch.no_grad()
def eval_refined(embed_model, refine_head, loader, E_db, XY_db, device, K=10, tau=0.07, amp=False):
    embed_model.eval()
    refine_head.eval()
    errs = []
    for images, _labels, gps in tqdm(loader, desc="Eval refined", leave=False):
        images = images.to(device)
        gps = gps.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type=="cuda"):
            e = embed_model(images)
        e = torch.nn.functional.normalize(e, p=2, dim=1)
        xy0 = retrieve_xy0(e, E_db, XY_db, K=K, tau=tau, device=device)
        delta = refine_head(e.float(), xy0.float())
        pred = xy0 + delta
        err = torch.norm(pred - gps, dim=1)
        errs.append(err.detach().cpu())
    errs = torch.cat(errs)
    return float(errs.mean()), float(errs.median())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/metadata1.csv")
    ap.add_argument("--img_dir", default="data/images")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs_embed", type=int, default=10)
    ap.add_argument("--epochs_refine", type=int, default=10)
    ap.add_argument("--emb_dim", type=int, default=512)
    ap.add_argument("--lr_embed", type=float, default=1e-4)
    ap.add_argument("--lr_refine", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--pos", type=float, default=30.0)
    ap.add_argument("--neg", type=float, default=80.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_embed", default="embed.pt")
    ap.add_argument("--save_refine", default="refine.pt")
    args = ap.parse_args()

    seed_everything(args.seed)

    df = pd.read_csv(args.csv)
    df["sector_label"] = pd.factorize(df["sector_label"])[0].astype(int)

    train_df = df.sample(frac=0.9, random_state=args.seed)
    val_df = df.drop(train_df.index)

    # IMPORTANT:
    # For retrieval DB building, you want *no heavy augmentation*.
    # So we use mode="val" for the DB loader too.
    train_loader_aug = get_dataloader(train_df, args.img_dir, batch_size=args.batch_size, mode="train")
    train_loader_db  = get_dataloader(train_df, args.img_dir, batch_size=args.batch_size, mode="val")
    val_loader       = get_dataloader(val_df,   args.img_dir, batch_size=args.batch_size, mode="val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_model = EmbedNet(emb_dim=args.emb_dim).to(device)
    triplet = GPSTripletLoss(pos_thresh=args.pos, neg_thresh=args.neg).to(device)
    opt = optim.AdamW(embed_model.parameters(), lr=args.lr_embed, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=="cuda")

    # ---- Phase A: train embedding ----
    for ep in range(1, args.epochs_embed + 1):
        embed_model.train()
        pbar = tqdm(train_loader_aug, desc=f"Embed Epoch {ep}/{args.epochs_embed}")
        for images, _labels, gps in pbar:
            images = images.to(device)
            gps = gps.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device.type=="cuda"):
                e = embed_model(images)
                loss = triplet(e, gps)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(triplet=float(loss))

        # build DB + eval retrieval each epoch
        E_db, XY_db = build_db(embed_model, train_loader_db, device, amp=args.amp)
        mean_e, med_e = eval_retrieval(embed_model, val_loader, E_db, XY_db, device, K=args.K, tau=args.tau, amp=args.amp)
        print(f"[Embed Epoch {ep}] Retrieval MeanErr(m): {mean_e:.2f} | MedErr(m): {med_e:.2f}")

    torch.save(embed_model.state_dict(), args.save_embed)
    print(f"Saved embedding model: {args.save_embed}")

    # ---- Phase B: train refinement head (freeze embedding) ----
    for p in embed_model.parameters():
        p.requires_grad = False
    embed_model.eval()

    refine = RefineHead(emb_dim=args.emb_dim).to(device)
    opt_r = optim.AdamW(refine.parameters(), lr=args.lr_refine, weight_decay=1e-4)
    crit = nn.SmoothL1Loss()

    # Build fixed DB once
    E_db, XY_db = build_db(embed_model, train_loader_db, device, amp=args.amp)

    for ep in range(1, args.epochs_refine + 1):
        refine.train()
        pbar = tqdm(train_loader_db, desc=f"Refine Epoch {ep}/{args.epochs_refine}")
        for images, _labels, gps in pbar:
            images = images.to(device)
            gps = gps.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device.type=="cuda"):
                e = embed_model(images)
            e = torch.nn.functional.normalize(e, p=2, dim=1).float()

            xy0 = retrieve_xy0(e, E_db, XY_db, K=args.K, tau=args.tau, device=device)
            delta = refine(e, xy0)
            pred = xy0 + delta

            loss = crit(pred, gps.float())

            opt_r.zero_grad(set_to_none=True)
            loss.backward()
            opt_r.step()

            pbar.set_postfix(refine=float(loss))

        mean_r, med_r = eval_refined(embed_model, refine, val_loader, E_db, XY_db, device, K=args.K, tau=args.tau, amp=args.amp)
        print(f"[Refine Epoch {ep}] Refined MeanErr(m): {mean_r:.2f} | MedErr(m): {med_r:.2f}")

    torch.save(refine.state_dict(), args.save_refine)
    print(f"Saved refine head: {args.save_refine}")

if __name__ == "__main__":
    main()