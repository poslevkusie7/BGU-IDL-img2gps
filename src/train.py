import os, random, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from src.dataset import get_dataloader
from src.model import MDNResNet

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mdn_nll(pi_logits, mu, log_sigma, y):
    """
    Negative log-likelihood of y under mixture of diagonal Gaussians.
    pi_logits: [B,K]
    mu:        [B,K,2]
    log_sigma: [B,K,2]
    y:         [B,2]
    """
    # log pi
    log_pi = torch.log_softmax(pi_logits, dim=1)  # [B,K]

    # expand y to [B,K,2]
    y_exp = y.unsqueeze(1).expand_as(mu)

    # log N(y | mu, sigma)
    # diagonal Gaussian: sum over dims
    # log N = -0.5 * [ sum((y-mu)^2 / sigma^2) + sum(log(2pi*sigma^2)) ]
    sigma2 = torch.exp(2.0 * log_sigma)  # sigma^2
    diff2 = (y_exp - mu) ** 2
    log_norm = log_sigma + 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=y.device))
    # per-dim: -0.5 * (diff^2/sigma^2) - log(sigma*sqrt(2pi))
    log_comp = -0.5 * (diff2 / sigma2) - log_norm
    log_comp = log_comp.sum(dim=2)  # [B,K]

    # logsumexp over K components
    log_prob = torch.logsumexp(log_pi + log_comp, dim=1)  # [B]
    return (-log_prob).mean()

@torch.no_grad()
def mdn_predict(pi_logits, mu, log_sigma, mode="mean"):
    """
    mode:
      - "mean": E[y] = sum pi * mu
      - "map":  mu[argmax pi]
    """
    pi = torch.softmax(pi_logits, dim=1)  # [B,K]
    if mode == "mean":
        return (pi.unsqueeze(-1) * mu).sum(dim=1)  # [B,2]
    else:
        k = torch.argmax(pi, dim=1)                # [B]
        return mu[torch.arange(mu.size(0), device=mu.device), k]

@torch.no_grad()
def evaluate(model, loader, device, amp=False, pred_mode="mean"):
    model.eval()
    total_nll = 0.0
    total = 0
    errs = []

    for images, _labels, gps in loader:
        images = images.to(device)
        gps = gps.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type=="cuda"):
            pi_logits, mu, log_sigma = model(images)
            nll = mdn_nll(pi_logits, mu, log_sigma, gps)

        pred = mdn_predict(pi_logits, mu, log_sigma, mode=pred_mode)
        err = torch.norm(pred - gps, dim=1)  # meters
        errs.append(err.detach().cpu())

        bs = images.size(0)
        total_nll += float(nll) * bs
        total += bs

    errs = torch.cat(errs)
    mean_err = float(errs.mean())
    med_err = float(errs.median())
    return total_nll / max(total,1), mean_err, med_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/metadata1.csv")
    ap.add_argument("--img_dir", default="data/images")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="mdn.pt")
    ap.add_argument("--pred_mode", choices=["mean","map"], default="mean")
    args = ap.parse_args()

    seed_everything(args.seed)

    df = pd.read_csv(args.csv)
    df["sector_label"] = pd.factorize(df["sector_label"])[0].astype(int)

    # IMPORTANT: if you can, use stratified split for stability
    train_df = df.sample(frac=0.9, random_state=args.seed)
    val_df = df.drop(train_df.index)

    train_loader = get_dataloader(train_df, args.img_dir, batch_size=64, mode="train")
    val_loader   = get_dataloader(val_df,   args.img_dir, batch_size=64, mode="val")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MDNResNet(K=args.K).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=="cuda")

    for ep in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"MDN Epoch {ep}/{args.epochs}")
        for images, _labels, gps in pbar:
            images = images.to(device)
            gps = gps.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and device.type=="cuda"):
                pi_logits, mu, log_sigma = model(images)
                loss = mdn_nll(pi_logits, mu, log_sigma, gps)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(nll=float(loss))

        val_nll, mean_err, med_err = evaluate(model, val_loader, device, amp=args.amp, pred_mode=args.pred_mode)
        print(f"[Epoch {ep}] Val NLL: {val_nll:.4f} | MeanErr(m): {mean_err:.2f} | MedErr(m): {med_err:.2f}")

    torch.save(model.state_dict(), args.save)
    print(f"Saved: {args.save}")

if __name__ == "__main__":
    main()