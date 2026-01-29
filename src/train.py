import argparse
import math
import os
import random

import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .dataset import LocalizationDataset, compute_coord_stats
from .model import DinoV2CoordRegressor, SharedDinoMultiTask


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if isinstance(cfg, list):
        if not all(isinstance(item, dict) for item in cfg):
            raise ValueError("Config list entries must be mappings.")
        return cfg
    if not isinstance(cfg, dict):
        raise ValueError("Config file must define a mapping at the top level.")
    return cfg


def ensure_parent_dir(path):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def make_log_fn(log_path):
    if not log_path:
        return print, lambda: None
    ensure_parent_dir(log_path)
    log_file = open(log_path, "w", encoding="utf-8")

    def log_fn(message):
        tqdm.write(message)
        log_file.write(f"{message}\n")
        log_file.flush()

    return log_fn, log_file.close


def args_from_config(cfg, parser):
    args = parser.parse_args([])
    for key, value in cfg.items():
        setattr(args, key, value)
    return args


def expand_runs(cfg):
    if isinstance(cfg, list):
        return cfg
    if isinstance(cfg, dict) and "runs" in cfg:
        runs = cfg["runs"]
        if not isinstance(runs, list):
            raise ValueError("'runs' must be a list of mappings.")
        base_cfg = {k: v for k, v in cfg.items() if k != "runs"}
        merged = []
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError("Each run in 'runs' must be a mapping.")
            run_cfg = dict(base_cfg)
            run_cfg.update(run)
            merged.append(run_cfg)
        return merged
    return None


class Lookahead(optim.Optimizer):
    def __init__(self, optimizer, alpha=0.5, k=6):
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.defaults = optimizer.defaults
        self.alpha = alpha
        self.k = k
        self.state = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p] = {"slow_buffer": p.data.clone()}
        self._step = 0

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step += 1
        if self._step % self.k != 0:
            return loss
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                slow = state["slow_buffer"]
                slow.add_(self.alpha * (p.data - slow))
                p.data.copy_(slow)
        return loss


def build_transforms(mode="train", img_size=518, randaugment=False, ra_n=2, ra_m=9):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if mode == "train":
        ops = [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        ]
        if randaugment:
            ops.append(transforms.RandAugment(num_ops=ra_n, magnitude=ra_m))
        ops += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        ]
    else:
        ops = [transforms.Resize((img_size, img_size))]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(ops)




def denormalize_coords(coords, coord_stats, coord_norm):
    coords = coords.float()
    mean = torch.tensor(coord_stats["mean"], device=coords.device, dtype=torch.float32)
    std = torch.tensor(coord_stats["std"], device=coords.device, dtype=torch.float32)
    if coord_norm == "standard":
        return coords * std + mean
    if coord_norm == "center":
        return coords + mean
    return coords


def haversine_m(latlon_pred, latlon_true):
    # lat/lon in degrees -> meters
    lat1 = torch.deg2rad(latlon_pred[:, 0])
    lon1 = torch.deg2rad(latlon_pred[:, 1])
    lat2 = torch.deg2rad(latlon_true[:, 0])
    lon2 = torch.deg2rad(latlon_true[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a.clamp(min=0, max=1)))
    return 6371000.0 * c


def distance_m(preds, targets, coord_mode):
    if coord_mode == "utm":
        return torch.norm(preds - targets, dim=1)
    return haversine_m(preds, targets)


def get_optimizer(opt_name, params, lr, weight_decay):
    if opt_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "radam":
        return optim.RAdam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "radam_lookahead":
        base = optim.RAdam(params, lr=lr, weight_decay=weight_decay)
        return Lookahead(base, alpha=0.5, k=6)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def build_scheduler(optimizer, scheduler_name, warmup_steps, total_steps):
    """
    Returns scheduler and a flag indicating step-level stepping.
    """
    if scheduler_name == "none":
        return None, False
    if scheduler_name == "cosine_warmup":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), True
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps), True
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def compute_regression_loss(preds, targets):
    return nn.functional.mse_loss(preds, targets, reduction="mean")





def train_coords(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    amp,
    optimizer_name,
    coord_stats,
    coord_norm,
    coord_mode,
    freeze_epochs,
    scheduler_name,
    warmup_steps,
    save_best_cb,
    accum_steps,
    epoch_saver=None,
    log_fn=print,
):
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    total_steps = math.ceil(epochs * len(train_loader) / max(1, accum_steps))
    scheduler, per_step = build_scheduler(optimizer, scheduler_name, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_epochs and epoch <= freeze_epochs and hasattr(model, "freeze_backbone"):
            model.freeze_backbone()
        elif freeze_epochs and epoch == freeze_epochs + 1 and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()

        running_loss = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for step_idx, (images, _, coords) in enumerate(bar):
            images = images.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
                preds = model(images)
                loss = compute_regression_loss(preds, coords)

            loss_value = loss.item()
            loss = loss / max(1, accum_steps)
            scaler.scale(loss).backward()
            do_step = ((step_idx + 1) % max(1, accum_steps) == 0) or (step_idx + 1 == len(train_loader))
            if do_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and per_step:
                    scheduler.step()

            bs = images.size(0)
            running_loss += loss_value * bs
            seen += bs
            bar.set_postfix(loss=running_loss / max(seen, 1))

        if scheduler is not None and not per_step:
            scheduler.step()
        if val_loader is not None:
            val_loss, val_mse, val_dist_m, val_p10, val_p25 = evaluate_coords(
                model,
                val_loader,
                device,
                amp,
                coord_stats,
                coord_norm,
                coord_mode
            )
            log_fn(
                f"Epoch {epoch}: train_loss={running_loss / max(seen, 1):.6f} "
                f"val_loss={val_loss:.6f} val_mse={val_mse:.3e} "
                f"val_dist_m={val_dist_m:.2f} p10m={val_p10:.3f} p25m={val_p25:.3f}"
            )
            if save_best_cb is not None:
                save_best_cb(-val_dist_m, model, epoch)
        else:
            log_fn(f"Epoch {epoch}: train_loss={running_loss / max(seen, 1):.6f}")

        if epoch_saver is not None:
            epoch_saver(epoch, model)


def train_multitask(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    amp,
    optimizer_name,
    coord_stats,
    coord_norm,
    coord_mode,
    cls_weight,
    coord_weight,
    freeze_epochs,
    scheduler_name,
    warmup_steps,
    save_best_cb,
    accum_steps,
    epoch_saver=None,
    log_fn=print,
):
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr, weight_decay)
    total_steps = math.ceil(epochs * len(train_loader) / max(1, accum_steps))
    scheduler, per_step = build_scheduler(optimizer, scheduler_name, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_epochs and epoch <= freeze_epochs and hasattr(model, "freeze_backbones"):
            model.freeze_backbones()
        elif freeze_epochs and epoch == freeze_epochs + 1 and hasattr(model, "unfreeze_backbones"):
            model.unfreeze_backbones()

        running_cls_loss = 0.0
        running_reg_loss = 0.0
        running_cls_acc = 0.0
        running_dist = 0.0
        running_p10 = 0.0
        running_p25 = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for step_idx, (images, labels, coords) in enumerate(bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
                logits, preds = model(images)
                cls_loss = cls_criterion(logits, labels)
                reg_loss = compute_regression_loss(preds, coords)
                loss = cls_weight * cls_loss + coord_weight * reg_loss

            loss_value_cls = cls_loss.item()
            loss_value_reg = reg_loss.item()
            loss = loss / max(1, accum_steps)

            scaler.scale(loss).backward()
            do_step = ((step_idx + 1) % max(1, accum_steps) == 0) or (step_idx + 1 == len(train_loader))
            if do_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and per_step:
                    scheduler.step()

            preds_cls = logits.argmax(dim=1)
            acc = (preds_cls == labels).sum().item()
            denorm_preds = denormalize_coords(preds, coord_stats, coord_norm)
            denorm_targets = denormalize_coords(coords, coord_stats, coord_norm)
            dist = distance_m(denorm_preds, denorm_targets, coord_mode)

            bs = images.size(0)
            running_cls_loss += loss_value_cls * bs
            running_reg_loss += loss_value_reg * bs
            running_cls_acc += acc
            running_dist += dist.mean().item() * bs
            running_p10 += (dist <= 10.0).float().mean().item() * bs
            running_p25 += (dist <= 25.0).float().mean().item() * bs
            seen += bs
            bar.set_postfix(
                cls_loss=running_cls_loss / max(seen, 1),
                reg_loss=running_reg_loss / max(seen, 1),
            )

        if scheduler is not None and not per_step:
            scheduler.step()

        if val_loader is not None:
            (
                val_cls_loss,
                val_reg_loss,
                val_cls_acc,
                val_dist,
                val_p10,
                val_p25,
            ) = evaluate_multitask(
                model,
                val_loader,
                device,
                amp,
                coord_stats,
                coord_norm,
                coord_mode,
                cls_criterion
            )
            log_fn(
                f"Epoch {epoch}: cls_loss={running_cls_loss / max(seen,1):.4f} "
                f"reg_loss={running_reg_loss / max(seen,1):.4f} "
                f"val_cls_loss={val_cls_loss:.4f} val_reg_loss={val_reg_loss:.4f} "
                f"val_acc={val_cls_acc:.4f} val_dist_m={val_dist:.2f} "
                f"val_p10={val_p10:.3f} val_p25={val_p25:.3f}"
            )
            if save_best_cb is not None:
                save_best_cb(-val_dist, model, epoch)
        else:
            log_fn(
                f"Epoch {epoch}: cls_loss={running_cls_loss / max(seen,1):.4f} "
                f"reg_loss={running_reg_loss / max(seen,1):.4f} "
                f"train_acc={running_cls_acc / max(seen,1):.4f} "
                f"train_dist_m={running_dist / max(seen,1):.2f} "
                f"train_p10={running_p10 / max(seen,1):.3f} "
                f"train_p25={running_p25 / max(seen,1):.3f}"
            )

        if epoch_saver is not None:
            epoch_saver(epoch, model)
@torch.no_grad()
def evaluate_coords(
    model,
    dataloader,
    device,
    amp,
    coord_stats,
    coord_norm,
    coord_mode,
):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_dist = 0.0
    running_p10 = 0.0
    running_p25 = 0.0
    seen = 0
    for images, _, coords in dataloader:
        images = images.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            preds = model(images)
            reg_loss = compute_regression_loss(preds, coords)
        denorm_preds = denormalize_coords(preds, coord_stats, coord_norm)
        denorm_targets = denormalize_coords(coords, coord_stats, coord_norm)
        mse = nn.functional.mse_loss(denorm_preds, denorm_targets, reduction="mean")
        dist = distance_m(denorm_preds, denorm_targets, coord_mode)
        p10 = (dist <= 10.0).float().mean()
        p25 = (dist <= 25.0).float().mean()

        bs = images.size(0)
        running_loss += reg_loss.item() * bs
        running_mse += mse.item() * bs
        running_dist += dist.mean().item() * bs
        running_p10 += p10.item() * bs
        running_p25 += p25.item() * bs
        seen += bs

    return (
        running_loss / max(seen, 1),
        running_mse / max(seen, 1),
        running_dist / max(seen, 1),
        running_p10 / max(seen, 1),
        running_p25 / max(seen, 1),
    )


@torch.no_grad()
def evaluate_multitask(
    model,
    dataloader,
    device,
    amp,
    coord_stats,
    coord_norm,
    coord_mode,
    cls_criterion
):
    model.eval()
    running_cls_loss = 0.0
    running_reg_loss = 0.0
    running_cls_acc = 0.0
    running_dist = 0.0
    running_p10 = 0.0
    running_p25 = 0.0
    seen = 0
    for images, labels, coords in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        coords = coords.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            logits, preds = model(images)
            cls_loss = cls_criterion(logits, labels)
            reg_loss = compute_regression_loss(preds, coords)


        preds_cls = logits.argmax(dim=1)
        acc = (preds_cls == labels).float().mean()
        denorm_preds = denormalize_coords(preds, coord_stats, coord_norm)
        denorm_targets = denormalize_coords(coords, coord_stats, coord_norm)
        dist = distance_m(denorm_preds, denorm_targets, coord_mode)

        bs = images.size(0)
        running_cls_loss += cls_loss.item() * bs
        running_reg_loss += reg_loss.item() * bs
        running_cls_acc += acc.item() * bs
        running_dist += dist.mean().item() * bs
        running_p10 += (dist <= 10.0).float().mean().item() * bs
        running_p25 += (dist <= 25.0).float().mean().item() * bs
        seen += bs

    return (
        running_cls_loss / max(seen, 1),
        running_reg_loss / max(seen, 1),
        running_cls_acc / max(seen, 1),
        running_dist / max(seen, 1),
        running_p10 / max(seen, 1),
        running_p25 / max(seen, 1),
    )


def build_parser(config_parser=None):
    parents = [config_parser] if config_parser is not None else []
    parser = argparse.ArgumentParser(parents=parents)
    parser.add_argument("--task", choices=["coords", "multitask"], required=False)
    parser.add_argument("--csv-path", default="data/metadata1.csv")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--img-dir", default="data/images")
    parser.add_argument("--train-img-dir", default=None)
    parser.add_argument("--val-img-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "radam", "radam_lookahead"], default="adamw")
    parser.add_argument("--save-path", default="model.pt")
    parser.add_argument("--best-path", default="best_model.pt")
    parser.add_argument("--last-path", default=None)
    parser.add_argument("--prev-path", default=None)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--randaugment", action="store_true")
    parser.add_argument("--ra-n", type=int, default=2)
    parser.add_argument("--ra-m", type=int, default=9)
    parser.add_argument("--coord-mode", choices=["latlon", "utm"], default="latlon")
    parser.add_argument("--coord-norm", choices=["standard", "center", "none"], default="standard")
    parser.add_argument("--dinov2-name", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--coord-weight", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "cosine", "cosine_warmup"], default="cosine_warmup")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio when train/val CSVs are not provided.",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="When config defines runs, select a single run by index (0-based).",
    )
    return parser


def run_training(args):
    if args.task is None:
        raise ValueError("--task is required (coords, multitask).")

    log_fn, close_log = make_log_fn(args.log_path)
    try:
        seed_everything(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        df = pd.read_csv(args.csv_path)
        required = {"image_id", "sector_label", "lat", "lon"}
        if not required.issubset(df.columns):
            missing = sorted(required - set(df.columns))
            raise KeyError(f"CSV missing required columns: {missing}")

        label_map = None
        if not pd.api.types.is_integer_dtype(df["sector_label"]):
            codes, uniques = pd.factorize(df["sector_label"])
            df["sector_label"] = codes.astype(int)
            label_map = {label: idx for idx, label in enumerate(uniques)}

        def apply_label_map(sub_df):
            if sub_df is None:
                return None
            if label_map is None:
                if not pd.api.types.is_integer_dtype(sub_df["sector_label"]):
                    sub_df["sector_label"] = pd.factorize(sub_df["sector_label"])[0].astype(int)
                return sub_df
            mapped = sub_df["sector_label"].map(label_map)
            if mapped.isnull().any():
                missing = sub_df.loc[mapped.isnull(), "sector_label"].unique()
                raise ValueError(f"Unknown sector_label values in split: {missing[:5]}")
            sub_df["sector_label"] = mapped.astype(int)
            return sub_df

        if args.train_csv or args.val_csv:
            if not args.train_csv:
                raise ValueError("train_csv is required when using explicit train/val splits.")
            train_df = pd.read_csv(args.train_csv)
            val_df = pd.read_csv(args.val_csv) if args.val_csv else None
        else:
            if not 0.0 < args.val_split < 1.0:
                raise ValueError("val_split must be between 0 and 1.")
            train_df = df.sample(frac=1.0 - args.val_split, random_state=args.seed)
            val_df = df.drop(train_df.index)

        for split_name, split_df in (("train", train_df), ("val", val_df)):
            if split_df is None:
                continue
            if not required.issubset(split_df.columns):
                missing = sorted(required - set(split_df.columns))
                raise KeyError(f"{split_name} CSV missing required columns: {missing}")

        train_df = apply_label_map(train_df)
        val_df = apply_label_map(val_df)

        train_transform = build_transforms(
            mode="train",
            randaugment=args.randaugment,
            ra_n=args.ra_n,
            ra_m=args.ra_m,
        )
        val_transform = build_transforms(mode="val")

        coord_stats = (
            compute_coord_stats(train_df, coord_mode=args.coord_mode)
            if args.task in ("coords", "multitask")
            else None
        )

        train_img_dir = args.train_img_dir or args.img_dir
        val_img_dir = args.val_img_dir or args.img_dir

        train_ds = LocalizationDataset(
            train_df,
            train_img_dir,
            transform=train_transform,
            coord_mode=args.coord_mode,
            coord_norm=args.coord_norm,
            coord_stats=coord_stats,
        )
        val_ds = None
        if val_df is not None:
            val_ds = LocalizationDataset(
                val_df,
                val_img_dir,
                transform=val_transform,
                coord_mode=args.coord_mode,
                coord_norm=args.coord_norm,
                coord_stats=coord_stats,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

        def make_best_saver(base_payload, path):
            if not path:
                return None
            ensure_parent_dir(path)
            best_metric = -float("inf")

            def save(metric, model, epoch=None):
                nonlocal best_metric
                if metric > best_metric:
                    best_metric = metric
                    payload = dict(base_payload)
                    payload["model_state"] = model.state_dict()
                    if epoch is not None:
                        payload["epoch"] = epoch
                    torch.save(payload, path)
                    log_fn(f"Saved best checkpoint to {path} (metric={metric:.4f})")

            return save

        def make_epoch_saver(base_payload, last_path, prev_path, epochs):
            if not last_path and not prev_path:
                return None
            if last_path:
                ensure_parent_dir(last_path)
            if prev_path:
                ensure_parent_dir(prev_path)

            def save(epoch, model):
                if prev_path and epochs > 1 and epoch == epochs - 1:
                    payload = dict(base_payload)
                    payload["model_state"] = model.state_dict()
                    payload["epoch"] = epoch
                    torch.save(payload, prev_path)
                    log_fn(f"Saved prev checkpoint to {prev_path} (epoch={epoch})")
                if last_path and epoch == epochs:
                    payload = dict(base_payload)
                    payload["model_state"] = model.state_dict()
                    payload["epoch"] = epoch
                    torch.save(payload, last_path)
                    log_fn(f"Saved last checkpoint to {last_path} (epoch={epoch})")

            return save

        if args.task == "coords":
            coord_stats = train_ds.get_coord_stats()
            model = DinoV2CoordRegressor(
                pretrained=args.pretrained,
                model_name=args.dinov2_name,
            )
            model.to(device)
            base_payload = {
                "task": "coords",
                "coord_stats": coord_stats,
                "coord_norm": args.coord_norm,
                "coord_mode": args.coord_mode,
                "dinov2_name": args.dinov2_name,
            }
            save_best = make_best_saver(base_payload, args.best_path)
            epoch_saver = make_epoch_saver(
                base_payload, args.last_path, args.prev_path, args.epochs
            )
            train_coords(
                model,
                train_loader,
                val_loader,
                device,
                args.epochs,
                args.lr,
                args.weight_decay,
                args.amp,
                args.optimizer,
                coord_stats,
                args.coord_norm,
                args.coord_mode,
                args.freeze_epochs,
                args.scheduler,
                args.warmup_steps,
                save_best,
                args.accum_steps,
                epoch_saver=epoch_saver,
                log_fn=log_fn,
            )
            save_payload = dict(base_payload)
            save_payload["model_state"] = model.state_dict()
        else:
            num_classes = int(df["sector_label"].nunique())
            model = SharedDinoMultiTask(
                num_classes=num_classes,
                coord_model_name=args.dinov2_name,
                pretrained=args.pretrained,
            ).to(device)
            base_payload = {
                "task": "multitask",
                "num_classes": num_classes,
                "coord_stats": coord_stats,
                "coord_norm": args.coord_norm,
                "coord_mode": args.coord_mode,
                "dinov2_name": args.dinov2_name,
            }
            save_best = make_best_saver(base_payload, args.best_path)
            epoch_saver = make_epoch_saver(
                base_payload, args.last_path, args.prev_path, args.epochs
            )
            train_multitask(
                model,
                train_loader,
                val_loader,
                device,
                args.epochs,
                args.lr,
                args.weight_decay,
                args.amp,
                args.optimizer,
                coord_stats,
                args.coord_norm,
                args.coord_mode,
                args.cls_weight,
                args.coord_weight,
                args.freeze_epochs,
                args.scheduler,
                args.warmup_steps,
                save_best,
                args.accum_steps,
                epoch_saver=epoch_saver,
                log_fn=log_fn,
            )
            save_payload = dict(base_payload)
            save_payload["model_state"] = model.state_dict()

        if args.save_path:
            ensure_parent_dir(args.save_path)
            torch.save(save_payload, args.save_path)
            log_fn(f"Saved: {args.save_path}")
    finally:
        close_log()


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="Path to YAML config.", default=None)
    cfg_args, remaining = config_parser.parse_known_args()

    parser = build_parser(config_parser)

    if cfg_args.config:
        cfg = load_config(cfg_args.config)
        run_cfgs = expand_runs(cfg)
        if run_cfgs is not None:
            cli_args = parser.parse_args(remaining)
            if cli_args.run_index is not None:
                if cli_args.run_index < 0 or cli_args.run_index >= len(run_cfgs):
                    raise ValueError(f"run_index must be between 0 and {len(run_cfgs) - 1}.")
                run_cfgs = [run_cfgs[cli_args.run_index]]
            for run_cfg in run_cfgs:
                args = args_from_config(run_cfg, parser)
                run_training(args)
            return
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    if args.task is None:
        parser.error("--task is required (coords, multitask)")
    run_training(args)


if __name__ == "__main__":
    main()
