import re
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
# Set to the weights used during training.
CLS_WEIGHT = 0.3
COORD_WEIGHT = 1.0
COORDS_RE = re.compile(
    rf"Epoch\s+(?P<epoch>\d+):\s+train_loss=(?P<train_loss>{FLOAT_RE})\s+"
    rf"val_loss=(?P<val_loss>{FLOAT_RE})\s+val_mse=(?P<val_mse>{FLOAT_RE})\s+"
    rf"val_dist_m=(?P<val_dist_m>{FLOAT_RE})\s+p10m=(?P<p10m>{FLOAT_RE})\s+"
    rf"p25m=(?P<p25m>{FLOAT_RE})"
)
MULTIHEAD_RE = re.compile(
    rf"Epoch\s+(?P<epoch>\d+):\s+cls_loss=(?P<cls_loss>{FLOAT_RE})\s+"
    rf"reg_loss=(?P<reg_loss>{FLOAT_RE})\s+val_cls_loss=(?P<val_cls_loss>{FLOAT_RE})\s+"
    rf"val_reg_loss=(?P<val_reg_loss>{FLOAT_RE})\s+val_acc=(?P<val_acc>{FLOAT_RE})\s+"
    rf"val_dist_m=(?P<val_dist_m>{FLOAT_RE})\s+val_p10=(?P<val_p10>{FLOAT_RE})\s+"
    rf"val_p25=(?P<val_p25>{FLOAT_RE})"
)

def parse_log(path, pattern):
    data = {}
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        data.setdefault("epoch", []).append(int(match.group("epoch")))
        for key, value in match.groupdict().items():
            if key == "epoch":
                continue
            data.setdefault(key, []).append(float(value))
    return data


def plot_coords_train_val_loss(data, out_path):
    epochs = data["epoch"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epochs, data["train_loss"], label="train_loss")
    ax.plot(epochs, data["val_loss"], label="val_loss")
    ax.set_title("Regression-only: Train vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_multihead_total_loss(data, out_path):
    epochs = data["epoch"]
    train_total = [
        CLS_WEIGHT * cls + COORD_WEIGHT * reg
        for cls, reg in zip(data["cls_loss"], data["reg_loss"])
    ]
    val_total = [
        CLS_WEIGHT * cls + COORD_WEIGHT * reg
        for cls, reg in zip(data["val_cls_loss"], data["val_reg_loss"])
    ]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epochs, train_total, label="train_total_loss")
    ax.plot(epochs, val_total, label="val_total_loss")
    ax.set_title("Multihead: Total Loss (weighted)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_multihead_accuracy(data, out_path):
    epochs = data["epoch"]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epochs, data["val_acc"], label="val_acc")
    ax.set_title("Multihead: Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_comparison(coords_data, multihead_data, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(coords_data["epoch"], coords_data["val_dist_m"], label="regression-only")
    ax.plot(
        multihead_data["epoch"],
        multihead_data["val_dist_m"],
        label="multihead",
    )
    ax.set_title("Validation Distance (m)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Meters")
    ax.legend()

    ax = axes[1]
    ax.plot(coords_data["epoch"], coords_data["p10m"], label="regression-only")
    ax.plot(
        multihead_data["epoch"],
        multihead_data["val_p10"],
        label="multihead",
    )
    ax.set_title("P@10")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction")
    ax.legend()

    ax = axes[2]
    ax.plot(coords_data["epoch"], coords_data["p25m"], label="regression-only")
    ax.plot(
        multihead_data["epoch"],
        multihead_data["val_p25"],
        label="multihead",
    )
    ax.set_title("P@25")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction")
    ax.legend()

    fig.suptitle("Regression-only vs Multihead Comparison", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    coords_path = Path("runs/coords_log.txt")
    multihead_path = Path("runs/multitask_log.txt")
    out_dir = Path("runs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not coords_path.exists():
        print(f"Coords log not found: {coords_path}")
        return
    if not multihead_path.exists():
        print(f"Multihead log not found: {multihead_path}")
        return

    coords_data = parse_log(coords_path, COORDS_RE)
    multihead_data = parse_log(multihead_path, MULTIHEAD_RE)

    if not coords_data.get("epoch"):
        print(f"No coords epochs found in {coords_path}")
        return
    if not multihead_data.get("epoch"):
        print(f"No multihead epochs found in {multihead_path}")
        return

    plot_coords_train_val_loss(coords_data, out_dir / "coords_train_val_loss.png")
    print(f"Wrote {out_dir / 'coords_train_val_loss.png'}")

    plot_multihead_total_loss(multihead_data, out_dir / "multihead_total_loss.png")
    print(f"Wrote {out_dir / 'multihead_total_loss.png'}")

    plot_multihead_accuracy(multihead_data, out_dir / "multihead_accuracy.png")
    print(f"Wrote {out_dir / 'multihead_accuracy.png'}")

    plot_comparison(coords_data, multihead_data, out_dir / "comparison_metrics.png")
    print(f"Wrote {out_dir / 'comparison_metrics.png'}")


if __name__ == "__main__":
    main()
