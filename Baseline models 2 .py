
=============================================================================
  SKIN DISEASE LIGHTWEIGHT MODEL TRAINING  —  v2 (Clean Balanced Dataset)
  -------------------------------------------------------------------------
  Dataset  : SkinDisNet_Clean/  (output of dataset_preparation pipeline)
             Folder structure: one sub-folder per class, already balanced.
             No CSV / metadata merge needed — uses ImageFolder directly.

  Models   : 8 lightweight models covering CNN, hybrid, and ViT families.

  Speed    : Optimised for free-tier Colab T4 GPU
             • AMP (fp16) always enabled
             • num_workers=2, pin_memory, persistent_workers
             • torch.compile() when PyTorch ≥ 2.0
             • OneCycleLR — faster convergence than ReduceLROnPlateau
             • Batch size 64 — maximises GPU utilisation on T4
             • No per-batch tqdm; single summary line per epoch

  Saves    : MODELS_DIR/<ModelName>/best_model.pth   (weights only)
             MODELS_DIR/<ModelName>/history.csv       (epoch log)
             MODELS_DIR/training_summary.csv          (final benchmark)
=============================================================================
"""

# ── 0. INSTALL ────────────────────────────────────────────────────────────────
# !pip install -q timm==1.0.20 scikit-learn==1.7.1

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import os, gc, json, time, random, warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

warnings.filterwarnings("ignore")

# ── 2. PATHS ──────────────────────────────────────────────────────────────────
DATASET_DIR = "/content/drive/MyDrive/Research/SkinDisNet_Clean"
MODELS_DIR  = "/content/drive/MyDrive/Research/SkinDisNet_Models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 3. CONFIG ─────────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 64       # T4 handles 64 comfortably at 224px with fp16
NUM_WORKERS  = 2        # 2 is optimal for free Colab; 0 caused bottleneck
EPOCHS       = 20       # OneCycleLR converges faster; 20 is enough
PATIENCE     = 5        # early stopping patience
LR_MAX       = 3e-4     # OneCycleLR peak LR — well-tested for fine-tuning
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15     # 15 % validation from the balanced dataset
TEST_SPLIT   = 0.15     # 15 % held-out test set
IMG_SIZE     = 224      # all models normalised to 224 for fair comparison

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP  = DEVICE == "cuda"
HAS_COMPILE = hasattr(torch, "compile")   # PyTorch ≥ 2.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True   # auto-tune CUDA kernels

print(f"Device : {DEVICE}  |  AMP : {USE_AMP}  |  "
      f"torch.compile : {HAS_COMPILE}  |  Batch : {BATCH_SIZE}")

# ── 4. MODEL REGISTRY ─────────────────────────────────────────────────────────
# 8 models covering CNN (MobileNet, EfficientNet, GhostNet),
# hybrid (ConvNeXt, FastViT, MobileViT) and pure ViT (EfficientViT) families.
# All are ImageNet-pretrained; ConvNeXt uses the stronger IN-22K checkpoint.
MODEL_MAP = {
    # ── CNNs ──────────────────────────────────────────────────────────────────
    "MobileNetV3"    : "mobilenetv3_small_100.lamb_in1k",       # 2.5 M
    "MobileNetV4"    : "mobilenetv4_conv_small.e1200_r224_in1k", # 3.8 M
    "EfficientNet-B0": "efficientnet_b0.ra_in1k",               # 5.3 M
    "GhostNet"       : "ghostnet_100.in1k",                     # 5.2 M
    # ── Hybrid CNN-ViT ────────────────────────────────────────────────────────
    "ConvNeXt-Tiny"  : "convnext_tiny.fb_in22k_ft_in1k",        # 28 M  (best baseline)
    "FastViT-T8"     : "fastvit_t8.apple_in1k",                 # 3.6 M
    "MobileViT-XXS"  : "mobilevit_xxs.cvnets_in1k",             # 1.3 M
    # ── Lightweight ViT ───────────────────────────────────────────────────────
    "EfficientViT"   : "efficientvit_m0.r224_in1k",             # 2.4 M
}

# ── 5. DATA LOADING ───────────────────────────────────────────────────────────
# The balanced dataset uses a flat ImageFolder structure (class = folder name).
# We perform a stratified patient-agnostic split on file indices since
# the synthetic images do not carry patient IDs.

def make_splits(dataset_dir: str, val_split: float, test_split: float, seed: int):
    """
    Scan ImageFolder structure, return (train_df, val_df, test_df) DataFrames
    with columns [path, label] — stratified by class.
    """
    rows = []
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    label2id = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(dataset_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({
                    "path" : os.path.join(cls_dir, fname),
                    "label": label2id[cls],
                    "class": cls,
                })

    df = pd.DataFrame(rows)

    # Stratified split: test → val → train
    from sklearn.model_selection import train_test_split
    train_val, test_df = train_test_split(
        df, test_size=test_split, random_state=seed, stratify=df["label"]
    )
    rel_val = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=train_val["label"]
    )

    print(f"\nDataset  : {len(df)} images  |  Classes : {len(class_names)}")
    print(f"Train    : {len(train_df)}   Val : {len(val_df)}   Test : {len(test_df)}")
    for cls in class_names:
        tr = (train_df['class'] == cls).sum()
        vl = (val_df['class'] == cls).sum()
        te = (test_df['class'] == cls).sum()
        print(f"  {cls:4s}  train={tr:4d}  val={vl:4d}  test={te:4d}")

    return train_df, val_df, test_df, class_names, label2id


class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths  = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def get_loaders(train_df, val_df, test_df, model_name: str):
    """Build DataLoaders using timm's model-specific transform config."""
    # Resolve transforms from the model's pretraining config
    tmp = create_model(model_name, pretrained=False, num_classes=2)
    cfg = resolve_data_config({}, model=tmp)
    del tmp; gc.collect()

    train_tf = create_transform(
        input_size=cfg["input_size"], is_training=True,
        interpolation=cfg["interpolation"],
        mean=cfg["mean"], std=cfg["std"],
        # timm's auto-augment adds Rand/TrivialAugment — safe to enable
        auto_augment="rand-m7-n2-mstd0.5",
        re_prob=0.1,   # random erasing — replaces CoarseDropout, safe for medical
    )
    val_tf = create_transform(
        input_size=cfg["input_size"], is_training=False,
        interpolation=cfg["interpolation"],
        mean=cfg["mean"], std=cfg["std"],
    )

    loader_kw = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    train_loader = DataLoader(
        SkinDataset(train_df, train_tf), batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(
        SkinDataset(val_df, val_tf), batch_size=BATCH_SIZE,
        shuffle=False, **loader_kw
    )
    test_loader = DataLoader(
        SkinDataset(test_df, val_tf), batch_size=BATCH_SIZE,
        shuffle=False, **loader_kw
    )
    return train_loader, val_loader, test_loader

# ── 6. CLASS WEIGHTS ──────────────────────────────────────────────────────────
def compute_class_weights(train_df: pd.DataFrame, n_classes: int) -> torch.Tensor:
    counts = train_df["label"].value_counts().sort_index().values.astype(float)
    w = len(train_df) / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)

# ── 7. TRAINING UTILITIES ─────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int):
        self.patience   = patience
        self.best_score = None
        self.counter    = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter    = 0
            # Store on CPU to free GPU memory during training
            self.best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self): return self.counter >= self.patience


def train_epoch(model, loader, criterion, optimizer, scaler, scheduler):
    model.train()
    total_loss = n_correct = n_total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), \
                       labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()   # OneCycleLR steps every batch

        total_loss += loss.item() * imgs.size(0)
        n_correct  += (logits.argmax(1) == labels).sum().item()
        n_total    += imgs.size(0)

    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = n_correct = n_total = 0
    all_true, all_pred = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), \
                       labels.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        preds = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        n_correct  += (preds == labels).sum().item()
        n_total    += imgs.size(0)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / n_total
    acc      = n_correct / n_total
    _, _, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    )
    return avg_loss, acc, f1

# ── 8. SINGLE MODEL TRAINING RUN ─────────────────────────────────────────────
def run_model(label: str, model_name: str,
              train_df, val_df, test_df, n_classes: int):

    out_dir = os.path.join(MODELS_DIR, label.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    # ── Skip if already trained ──────────────────────────────────────────────
    done_path = os.path.join(out_dir, "best_model.pth")
    if os.path.exists(done_path):
        print(f"  [{label}] already trained — skipping.")
        res_path = os.path.join(out_dir, "result.json")
        if os.path.exists(res_path):
            with open(res_path) as f:
                return json.load(f)
        return None

    t0 = time.time()
    print(f"\n{'─'*60}")
    print(f"  {label}  ({model_name})")
    print(f"{'─'*60}")

    train_loader, val_loader, _ = get_loaders(train_df, val_df, test_df, model_name)

    # ── Model ────────────────────────────────────────────────────────────────
    model = create_model(model_name, pretrained=True, num_classes=n_classes)
    model = model.to(DEVICE)

    # Keep a reference to the bare (uncompiled) model for state_dict saves.
    # torch.compile wraps keys with '_orig_mod.' which breaks load_state_dict.
    bare_model = model

    # torch.compile — ~15-25 % speedup on T4 when available (PyTorch ≥ 2.0)
    if HAS_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass   # fallback silently if compile fails on this architecture

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_df, n_classes),
        label_smoothing=0.1   # reduces overconfidence; small but consistent gain
    )
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX / 25,
                            weight_decay=WEIGHT_DECAY)

    # OneCycleLR: warms up, peaks, then cosine anneals — best single-cycle recipe
    total_steps = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX,
        total_steps=total_steps,
        pct_start=0.15,         # 15 % warm-up
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1e4,
    )
    scaler  = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    stopper = EarlyStopping(patience=PATIENCE)
    history = []

    # ── Training loop (silent per-batch; one line per epoch) ─────────────────
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc          = train_epoch(model, train_loader, criterion,
                                               optimizer, scaler, scheduler)
        vl_loss, vl_acc, vl_f1  = eval_epoch(model, val_loader, criterion)
        improved = stopper.step(vl_f1, bare_model)   # always save from bare model

        history.append({
            "epoch": epoch, "lr": scheduler.get_last_lr()[0],
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss": round(vl_loss, 4),   "val_acc": round(vl_acc, 4),
            "val_f1_macro": round(vl_f1, 4),
        })
        flag = " ✓" if improved else ""
        print(f"  ep {epoch:02d}/{EPOCHS}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}  "
              f"val_f1={vl_f1:.3f}{flag}")

        if stopper.should_stop:
            print(f"  Early stop at epoch {epoch}.")
            break

    # ── Restore best weights to bare model and save ──────────────────────────
    if stopper.best_state:
        bare_model.load_state_dict(stopper.best_state)
        torch.save(stopper.best_state, done_path)

    train_time = round(time.time() - t0, 1)
    params_m   = round(
        sum(p.numel() for p in bare_model.parameters() if p.requires_grad) / 1e6, 2
    )

    result = {
        "model"          : label,
        "resolved_name"  : model_name,
        "params_million" : params_m,
        "train_time_sec" : train_time,
        "best_val_f1"    : round(stopper.best_score, 4),
        "epochs_run"     : len(history),
        "status"         : "done",
    }

    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, "history.csv"), index=False
    )
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Done in {train_time}s  |  best val F1 = {stopper.best_score:.4f}  "
          f"|  saved → {done_path}")

    del model, bare_model, optimizer, scheduler, scaler, stopper
    gc.collect()
    torch.cuda.empty_cache()

    return result

# ── 9. MAIN ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SKIN DISEASE TRAINING PIPELINE  —  v2")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Models  : {MODELS_DIR}")
    print("=" * 60)

    # Data splits (done once, shared across all models)
    train_df, val_df, test_df, class_names, label2id = make_splits(
        DATASET_DIR, VAL_SPLIT, TEST_SPLIT, SEED
    )
    n_classes = len(class_names)

    # Save split indices for reproducible evaluation later
    split_path = os.path.join(MODELS_DIR, "test_split.csv")
    if not os.path.exists(split_path):
        test_df.to_csv(split_path, index=False)
        pd.DataFrame({"class": class_names}).to_csv(
            os.path.join(MODELS_DIR, "class_names.csv"), index=False
        )
        print(f"\n  Test split saved → {split_path}")

    # Train all models
    all_results = []
    for label, model_name in MODEL_MAP.items():
        try:
            res = run_model(label, model_name,
                            train_df, val_df, test_df, n_classes)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"\n  [ERROR] {label}: {e}")
            all_results.append({
                "model": label, "resolved_name": model_name,
                "status": "failed", "error": str(e)
            })
        gc.collect()
        torch.cuda.empty_cache()

    # Final summary table
    if all_results:
        summary = (pd.DataFrame(all_results)
                   .sort_values("best_val_f1", ascending=False,
                                key=lambda x: pd.to_numeric(x, errors="coerce"))
                   .reset_index(drop=True))
        summary_path = os.path.join(MODELS_DIR, "training_summary.csv")
        summary.to_csv(summary_path, index=False)

        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE — SUMMARY")
        print("=" * 60)
        cols = ["model", "params_million", "train_time_sec",
                "best_val_f1", "epochs_run", "status"]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
        print(f"\n  Summary saved → {summary_path}")

    print("\n  ✅  All models trained. "
          "Run evaluation script next for test-set metrics.")


if __name__ == "__main__":
    main()