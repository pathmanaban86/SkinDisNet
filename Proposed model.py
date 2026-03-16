#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SkinDisNet-LGF-Lite
A lightweight skin disease classifier with:
- MobileNetV3-small backbone (timm, features_only)
- Multi-scale fusion
- Local texture/boundary refinement
- Lightweight dual attention

Dataset layout expected:
DATA_DIR/
  train/
  val/
  test/

Default paths:
  /content/drive/MyDrive/Christy/SkinDisNet_Split
Outputs:
  /content/drive/MyDrive/Christy/chatgp/SkinDisNet_Split/modified
"""

import os
import json
import time
import math
import random
import warnings
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

try:
    import timm
except ImportError as e:
    raise ImportError("Please install timm first: pip install timm") from e

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

CFG = dict(
    data_dir="/content/drive/MyDrive/Research/SkinDisNet_Split",
    output_dir="/content/drive/MyDrive/Research/SkinDisNet_Split/modified",
    img_size=256,
    batch_size=32,
    num_workers=2,
    epochs=80,
    patience=18,
    lr_backbone=4e-4,
    lr_novel=8e-4,
    weight_decay_backbone=1e-4,
    weight_decay_novel=5e-4,
    warmup_epochs=3,
    label_smoothing=0.05,
    dropout=0.35,
    mixup_alpha=0.10,
    use_sampler=True,
    pretrained=True,
    seed=42,
    backbone_name="mobilenetv3_small_100",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0
    @property
    def avg(self):
        return self.sum / max(self.count, 1)
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_curves(history, out_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.plot(history["val_f1"], label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy / F1 Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_curve.png"), dpi=200)
    plt.close()

def plot_confmat(cm, class_names, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(7.2, 6.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

def build_loaders(cfg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(cfg["img_size"], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.08)),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize(int(cfg["img_size"] * 1.10)),
        transforms.CenterCrop(cfg["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(cfg["data_dir"], "train"), transform=train_tfm)
    val_ds = datasets.ImageFolder(os.path.join(cfg["data_dir"], "val"), transform=eval_tfm)
    test_ds = datasets.ImageFolder(os.path.join(cfg["data_dir"], "test"), transform=eval_tfm)

    class_names = train_ds.classes
    counts = {c: 0 for c in class_names}
    for _, lab in train_ds.samples:
        counts[class_names[lab]] += 1

    print(f"Device: {cfg['device']}")
    print(f"  Classes : {class_names}")
    print(f"  Train/Val/Test : {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"  Counts  : {counts}")

    sampler = None
    if cfg["use_sampler"]:
        labels = [lab for _, lab in train_ds.samples]
        class_count = np.bincount(labels)
        weights = 1.0 / class_count[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    return train_loader, val_loader, test_loader, class_names

def mixup_data(x, y, alpha=0.1):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MultiScaleContextBlock(nn.Module):
    def __init__(self, ch_p3, ch_p4, ch_final, out_ch=128, reduction=8):
        super().__init__()
        def _proj(in_ch):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(7),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )
        self.proj_p3 = _proj(ch_p3)
        self.proj_p4 = _proj(ch_p4)
        self.proj_final = _proj(ch_final)
        self.scale_w = nn.Parameter(torch.ones(3))
        mid = max(out_ch // reduction, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_ch, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, out_ch, bias=False),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, p3, p4, final):
        w = F.softmax(self.scale_w, dim=0)
        fused = self.proj_p3(p3) * w[0] + self.proj_p4(p4) * w[1] + self.proj_final(final) * w[2]
        fused = self.bn(fused)
        se_w = self.se(fused).view(fused.size(0), fused.size(1), 1, 1)
        return fused * se_w

class LocalRefineBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        squeeze = max(channels // 8, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        res = x
        x = self.dw3(x) + self.dw5(x)
        x = self.bn(x)
        x = self.pw(x)
        x = x * self.se(x)
        return x + res

class DualAttention(nn.Module):
    def __init__(self, channels=128, reduction=8, init_alpha=0.1):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.ch_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.blend = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x):
        ch = self.ch_fc(x).view(x.size(0), x.size(1), 1, 1)
        sp = self.sp_conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1))
        w = F.softmax(self.blend, dim=0)
        mask = w[0] * ch + w[1] * sp
        return x * (1.0 + self.alpha * mask)

class SkinDisNetLGFLite(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg["backbone_name"], pretrained=cfg["pretrained"],
            features_only=True, out_indices=(1, 2, 4),
        )
        chs = self.backbone.feature_info.channels()
        ch_p3, ch_p4, ch_final = chs[0], chs[1], chs[2]
        self.mscb = MultiScaleContextBlock(ch_p3, ch_p4, ch_final, out_ch=128)
        self.local_refine = LocalRefineBlock(channels=128)
        self.dual_attn = DualAttention(channels=128, reduction=8, init_alpha=0.1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(128),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        p3, p4, final = self.backbone(x)
        x = self.mscb(p3, p4, final)
        x = self.local_refine(x)
        x = self.dual_attn(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

def build_optimizer(model, cfg):
    backbone_params, novel_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ["mscb", "local_refine", "dual_attn", "classifier"]):
            novel_params.append(p)
        else:
            backbone_params.append(p)

    bb_count = sum(p.numel() for p in backbone_params)
    novel_count = sum(p.numel() for p in novel_params)
    total_count = bb_count + novel_count
    print(f"  Params → total={total_count/1e6:.2f}M  backbone={bb_count/1e6:.2f}M  novel={novel_count/1e6:.2f}M")
    print(f"  Optimizer | bb_lr={cfg['lr_backbone']}  novel_lr={cfg['lr_novel']} | bb={bb_count/1e6:.2f}M  novel={novel_count/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["lr_backbone"], "weight_decay": cfg["weight_decay_backbone"]},
            {"params": novel_params, "lr": cfg["lr_novel"], "weight_decay": cfg["weight_decay_novel"]},
        ]
    )
    return optimizer

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lrs):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = total_epochs
        self.base_lrs = list(base_lrs)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = float(epoch + 1) / float(self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = max(factor, 1e-3)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor

    def get_lrs(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_targets, all_preds = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return loss_meter.avg, acc, macro_f1, np.array(all_targets), np.array(all_preds)

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, mixup_alpha=0.1):
    model.train()
    loss_meter = AverageMeter()
    all_targets, all_preds = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_alpha > 0:
            images, y_a, y_b, lam = mixup_data(images, targets, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam) if mixup_alpha > 0 else criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    return loss_meter.avg, acc

def save_reports(targets, preds, class_names, out_prefix):
    report = classification_report(targets, preds, target_names=class_names, digits=4, output_dict=True)
    save_json(report, out_prefix + "_classification_report.json")
    cm = confusion_matrix(targets, preds)
    plot_confmat(cm, class_names, out_prefix + "_confusion_matrix.png",
                 title=os.path.basename(out_prefix).replace("_", " ").title())

def main(cfg):
    seed_everything(cfg["seed"])
    ensure_dir(cfg["output_dir"])
    device = torch.device(cfg["device"])

    train_loader, val_loader, test_loader, class_names = build_loaders(cfg)
    num_classes = len(class_names)

    model = SkinDisNetLGFLite(num_classes=num_classes, cfg=cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["epochs"],
        base_lrs=[cfg["lr_backbone"], cfg["lr_novel"]],
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr_backbone": [], "lr_novel": []}
    best_epoch, best_val_f1, wait = -1, -1.0, 0
    best_state = None

    start = time.time()
    print("\n   Epoch    TrLoss    TrAcc    VaLoss    VaAcc     VaF1      LR_bb")
    print("  --------------------------------------------------------------")
    print("  All layers active from epoch 1  (no staged fine-tuning)")

    for epoch in range(cfg["epochs"]):
        scheduler.step(epoch)
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, cfg["mixup_alpha"])
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        lrs = scheduler.get_lrs()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr_backbone"].append(lrs[0])
        history["lr_novel"].append(lrs[1])

        print(f"{epoch+1:>5}/{cfg['epochs']:<3}  {train_loss:>8.4f}  {train_acc:>7.4f}  {val_loss:>8.4f}  {val_acc:>7.4f}  {val_f1:>7.4f}   {lrs[0]:>9.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            wait = 0
            best_state = deepcopy(model.state_dict())
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "cfg": cfg,
                "class_names": class_names,
            }, os.path.join(cfg["output_dir"], "best_model.pt"))
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"\nEarly stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch}")
                break

    train_time_min = (time.time() - start) / 60.0
    print(f"\nTraining complete in {train_time_min:.2f} min. Best epoch: {best_epoch}")

    model.load_state_dict(best_state)

    val_loss, val_acc, val_f1, val_targets, val_preds = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc, test_f1, test_targets, test_preds = evaluate(model, test_loader, criterion, device)

    plot_curves(history, cfg["output_dir"])
    np.savetxt(
        os.path.join(cfg["output_dir"], "training_log.csv"),
        np.column_stack([
            np.arange(1, len(history["train_loss"]) + 1),
            history["train_loss"], history["train_acc"], history["val_loss"],
            history["val_acc"], history["val_f1"], history["lr_backbone"], history["lr_novel"]
        ]),
        delimiter=",",
        header="epoch,train_loss,train_acc,val_loss,val_acc,val_f1,lr_backbone,lr_novel",
        comments=""
    )

    save_reports(val_targets, val_preds, class_names, os.path.join(cfg["output_dir"], "val"))
    save_reports(test_targets, test_preds, class_names, os.path.join(cfg["output_dir"], "test"))

    total_params = sum(p.numel() for p in model.parameters())
    summary = {
        "best_epoch": best_epoch,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "num_classes": num_classes,
        "class_names": class_names,
        "train_count": len(train_loader.dataset),
        "val_count": len(val_loader.dataset),
        "test_count": len(test_loader.dataset),
        "total_params": int(total_params),
        "train_time_minutes": float(train_time_min),
    }
    save_json(summary, os.path.join(cfg["output_dir"], "summary.json"))

    print("\nFinal summary")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main(CFG)
