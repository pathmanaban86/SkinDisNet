import os
import json
import time
import math
import random
import warnings
from copy import deepcopy

import numpy as np
from PIL import ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import timm
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


CFG = dict(
    data_dir="/content/drive/MyDrive/Research/SkinDisNet_Split",
    output_dir="/content/drive/MyDrive/Research/SkinDisNet_Split/comparison_models",
    img_size=256,
    batch_size=32,
    num_workers=2,
    epochs=80,
    patience=18,
    lr_backbone=4e-4,
    weight_decay_backbone=1e-4,
    warmup_epochs=3,
    label_smoothing=0.05,
    dropout=0.35,
    mixup_alpha=0.10,
    use_sampler=True,
    pretrained=True,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

os.makedirs(CFG["output_dir"], exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG["seed"])


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
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
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


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = float(epoch + 1) / float(self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = max(factor, 1e-3)

        lr = self.base_lr * factor
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, mixup_alpha=0.10):
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
            if mixup_alpha > 0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    return loss_meter.avg, acc


@torch.no_grad()
def evaluate_torch_model(model, loader, criterion, device, class_names):
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

    return {
        "loss": float(loss_meter.avg),
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "f1_macro": float(f1_score(all_targets, all_preds, average="macro")),
        "precision_macro": float(precision_score(all_targets, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_targets, all_preds, average="macro", zero_division=0)),
        "classification_report": classification_report(
            all_targets, all_preds, target_names=class_names, digits=4, zero_division=0
        )
    }


def train_convnext_tiny(cfg):
    print("\n" + "=" * 70)
    print("Training ConvNeXt-Tiny")
    print("=" * 70)

    out_dir = os.path.join(cfg["output_dir"], "convnext_tiny")
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = build_loaders(cfg)
    num_classes = len(class_names)
    device = torch.device(cfg["device"])

    model = timm.create_model(
        "convnext_tiny.fb_in22k_ft_in1k",
        pretrained=cfg["pretrained"],
        num_classes=num_classes,
        drop_rate=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr_backbone"],
        weight_decay=cfg["weight_decay_backbone"]
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["epochs"],
        base_lr=cfg["lr_backbone"]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_epoch = -1
    best_val_f1 = -1.0
    best_state = None
    wait = 0
    start = time.time()

    for epoch in range(cfg["epochs"]):
        scheduler.step(epoch)

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, cfg["mixup_alpha"]
        )
        val_metrics = evaluate_torch_model(model, val_loader, criterion, device, class_names)

        print(
            f"Epoch {epoch+1:>3}/{cfg['epochs']} | "
            f"TrLoss {tr_loss:.4f} | TrAcc {tr_acc:.4f} | "
            f"VaAcc {val_metrics['accuracy']:.4f} | VaF1 {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch + 1
            wait = 0
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(out_dir, "best.pt"))
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}, best epoch {best_epoch}")
                break

    model.load_state_dict(best_state)

    val_metrics = evaluate_torch_model(model, val_loader, criterion, device, class_names)
    test_metrics = evaluate_torch_model(model, test_loader, criterion, device, class_names)

    summary = {
        "model": "convnext_tiny",
        "best_epoch": best_epoch,
        "train_time_minutes": (time.time() - start) / 60.0,
        "val": val_metrics,
        "test": test_metrics,
        "params_million": sum(p.numel() for p in model.parameters()) / 1e6,
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))
    with open(os.path.join(out_dir, "test_report.txt"), "w") as f:
        f.write(test_metrics["classification_report"])

    return summary

def train_yolo_variant(model_name, pretrained_pt, cfg):
    print("\n" + "=" * 70)
    print(f"Training {model_name}")
    print("=" * 70)

    out_dir = os.path.join(cfg["output_dir"], model_name)
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(pretrained_pt)

    # Note:
    # Ultralytics classification pipeline has its own internal transforms.
    # These settings are matched as closely as possible to the custom CFG.
    results = model.train(
        data=cfg["data_dir"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        imgsz=cfg["img_size"],
        batch=cfg["batch_size"],
        workers=cfg["num_workers"],
        device=0 if "cuda" in cfg["device"] else "cpu",
        pretrained=cfg["pretrained"],
        project=cfg["output_dir"],
        name=model_name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=cfg["lr_backbone"],
        weight_decay=cfg["weight_decay_backbone"],
        dropout=cfg["dropout"],
        label_smoothing=cfg["label_smoothing"],
        seed=cfg["seed"],
        amp=True,
        verbose=True,

        # keep augmentation moderate, closer to your proposed model
        augment=True,
        degrees=12.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.02,
        hsv_s=0.10,
        hsv_v=0.15,
        translate=0.0,
        scale=0.10,
        erasing=0.10,
        mixup=cfg["mixup_alpha"],
        auto_augment=None,
    )

    best_pt = os.path.join(cfg["output_dir"], model_name, "weights", "best.pt")

    # Evaluate on val
    val_metrics = model.val(
        data=cfg["data_dir"],
        split="val",
        imgsz=cfg["img_size"],
        batch=cfg["batch_size"],
        device=0 if "cuda" in cfg["device"] else "cpu",
    )

    # Evaluate on test
    test_metrics = model.val(
        data=cfg["data_dir"],
        split="test",
        imgsz=cfg["img_size"],
        batch=cfg["batch_size"],
        device=0 if "cuda" in cfg["device"] else "cpu",
    )

    summary = {
        "model": model_name,
        "best_weights": best_pt,
        "val_top1": float(val_metrics.top1),
        "val_top5": float(val_metrics.top5),
        "test_top1": float(test_metrics.top1),
        "test_top5": float(test_metrics.top5),
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))
    return summary


if __name__ == "__main__":
    all_results = {}

    # YOLO11n
    all_results["yolo11n_cls"] = train_yolo_variant(
        "yolo11n_cls",
        "yolo11n-cls.pt",
        CFG
    )

    # YOLO11s
    all_results["yolo11s_cls"] = train_yolo_variant(
        "yolo11s_cls",
        "yolo11s-cls.pt",
        CFG
    )

    # ConvNeXt-Tiny
    all_results["convnext_tiny"] = train_convnext_tiny(CFG)

    save_json(all_results, os.path.join(CFG["output_dir"], "all_model_summaries.json"))

    print("\nFinal comparison summaries saved to:")
    print(os.path.join(CFG["output_dir"], "all_model_summaries.json"))