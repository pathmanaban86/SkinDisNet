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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# CONFIG
# =========================================================
CFG = dict(
    data_dir="/content/drive/MyDrive/Research/SkinDisNet_Split",
    output_dir="/content/drive/MyDrive/Research/SkinDisNet_Split/ablation",
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

ABLATIONS = [
    {"name": "backbone_only", "use_mscb": False, "use_local": False, "use_attn": False},
    {"name": "backbone_mscb", "use_mscb": True,  "use_local": False, "use_attn": False},
    {"name": "backbone_mscb_local", "use_mscb": True, "use_local": True, "use_attn": False},
    {"name": "full_model", "use_mscb": True, "use_local": True, "use_attn": True},
]

# =========================================================
# UTILS
# =========================================================
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


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


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


# =========================================================
# DATA
# =========================================================
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
    val_ds   = datasets.ImageFolder(os.path.join(cfg["data_dir"], "val"), transform=eval_tfm)
    test_ds  = datasets.ImageFolder(os.path.join(cfg["data_dir"], "test"), transform=eval_tfm)

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
            replacement=True
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    return train_loader, val_loader, test_loader, class_names


# =========================================================
# MIXUP
# =========================================================
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


# =========================================================
# BLOCKS
# =========================================================
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


# =========================================================
# MODEL
# =========================================================
class AblationModel(nn.Module):
    def __init__(self, num_classes, cfg, use_mscb=True, use_local=True, use_attn=True):
        super().__init__()
        self.use_mscb = use_mscb
        self.use_local = use_local
        self.use_attn = use_attn

        self.backbone = timm.create_model(
            cfg["backbone_name"],
            pretrained=cfg["pretrained"],
            features_only=True,
            out_indices=(1, 2, 4),
        )

        chs = self.backbone.feature_info.channels()
        ch_p3, ch_p4, ch_final = chs[0], chs[1], chs[2]

        if self.use_mscb:
            self.mscb = MultiScaleContextBlock(ch_p3, ch_p4, ch_final, out_ch=128)
            feat_dim = 128
        else:
            self.global_proj = nn.Sequential(
                nn.Conv2d(ch_final, 128, kernel_size=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
            )
            feat_dim = 128

        if self.use_local:
            self.local_refine = LocalRefineBlock(channels=feat_dim)

        if self.use_attn:
            self.dual_attn = DualAttention(channels=feat_dim, reduction=8, init_alpha=0.1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(feat_dim),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        p3, p4, final = self.backbone(x)

        if self.use_mscb:
            x = self.mscb(p3, p4, final)
        else:
            x = self.global_proj(final)

        if self.use_local:
            x = self.local_refine(x)

        if self.use_attn:
            x = self.dual_attn(x)

        x = self.pool(x)
        x = self.classifier(x)
        return x


# =========================================================
# OPTIMIZER + SCHEDULER
# =========================================================
def build_optimizer(model, cfg):
    backbone_params, novel_params = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        else:
            novel_params.append(p)

    bb_count = sum(p.numel() for p in backbone_params)
    novel_count = sum(p.numel() for p in novel_params)
    total_count = bb_count + novel_count

    print(f"  Params → total={total_count/1e6:.2f}M  backbone={bb_count/1e6:.2f}M  novel={novel_count/1e6:.2f}M")
    print(f"  Optimizer | bb_lr={cfg['lr_backbone']}  novel_lr={cfg['lr_novel']}")

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg["lr_backbone"], "weight_decay": cfg["weight_decay_backbone"]},
        {"params": novel_params, "lr": cfg["lr_novel"], "weight_decay": cfg["weight_decay_novel"]},
    ])
    return optimizer, total_count


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


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, mixup_alpha):
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

    train_acc = accuracy_score(all_targets, all_preds)
    return loss_meter.avg, train_acc


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


# =========================================================
# SINGLE RUN
# =========================================================
def run_one_ablation(ab_cfg, cfg, train_loader, val_loader, test_loader, class_names):
    name = ab_cfg["name"]
    out_dir = os.path.join(cfg["output_dir"], name)
    ensure_dir(out_dir)

    print("\n" + "=" * 70)
    print(f"Running ablation: {name}")
    print("=" * 70)

    device = torch.device(cfg["device"])

    model = AblationModel(
        num_classes=len(class_names),
        cfg=cfg,
        use_mscb=ab_cfg["use_mscb"],
        use_local=ab_cfg["use_local"],
        use_attn=ab_cfg["use_attn"],
    ).to(device)

    optimizer, total_params = build_optimizer(model, cfg)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=cfg["warmup_epochs"],
        total_epochs=cfg["epochs"],
        base_lrs=[cfg["lr_backbone"], cfg["lr_novel"]],
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_epoch = -1
    best_val_f1 = -1.0
    best_state = None
    wait = 0
    start = time.time()

    history = []

    for epoch in range(cfg["epochs"]):
        scheduler.step(epoch)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, cfg["mixup_alpha"]
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        })

        print(f"{epoch+1:>3}/{cfg['epochs']} | TrLoss {train_loss:.4f} | TrAcc {train_acc:.4f} | VaAcc {val_acc:.4f} | VaF1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            wait = 0
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(out_dir, "best_model_state.pt"))
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch+1}, best epoch {best_epoch}")
                break

    train_time_min = (time.time() - start) / 60.0
    model.load_state_dict(best_state)

    val_loss, val_acc, val_f1, val_targets, val_preds = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc, test_f1, test_targets, test_preds = evaluate(model, test_loader, criterion, device)

    val_cm = confusion_matrix(val_targets, val_preds)
    test_cm = confusion_matrix(test_targets, test_preds)

    summary = {
        "name": name,
        "best_epoch": best_epoch,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "total_params": int(total_params),
        "train_time_minutes": float(train_time_min),
        "use_mscb": ab_cfg["use_mscb"],
        "use_local": ab_cfg["use_local"],
        "use_attn": ab_cfg["use_attn"],
    }

    save_json(summary, os.path.join(out_dir, "summary.json"))
    save_json(history, os.path.join(out_dir, "history.json"))

    with open(os.path.join(out_dir, "val_classification_report.txt"), "w") as f:
        f.write(classification_report(val_targets, val_preds, target_names=class_names, digits=4))
    with open(os.path.join(out_dir, "test_classification_report.txt"), "w") as f:
        f.write(classification_report(test_targets, test_preds, target_names=class_names, digits=4))

    np.savetxt(os.path.join(out_dir, "val_confusion_matrix.csv"), val_cm, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(out_dir, "test_confusion_matrix.csv"), test_cm, delimiter=",", fmt="%d")

    print(json.dumps(summary, indent=2))
    return summary


# =========================================================
# MAIN
# =========================================================
def main():
    seed_everything(CFG["seed"])
    ensure_dir(CFG["output_dir"])

    train_loader, val_loader, test_loader, class_names = build_loaders(CFG)

    all_results = []
    for ab in ABLATIONS:
        result = run_one_ablation(ab, CFG, train_loader, val_loader, test_loader, class_names)
        all_results.append(result)

    all_results = sorted(all_results, key=lambda x: x["test_macro_f1"], reverse=True)

    summary_csv = os.path.join(CFG["output_dir"], "ablation_summary.csv")
    with open(summary_csv, "w") as f:
        f.write("name,use_mscb,use_local,use_attn,total_params,best_epoch,val_acc,val_macro_f1,test_acc,test_macro_f1,train_time_minutes\n")
        for r in all_results:
            f.write(
                f"{r['name']},{r['use_mscb']},{r['use_local']},{r['use_attn']},"
                f"{r['total_params']},{r['best_epoch']},{r['val_acc']:.6f},{r['val_macro_f1']:.6f},"
                f"{r['test_acc']:.6f},{r['test_macro_f1']:.6f},{r['train_time_minutes']:.2f}\n"
            )

    save_json(all_results, os.path.join(CFG["output_dir"], "ablation_summary.json"))

    print("\nFinal Ablation Ranking")
    for i, r in enumerate(all_results, 1):
        print(
            f"{i}. {r['name']} | "
            f"Test Acc: {r['test_acc']:.4f} | "
            f"Test F1: {r['test_macro_f1']:.4f} | "
            f"Params: {r['total_params']}"
        )


if __name__ == "__main__":
    main()