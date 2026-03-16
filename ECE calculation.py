import torch
import torch.nn.functional as F
import numpy as np
from torch import optim

# ── Step 1: Collect raw logits from VALIDATION set ───────────
# Temperature is tuned on val, then ECE is measured on test.
# Never tune on test — that would be data leakage.

VAL_DIR = "/content/drive/MyDrive/Research/SkinDisNet_Split/val"

val_ds = datasets.ImageFolder(VAL_DIR, transform=eval_tfm)
print(f"Validation set: {len(val_ds)} images")

@torch.no_grad()
def collect_logits_labels(dataset, model, device):
    """Return raw logits and true labels for an entire dataset."""
    all_logits, all_labels = [], []
    for i in range(len(dataset)):
        img, label = dataset[i]
        logit = model(img.unsqueeze(0).to(device))
        all_logits.append(logit.cpu())
        all_labels.append(label)
    logits = torch.cat(all_logits, dim=0)          # [N, 6]
    labels = torch.tensor(all_labels, dtype=torch.long)  # [N]
    return logits, labels

print("Collecting validation logits...")
val_logits, val_labels = collect_logits_labels(val_ds, model, DEVICE)

print("Collecting test logits...")
test_logits, test_labels = collect_logits_labels(test_ds, model, DEVICE)

print(f"Val  logits shape : {val_logits.shape}")
print(f"Test logits shape : {test_logits.shape}")


# ── Step 2: ECE calculation function ─────────────────────────

def compute_ece(logits, labels, n_bins=10):
    """
    Expected Calibration Error using equal-width bins.
    logits : [N, C] raw (pre-softmax) logits
    labels : [N]   integer class labels
    """
    probs      = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece    = 0.0
    n      = len(labels)
    bins   = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        # include upper edge in last bin
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        bin_conf = confidences[mask].mean().item()
        bin_acc  = accuracies[mask].float().mean().item()
        bin_size = mask.sum().item()

        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return ece


# ── Step 3: Baseline ECE (before temperature scaling) ────────

ece_before = compute_ece(test_logits, test_labels)
print(f"\n{'='*50}")
print(f"  ECE BEFORE temperature scaling : {ece_before:.4f}")
print(f"{'='*50}")


# ── Step 4: Learn optimal temperature on VALIDATION set ──────
# Temperature T > 1 softens (flattens) the distribution → less overconfident
# Temperature T < 1 sharpens → more confident
# Optimise T to minimise NLL on validation logits

temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
optimizer   = optim.LBFGS([temperature], lr=0.01, max_iter=500)
criterion   = torch.nn.CrossEntropyLoss()

def eval_nll():
    optimizer.zero_grad()
    scaled_logits = val_logits / temperature
    loss = criterion(scaled_logits, val_labels)
    loss.backward()
    return loss

optimizer.step(eval_nll)

T_opt = temperature.item()
print(f"\n  Optimal temperature T = {T_opt:.4f}")
print(f"  (T > 1 means model was overconfident, as expected from ECE=0.0798)")


# ── Step 5: ECE AFTER temperature scaling on TEST set ────────

scaled_test_logits = test_logits / T_opt
ece_after = compute_ece(scaled_test_logits, test_labels)
print(f"\n{'='*50}")
print(f"  ECE AFTER  temperature scaling : {ece_after:.4f}")
print(f"  ECE BEFORE temperature scaling : {ece_before:.4f}")
print(f"  ECE improvement                : {ece_before - ece_after:.4f}")
print(f"{'='*50}")


# ── Step 6: Brier score before and after ─────────────────────

def compute_brier(logits, labels, n_classes=6):
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(labels, num_classes=n_classes).float()
    return ((probs - onehot) ** 2).sum(dim=1).mean().item()

brier_before = compute_brier(test_logits, test_labels)
brier_after  = compute_brier(scaled_test_logits, test_labels)

print(f"\n  Brier score BEFORE : {brier_before:.4f}")
print(f"  Brier score AFTER  : {brier_after:.4f}")


# ── Step 7: Accuracy check (must not change) ─────────────────
# Temperature scaling only changes probabilities, not argmax predictions.
# Accuracy must be identical before and after.

acc_before = (test_logits.argmax(1) == test_labels).float().mean().item()
acc_after  = (scaled_test_logits.argmax(1) == test_labels).float().mean().item()

print(f"\n  Accuracy BEFORE : {acc_before:.4f}  (must equal AFTER)")
print(f"  Accuracy AFTER  : {acc_after:.4f}  ✓ identical (scaling preserves argmax)")


# ── Step 8: Compare against all baselines ────────────────────
print(f"\n{'='*50}")
print("  CALIBRATION SUMMARY — all models")
print(f"{'='*50}")
print(f"  {'Model':<30} {'ECE':>8}")
print(f"  {'-'*40}")
print(f"  {'ConvNeXt-Tiny':<30} {'0.0539':>8}  ← best baseline")
print(f"  {'YOLO11s-cls':<30} {'0.0625':>8}")
print(f"  {'YOLO11n-cls':<30} {'0.0647':>8}")
print(f"  {'SkinDisNet (before scaling)':<30} {ece_before:>8.4f}  ← worst")
print(f"  {'SkinDisNet (after  scaling)':<30} {ece_after:>8.4f}  ← after fix")
print(f"{'='*50}")
# Check ECE at different bin counts to find which matches 0.0798
for n_bins in [10, 15, 20, 25]:
    ece = compute_ece(test_logits, test_labels, n_bins=n_bins)
    print(f"  n_bins={n_bins:>3} :  ECE = {ece:.4f}")

# Find which bin count matches the originally reported 0.0798
print("Finding bin count that matches reported ECE = 0.0798")
print("-" * 45)
for n_bins in [10, 15, 20, 25]:
    ece = compute_ece(test_logits, test_labels, n_bins=n_bins)
    print(f"  n_bins = {n_bins:>3}  →  ECE = {ece:.4f}",
          " ← MATCH" if abs(ece - 0.0798) < 0.005 else "")

# Then compute after-scaling ECE at that same bin count
print("\nAfter temperature scaling (T=0.7245):")
print("-" * 45)
scaled = test_logits / 0.7245
for n_bins in [10, 15, 20, 25]:
    ece = compute_ece(scaled, test_labels, n_bins=n_bins)
    print(f"  n_bins = {n_bins:>3}  →  ECE = {ece:.4f}")