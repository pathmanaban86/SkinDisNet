import os
import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ── Paths ────────────────────────────────────────────────────
TEST_DIR  = "/content/drive/MyDrive/Research/SkinDisNet_Split/test"
CKPT_PATH = "/content/drive/MyDrive/Research/SkinDisNet_Split/modified/best_model.pt"
# ↑ Adjust filename if yours is named differently, e.g. best_skindisnet.pth

IMG_SIZE  = CFG["img_size"]   # 256
MEAN      = [0.485, 0.456, 0.406]
STD       = [0.229, 0.224, 0.225]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ───────────────────────────────────────────────
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(DEVICE)
model.eval()
print(f"Model loaded — epoch {checkpoint['epoch']}")
print(f"Classes in checkpoint: {checkpoint['class_names']}")

# ── Load test dataset (no augmentation — eval transforms only) ──
eval_tfm = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.10)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tfm)
class_names = test_ds.classes
print(f"\nTest set total : {len(test_ds)} images")
print(f"Classes        : {class_names}")

# ── Split indices by filename prefix ─────────────────────────
# Real images keep their original name (e.g. AD_001.jpg)
# Synthetic images are prefixed with "syn_" by the pipeline
real_idx = [i for i, (p, _) in enumerate(test_ds.samples)
            if not os.path.basename(p).startswith("syn_")]
synt_idx = [i for i, (p, _) in enumerate(test_ds.samples)
            if os.path.basename(p).startswith("syn_")]

print(f"\nReal test images      : {len(real_idx)}")
print(f"Synthetic test images : {len(synt_idx)}")
print(f"Total                 : {len(real_idx) + len(synt_idx)}")

# Sanity check — should match total
assert len(real_idx) + len(synt_idx) == len(test_ds), \
    "WARNING: Some images were not categorised — check filename convention."

# ── Inference function ────────────────────────────────────────
@torch.no_grad()
def run_subset(indices, dataset, model, device):
    preds, labels = [], []
    for i in indices:
        img, label = dataset[i]
        logit = model(img.unsqueeze(0).to(device))
        preds.append(logit.argmax(1).item())
        labels.append(label)
    return np.array(labels), np.array(preds)

# ── Run inference on all three subsets ───────────────────────
print("\nRunning inference... (this may take a few minutes)")

print("\n" + "="*60)
print("  FULL TEST SET  (currently reported in paper)")
print("="*60)
y_true_all, y_pred_all = run_subset(
    list(range(len(test_ds))), test_ds, model, DEVICE)
print(classification_report(y_true_all, y_pred_all,
                             target_names=class_names, digits=4))
print(f"  Accuracy : {accuracy_score(y_true_all, y_pred_all):.4f}")
print(f"  Macro-F1 : {f1_score(y_true_all, y_pred_all, average='macro'):.4f}")

print("\n" + "="*60)
print("  REAL IMAGES ONLY  (key for reviewer)")
print("="*60)
y_true_r, y_pred_r = run_subset(real_idx, test_ds, model, DEVICE)
print(classification_report(y_true_r, y_pred_r,
                             target_names=class_names, digits=4))
print(f"  Accuracy : {accuracy_score(y_true_r, y_pred_r):.4f}")
print(f"  Macro-F1 : {f1_score(y_true_r, y_pred_r, average='macro'):.4f}")

print("\n" + "="*60)
print("  SYNTHETIC IMAGES ONLY")
print("="*60)
y_true_s, y_pred_s = run_subset(synt_idx, test_ds, model, DEVICE)
print(classification_report(y_true_s, y_pred_s,
                             target_names=class_names, digits=4))
print(f"  Accuracy : {accuracy_score(y_true_s, y_pred_s):.4f}")
print(f"  Macro-F1 : {f1_score(y_true_s, y_pred_s, average='macro'):.4f}")

# ── Per-class real count summary ─────────────────────────────
print("\n" + "="*60)
print("  PER-CLASS BREAKDOWN — test set composition")
print("="*60)
print(f"  {'Class':<6}  {'Real':>6}  {'Synthetic':>10}  {'Total':>6}")
print("  " + "-"*34)
for i, cls in enumerate(class_names):
    n_real = sum(1 for j in real_idx if test_ds.samples[j][1] == i)
    n_synt = sum(1 for j in synt_idx if test_ds.samples[j][1] == i)
    print(f"  {cls:<6}  {n_real:>6}  {n_synt:>10}  {n_real+n_synt:>6}")