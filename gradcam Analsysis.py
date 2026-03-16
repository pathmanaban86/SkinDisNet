#!/usr/bin/env python3
# =============================================================================
# ALL-MODELS GradCAM — ConvNeXt-Tiny + Proposed SkinDisNet
# (YOLO is handled separately in gradcam_yolo_only.py)
# Outputs:
#   1. overview_convnext_proposed.png   — 6 classes × 2 models
#   2. proposed_progression_<CLS>.png  — 4-stage pipeline per class
#   3. summary_grid_<CLS>.png          — 3 samples × 2 models per class
# =============================================================================

!pip install -q timm

import os, gc, random, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import timm

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR        = "/content/drive/MyDrive/Research/SkinDisNet_Split"
COMPARISON_DIR  = "/content/drive/MyDrive/Research/SkinDisNet_Split/comparison_models"
PROPOSED_DIR    = "/content/drive/MyDrive/Research/SkinDisNet_Split/modified"
OUT_DIR         = "/content/drive/MyDrive/Research/SkinDisNet_Split/DeepEval"
CAM_DIR         = os.path.join(OUT_DIR, "gradcam_torch_models")
os.makedirs(CAM_DIR, exist_ok=True)

DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE          = 256
NUM_CLASSES       = 6
SAMPLES_PER_CLASS = 3
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# =============================================================================
# FONT SIZES — change once, applies everywhere
# =============================================================================
FS_SUPERTITLE = 22
FS_COL_HEADER = 17
FS_ROW_LABEL  = 15
FS_PRED_LABEL = 14
FIG_BG        = "#F5F5F5"

BORDER_COLORS = {
    "ConvNeXt-Tiny"       : "#2166AC",   # blue
    "Proposed SkinDisNet" : "#D6604D",   # red-orange
}

# 4 stage colours (cool → hot) for progression figure
STAGE_COLORS = ["#4575B4", "#74ADD1", "#F46D43", "#D73027"]


# =============================================================================
# PROPOSED MODEL  (exact match to training script)
# =============================================================================
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
        self.proj_p3    = _proj(ch_p3)
        self.proj_p4    = _proj(ch_p4)
        self.proj_final = _proj(ch_final)
        self.scale_w    = nn.Parameter(torch.ones(3))
        mid = max(out_ch // reduction, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(out_ch, mid, bias=False), nn.SiLU(inplace=True),
            nn.Linear(mid, out_ch, bias=False), nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, p3, p4, final):
        w     = F.softmax(self.scale_w, dim=0)
        fused = self.proj_p3(p3)*w[0] + self.proj_p4(p4)*w[1] + self.proj_final(final)*w[2]
        fused = self.bn(fused)
        se_w  = self.se(fused).view(fused.size(0), fused.size(1), 1, 1)
        return fused * se_w


class LocalRefineBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.bn  = nn.BatchNorm2d(channels)
        self.pw  = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(inplace=True),
        )
        sq = max(channels // 8, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, sq, 1), nn.SiLU(inplace=True),
            nn.Conv2d(sq, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        res = x
        x   = self.dw3(x) + self.dw5(x)
        x   = self.bn(x)
        x   = self.pw(x)
        x   = x * self.se(x)
        return x + res


class DualAttention(nn.Module):
    def __init__(self, channels=128, reduction=8, init_alpha=0.1):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.ch_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid, bias=False), nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid(),
        )
        self.sp_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid(),
        )
        self.blend = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        ch   = self.ch_fc(x).view(x.size(0), x.size(1), 1, 1)
        sp   = self.sp_conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1))
        w    = F.softmax(self.blend, dim=0)
        mask = w[0]*ch + w[1]*sp
        return x * (1.0 + self.alpha * mask)


class SkinDisNetLGFLite(nn.Module):
    def __init__(self, num_classes=6, backbone_name="mobilenetv3_small_100",
                 pretrained=False, dropout=0.35):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained,
            features_only=True, out_indices=(1, 2, 4),
        )
        chs = self.backbone.feature_info.channels()
        self.mscb         = MultiScaleContextBlock(chs[0], chs[1], chs[2], out_ch=128)
        self.local_refine = LocalRefineBlock(128)
        self.dual_attn    = DualAttention(128, reduction=8, init_alpha=0.1)
        self.pool         = nn.AdaptiveAvgPool2d(1)
        self.classifier   = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(128), nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        p3, p4, final = self.backbone(x)
        x = self.mscb(p3, p4, final)
        x = self.local_refine(x)
        x = self.dual_attn(x)
        x = self.pool(x)
        return self.classifier(x)


# =============================================================================
# DATASET
# =============================================================================
eval_tfm = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.10)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_ds     = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_tfm)
class_names = test_ds.classes
print("Classes:", class_names)


# =============================================================================
# MANUAL GRADCAM++  (pure PyTorch — no external library)
# =============================================================================
def manual_gradcam_pp(model, target_layer, tensor, target_cls):
    act = {}; grad = {}

    h1 = target_layer.register_forward_hook(
        lambda m, i, o: act.__setitem__('x', o[0] if isinstance(o, tuple) else o))
    h2 = target_layer.register_full_backward_hook(
        lambda m, gi, go: grad.__setitem__('x', go[0] if isinstance(go, tuple) else go))

    model.zero_grad()
    try:
        logits = model(tensor)
        logits[0, target_cls].backward()
    except Exception as e:
        h1.remove(); h2.remove()
        print(f"    ✗ backward: {e}")
        return None

    h1.remove(); h2.remove()

    a = act.get('x'); g = grad.get('x')
    if a is None or g is None or a.dim() != 4 or a.shape[2] <= 1:
        print(f"    ✗ hook miss or non-spatial  shape={a.shape if a is not None else None}")
        return None

    a = a.detach().float(); g = g.detach().float()
    g2 = g**2; g3 = g**3
    alpha   = g2 / (2*g2 + a.sum(dim=[2,3], keepdim=True)*g3 + 1e-8)
    weights = (alpha * F.relu(g)).mean(dim=[2,3], keepdim=True)
    cam = F.relu((weights * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    if cam.max() < 1e-7:
        return None
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(rgb, cam, alpha=0.55):
    import matplotlib.cm as cm
    heat = cm.inferno(cam)[..., :3]
    return np.clip(alpha*heat + (1-alpha)*rgb, 0, 1)


# =============================================================================
# HELPERS
# =============================================================================
def load_image(path):
    img  = Image.open(path).convert("RGB")
    rgb  = np.array(img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    t    = eval_tfm(img).unsqueeze(0).to(DEVICE)
    return t, rgb

def get_pred(model, tensor):
    with torch.no_grad():
        idx = model(tensor).argmax(1).item()
    return idx, class_names[idx]

def add_spine(ax, color, lw=6):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(lw)

def hide_ticks(ax):
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def sample_per_class(ds, n=SAMPLES_PER_CLASS, seed=42):
    random.seed(seed)
    by = {}
    for path, lbl in ds.samples:
        by.setdefault(lbl, []).append(path)
    out = []
    for lbl, paths in sorted(by.items()):
        for p in random.sample(paths, min(n, len(paths))):
            out.append((p, lbl))
    return out


# =============================================================================
# LOAD MODELS
# =============================================================================
print("\n─── Loading models ───")
model_entries = []   # (name, model, target_layer)

# ConvNeXt-Tiny
conv_path = os.path.join(COMPARISON_DIR, "convnext_tiny", "best.pt")
if os.path.exists(conv_path):
    conv = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False,
                              num_classes=NUM_CLASSES, drop_rate=0.35)
    conv.load_state_dict(torch.load(conv_path, map_location="cpu"))
    conv = conv.to(DEVICE).eval()
    tl_conv = conv.stages[-1].blocks[-1].conv_dw
    print(f"ConvNeXt-Tiny target layer: {tl_conv.__class__.__name__}  k={tl_conv.kernel_size}  ch={tl_conv.out_channels}")
    model_entries.append(("ConvNeXt-Tiny", conv, tl_conv))

# Proposed
prop_path = os.path.join(PROPOSED_DIR, "best_model.pt")
proposed = None
if os.path.exists(prop_path):
    ckpt     = torch.load(prop_path, map_location="cpu")
    proposed = SkinDisNetLGFLite(num_classes=NUM_CLASSES, backbone_name="mobilenetv3_small_100",
                                  pretrained=False, dropout=0.35)
    proposed.load_state_dict(ckpt.get("model_state_dict", ckpt))
    proposed = proposed.to(DEVICE).eval()
    tl_prop  = proposed.local_refine.dw3
    print(f"Proposed SkinDisNet target: {tl_prop.__class__.__name__}  k={tl_prop.kernel_size}  ch={tl_prop.out_channels}")
    model_entries.append(("Proposed SkinDisNet", proposed, tl_prop))

print(f"\nModels ready: {[m[0] for m in model_entries]}")


# =============================================================================
# 4-STAGE PROGRESSION LAYERS for Proposed model
# Shows how each novel block progressively sharpens the attention map
# =============================================================================
PROG_STAGES = []
if proposed is not None:
    PROG_STAGES = [
        {
            "name"  : "Stage 1\nBackbone Projection\n(Deep feature → 128ch)",
            "layer" : proposed.mscb.proj_final[1],      # Conv2d(ch_final→128)
            "color" : STAGE_COLORS[0],
            "note"  : "Raw deep backbone\nfeatures (unfused)",
        },
        {
            "name"  : "Stage 2\nMSCB Fusion\n(P3+P4+Final blend)",
            "layer" : proposed.mscb.bn,                  # After weighted multi-scale fusion
            "color" : STAGE_COLORS[1],
            "note"  : "Multi-scale weighted\nfusion output",
        },
        {
            "name"  : "Stage 3\nLocal Refinement\n(3×3 depthwise texture)",
            "layer" : proposed.local_refine.dw3,         # 3×3 depthwise — texture boundary
            "color" : STAGE_COLORS[2],
            "note"  : "Lesion boundary &\ntexture captured",
        },
        {
            "name"  : "Stage 4\nChannel-Mix\n(1×1 pointwise pre-attn)",
            "layer" : proposed.local_refine.pw[0],       # 1×1 pointwise
            "color" : STAGE_COLORS[3],
            "note"  : "Final channel-mixed\nfeatures (sharpest)",
        },
    ]


# =============================================================================
# SAMPLE IMAGES
# =============================================================================
samples        = sample_per_class(test_ds)
by_cls_samples = {}
for p, lbl in samples:
    by_cls_samples.setdefault(lbl, []).append(p)


# =============================================================================
# FIGURE 1 — OVERVIEW: 6 classes × (Original + ConvNeXt + Proposed)
# =============================================================================
print("\n─── Figure 1: Overview ───")

n_models = len(model_entries)
ncols    = 1 + n_models
nrows    = len(class_names)

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(6 * ncols, 6 * nrows),
                          squeeze=False)
fig.patch.set_facecolor(FIG_BG)

for col, lbl in enumerate(["Original"] + [e[0] for e in model_entries]):
    color = "#222" if col == 0 else BORDER_COLORS.get(lbl, "#333")
    axes[0][col].set_title(lbl, fontsize=FS_COL_HEADER, fontweight="bold", color=color, pad=14)

for row, cls_idx in enumerate(range(len(class_names))):
    cls_name = class_names[cls_idx]
    paths    = by_cls_samples.get(cls_idx, [])
    if not paths: continue
    tensor, rgb_np = load_image(paths[0])

    ax0 = axes[row][0]
    ax0.imshow(rgb_np)
    ax0.set_ylabel(cls_name, fontsize=FS_ROW_LABEL+2, fontweight="bold", rotation=90, labelpad=10)
    hide_ticks(ax0)
    for sp in ax0.spines.values(): sp.set_visible(False)

    for col, (name, model, tl) in enumerate(model_entries, start=1):
        ax  = axes[row][col]
        t   = tensor.clone().detach().requires_grad_(True)
        cam = manual_gradcam_pp(model, tl, t, cls_idx)

        if cam is not None:
            vis = overlay(rgb_np, cam)
            pidx, pname = get_pred(model, tensor)
            ok   = "✓" if pidx == cls_idx else "✗"
            pcol = "#1a7f2e" if pidx == cls_idx else "#c0392b"
            ax.imshow(vis)
            ax.set_xlabel(f"Pred: {pname} {ok}", fontsize=FS_PRED_LABEL+1,
                          fontweight="bold", color=pcol, labelpad=6)
            add_spine(ax, BORDER_COLORS.get(name, "#888"), lw=6)
        else:
            ax.set_facecolor("#fff0f0")
            ax.text(0.5, 0.5, "GradCAM\nfailed", ha="center", va="center",
                    fontsize=FS_PRED_LABEL, color="#c0392b", transform=ax.transAxes)
        hide_ticks(ax)

fig.suptitle("GradCAM++ — ConvNeXt-Tiny vs. Proposed SkinDisNet  |  All Classes",
             fontsize=FS_SUPERTITLE, fontweight="bold", y=1.002)
plt.tight_layout(h_pad=2.5, w_pad=1.2)
p1 = os.path.join(CAM_DIR, "overview_convnext_proposed.png")
fig.savefig(p1, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
plt.close(fig)
print(f"  ✅ Overview → {p1}")


# =============================================================================
# FIGURE 2 — PROPOSED 4-STAGE PIPELINE PROGRESSION (per class)
# =============================================================================
if PROG_STAGES and proposed is not None:
    print("\n─── Figure 2: Proposed 4-Stage Progression ───")
    n_stages = len(PROG_STAGES)

    for cls_idx, cls_name in enumerate(class_names):
        paths = by_cls_samples.get(cls_idx, [])
        if not paths: continue
        nr = len(paths)
        nc = 1 + n_stages

        fig, axes = plt.subplots(nr, nc,
                                  figsize=(6 * nc, 6 * nr),
                                  squeeze=False)
        fig.patch.set_facecolor(FIG_BG)

        # Col headers
        axes[0][0].set_title("Original\nImage", fontsize=FS_COL_HEADER,
                              fontweight="bold", color="#222", pad=14)
        for si, stg in enumerate(PROG_STAGES, start=1):
            axes[0][si].set_title(stg["name"], fontsize=FS_COL_HEADER,
                                   fontweight="bold", color=stg["color"], pad=14)

        for row, img_path in enumerate(paths):
            tensor, rgb_np = load_image(img_path)
            pidx, pname    = get_pred(proposed, tensor)
            ok    = "✓" if pidx == cls_idx else "✗"
            pcol  = "#1a7f2e" if pidx == cls_idx else "#c0392b"

            ax0 = axes[row][0]
            ax0.imshow(rgb_np)
            ax0.set_ylabel(f"Sample {row+1}", fontsize=FS_ROW_LABEL,
                           fontweight="bold", rotation=90, labelpad=10)
            ax0.set_xlabel(f"True: {cls_name}", fontsize=FS_PRED_LABEL,
                           fontweight="bold", color="#333", labelpad=6)
            hide_ticks(ax0)
            for sp in ax0.spines.values(): sp.set_visible(False)

            for si, stg in enumerate(PROG_STAGES, start=1):
                ax  = axes[row][si]
                t   = tensor.clone().detach().requires_grad_(True)
                cam = manual_gradcam_pp(proposed, stg["layer"], t, cls_idx)

                if cam is not None:
                    vis = overlay(rgb_np, cam)
                    ax.imshow(vis)
                    # Show pred label on last stage only
                    if si == n_stages:
                        ax.set_xlabel(f"Pred: {pname} {ok}",
                                      fontsize=FS_PRED_LABEL+1, fontweight="bold",
                                      color=pcol, labelpad=6)
                    # Small annotation inside cell
                    ax.text(0.03, 0.97, stg["note"], transform=ax.transAxes,
                            fontsize=10, color="white", va="top", ha="left",
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.55))
                    add_spine(ax, stg["color"], lw=7)
                else:
                    ax.set_facecolor("#fff0f0")
                    ax.text(0.5, 0.5, "failed", ha="center", va="center",
                            fontsize=FS_PRED_LABEL, color="#c0392b", transform=ax.transAxes)
                hide_ticks(ax)

        fig.suptitle(
            f"Proposed SkinDisNet — GradCAM++ Block-by-Block Progression  |  Class: {cls_name}\n"
            "Stage 1: Backbone Proj → Stage 2: MSCB Fusion → "
            "Stage 3: 3×3 Local Texture → Stage 4: 1×1 Channel-Mix",
            fontsize=FS_SUPERTITLE - 1, fontweight="bold", y=1.02,
        )
        plt.tight_layout(h_pad=2.5, w_pad=1.2)
        pp = os.path.join(CAM_DIR, f"proposed_progression_{cls_name}.png")
        fig.savefig(pp, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
        plt.close(fig)
        print(f"  ✅ Progression [{cls_name}] → {pp}")


# =============================================================================
# FIGURE 3 — PER-CLASS SUMMARY GRIDS (3 samples × 2 models)
# =============================================================================
print("\n─── Figure 3: Summary Grids ───")

for cls_idx, cls_name in enumerate(class_names):
    paths = by_cls_samples.get(cls_idx, [])
    if not paths: continue
    nr = len(paths)
    nc = 1 + n_models

    fig, axes = plt.subplots(nr, nc,
                              figsize=(6 * nc, 6 * nr),
                              squeeze=False)
    fig.patch.set_facecolor(FIG_BG)
    fig.suptitle(f"GradCAM++ Summary — Class: {cls_name}",
                 fontsize=FS_SUPERTITLE, fontweight="bold", y=1.01)

    for col, lbl in enumerate(["Original"] + [e[0] for e in model_entries]):
        color = "#222" if col == 0 else BORDER_COLORS.get(lbl, "#333")
        axes[0][col].set_title(lbl, fontsize=FS_COL_HEADER, fontweight="bold", color=color, pad=12)

    for row, img_path in enumerate(paths):
        tensor, rgb_np = load_image(img_path)

        ax0 = axes[row][0]
        ax0.imshow(rgb_np)
        ax0.set_ylabel(f"Sample {row+1}", fontsize=FS_ROW_LABEL,
                       fontweight="bold", rotation=90, labelpad=10)
        hide_ticks(ax0)
        for sp in ax0.spines.values(): sp.set_visible(False)

        for col, (name, model, tl) in enumerate(model_entries, start=1):
            ax  = axes[row][col]
            t   = tensor.clone().detach().requires_grad_(True)
            cam = manual_gradcam_pp(model, tl, t, cls_idx)

            if cam is not None:
                vis = overlay(rgb_np, cam)
                pidx, pname = get_pred(model, tensor)
                ok   = "✓" if pidx == cls_idx else "✗"
                pcol = "#1a7f2e" if pidx == cls_idx else "#c0392b"
                ax.imshow(vis)
                ax.set_xlabel(f"Pred: {pname} {ok}", fontsize=FS_PRED_LABEL+1,
                              fontweight="bold", color=pcol, labelpad=6)
                add_spine(ax, BORDER_COLORS.get(name, "#888"), lw=6)
            else:
                ax.set_facecolor("#fff0f0")
                ax.text(0.5, 0.5, "failed", ha="center", va="center",
                        fontsize=FS_PRED_LABEL, color="#c0392b", transform=ax.transAxes)
            hide_ticks(ax)

    plt.tight_layout(h_pad=2.5, w_pad=1.2)
    gp = os.path.join(CAM_DIR, f"summary_grid_{cls_name}.png")
    fig.savefig(gp, dpi=160, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"  ✅ Grid [{cls_name}] → {gp}")


# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print(f"All torch-model GradCAM outputs → {CAM_DIR}")
for f in sorted(os.listdir(CAM_DIR)):
    print(f"  {f}")
print(f"{'='*60}")