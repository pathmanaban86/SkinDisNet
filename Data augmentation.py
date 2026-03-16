import os
import cv2
import json
import shutil
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

SOURCE_DIR  = "/content/drive/MyDrive/Research/Skin disease"

OUTPUT_DIR  = "/content/drive/MyDrive/Research/SkinDisNet_Clean"

CLASSES     = ["AD", "CD", "EC", "SC", "SD", "TC"]

# ▸ TARGET images per class after balancing
#   Chosen as ~500: above the largest natural class (CD=477) so every class
#   grows, but not so large that minority classes are over-repeated.
TARGET_PER_CLASS = 500

# ▸ Image quality thresholds (derived from image_quality_groups_1.csv analysis)
#   Blur score is Laplacian variance; higher = sharper.
BLUR_VERY_BLURRY_MAX  = 80.0    # images below this are excluded entirely
#   Raised from 50 → 80: the sample image (AD_00204) confirmed that images
#   in the 50–80 range are dominated by noise rather than recoverable blur.
#   Unsharp mask cannot reconstruct signal from a noise floor.
BLUR_BLURRY_MAX       = 120.0   # images in [80,120] get unsharp-mask repair

# ▸ Minimum blur score a CLEAN image must have to be used as an augmentation
#   SOURCE. Even if an image passes the rejection threshold (>80), we only
#   augment FROM images that have enough real texture to produce meaningful
#   synthetic variants. Images with score 80–120 are kept in the dataset as
#   originals (after repair) but are NOT used as augmentation seeds.
AUG_SOURCE_MIN_BLUR   = 120.0
BRIGHTNESS_VERY_DARK  = 40.0    # gray_mean below this → CLAHE correction
BRIGHTNESS_VERY_BRIGHT= 215.0   # gray_mean above this → gamma correction

# ▸ Output image format
OUT_EXT     = ".jpg"
JPEG_QUALITY= 95

# ▸ Reproducibility
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

# ── 3. HELPER FUNCTIONS ───────────────────────────────────────────────────────

def laplacian_variance(img_bgr: np.ndarray) -> float:
    """Compute Laplacian variance (sharpness proxy). Higher = sharper."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def gray_mean(img_bgr: np.ndarray) -> float:
    """Mean pixel intensity of the grayscale image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def apply_unsharp_mask(img_bgr: np.ndarray,
                       strength: float = 1.5,
                       blur_ksize: int = 5) -> np.ndarray:
    """
    Unsharp masking: sharpened = original + strength * (original - blurred).
    Effective for recovering moderate blur in clinical photography.
    """
    blurred = cv2.GaussianBlur(img_bgr, (blur_ksize, blur_ksize), 0)
    sharpened = cv2.addWeighted(img_bgr, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_clahe(img_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel
    of LAB color space. Brightens dark images while preserving color fidelity.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(img_bgr: np.ndarray,
                            gamma: float = 1.5) -> np.ndarray:
    """
    Gamma correction to reduce overexposure. gamma > 1 darkens; gamma < 1 brightens.
    Used for very-bright images.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, table)


def repair_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Run the full quality-repair chain on a single image:
      Step A – fix brightness (dark → CLAHE, bright → gamma)
      Step B – fix blur (blurry → unsharp mask)
    The repaired image is still stored at 512×512 to maintain uniformity.
    """
    bv = gray_mean(img_bgr)
    lv = laplacian_variance(img_bgr)

    # --- Step A: Brightness correction ---
    if bv < BRIGHTNESS_VERY_DARK:
        img_bgr = apply_clahe(img_bgr, clip_limit=3.0)
    elif bv > BRIGHTNESS_VERY_BRIGHT:
        img_bgr = apply_gamma_correction(img_bgr, gamma=1.8)

    # --- Step B: Sharpness correction ---
    if lv < BLUR_BLURRY_MAX:
        # Two-pass unsharp for blurry; single-pass for moderate-blurry
        passes = 2 if lv < BLUR_VERY_BLURRY_MAX * 1.5 else 1
        for _ in range(passes):
            img_bgr = apply_unsharp_mask(img_bgr, strength=1.2)

    return img_bgr


def load_image_bgr(path: str) -> np.ndarray:
    """Load an image with OpenCV (BGR). Returns None on failure."""
    img = cv2.imread(path)
    if img is None:
        img_pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img


def save_image(img_bgr: np.ndarray, out_path: str) -> None:
    """Save a BGR OpenCV image as JPEG at the configured quality."""
    cv2.imwrite(out_path, img_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


# ── 4. AUGMENTATION PIPELINES ─────────────────────────────────────────────────
#
# Three intensity levels are defined:
#   LIGHT  → used for majority classes (CD, EC) to add slight variety
#   MEDIUM → used for mid-size classes (SC, TC)
#   HEAVY  → used for minority classes (AD, SD) to maximally diversify
#
# All augmentations are clinically plausible:
#   ✓ Flips / rotations     — dermoscope orientation varies
#   ✓ Color jitter          — lighting conditions vary between clinics
#   ✓ Elastic distortion    — skin surface texture deformation
#   ✗ CoarseDropout REMOVED — destroys diagnostic texture; harmful in medical imaging
#   ✗ ISONoise REMOVED      — adds artificial noise on top of already-noisy images
#   ✗ NO synthetic textures or GAN artifacts that alter disease morphology

def get_augmentation_pipeline(intensity: str) -> A.Compose:
    """
    Return an Albumentations pipeline at the given intensity level.
    All pipelines output 512×512 RGB images matching the source resolution.
    """

    shared_base = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.7),
    ]

    color_light = [
        A.ColorJitter(brightness=0.15, contrast=0.15,
                      saturation=0.10, hue=0.05, p=0.5),
    ]

    color_medium = [
        A.ColorJitter(brightness=0.25, contrast=0.25,
                      saturation=0.20, hue=0.08, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=20,
                             val_shift_limit=15, p=0.4),
    ]

    color_heavy = [
        A.ColorJitter(brightness=0.35, contrast=0.35,
                      saturation=0.30, hue=0.10, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15,
                             sat_shift_limit=30,
                             val_shift_limit=20, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
    ]

    spatial_light = [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10,
                           rotate_limit=20, p=0.4),
    ]

    spatial_medium = [
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15,
                           rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=40, sigma=5, p=0.3),
    ]

    spatial_heavy = [
        A.ShiftScaleRotate(shift_limit=0.12, scale_limit=0.20,
                           rotate_limit=45, p=0.6),
        A.ElasticTransform(alpha=60, sigma=6, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
    ]

    noise_medium = [
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.25),
    ]

    # NOTE: For heavy intensity (minority medical classes), noise augmentation
    # is intentionally minimal. Adding aggressive noise to already low-quality
    # source images (AD=42 images, SD=56 images) produces meaningless outputs
    # as demonstrated by the syn_AD_00204.jpg artifact.
    # CoarseDropout and ISONoise have been removed entirely from this level.
    noise_heavy = [
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    ]

    resize = [A.Resize(512, 512)]

    if intensity == "light":
        transforms = shared_base + color_light + spatial_light + resize
    elif intensity == "medium":
        transforms = shared_base + color_medium + spatial_medium + noise_medium + resize
    else:  # heavy
        transforms = shared_base + color_heavy + spatial_heavy + noise_heavy + resize

    return A.Compose(transforms)


# Maps each class to its augmentation intensity based on natural count
def get_intensity(natural_count: int) -> str:
    if natural_count < 80:
        return "heavy"
    elif natural_count < 250:
        return "medium"
    else:
        return "light"


# ── 5. PHASE 1 & 2: SCAN, FILTER, REPAIR, COPY ───────────────────────────────

def build_clean_base(source_dir: str,
                     output_dir: str,
                     classes: list) -> pd.DataFrame:
    """
    Scan each class folder, filter out truly unusable (very blurry) images,
    repair remaining images, and copy them into OUTPUT_DIR/<class>/.
    Returns a DataFrame with one row per accepted image and its metadata.
    """
    records = []

    for cls in classes:
        src_cls_dir = os.path.join(source_dir, cls)
        out_cls_dir = os.path.join(output_dir, cls)
        os.makedirs(out_cls_dir, exist_ok=True)

        image_files = [
            f for f in sorted(os.listdir(src_cls_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png",
                                   ".bmp", ".tif", ".tiff"))
        ]

        accepted = 0
        rejected = 0

        for fname in tqdm(image_files, desc=f"[Phase 1-2] {cls}", leave=False):
            src_path = os.path.join(src_cls_dir, fname)
            img = load_image_bgr(src_path)
            if img is None:
                rejected += 1
                continue

            lv = laplacian_variance(img)

            # ── FILTER: discard very blurry images ──────────────────────────
            if lv < BLUR_VERY_BLURRY_MAX:
                rejected += 1
                records.append({
                    "class": cls, "file": fname,
                    "action": "rejected_very_blurry",
                    "blur_score": round(lv, 2),
                    "out_path": None
                })
                continue

            # ── REPAIR: fix remaining brightness/blur issues ─────────────────
            img_repaired = repair_image(img)

            stem = os.path.splitext(fname)[0]
            out_name = f"{stem}{OUT_EXT}"
            out_path = os.path.join(out_cls_dir, out_name)
            save_image(img_repaired, out_path)

            action = "copied_clean" if lv >= BLUR_BLURRY_MAX else "repaired_blur"
            accepted += 1
            records.append({
                "class": cls, "file": fname,
                "action": action,
                "blur_score": round(lv, 2),   # original pre-repair score retained
                "out_path": out_path
            })

        print(f"  {cls}: {accepted} accepted, {rejected} rejected (very blurry)")

    return pd.DataFrame(records)


# ── 6. PHASE 3: BALANCE VIA AUGMENTATION ─────────────────────────────────────

def augment_to_target(clean_df: pd.DataFrame,
                      output_dir: str,
                      classes: list,
                      target: int) -> pd.DataFrame:
    """
    For each class, generate augmented synthetic images until the class folder
    reaches TARGET_PER_CLASS total images.
    Returns a DataFrame with metadata for every synthetic image produced.
    """
    syn_records = []

    for cls in classes:
        cls_df = clean_df[
            (clean_df["class"] == cls) &
            (clean_df["action"].isin(["copied_clean", "repaired_blur"]))
        ].reset_index(drop=True)

        out_cls_dir = os.path.join(output_dir, cls)
        natural_count = len(cls_df)
        needed = target - natural_count

        if needed <= 0:
            # Class already meets or exceeds target (shouldn't happen here,
            # but handle gracefully — keep all natural images, no extra)
            print(f"  {cls}: already {natural_count} images, no augmentation needed.")
            continue

        intensity = get_intensity(natural_count)
        aug_pipeline = get_augmentation_pipeline(intensity)

        # ── SOURCE POOL: only use images above AUG_SOURCE_MIN_BLUR as seeds ──
        # This prevents augmenting FROM noisy/blurry originals, which would
        # only amplify noise rather than create meaningful new variants.
        best_df = cls_df[cls_df["blur_score"] >= AUG_SOURCE_MIN_BLUR]

        if len(best_df) == 0:
            # Fallback: if no image meets the high bar, use the top-50% by score
            median_score = cls_df["blur_score"].median()
            best_df = cls_df[cls_df["blur_score"] >= median_score]
            print(f"  ⚠  {cls}: No images above AUG_SOURCE_MIN_BLUR={AUG_SOURCE_MIN_BLUR}. "
                  f"Falling back to top 50% (score ≥ {median_score:.1f}, n={len(best_df)}).")

        src_paths = best_df["out_path"].tolist()

        print(f"  {cls}: {natural_count} clean → generating {needed} "
              f"synthetic (intensity={intensity}, "
              f"aug_sources={len(src_paths)}/{natural_count})")
        generated = 0

        with tqdm(total=needed, desc=f"[Phase 3] {cls} augment", leave=False) as pbar:
            attempts = 0
            while generated < needed:
                src_path = src_paths[generated % len(src_paths)]
                img_bgr = load_image_bgr(src_path)
                if img_bgr is None:
                    attempts += 1
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                augmented = aug_pipeline(image=img_rgb)["image"]
                aug_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

                # ── POST-AUGMENTATION QUALITY GATE ───────────────────────────
                # Verify the augmented image still has recoverable texture.
                # Spatial transforms (elastic, perspective) can occasionally
                # produce very low-contrast results; discard and retry.
                aug_lv = laplacian_variance(aug_bgr)
                if aug_lv < BLUR_VERY_BLURRY_MAX:
                    attempts += 1
                    if attempts > needed * 3:
                        # Safety valve: avoid infinite loop on very small pools
                        print(f"  ⚠  {cls}: Stopping early at {generated} synthetic "
                              f"images (too many low-quality augmentation results).")
                        break
                    continue  # discard this result and try again

                out_name = f"syn_{cls}_{generated:05d}{OUT_EXT}"
                out_path = os.path.join(out_cls_dir, out_name)
                save_image(aug_bgr, out_path)

                syn_records.append({
                    "class": cls,
                    "file": out_name,
                    "action": f"synthetic_{intensity}",
                    "source_file": os.path.basename(src_path),
                    "out_path": out_path
                })

                generated += 1
                pbar.update(1)

    return pd.DataFrame(syn_records)


# ── 7. PHASE 4: SUMMARY REPORT ───────────────────────────────────────────────

def generate_report(clean_df: pd.DataFrame,
                    syn_df: pd.DataFrame,
                    output_dir: str,
                    classes: list) -> None:
    """
    Print a per-class summary and save JSON + CSV reports to OUTPUT_DIR.
    """
    print("\n" + "=" * 65)
    print("  DATASET PREPARATION REPORT")
    print("=" * 65)

    report_rows = []

    for cls in classes:
        src_total_approx = len(clean_df[clean_df["class"] == cls])
        rejected = len(clean_df[
            (clean_df["class"] == cls) &
            (clean_df["action"] == "rejected_very_blurry")
        ])
        repaired = len(clean_df[
            (clean_df["class"] == cls) &
            (clean_df["action"] == "repaired_blur")
        ])
        clean_copied = len(clean_df[
            (clean_df["class"] == cls) &
            (clean_df["action"] == "copied_clean")
        ])

        synthetic = 0
        if not syn_df.empty:
            synthetic = len(syn_df[syn_df["class"] == cls])

        total_out = clean_copied + repaired + synthetic
        out_dir_cls = os.path.join(output_dir, cls)
        actual_files = len([f for f in os.listdir(out_dir_cls)
                            if f.lower().endswith(OUT_EXT)])

        print(f"\n  CLASS: {cls}")
        print(f"    Source scanned       : {src_total_approx}")
        print(f"    Rejected (v.blurry)  : {rejected}")
        print(f"    Repaired (blur/brt)  : {repaired}")
        print(f"    Clean originals kept : {clean_copied}")
        print(f"    Synthetic generated  : {synthetic}")
        print(f"    Total in output dir  : {actual_files}")

        report_rows.append({
            "class": cls,
            "source_scanned": src_total_approx,
            "rejected_very_blurry": rejected,
            "repaired": repaired,
            "clean_originals": clean_copied,
            "synthetic": synthetic,
            "total_output": actual_files
        })

    print("\n" + "=" * 65)

    # Save CSV
    report_df = pd.DataFrame(report_rows)
    csv_path = os.path.join(output_dir, "dataset_report.csv")
    report_df.to_csv(csv_path, index=False)

    # Save JSON
    report_dict = {
        "output_dir": output_dir,
        "target_per_class": TARGET_PER_CLASS,
        "blur_rejection_threshold": BLUR_VERY_BLURRY_MAX,
        "classes": report_rows
    }
    json_path = os.path.join(output_dir, "dataset_report.json")
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\n  Reports saved to:")
    print(f"    {csv_path}")
    print(f"    {json_path}")
    print("=" * 65)


# ── 8. MAIN PIPELINE ──────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SKIN DISEASE DATASET PREPARATION PIPELINE")
    print(f"  Source : {SOURCE_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Target : {TARGET_PER_CLASS} images per class")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Phase 1 & 2: Quality filter + repair + copy ──────────────────────────
    print("\n[Phase 1 & 2] Quality audit, repair, and clean copy...")
    clean_df = build_clean_base(SOURCE_DIR, OUTPUT_DIR, CLASSES)

    # ── Phase 3: Augment minority and mid classes to TARGET_PER_CLASS ─────────
    print("\n[Phase 3] Balancing classes via augmentation...")
    syn_df = augment_to_target(clean_df, OUTPUT_DIR, CLASSES, TARGET_PER_CLASS)

    # ── Phase 4: Summary report ───────────────────────────────────────────────
    print("\n[Phase 4] Generating summary report...")
    generate_report(clean_df, syn_df, OUTPUT_DIR, CLASSES)

    print("\n✅  Dataset preparation complete.")
    print(f"    Clean balanced dataset is ready at:\n    {OUTPUT_DIR}")


if __name__ == "__main__":
    main()