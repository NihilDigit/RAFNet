#!/usr/bin/env python3
"""Generate qualitative-results visualizations: Appearance-only vs PRISM fusion.

Layout (one scene per row):
  Left column:   Ground Truth
  Middle column: ConvNeXt-only (Appearance)
  Right column:  PRISM (Relation + Appearance, C+Spatial)
  Below each column is a zoomed-in region that highlights improvements from fusion.

Usage:
    pixi run python prism/scripts/qualitative_results.py --gpu 0
    pixi run python prism/scripts/qualitative_results.py --gpu 0 --n-images 3 --seed 43
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["numpy._core"] = np.core
sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

from prism.models import MODELS  # noqa: E402
from prism.utils import load_config  # noqa: E402

# ---------- Constants ----------
LABEL_NAMES = ["Bowing Head", "Listening", "Reading/Writing", "Null", "Using Phone"]
LABEL_COLORS_RGB = {
    0: (76, 175, 80),    # Green  - Bowing Head
    1: (33, 150, 243),   # Blue   - Listening
    2: (255, 152, 0),    # Orange - Reading/Writing
    3: (158, 158, 158),  # Gray   - Null
    4: (244, 67, 54),    # Red    - Using Phone
}
LABEL_COLORS_MPL = {k: tuple(c / 255.0 for c in v) for k, v in LABEL_COLORS_RGB.items()}

DATA_ROOT = ROOT_DIR / "data" / "datasets" / "ncst_classroom"
FEATURES_DIR = ROOT_DIR / "output" / "prism_features" / "20260209_200317"

# The two model configurations we compare
PRISM_MODEL = "grouprec_convnext_gated_c_spatial_graph_loss_tuned"
APPEAR_MODEL = "convnext_only_loss_tuned"

OUTPUT_DIR = ROOT_DIR / "Apr01_Submission" / "Springer_Nature_LaTeX_Template" / "figs"


def load_test_data():
    """Load the NCST test set: features, bbox coordinates, and labels."""
    feat_path = FEATURES_DIR / "test_features_with_context.pkl"
    with open(feat_path, "rb") as f:
        data = pickle.load(f)

    meta = data["metadata"]
    ctx = data["context"]

    grouprec = np.array(data["grouprec"])
    convnext = np.array(data["convnext"])
    labels = np.array(data["labels"])
    img_paths = meta["img_paths"]
    bbox_ids = meta["bbox_ids"]
    bbox_xyxy = np.array(ctx["bbox_xyxy_abs"])
    bbox_ctx = np.array(ctx["bbox_cxcywh_norm"])

    # img_uid
    uid_map = {}
    next_uid = 0
    img_uids = []
    for p in img_paths:
        if p not in uid_map:
            uid_map[p] = next_uid
            next_uid += 1
        img_uids.append([float(uid_map[p])])
    img_uids = np.array(img_uids)

    return {
        "grouprec": grouprec,
        "convnext": convnext,
        "bbox_ctx": bbox_ctx,
        "img_uid": img_uids,
        "labels": labels,
        "img_paths": img_paths,
        "bbox_ids": bbox_ids,
        "bbox_xyxy": bbox_xyxy,
    }


def load_model(model_name: str, seed: int, device: str = "cpu"):
    """Load the specified model."""
    cfg_path = ROOT_DIR / "prism" / "configs" / f"{model_name}.yaml"
    cfg = load_config(str(cfg_path))
    model_cfg = cfg["model"]
    model = MODELS.build(model_cfg["name"], **model_cfg["params"])

    ckpt_path = ROOT_DIR / "results" / "training" / model_name / f"seed_{seed}" / "best_model.pth"
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, model_cfg


def predict_prism(model, data: dict, device: str = "cpu") -> np.ndarray:
    """PRISM (C+Spatial): grouprec + convnext + bbox_ctx + img_uid."""
    features = np.concatenate(
        [data["grouprec"], data["convnext"], data["bbox_ctx"], data["img_uid"]], axis=1,
    )
    return _run_inference(model, features, device, has_entropy=True)


def predict_appear(model, data: dict, device: str = "cpu") -> np.ndarray:
    """ConvNeXt-only: uses only convnext features."""
    return _run_inference(model, data["convnext"], device, has_entropy=False)


def _run_inference(model, features: np.ndarray, device: str, has_entropy: bool) -> np.ndarray:
    features_t = torch.tensor(features, dtype=torch.float32).to(device)
    preds = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(features_t), bs):
            batch = features_t[i : i + bs]
            out = model(batch)
            logits = out[0] if has_entropy else out
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def group_by_image(data: dict, preds_appear: np.ndarray, preds_prism: np.ndarray):
    """Group per image, storing GT, appearance predictions, and PRISM predictions."""
    groups = defaultdict(lambda: {"bboxes": [], "labels": [], "appear": [], "prism": []})
    for i, img_path in enumerate(data["img_paths"]):
        groups[img_path]["bboxes"].append(data["bbox_xyxy"][i])
        groups[img_path]["labels"].append(data["labels"][i])
        groups[img_path]["appear"].append(preds_appear[i])
        groups[img_path]["prism"].append(preds_prism[i])

    for g in groups.values():
        g["bboxes"] = np.array(g["bboxes"])
        g["labels"] = np.array(g["labels"])
        g["appear"] = np.array(g["appear"])
        g["prism"] = np.array(g["prism"])

    return groups


def draw_bboxes(img: np.ndarray, bboxes: np.ndarray, classes: np.ndarray,
                thickness: int = 2, font_scale: float = 0.45) -> np.ndarray:
    """Draw bounding boxes on the image."""
    img_draw = img.copy()
    for bbox, cls in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox.astype(int)
        color = LABEL_COLORS_RGB[int(cls)]
        color_bgr = color[::-1]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color_bgr, thickness)
        label_text = LABEL_NAMES[int(cls)]
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img_draw, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(img_draw, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img_draw


def find_zoom_region(g: dict, img_shape: tuple):
    """Pick a zoom region, preferring dense areas where fusion fixed appearance errors."""
    H, W = img_shape[:2]
    bboxes = g["bboxes"]
    labels = g["labels"]
    appear = g["appear"]
    prism = g["prism"]

    # Find bboxes that fusion corrected (appearance wrong, PRISM right)
    fixed_mask = (appear != labels) & (prism == labels)
    if fixed_mask.sum() == 0:
        # Fallback: use the region where PRISM predicts correctly
        fixed_mask = prism == labels

    fixed_indices = np.where(fixed_mask)[0]
    if len(fixed_indices) == 0:
        # Fallback: center of the image
        cx, cy = W // 2, H // 2
    else:
        # Center on the region with the most corrections
        centers = (bboxes[fixed_indices, :2] + bboxes[fixed_indices, 2:]) / 2
        cx, cy = centers.mean(axis=0).astype(int)

    # Zoom region size: roughly 35% of the image width
    zoom_w = int(W * 0.38)
    zoom_h = int(H * 0.45)
    x1 = max(0, min(cx - zoom_w // 2, W - zoom_w))
    y1 = max(0, min(cy - zoom_h // 2, H - zoom_h))
    x2 = x1 + zoom_w
    y2 = y1 + zoom_h

    return (x1, y1, x2, y2)


def select_images(groups: dict, n: int = 3, rng_seed: int = 42):
    """Select the most illustrative images: prefer those where fusion fixed the most appearance errors."""
    scored = []
    for img_path, g in groups.items():
        n_people = len(g["labels"])
        n_fixed = ((g["appear"] != g["labels"]) & (g["prism"] == g["labels"])).sum()
        n_classes = len(set(g["labels"].tolist()))
        appear_acc = (g["appear"] == g["labels"]).mean()
        prism_acc = (g["prism"] == g["labels"]).mean()
        gain = prism_acc - appear_acc

        # High-score criteria: many corrections, many classes, moderate number of people, positive gain
        score = n_fixed * 8 + n_classes * 5 + gain * 30
        if 15 <= n_people <= 60:
            score += 10
        elif n_people < 8 or n_people > 90:
            score -= 15
        scored.append((score, img_path))

    scored.sort(key=lambda x: -x[0])
    rng = np.random.RandomState(rng_seed)
    candidates = [p for _, p in scored[:max(n * 5, 20)]]
    selected = list(rng.choice(candidates, size=min(n, len(candidates)), replace=False))
    return selected


def make_figure(groups: dict, selected_paths: list, output_path: Path, dpi: int = 200):
    """Build the comparison figure. Each row is one scene; the upper row is the full image
    and the lower row is the zoom crop. Columns: GT | Appearance | PRISM.
    """
    n = len(selected_paths)
    # Each scene takes 2 rows (full + zoom) and 3 columns
    fig, axes = plt.subplots(n * 2, 3, figsize=(15, 4.2 * n), dpi=dpi,
                             gridspec_kw={"height_ratios": [3, 2] * n, "hspace": 0.08, "wspace": 0.03})
    if n == 1:
        axes = axes.reshape(2, 3)
    else:
        axes = axes.reshape(n * 2, 3)

    col_titles = ["Ground Truth", "ConvNeXt-only (Appearance)", "PRISM (Ours)"]

    for idx, img_path in enumerate(selected_paths):
        g = groups[img_path]
        full_path = DATA_ROOT / img_path
        img = cv2.imread(str(full_path))
        if img is None:
            print(f"Warning: cannot read {full_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        appear_acc = (g["appear"] == g["labels"]).mean() * 100
        prism_acc = (g["prism"] == g["labels"]).mean() * 100

        gt_img = draw_bboxes(img_rgb, g["bboxes"], g["labels"])
        appear_img = draw_bboxes(img_rgb, g["bboxes"], g["appear"])
        prism_img = draw_bboxes(img_rgb, g["bboxes"], g["prism"])

        imgs = [gt_img, appear_img, prism_img]
        titles = [
            "Ground Truth",
            f"ConvNeXt-only ({appear_acc:.0f}%)",
            f"PRISM ({prism_acc:.0f}%)",
        ]

        # Zoom region
        zoom = find_zoom_region(g, img_rgb.shape)
        zx1, zy1, zx2, zy2 = zoom

        row_full = idx * 2
        row_zoom = idx * 2 + 1

        for col in range(3):
            # Full image
            ax = axes[row_full, col]
            ax.imshow(imgs[col])
            if idx == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight="bold", pad=6)
            # Draw the zoom-region rectangle
            rect = plt.Rectangle((zx1, zy1), zx2 - zx1, zy2 - zy1,
                                 linewidth=2, edgecolor="yellow", facecolor="none",
                                 linestyle="--")
            ax.add_patch(rect)
            ax.axis("off")

            # Zoom crop
            ax_z = axes[row_zoom, col]
            crop = imgs[col][zy1:zy2, zx1:zx2]
            ax_z.imshow(crop)
            # Yellow border indicates this is the zoom region
            for spine in ax_z.spines.values():
                spine.set_edgecolor("yellow")
                spine.set_linewidth(2)
            ax_z.set_xticks([])
            ax_z.set_yticks([])

            # Annotate accuracy in the bottom-right corner of the zoom crop (middle and right columns)
            if col >= 1:
                acc_val = appear_acc if col == 1 else prism_acc
                ax_z.text(0.97, 0.03, f"{acc_val:.0f}%", transform=ax_z.transAxes,
                          fontsize=11, fontweight="bold", color="white",
                          ha="right", va="bottom",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

        # Scene label (a), (b), (c), ...
        scene_label = chr(ord("a") + idx)
        axes[row_full, 0].text(-0.02, 0.5, f"({scene_label})", transform=axes[row_full, 0].transAxes,
                               fontsize=13, fontweight="bold", va="center", ha="right")

    # Legend
    legend_patches = [
        mpatches.Patch(color=LABEL_COLORS_MPL[i], label=LABEL_NAMES[i])
        for i in range(5)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.subplots_adjust(bottom=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative comparison figure: Appearance vs PRISM")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading test data...")
    data = load_test_data()
    print(f"  Samples: {len(data['labels'])}, Images: {len(set(data['img_paths']))}")

    # Load both models
    print(f"Loading ConvNeXt-only model (seed {args.seed})...")
    model_appear, _ = load_model(APPEAR_MODEL, args.seed, device)

    print(f"Loading PRISM model (seed {args.seed})...")
    model_prism, _ = load_model(PRISM_MODEL, args.seed, device)

    # Inference
    print("Running inference (ConvNeXt-only)...")
    preds_appear = predict_appear(model_appear, data, device)
    appear_acc = (preds_appear == data["labels"]).mean() * 100
    print(f"  ConvNeXt-only accuracy: {appear_acc:.2f}%")

    print("Running inference (PRISM)...")
    preds_prism = predict_prism(model_prism, data, device)
    prism_acc = (preds_prism == data["labels"]).mean() * 100
    print(f"  PRISM accuracy: {prism_acc:.2f}%")

    print("Grouping by image...")
    groups = group_by_image(data, preds_appear, preds_prism)

    print(f"Selecting {args.n_images} images (fusion gain focus)...")
    selected = select_images(groups, n=args.n_images)
    for p in selected:
        g = groups[p]
        a_acc = (g["appear"] == g["labels"]).mean() * 100
        p_acc = (g["prism"] == g["labels"]).mean() * 100
        n_fixed = ((g["appear"] != g["labels"]) & (g["prism"] == g["labels"])).sum()
        print(f"  {p}: {len(g['labels'])} people, appear={a_acc:.0f}%, prism={p_acc:.0f}%, fixed={n_fixed}")

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "ch4_qualitative_results.pdf"
    make_figure(groups, selected, output_path)


if __name__ == "__main__":
    main()
