import os
import pickle
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

DATA_ROOT = Path('data/datasets/ncst_classroom')
PICKLE_PATH = DATA_ROOT / 'train.pkl'
IMAGES_DIR = DATA_ROOT
ASSETS_DIR = Path('docs/pipeline/assets')
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_sample(n_try: int = 20):
    with open(PICKLE_PATH, 'rb') as f:
        d = pickle.load(f)
    images = d['images']
    # pick one with enough people (>= 10)
    for item in images[:n_try]:
        if len(item.get('bboxes', [])) >= 12:
            return item
    return images[0]


def crop_patches(img: np.ndarray, bboxes: List[dict], k: int = 6) -> List[np.ndarray]:
    # sort by area descending, take top-k and center-crop with a little margin
    def area(bb):
        (x1, y1), (x2, y2) = bb['bbox']
        return max(0, (x2 - x1)) * max(0, (y2 - y1))

    sel = sorted(bboxes, key=area, reverse=True)[:k]
    patches_img = []
    H, W = img.shape[:2]
    for bb in sel:
        (x1, y1), (x2, y2) = bb['bbox']
        x1 = int(max(0, x1)); y1 = int(max(0, y1))
        x2 = int(min(W - 1, x2)); y2 = int(min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        patches_img.append(crop)
    return patches_img


def draw_pipeline_figure(class_img: np.ndarray, patches_img: List[np.ndarray], out_path: Path):
    fig = plt.figure(figsize=(16, 2.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Panel (a) boundary
    ax.add_patch(patches.Rectangle((0.01, 0.05), 0.36, 0.90, fill=False, ls='--', lw=2, ec='#c76f28'))
    ax.text(0.02, 0.04, '(a) Individual Multimodal Feature Extraction', fontsize=9, color='#c76f28')

    # Classroom image
    ax_im = fig.add_axes([0.03, 0.50, 0.12, 0.42])
    ax_im.imshow(class_img)
    ax_im.set_title('Classroom Image', fontsize=8)
    ax_im.axis('off')

    # Extracting students arrow
    ax.annotate('', xy=(0.145, 0.51), xytext=(0.09, 0.51),
                arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'))
    ax.text(0.10, 0.47, 'Extracting Students', fontsize=7, color='gray')

    # Extracted students grid
    grid_x0, grid_y0 = 0.03, 0.06
    gw, gh = 0.06, 0.18
    for i, p in enumerate(patches_img[:6]):
        gx = grid_x0 + (i % 3) * (gw + 0.01)
        gy = grid_y0 + (i // 3) * (gh + 0.02)
        axi = fig.add_axes([gx, gy, gw, gh])
        axi.imshow(p)
        axi.axis('off')
    ax.text(grid_x0, grid_y0 - 0.02, 'Extracted Students', fontsize=8)

    # Feature extracting column
    ax.text(0.19, 0.62, 'Feature Extracting', fontsize=8, color='gray')
    # GroupRec 1024D
    ax.add_patch(patches.FancyBboxPatch((0.19, 0.54), 0.12, 0.09, boxstyle='round,pad=0.006',
                                        fc='#e8f5e9', ec='#2e7d32'))
    ax.text(0.19 + 0.06, 0.585, 'GroupRec 1024D', ha='center', va='center', fontsize=8)
    # ConvNeXt 1024D
    ax.add_patch(patches.FancyBboxPatch((0.19, 0.42), 0.12, 0.09, boxstyle='round,pad=0.006',
                                        fc='#e3f2fd', ec='#1565c0'))
    ax.text(0.19 + 0.06, 0.465, 'ConvNeXt 1024D', ha='center', va='center', fontsize=8)
    # Arrows from patches to features
    ax.annotate('', xy=(0.19, 0.58), xytext=(0.12, 0.58), arrowprops=dict(arrowstyle='-|>', lw=1.8))
    ax.annotate('', xy=(0.19, 0.47), xytext=(0.12, 0.47), arrowprops=dict(arrowstyle='-|>', lw=1.8))

    # Concatenate block
    ax.add_patch(patches.FancyBboxPatch((0.33, 0.45), 0.04, 0.20, boxstyle='round,pad=0.01', fc='#e3eafc', ec='#3f51b5'))
    ax.text(0.35, 0.55, 'Concatenate', rotation=90, fontsize=8, ha='center', va='center')
    ax.annotate('', xy=(0.33, 0.58), xytext=(0.31, 0.58), arrowprops=dict(arrowstyle='-|>', lw=1.6))
    ax.annotate('', xy=(0.33, 0.47), xytext=(0.31, 0.47), arrowprops=dict(arrowstyle='-|>', lw=1.6))

    # Panel (b) boundary
    ax.add_patch(patches.Rectangle((0.38, 0.05), 0.42, 0.90, fill=False, ls='--', lw=2, ec='#0d47a1'))
    ax.text(0.39, 0.04, '(b) Group Relation Reasoning + Fusion', fontsize=9, color='#0d47a1')

    # Gated Fusion branch box
    ax.add_patch(patches.FancyBboxPatch((0.41, 0.60), 0.18, 0.28, boxstyle='round,pad=0.01', fc='#fffde7', ec='#f9a825'))
    ax.text(0.41 + 0.09, 0.86 - 0.15, 'Gated Fusion', ha='center', va='center', fontsize=9)
    ax.text(0.41 + 0.09, 0.86 - 0.21, 'LN+Proj; Gate+Softmax(T)', ha='center', va='center', fontsize=7)
    ax.text(0.41 + 0.09, 0.86 - 0.29, '→ MLP → logits L^(G)', ha='center', va='center', fontsize=7)
    ax.annotate('', xy=(0.41, 0.74), xytext=(0.37, 0.53), arrowprops=dict(arrowstyle='-|>', lw=1.6))

    # Transformer Fusion branch box
    ax.add_patch(patches.FancyBboxPatch((0.41, 0.20), 0.18, 0.28, boxstyle='round,pad=0.01', fc='#e8eaf6', ec='#3949ab'))
    ax.text(0.41 + 0.09, 0.20 + 0.24, 'Transformer Fusion', ha='center', va='center', fontsize=9)
    ax.text(0.41 + 0.09, 0.20 + 0.18, 'Tokens; Encoders×2', ha='center', va='center', fontsize=7)
    ax.text(0.41 + 0.09, 0.20 + 0.10, '→ MLP → logits L^(T)', ha='center', va='center', fontsize=7)
    ax.annotate('', xy=(0.41, 0.35), xytext=(0.37, 0.53), arrowprops=dict(arrowstyle='-|>', lw=1.6))

    # Seed averaging notes
    ax.text(0.60, 0.77, 'Seed-wise average L^(G)', fontsize=7)
    ax.text(0.60, 0.33, 'Seed-wise average L^(T)', fontsize=7)

    # Ensemble merge
    ax.add_patch(patches.FancyBboxPatch((0.63, 0.46), 0.15, 0.18, boxstyle='round,pad=0.01', fc='#e3f2fd', ec='#1976d2'))
    ax.text(0.705, 0.55, 'Ensemble logits\n0.5·L^(G) + 0.5·L^(T)', ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(0.63, 0.55), xytext=(0.59, 0.74), arrowprops=dict(arrowstyle='-|>', lw=1.6))
    ax.annotate('', xy=(0.63, 0.55), xytext=(0.59, 0.32), arrowprops=dict(arrowstyle='-|>', lw=1.6))

    # Panel (c) boundary
    ax.add_patch(patches.Rectangle((0.81, 0.05), 0.18, 0.90, fill=False, ls='--', lw=2, ec='#2e7d32'))
    ax.text(0.82, 0.04, '(c) Classification', fontsize=9, color='#2e7d32')

    # Softmax and prediction
    ax.add_patch(patches.FancyBboxPatch((0.83, 0.53), 0.06, 0.08, boxstyle='round,pad=0.01', fc='#f1f8e9', ec='#388e3c'))
    ax.text(0.86, 0.57, 'softmax', ha='center', va='center', fontsize=8)
    ax.add_patch(patches.FancyBboxPatch((0.90, 0.53), 0.06, 0.08, boxstyle='round,pad=0.01', fc='#f1f8e9', ec='#388e3c'))
    ax.text(0.93, 0.57, 'argmax', ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(0.90, 0.57), xytext=(0.89, 0.57), arrowprops=dict(arrowstyle='-|>', lw=1.6))
    ax.annotate('', xy=(0.83, 0.57), xytext=(0.78, 0.55), arrowprops=dict(arrowstyle='-|>', lw=1.6))

    # Class list bullets
    classes = ['Bowing Head', 'Listening', 'Reading/Writing', 'Null', 'Using Phone']
    for i, name in enumerate(classes):
        y = 0.44 - i * 0.06
        circ = patches.Circle((0.905, y), 0.008, fc='#66bb6a', ec='none')
        ax.add_patch(circ)
        ax.text(0.915, y, ' ' + name, va='center', fontsize=8)

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def main():
    item = load_sample()
    img_path = IMAGES_DIR / item.get('img_path')
    img = imageio.imread(img_path)
    patches_img = crop_patches(img, item['bboxes'], k=6)
    out = ASSETS_DIR / 'pipeline_ensemble_mpl.png'
    draw_pipeline_figure(img, patches_img, out)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()

