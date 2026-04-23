#!/usr/bin/env python3
"""RAFNet Quick Run demo.

Renders paper Fig 4 style side-by-side comparison (GT / ConvNeXt-only / RAFNet)
using pre-extracted features and the released fusion-head checkpoints. Does NOT
touch the upstream GroupRec / SMPL / YOLOX pipeline — all heavy work was done
offline, this script only forwards the lightweight fusion head.

Defaults point to bundled demo samples under ``demo/``:

    demo/
    ├── features.pkl                        # pre-extracted, includes GroupRec + ConvNeXt + bbox_ctx + img_uid
    ├── images/                             # faces deface'd; referenced in features.pkl by basename
    └── ckpts/
        ├── c_spatial_seed_43.pth           # RAFNet (C+Spatial) fusion head
        └── convnext_only_seed_43.pth       # ConvNeXt-only baseline head

Usage:
    pixi run demo                           # defaults, bundled samples
    pixi run demo --output demo_out/        # custom output dir
    pixi run demo --features my.pkl \\
                  --images-dir my_images/   # run on your own features

Output:
    <output>/
    ├── <image_name>_comparison.png         # one 3-panel PNG per sample image
    └── summary.txt                         # per-image GT / Appear / RAFNet accuracy
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe; the demo writes PNG files, no interactive backend needed.
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Compat shim for older numpy pickles (NCST features were dumped under numpy>=1.25
# which introduced ``numpy._core``; recreate the alias for load under 1.24).
sys.modules["numpy._core"] = np.core
sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

from prism.models import MODELS  # noqa: E402
from prism.utils.config import load_config  # noqa: E402

LABEL_NAMES = ["Bowing Head", "Listening", "Reading/Writing", "Null", "Using Phone"]
LABEL_COLORS_RGB = {
    0: (76, 175, 80),    # Green  — Bowing Head
    1: (33, 150, 243),   # Blue   — Listening
    2: (255, 152, 0),    # Orange — Reading/Writing
    3: (158, 158, 158),  # Gray   — Null
    4: (244, 67, 54),    # Red    — Using Phone
}
LABEL_COLORS_MPL = {k: tuple(c / 255.0 for c in v) for k, v in LABEL_COLORS_RGB.items()}

RAFNet_MODEL_NAME = "grouprec_convnext_gated_c_spatial_graph_loss_tuned"
APPEAR_MODEL_NAME = "convnext_only_loss_tuned"


def load_features(pkl_path: Path) -> dict:
    """Load pre-extracted features. Expects the format produced by
    prism/scripts/build_feature_context_sidecar.py (with_context variant)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    meta = data["metadata"]
    ctx = data["context"]

    grouprec = np.asarray(data["grouprec"], dtype=np.float32)
    convnext = np.asarray(data["convnext"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
    img_paths = list(meta["img_paths"])
    bbox_xyxy = np.asarray(ctx["bbox_xyxy_abs"], dtype=np.float32)
    bbox_ctx = np.asarray(ctx["bbox_cxcywh_norm"], dtype=np.float32)

    # Assign a stable integer uid per unique image path (the C+Spatial model
    # conditions its same-image spatial graph on this id).
    uid_map: dict[str, int] = {}
    img_uids = []
    for p in img_paths:
        if p not in uid_map:
            uid_map[p] = len(uid_map)
        img_uids.append([float(uid_map[p])])
    img_uids = np.asarray(img_uids, dtype=np.float32)

    return {
        "grouprec": grouprec,
        "convnext": convnext,
        "bbox_ctx": bbox_ctx,
        "img_uid": img_uids,
        "labels": labels,
        "img_paths": img_paths,
        "bbox_xyxy": bbox_xyxy,
    }


def load_fusion_head(model_name: str, ckpt_path: Path, device: str):
    cfg_path = ROOT_DIR / "prism" / "configs" / f"{model_name}.yaml"
    cfg = load_config(str(cfg_path))
    model_cfg = cfg["model"]
    model = MODELS.build(model_cfg["name"], **model_cfg["params"])
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def predict(model, features: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    x = torch.tensor(features, dtype=torch.float32, device=device)
    out_chunks = []
    for i in range(0, len(x), batch_size):
        out = model(x[i : i + batch_size])
        logits = out[0] if isinstance(out, tuple) else out
        out_chunks.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(out_chunks)


def predict_prism(model, data: dict, device: str) -> np.ndarray:
    features = np.concatenate(
        [data["grouprec"], data["convnext"], data["bbox_ctx"], data["img_uid"]], axis=1
    )
    return predict(model, features, device)


def predict_appear(model, data: dict, device: str) -> np.ndarray:
    return predict(model, data["convnext"], device)


def group_by_image(data: dict, preds_appear: np.ndarray, preds_prism: np.ndarray):
    groups: dict[str, dict] = defaultdict(
        lambda: {"bboxes": [], "labels": [], "appear": [], "prism": []}
    )
    for i, img_path in enumerate(data["img_paths"]):
        g = groups[img_path]
        g["bboxes"].append(data["bbox_xyxy"][i])
        g["labels"].append(data["labels"][i])
        g["appear"].append(preds_appear[i])
        g["prism"].append(preds_prism[i])
    for g in groups.values():
        for k in ("bboxes", "labels", "appear", "prism"):
            g[k] = np.asarray(g[k])
    return groups


def draw_bboxes(img: np.ndarray, bboxes: np.ndarray, classes: np.ndarray,
                gt_classes: np.ndarray | None = None,
                thickness: int = 2, font_scale: float = 0.45) -> np.ndarray:
    """Draw bboxes with class labels. If ``gt_classes`` is given, boxes that
    disagree with GT are drawn with a dashed-style heavier outline to mark the
    error."""
    out = img.copy()
    for i, (bbox, cls) in enumerate(zip(bboxes, classes)):
        x1, y1, x2, y2 = bbox.astype(int)
        color_rgb = LABEL_COLORS_RGB[int(cls)]
        color_bgr = color_rgb[::-1]
        is_wrong = gt_classes is not None and int(cls) != int(gt_classes[i])
        box_thickness = thickness + 1 if is_wrong else thickness
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, box_thickness)
        label = LABEL_NAMES[int(cls)]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_comparison(
    img_bgr: np.ndarray, group: dict, image_name: str, dpi: int = 150
) -> plt.Figure:
    """Produce a 1-row 3-column figure: GT | ConvNeXt-only | RAFNet."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    panels = [
        ("Ground Truth", group["labels"], None),
        ("ConvNeXt-only", group["appear"], group["labels"]),
        ("RAFNet (C+Spatial)", group["prism"], group["labels"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi)
    for ax, (title, classes, gt) in zip(axes, panels):
        drawn = draw_bboxes(img_rgb, group["bboxes"], classes, gt_classes=gt)
        ax.imshow(drawn)
        ax.set_title(title, fontsize=13)
        ax.set_axis_off()
        if gt is not None:
            acc = (classes == gt).mean() * 100
            ax.set_xlabel(f"accuracy: {acc:.1f}%", fontsize=10)
            ax.xaxis.set_visible(True)
            ax.set_xticks([])

    # Legend strip at the bottom.
    from matplotlib.patches import Patch
    handles = [Patch(color=LABEL_COLORS_MPL[i], label=LABEL_NAMES[i]) for i in range(5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(image_name, fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAFNet Quick Run demo — side-by-side comparison on sample images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    demo_dir = ROOT_DIR / "demo"
    p.add_argument("--features", type=Path,
                   default=demo_dir / "features.pkl",
                   help="path to pre-extracted features pkl")
    p.add_argument("--images-dir", type=Path,
                   default=demo_dir / "images",
                   help="directory containing the sample images (matched by basename)")
    p.add_argument("--ckpt-prism", type=Path,
                   default=demo_dir / "ckpts" / "c_spatial_seed_43.pth",
                   help="RAFNet (C+Spatial) fusion head checkpoint")
    p.add_argument("--ckpt-appear", type=Path,
                   default=demo_dir / "ckpts" / "convnext_only_seed_43.pth",
                   help="ConvNeXt-only baseline fusion head checkpoint")
    p.add_argument("--output", type=Path,
                   default=ROOT_DIR / "demo_out",
                   help="output directory for PNGs + summary")
    p.add_argument("--device", type=str, default="auto",
                   help="'cuda', 'cpu', or 'auto' (picks cuda if available)")
    p.add_argument("--max-images", type=int, default=None,
                   help="limit the number of images rendered (useful when features "
                        "pkl contains a large dataset; default = render all)")
    p.add_argument("--show-gates", action="store_true",
                   help="(experimental) also render per-bbox gate weights — not yet implemented")
    return p.parse_args()


def resolve_image_path(img_ref: str, images_dir: Path) -> Path | None:
    """Find the sample image for a given reference (basename or relative path)."""
    candidate = images_dir / Path(img_ref).name
    if candidate.exists():
        return candidate
    direct = images_dir / img_ref
    if direct.exists():
        return direct
    return None


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    for label, path in [
        ("features", args.features),
        ("C+Spatial ckpt", args.ckpt_prism),
        ("ConvNeXt ckpt", args.ckpt_appear),
    ]:
        if not path.exists():
            sys.exit(f"[error] {label} not found: {path}\n"
                     f"        See README 'Quick Run' for how to download demo assets.")

    using_bundled = (
        args.features.resolve() == (ROOT_DIR / "demo" / "features.pkl").resolve()
    )
    if using_bundled:
        print("[notice] Using bundled demo assets.")
        print("         demo/images/*.jpg are face-blurred (anonymized).")
        print("         demo/features.pkl was extracted offline from the *original*")
        print("         non-anonymized images — the shown pixels and cached features")
        print("         do NOT come from the same image state. This matches the")
        print("         recipe used for paper Fig 4. See the top-level README (Quick")
        print("         Run → Bundled demo assets) for full provenance details.")
        print()

    print(f"Loading features from {args.features} ...")
    data = load_features(args.features)
    n_samples = len(data["labels"])
    n_images = len(set(data["img_paths"]))
    print(f"  {n_samples} instances across {n_images} image(s)")

    print(f"Loading RAFNet (C+Spatial) checkpoint ...")
    model_prism = load_fusion_head(RAFNet_MODEL_NAME, args.ckpt_prism, device)

    print(f"Loading ConvNeXt-only checkpoint ...")
    model_appear = load_fusion_head(APPEAR_MODEL_NAME, args.ckpt_appear, device)

    print("Running fusion heads ...")
    preds_prism = predict_prism(model_prism, data, device)
    preds_appear = predict_appear(model_appear, data, device)

    acc_prism = (preds_prism == data["labels"]).mean() * 100
    acc_appear = (preds_appear == data["labels"]).mean() * 100
    print(f"  Overall: ConvNeXt-only {acc_appear:.2f}% | RAFNet {acc_prism:.2f}%")

    groups = group_by_image(data, preds_appear, preds_prism)

    args.output.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "RAFNet Quick Run demo summary",
        "=" * 60,
        f"device          : {device}",
        f"features pkl    : {args.features}",
        f"samples         : {n_samples}",
        f"images          : {n_images}",
        f"overall accuracy: ConvNeXt-only {acc_appear:.2f}%, RAFNet {acc_prism:.2f}%",
    ]
    if using_bundled:
        summary_lines += [
            "",
            "Provenance: demo/images/*.jpg are face-blurred; demo/features.pkl was",
            "extracted offline from the original non-anonymized images. Shown pixels",
            "and cached features do not come from the same image state. Features are",
            "1024-D embeddings and are not face-recoverable. See top-level README.",
        ]
    summary_lines += [
        "",
        f"{'image':<40} {'#ppl':>5} {'Appear':>8} {'RAFNet':>8} {'fixed':>6}",
        "-" * 80,
    ]
    rendered = 0
    skipped_no_image = []
    image_items = sorted(groups.items())
    if args.max_images is not None:
        image_items = image_items[: args.max_images]
    for img_ref, g in image_items:
        n_ppl = len(g["labels"])
        a_acc = (g["appear"] == g["labels"]).mean() * 100
        p_acc = (g["prism"] == g["labels"]).mean() * 100
        n_fixed = int(((g["appear"] != g["labels"]) & (g["prism"] == g["labels"])).sum())
        summary_lines.append(
            f"{Path(img_ref).name:<40} {n_ppl:>5d} {a_acc:>7.1f}% {p_acc:>7.1f}% {n_fixed:>6d}"
        )

        img_path = resolve_image_path(img_ref, args.images_dir)
        if img_path is None:
            skipped_no_image.append(img_ref)
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            skipped_no_image.append(img_ref)
            continue

        stem = Path(img_ref).stem
        fig = render_comparison(img_bgr, g, image_name=Path(img_ref).name)
        out_png = args.output / f"{stem}_comparison.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        rendered += 1
        print(f"  wrote {out_png}")

    summary_path = args.output / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"\nRendered {rendered}/{n_images} image(s). Summary: {summary_path}")
    if skipped_no_image:
        print(f"  [warn] {len(skipped_no_image)} image(s) referenced in features pkl "
              f"were not found in {args.images_dir}: "
              + ", ".join(Path(p).name for p in skipped_no_image[:3])
              + (" ..." if len(skipped_no_image) > 3 else ""))

    if args.show_gates:
        print("\n[note] --show-gates is not yet implemented in v1 (planned for v2 "
              "once gate-weight hooks are added to GatedFusionClassifier).")


if __name__ == "__main__":
    main()
