#!/usr/bin/env python3
"""Tier-2 diagnostics: per-class report, confusion matrix, params and throughput.

Example:
  pixi run python prism/scripts/tier2_diagnostics.py \
    --model grouprec_convnext_gated_c_spatial_graph_loss_tuned \
    --seed best --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as _np  # type: ignore

sys.modules["numpy._core"] = _np.core
sys.modules["numpy._core._multiarray_umath"] = _np.core._multiarray_umath


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier-2 diagnostics exporter")
    p.add_argument("--model", required=True, help="model config name, e.g. grouprec_convnext_gated_loss_tuned")
    p.add_argument("--seed", default="best", help="'best' or explicit seed id, e.g. 43")
    p.add_argument("--device", default="cpu", help="cpu or cuda:0")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--features-dir",
        default=None,
        help="override feature directory; keeps original filenames (e.g., train_features.pkl)",
    )
    p.add_argument("--warmup", type=int, default=1, help="throughput warmup passes")
    p.add_argument("--repeats", type=int, default=3, help="throughput measured passes")
    p.add_argument("--batch-size", type=int, default=256, help="inference batch size")
    p.add_argument("--use-amp", action="store_true", help="enable amp (cuda only)")
    p.add_argument(
        "--label-names",
        default="Bowing Head,Listening,Reading/Writing,Null,Using Phone",
        help="comma-separated class names",
    )
    return p.parse_args()


def pick_seed(model: str, seed_arg: str) -> int:
    if seed_arg != "best":
        return int(seed_arg)

    model_dir = ROOT_DIR / "results" / "training" / model
    best_seed = None
    best_score = -1e18
    for sd in sorted(model_dir.glob("seed_*")):
        rp = sd / "results.json"
        if not rp.exists():
            continue
        try:
            data = json.loads(rp.read_text())
            score = float(data.get("test_macro_f1_no_null", data.get("test_macro_f1", -1e18)))
            seed = int(sd.name.split("_")[1])
            if score > best_score:
                best_score = score
                best_seed = seed
        except Exception:
            continue
    if best_seed is None:
        raise FileNotFoundError(f"No valid seed results found for model: {model}")
    return best_seed


def build_loader(cfg: dict[str, Any], split: str, batch_size: int):
    import torch
    from torch.utils.data import DataLoader
    from prism.data import DATASETS

    ds_cfg = cfg["datasets"][split]
    dataset = DATASETS.build(ds_cfg["name"], **ds_cfg.get("params", {}))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


def load_model(model_name: str, seed: int, device: str):
    import torch
    from prism.utils import load_config
    from prism.evaluation.ensemble import build_model_from_config

    cfg = load_config(f"prism/configs/{model_name}.yaml")
    ckpt = ROOT_DIR / "results" / "training" / model_name / f"seed_{seed}" / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")

    dev = torch.device(device if (not device.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    model = build_model_from_config(cfg).to(dev)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return cfg, model, dev


def run_inference(model, loader, device, use_amp: bool):
    import torch

    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(features)
            else:
                outputs = model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)


def save_per_class_csv(out_path: Path, report: dict[str, Any], label_names: list[str]) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "precision", "recall", "f1", "support"])
        for i, name in enumerate(label_names):
            r = report.get(name, report.get(str(i), {}))
            writer.writerow(
                [
                    i,
                    name,
                    f"{float(r.get('precision', 0.0)):.4f}",
                    f"{float(r.get('recall', 0.0)):.4f}",
                    f"{float(r.get('f1-score', 0.0)):.4f}",
                    int(r.get("support", 0)),
                ]
            )
        for key in ("macro avg", "weighted avg"):
            if key in report:
                r = report[key]
                writer.writerow(
                    [
                        key,
                        key,
                        f"{float(r.get('precision', 0.0)):.4f}",
                        f"{float(r.get('recall', 0.0)):.4f}",
                        f"{float(r.get('f1-score', 0.0)):.4f}",
                        int(r.get("support", 0)),
                    ]
                )


def save_confusion_figs(cm: np.ndarray, labels: list[str], out_pdf: Path, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return
    # Prefer SciencePlots for paper-ready styling; fallback silently if unavailable.
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        pass

    fig = plt.figure(figsize=(6.6, 5.2))
    ax = fig.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def benchmark_throughput(model, loader, device, warmup: int, repeats: int, use_amp: bool) -> dict[str, float]:
    import torch

    def one_pass() -> int:
        seen = 0
        with torch.no_grad():
            for features, _ in loader:
                features = features.to(device)
                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(features)
                else:
                    outputs = model(features)
                _ = outputs[0] if isinstance(outputs, tuple) else outputs
                seen += int(features.shape[0])
        return seen

    for _ in range(max(0, warmup)):
        _ = one_pass()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_samples = 0
    for _ in range(max(1, repeats)):
        total_samples += one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    sps = total_samples / max(dt, 1e-9)
    return {
        "total_samples": float(total_samples),
        "total_time_sec": float(dt),
        "samples_per_sec": float(sps),
        "ms_per_sample": float(1000.0 / max(sps, 1e-9)),
    }


def main() -> None:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    args = parse_args()
    labels = [x.strip() for x in args.label_names.split(",")]
    seed = pick_seed(args.model, args.seed)

    cfg, model, device = load_model(args.model, seed, args.device)
    if args.features_dir:
        froot = Path(args.features_dir).expanduser().resolve()
        for sp in ("train", "val", "test"):
            old = Path(cfg["datasets"][sp]["params"]["features_path"])
            cfg["datasets"][sp]["params"]["features_path"] = str(froot / old.name)
    loader = build_loader(cfg, args.split, args.batch_size)
    logits, y_true = run_inference(model, loader, device, use_amp=args.use_amp)
    y_pred = np.argmax(logits, axis=1)

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    null_idx = labels.index("Null") if "Null" in labels else None
    if null_idx is None:
        macro_f1_no_null = float(report["macro avg"]["f1-score"])
    else:
        mask = y_true != null_idx
        macro_f1_no_null = (
            float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))
            if np.any(mask)
            else 0.0
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    speed = benchmark_throughput(
        model=model,
        loader=loader,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
        use_amp=args.use_amp,
    )

    out_dir = ROOT_DIR / "results" / "evaluation" / "tier2" / args.model / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model,
        "seed": seed,
        "split": args.split,
        "device": str(device),
        "overall": {
            "accuracy": float((y_pred == y_true).mean()),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "macro_f1_no_null": macro_f1_no_null,
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
        },
        "params": {
            "total": int(total_params),
            "trainable": int(trainable_params),
        },
        "throughput": speed,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist(), indent=2))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_per_class_csv(out_dir / "per_class_metrics.csv", report, labels)
    save_confusion_figs(
        cm=cm,
        labels=labels,
        out_pdf=out_dir / "confusion_matrix.pdf",
        out_png=out_dir / "confusion_matrix.png",
    )

    # Compare with stored test metric to catch feature-path drift.
    rs_path = ROOT_DIR / "results" / "training" / args.model / f"seed_{seed}" / "results.json"
    if rs_path.exists():
        try:
            rs = json.loads(rs_path.read_text())
            stored = float(rs.get("test_macro_f1_no_null", rs.get("test_macro_f1", np.nan)))
            live = summary["overall"]["macro_f1_no_null"] * 100.0
            summary["stored_metric_hint"] = {"test_macro_f1_or_no_null": stored}
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            if np.isfinite(stored) and abs(live - stored) > 5.0:
                print(
                    "  [warn] live metric differs from stored checkpoint metric by >5 points; "
                    "check feature path (use --features-dir for exact training feature set)."
                )
        except Exception:
            pass

    print(f"[OK] model={args.model} seed={seed} split={args.split}")
    print(f"     out: {out_dir}")
    print(
        f"     macro_f1={summary['overall']['macro_f1']*100:.2f}% "
        f"macro_f1_no_null={summary['overall']['macro_f1_no_null']*100:.2f}% "
        f"acc={summary['overall']['accuracy']*100:.2f}% "
        f"samples/s={speed['samples_per_sec']:.2f}"
    )


if __name__ == "__main__":
    main()
