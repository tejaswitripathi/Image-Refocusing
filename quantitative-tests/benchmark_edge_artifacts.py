#!/usr/bin/env python3
"""Benchmark edge artifacts for the prototype rendering pipeline.

For each JPG in cache/, runs the same NN + Gaussian-background pipeline as
prototype.py, then compares edge artifacts against the original image:

  1. Flat Gaussian background vs original
  2. CoC-weighted NN render vs original

Metrics are computed in regions where compositing is most likely to leave
visible halos: in-focus pixels (should match the original), the CoC transition
band, and a narrow ring around the Gaussian mask boundary (CoC threshold).

Usage (from repo root):
    python quantitative-tests/benchmark_edge_artifacts.py
    python quantitative-tests/benchmark_edge_artifacts.py --max-side 1536
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import boto3
import numpy as np
import torch
from PIL import Image
from skimage.filters import gaussian, sobel
from skimage.morphology import dilation, disk, erosion
from skimage.transform import resize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_preprocessing import (  # noqa: E402
    COC_PX_MAX,
    FOCAL_LENGTH_MM_MAX,
    F_STOP_MAX,
    TARGET_SIZE,
)
from depth_anything_inference import (  # noqa: E402
    generate_pseudo_coc_from_relative_depth,
    load_depth_anything,
    predict_relative_depth,
)
from models.RendererNet import RendererNet  # noqa: E402


# Match prototype.py defaults.
S3_BUCKET = "tejas-blender-bucket"
RENDERER_KEY = "defocus-checkpoints/renderer-net/best_renderer.pth"
F_STOP = 1.2
FOCAL_LENGTH_MM = 6.765
DEPTH_ENCODER = "vitb"
DEPTH_CHECKPOINT = REPO_ROOT / "checkpoints" / "depth_anything_v2_vitb.pth"
COC_MAX_PX = 4.0
COC_FOCUS_THRESHOLD_PX = 0.4
GAUSSIAN_COC_THRESHOLD_PX = 1.0
GAUSSIAN_SIGMA_PX = 12.0
TRANSITION_COC_LOW = COC_FOCUS_THRESHOLD_PX
TRANSITION_COC_HIGH = GAUSSIAN_COC_THRESHOLD_PX
BOUNDARY_RING_RADIUS_PX = 3


@dataclass
class EdgeArtifactMetrics:
    in_focus_mean_abs_diff: float
    in_focus_mean_grad_excess: float
    transition_mean_abs_diff: float
    transition_mean_grad_excess: float
    boundary_ring_mean_abs_diff: float
    boundary_ring_mean_grad_excess: float
    boundary_ring_p95_abs_diff: float
    global_mean_grad_excess: float


@dataclass
class ImageBenchmark:
    image: str
    width: int
    height: int
    in_focus_fraction: float
    transition_fraction: float
    boundary_ring_fraction: float
    gaussian: EdgeArtifactMetrics
    nn_render: EdgeArtifactMetrics
    nn_vs_gaussian_improvement: dict[str, float]


def list_cache_jpgs() -> list[Path]:
    cache_dir = REPO_ROOT / "cache"
    return sorted(p for p in cache_dir.glob("*.jpg") if p.is_file())


def load_checkpoint_from_s3(s3_key: str, map_location: str) -> object:
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, s3_key, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location=map_location)


def load_renderer(device: str) -> RendererNet:
    checkpoint = load_checkpoint_from_s3(RENDERER_KEY, map_location=device)
    model = RendererNet(in_channels=6, out_channels=3).to(device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def fit_max_side(rgb: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None:
        return rgb
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    return resize(
        rgb, (new_h, new_w), anti_aliasing=True, preserve_range=True
    ).astype(np.float32)


def make_param_maps(size: int) -> tuple[np.ndarray, np.ndarray]:
    fstop_map = np.ones((1, size, size), dtype=np.float32) * (F_STOP / F_STOP_MAX)
    focal_map = np.ones((1, size, size), dtype=np.float32) * (
        FOCAL_LENGTH_MM / FOCAL_LENGTH_MM_MAX
    )
    return fstop_map, focal_map


def to_chw_resized(rgb: np.ndarray, size: int) -> np.ndarray:
    rs = resize(
        rgb, (size, size), anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    return np.transpose(rs, (2, 0, 1))


def coc_px_to_norm_512(coc_px: np.ndarray) -> np.ndarray:
    coc_norm = np.clip(coc_px, 0, COC_PX_MAX) / COC_PX_MAX
    return resize(
        coc_norm,
        (TARGET_SIZE, TARGET_SIZE),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


@torch.no_grad()
def run_renderer(
    model: RendererNet,
    device: str,
    rgb: np.ndarray,
    coc_norm_512: np.ndarray,
    out_size: tuple[int, int],
) -> np.ndarray:
    chw = to_chw_resized(rgb, TARGET_SIZE)
    fstop_map, focal_map = make_param_maps(TARGET_SIZE)
    coc_channel = coc_norm_512[None, :, :]

    x = np.concatenate([chw, fstop_map, focal_map, coc_channel], axis=0)[None]
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    x = torch.from_numpy(x).to(device)

    out = model(x)[0].cpu().numpy()
    out = np.clip(np.transpose(out, (1, 2, 0)), 0, 1)
    return resize(
        out, out_size, anti_aliasing=True, preserve_range=True
    ).astype(np.float32)


def pseudo_coc_px(
    rel_depth: np.ndarray, focus_y: int, focus_x: int
) -> np.ndarray:
    coc_px = generate_pseudo_coc_from_relative_depth(
        rel_depth,
        focus_y,
        focus_x,
        coc_max_px=COC_MAX_PX,
        blur_strength=1.0,
    )
    return np.clip(coc_px, 0, COC_MAX_PX).astype(np.float32)


def coc_blend_weight(coc_px: np.ndarray) -> np.ndarray:
    span = max(COC_MAX_PX - COC_FOCUS_THRESHOLD_PX, 1e-6)
    t = np.clip((coc_px - COC_FOCUS_THRESHOLD_PX) / span, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def render_nn(
    model: RendererNet,
    device: str,
    rgb: np.ndarray,
    coc_px: np.ndarray,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    coc_norm_512 = coc_px_to_norm_512(coc_px)
    nn_render = run_renderer(model, device, rgb, coc_norm_512, (h, w))
    weight = coc_blend_weight(coc_px)[:, :, None]
    blended = (1.0 - weight) * rgb + weight * nn_render
    return np.clip(blended, 0, 1).astype(np.float32)


def render_gaussian_background(rgb: np.ndarray, coc_px: np.ndarray) -> np.ndarray:
    blurred = gaussian(
        rgb, sigma=GAUSSIAN_SIGMA_PX, channel_axis=-1, preserve_range=True
    ).astype(np.float32)
    mask = (coc_px > GAUSSIAN_COC_THRESHOLD_PX)[:, :, None].astype(np.float32)
    composite = (1.0 - mask) * rgb + mask * blurred
    return np.clip(composite, 0, 1).astype(np.float32)


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)


def grad_magnitude(gray: np.ndarray) -> np.ndarray:
    return sobel(gray).astype(np.float32)


def region_masks(coc_px: np.ndarray) -> dict[str, np.ndarray]:
    in_focus = coc_px <= COC_FOCUS_THRESHOLD_PX
    transition = (coc_px > TRANSITION_COC_LOW) & (coc_px <= TRANSITION_COC_HIGH)

    gaussian_mask = coc_px > GAUSSIAN_COC_THRESHOLD_PX
    footprint = disk(BOUNDARY_RING_RADIUS_PX)
    boundary_ring = dilation(gaussian_mask, footprint) ^ erosion(
        gaussian_mask, footprint
    )

    return {
        "in_focus": in_focus,
        "transition": transition,
        "boundary_ring": boundary_ring,
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def masked_p95(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.percentile(values[mask], 95))


def compute_edge_artifact_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    coc_px: np.ndarray,
) -> EdgeArtifactMetrics:
    diff = np.abs(processed - original)
    diff_gray = rgb_to_luminance(diff)

    orig_gray = rgb_to_luminance(original)
    proc_gray = rgb_to_luminance(processed)
    grad_excess = np.abs(grad_magnitude(proc_gray) - grad_magnitude(orig_gray))

    masks = region_masks(coc_px)

    return EdgeArtifactMetrics(
        in_focus_mean_abs_diff=masked_mean(diff_gray, masks["in_focus"]),
        in_focus_mean_grad_excess=masked_mean(grad_excess, masks["in_focus"]),
        transition_mean_abs_diff=masked_mean(diff_gray, masks["transition"]),
        transition_mean_grad_excess=masked_mean(grad_excess, masks["transition"]),
        boundary_ring_mean_abs_diff=masked_mean(diff_gray, masks["boundary_ring"]),
        boundary_ring_mean_grad_excess=masked_mean(
            grad_excess, masks["boundary_ring"]
        ),
        boundary_ring_p95_abs_diff=masked_p95(diff_gray, masks["boundary_ring"]),
        global_mean_grad_excess=float(np.mean(grad_excess)),
    )


def improvement_ratio(gaussian_value: float, nn_value: float) -> float:
    if nn_value <= 1e-12:
        return float("inf") if gaussian_value > 1e-12 else 1.0
    return gaussian_value / nn_value


def benchmark_image(
    image_path: Path,
    renderer: RendererNet,
    depth_model,
    device: str,
    max_side: int | None,
) -> ImageBenchmark:
    rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    rgb = fit_max_side(rgb, max_side)
    h, w = rgb.shape[:2]

    rel_depth = predict_relative_depth(str(image_path), depth_model, normalize=True)
    rel_depth = resize(
        rel_depth, (h, w), order=1, anti_aliasing=True, preserve_range=True
    ).astype(np.float32)

    focus_y, focus_x = h // 2, w // 2
    coc_px = pseudo_coc_px(rel_depth, focus_y, focus_x)

    nn_image = render_nn(renderer, device, rgb, coc_px)
    gaussian_image = render_gaussian_background(rgb, coc_px)

    gaussian_metrics = compute_edge_artifact_metrics(rgb, gaussian_image, coc_px)
    nn_metrics = compute_edge_artifact_metrics(rgb, nn_image, coc_px)

    masks = region_masks(coc_px)
    pixel_count = float(h * w)
    improvement = {
        "boundary_ring_mean_abs_diff": improvement_ratio(
            gaussian_metrics.boundary_ring_mean_abs_diff,
            nn_metrics.boundary_ring_mean_abs_diff,
        ),
        "boundary_ring_mean_grad_excess": improvement_ratio(
            gaussian_metrics.boundary_ring_mean_grad_excess,
            nn_metrics.boundary_ring_mean_grad_excess,
        ),
        "transition_mean_grad_excess": improvement_ratio(
            gaussian_metrics.transition_mean_grad_excess,
            nn_metrics.transition_mean_grad_excess,
        ),
        "in_focus_mean_grad_excess": improvement_ratio(
            gaussian_metrics.in_focus_mean_grad_excess,
            nn_metrics.in_focus_mean_grad_excess,
        ),
    }

    return ImageBenchmark(
        image=image_path.name,
        width=w,
        height=h,
        in_focus_fraction=float(np.sum(masks["in_focus"]) / pixel_count),
        transition_fraction=float(np.sum(masks["transition"]) / pixel_count),
        boundary_ring_fraction=float(np.sum(masks["boundary_ring"]) / pixel_count),
        gaussian=gaussian_metrics,
        nn_render=nn_metrics,
        nn_vs_gaussian_improvement=improvement,
    )


def average_metrics(rows: list[ImageBenchmark], method: str) -> EdgeArtifactMetrics:
    values = [getattr(row, method) for row in rows]
    return EdgeArtifactMetrics(
        **{
            field: float(np.mean([getattr(v, field) for v in values]))
            for field in EdgeArtifactMetrics.__dataclass_fields__
        }
    )


def print_metric_block(title: str, metrics: EdgeArtifactMetrics) -> None:
    print(title)
    print(f"  in_focus_mean_abs_diff        : {metrics.in_focus_mean_abs_diff:.6f}")
    print(f"  in_focus_mean_grad_excess     : {metrics.in_focus_mean_grad_excess:.6f}")
    print(f"  transition_mean_abs_diff      : {metrics.transition_mean_abs_diff:.6f}")
    print(f"  transition_mean_grad_excess   : {metrics.transition_mean_grad_excess:.6f}")
    print(f"  boundary_ring_mean_abs_diff   : {metrics.boundary_ring_mean_abs_diff:.6f}")
    print(
        f"  boundary_ring_mean_grad_excess: {metrics.boundary_ring_mean_grad_excess:.6f}"
    )
    print(f"  boundary_ring_p95_abs_diff    : {metrics.boundary_ring_p95_abs_diff:.6f}")
    print(f"  global_mean_grad_excess       : {metrics.global_mean_grad_excess:.6f}")


def write_csv(path: Path, rows: list[ImageBenchmark]) -> None:
    metric_fields = list(EdgeArtifactMetrics.__dataclass_fields__)
    improvement_fields = [
        "boundary_ring_mean_abs_diff",
        "boundary_ring_mean_grad_excess",
        "transition_mean_grad_excess",
        "in_focus_mean_grad_excess",
    ]

    fieldnames = [
        "image",
        "width",
        "height",
        "in_focus_fraction",
        "transition_fraction",
        "boundary_ring_fraction",
    ]
    for prefix in ("gaussian", "nn"):
        for field in metric_fields:
            fieldnames.append(f"{prefix}_{field}")
    for field in improvement_fields:
        fieldnames.append(f"improvement_{field}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "image": row.image,
                "width": row.width,
                "height": row.height,
                "in_focus_fraction": row.in_focus_fraction,
                "transition_fraction": row.transition_fraction,
                "boundary_ring_fraction": row.boundary_ring_fraction,
            }
            for prefix, metrics in (
                ("gaussian", row.gaussian),
                ("nn", row.nn_render),
            ):
                for field in metric_fields:
                    record[f"{prefix}_{field}"] = getattr(metrics, field)
            for field in improvement_fields:
                record[f"improvement_{field}"] = row.nn_vs_gaussian_improvement[field]
            writer.writerow(record)


def write_json(path: Path, rows: list[ImageBenchmark]) -> None:
    payload = {
        "config": {
            "f_stop": F_STOP,
            "focal_length_mm": FOCAL_LENGTH_MM,
            "coc_max_px": COC_MAX_PX,
            "coc_focus_threshold_px": COC_FOCUS_THRESHOLD_PX,
            "gaussian_coc_threshold_px": GAUSSIAN_COC_THRESHOLD_PX,
            "gaussian_sigma_px": GAUSSIAN_SIGMA_PX,
            "transition_coc_band": [TRANSITION_COC_LOW, TRANSITION_COC_HIGH],
            "boundary_ring_radius_px": BOUNDARY_RING_RADIUS_PX,
        },
        "images": [asdict(row) for row in rows],
        "averages": {
            "gaussian": asdict(average_metrics(rows, "gaussian")),
            "nn_render": asdict(average_metrics(rows, "nn_render")),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark edge artifacts for prototype.py rendering."
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Resize longest image side before inference (default: 1024).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "quantitative-tests" / "results",
        help="Directory for CSV/JSON benchmark outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list_cache_jpgs()
    if not images:
        raise SystemExit("No JPG files found in cache/.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Benchmarking {len(images)} images from cache/")

    renderer = load_renderer(device)
    depth_model, _depth_device = load_depth_anything(
        encoder=DEPTH_ENCODER, checkpoint_path=str(DEPTH_CHECKPOINT)
    )

    rows: list[ImageBenchmark] = []
    for image_path in images:
        print(f"\nProcessing {image_path.name} ...")
        row = benchmark_image(
            image_path,
            renderer=renderer,
            depth_model=depth_model,
            device=device,
            max_side=args.max_side,
        )
        rows.append(row)

        print_metric_block("  Gaussian vs original:", row.gaussian)
        print_metric_block("  NN render vs original:", row.nn_render)
        print("  NN improvement ratios (Gaussian / NN, higher is better):")
        for key, value in row.nn_vs_gaussian_improvement.items():
            print(f"    {key}: {value:.3f}x")

    avg_gaussian = average_metrics(rows, "gaussian")
    avg_nn = average_metrics(rows, "nn_render")

    print("\n" + "=" * 72)
    print("Averages across cache JPGs")
    print_metric_block("Gaussian vs original:", avg_gaussian)
    print_metric_block("NN render vs original:", avg_nn)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "edge_artifact_benchmark.csv"
    json_path = args.output_dir / "edge_artifact_benchmark.json"
    write_csv(csv_path, rows)
    write_json(json_path, rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
