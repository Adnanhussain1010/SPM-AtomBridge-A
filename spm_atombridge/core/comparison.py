"""Quantitative comparison of images and scalar fields."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class ComparisonResult:
    correlation: float
    rmsd: float
    residual_map: np.ndarray
    discrepancy_mask: np.ndarray
    threshold: float


def compare_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    threshold_sigma: float = 2.0,
    smooth_sigma: float = 1.0,
) -> ComparisonResult:
    """Compare two 2D arrays. Returns metrics and a discrepancy mask."""
    if img_a.shape != img_b.shape:
        raise ValueError(f"Shape mismatch: {img_a.shape} vs {img_b.shape}")

    a = np.nan_to_num(img_a, nan=np.nanmean(img_a))
    b = np.nan_to_num(img_b, nan=np.nanmean(img_b))

    if smooth_sigma > 0:
        a = gaussian_filter(a, smooth_sigma)
        b = gaussian_filter(b, smooth_sigma)

    a_flat = a.flatten() - np.mean(a)
    b_flat = b.flatten() - np.mean(b)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    corr = float(np.dot(a_flat, b_flat) / denom) if denom > 0 else 0.0

    residual = a - b
    rmsd = float(np.sqrt(np.mean(residual**2)))

    sigma = np.std(residual)
    threshold = threshold_sigma * sigma
    mask = np.abs(residual) > threshold

    return ComparisonResult(
        correlation=corr,
        rmsd=rmsd,
        residual_map=residual,
        discrepancy_mask=mask,
        threshold=threshold,
    )
