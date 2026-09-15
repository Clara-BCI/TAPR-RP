"""
FBCNet backbone adapter.

This module implements a compact filter-bank convolutional extractor for the
existing (B, C, T) -> (B, feat_dim) contract used by IDRemovalNet.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args: Any, max_norm: float = 2.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


def _as_band_list(value: Any) -> List[Tuple[float, float]]:
    if value is None:
        return [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)]
    out: List[Tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict):
            out.append((float(item["fmin"]), float(item["fmax"])))
        elif isinstance(item, Iterable):
            lo, hi = list(item)[:2]
            out.append((float(lo), float(hi)))
        else:
            raise TypeError(f"Unsupported FBCNet band spec: {item!r}")
    return out


def _sinc_filter(fmin: float, fmax: float, fs: float, kernel_size: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    nyq = fs / 2.0
    lo = max(0.1, float(fmin)) / nyq
    hi = min(nyq - 0.1, float(fmax)) / nyq
    if not 0.0 < lo < hi < 1.0:
        raise ValueError(f"Invalid FBCNet band [{fmin}, {fmax}] for fs={fs}.")

    n = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    band = 2 * hi * torch.sinc(2 * hi * n) - 2 * lo * torch.sinc(2 * lo * n)
    window = torch.hamming_window(kernel_size, periodic=False)
    band = band * window
    band = band / band.abs().sum().clamp_min(1e-8)
    return band


class FixedFilterBank(nn.Module):
    def __init__(self, bands: List[Tuple[float, float]], fs: float, kernel_size: int) -> None:
        super().__init__()
        filters = [_sinc_filter(lo, hi, fs, kernel_size) for lo, hi in bands]
        weight = torch.stack(filters).view(len(filters), 1, 1, -1)
        self.register_buffer("weight", weight)
        self.padding = (kernel_size // 2, 0)
        self.n_bands = len(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        y = F.conv2d(
            x.reshape(b * c, 1, 1, t),
            self.weight,
            padding=(0, self.padding[0]),
        )
        return y.reshape(b, c, self.n_bands, t).permute(0, 2, 1, 3).contiguous()


class LogVarLayer(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.var(x, dim=-1, unbiased=False).clamp_min(self.eps))


class FBCNetFeatureExtractor(nn.Module):
    """
    FBCNet-style extractor: fixed EEG filter bank -> per-band temporal mixing ->
    depthwise spatial filtering -> log-variance features.
    """

    def __init__(self, n_channels: int, n_times: int, fbc_params: Dict[str, Any]) -> None:
        super().__init__()
        p = fbc_params
        bands = _as_band_list(p.get("frequency_bands"))
        fs = float(p.get("fs", 250))
        fb_kernel = int(p.get("filterbank_kernel_size", 129))
        temporal_filters = int(p.get("temporal_filters", 8))
        spatial_multiplier = int(p.get("spatial_multiplier", 2))
        stride = int(p.get("stride", 4))
        dropout = float(p.get("dropout_rate", 0.5))
        bn_momentum = float(p.get("bn_momentum", 0.1))
        bn_eps = float(p.get("bn_eps", 1e-5))
        max_norm_spatial = float(p.get("max_norm_spatial_conv", 2.0))
        log_eps = float(p.get("log_eps", 1e-6))

        self.filter_bank = FixedFilterBank(bands, fs, fb_kernel)
        n_bands = len(bands)
        self.temporal = nn.Sequential(
            nn.Conv2d(
                n_bands,
                n_bands * temporal_filters,
                kernel_size=(1, int(p.get("temporal_kernel_size", 25))),
                padding=(0, int(p.get("temporal_kernel_size", 25)) // 2),
                groups=n_bands,
                bias=False,
            ),
            nn.BatchNorm2d(n_bands * temporal_filters, momentum=bn_momentum, eps=bn_eps),
            nn.ELU(),
        )
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(
                n_bands * temporal_filters,
                n_bands * temporal_filters * spatial_multiplier,
                kernel_size=(n_channels, 1),
                groups=n_bands * temporal_filters,
                bias=False,
                max_norm=max_norm_spatial,
            ),
            nn.BatchNorm2d(
                n_bands * temporal_filters * spatial_multiplier,
                momentum=bn_momentum,
                eps=bn_eps,
            ),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, max(1, stride)), stride=(1, max(1, stride))),
            nn.Dropout(dropout),
        )
        self.logvar = LogVarLayer(log_eps)

        with torch.no_grad():
            probe = torch.zeros(1, n_channels, n_times)
            out = self._forward_features(probe)
        self.feat_dim = int(out.reshape(1, -1).shape[1])
        if self.feat_dim <= 0:
            raise ValueError("FBCNet: invalid feature dimension.")

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.filter_bank(x)
        y = self.temporal(y)
        y = self.spatial(y)
        y = self.logvar(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self._forward_features(x).reshape(b, -1)
