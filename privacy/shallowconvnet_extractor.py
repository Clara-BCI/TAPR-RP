"""
ShallowConvNet backbone adapter.

The module follows the classic ShallowConvNet path:
temporal convolution -> spatial convolution -> square -> average pooling -> safe log.
It returns flat features instead of logits so the existing task/privacy heads can be reused.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args: Any, max_norm: float = 2.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class SquareActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(x)


class SafeLogActivation(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowConvNetFeatureExtractor(nn.Module):
    """
    ShallowConvNet as a (B, C, T) -> (B, feat_dim) feature extractor.

    This mirrors the BioDecode ShallowConvNet implementation while removing its
    final classifier, matching the EEGNet/MSVTNet extractor contract in this project.
    """

    def __init__(self, n_channels: int, n_times: int, shallow_params: Dict[str, Any]) -> None:
        super().__init__()
        p = shallow_params
        f1 = int(p.get("F1", 40))
        conv1_size = int(p.get("conv1_size", 25))
        avg_pool_size = int(p.get("avg_pool_size", 75))
        avg_pool_stride = int(p.get("avg_pool_stride", 15))
        drop = float(p.get("dropout_rate", 0.5))
        bn_momentum = float(p.get("bn_momentum", 0.1))
        bn_eps = float(p.get("bn_eps", 1e-5))
        max_norm_temporal = float(p.get("max_norm_temporal_conv", 2.0))
        max_norm_spatial = float(p.get("max_norm_spatial_conv", 2.0))
        log_eps = float(p.get("log_eps", 1e-6))

        if conv1_size > n_times:
            raise ValueError(
                f"ShallowConvNet: conv1_size={conv1_size} exceeds n_times={n_times}."
            )

        self.features = nn.Sequential(
            Conv2dWithConstraint(
                1,
                f1,
                kernel_size=(1, conv1_size),
                max_norm=max_norm_temporal,
            ),
            Conv2dWithConstraint(
                f1,
                f1,
                kernel_size=(n_channels, 1),
                bias=False,
                max_norm=max_norm_spatial,
            ),
            nn.BatchNorm2d(f1, momentum=bn_momentum, eps=bn_eps),
            SquareActivation(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_size), stride=(1, avg_pool_stride)),
            SafeLogActivation(log_eps),
            nn.Dropout(drop),
        )

        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, n_times)
            out = self.features(probe)
        self.feat_dim = int(out.reshape(1, -1).shape[1])
        if self.feat_dim <= 0:
            raise ValueError(
                "ShallowConvNet: invalid feature dimension; please reduce conv/pool sizes."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        y = self.features(x.view(b, 1, c, t))
        return y.reshape(b, -1)
