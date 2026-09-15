"""
MSVTNet 主干（多分支时域卷积 + Transformer）作为 (B,C,T) -> (B,feat_dim) 特征提取器，
用于替换 EEGNet 式 backbone，与 IDRemovalNet 其余头兼容。

实现对应论文/代码中 patch_embedding 为 "conv" 的分支（MSVTNet_TSConv），
不含 SincConv / hybrid / MTMBCNN。依赖 einops（项目 requirements 已包含）。
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class MSVTNet_TSConv(nn.Sequential):
    """原 MSVTNet 中的时域-空域可分离卷积块。"""

    def __init__(
        self,
        n_ch: int,
        f: int,
        c1: int,
        c2: int,
        d: int,
        p1: int,
        p2: int,
        pc: float,
        *,
        bn_momentum: float = 0.01,
        bn_eps: float = 1e-3,
    ) -> None:
        super().__init__(
            nn.Conv2d(1, f, (1, c1), padding="same", bias=False),
            nn.BatchNorm2d(f, momentum=bn_momentum, eps=bn_eps),
            nn.Conv2d(f, f * d, (n_ch, 1), groups=f, bias=False),
            nn.BatchNorm2d(f * d, momentum=bn_momentum, eps=bn_eps),
            nn.ELU(),
            nn.AvgPool2d((1, p1)),
            nn.Dropout(pc),
            nn.Conv2d(f * d, f * d, (1, c2), padding="same", groups=f * d, bias=False),
            nn.BatchNorm2d(f * d, momentum=bn_momentum, eps=bn_eps),
            nn.ELU(),
            nn.AvgPool2d((1, p2)),
            nn.Dropout(pc),
        )


class MSVTNet_PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class MSVTNet_Transformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        ff_ratio: int,
        pt: float = 0.5,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = MSVTNet_PositionalEncoding(seq_len + 1, d_model)
        dim_ff = int(d_model * max(1, ff_ratio))
        self.dropout = nn.Dropout(pt)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=pt,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trans = nn.TransformerEncoder(enc_layer, num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = torch.cat((self.cls_embedding.expand(b, -1, -1), x), dim=1)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        return self.trans(x)[:, 0]


class MSVTNetFeatureExtractor(nn.Module):
    """
    仅保留 MSVTNet 中「多分支 TSConv + 沿特征维拼接 + Transformer 取 CLS」路径，
    输出 flat 特征供 CrossEntropy 的线性头使用（无 LogSoftmax）。

    注意：完整论文实现里还有各分支 ``branch_head`` + 主 ``last_head`` 的 **NLL 多损失**；
    本 baseline / 隐私管线只用 **单一线性分类头 + CE**，多分支卷积的监督较弱，
    在 LOSO 下任务准确率通常会 **低于** 同数据上的纯 EEGNet，属方法差异而非 dataloader 错误。
    """

    def __init__(self, n_channels: int, n_times: int, msvt_params: Dict[str, Any]) -> None:
        super().__init__()
        self.n_ch = n_channels
        self.n_time = n_times
        p = msvt_params
        f_list: List[int] = list(p["F"])
        c1_list: List[int] = list(p["C1"])
        assert len(f_list) == len(c1_list), "MSVTNet: len(F) must equal len(C1)"

        c2 = int(p.get("C2", 15))
        d = int(p.get("D", 2))
        p1 = int(p.get("P1", 8))
        p2 = int(p.get("P2", 7))
        # 卷积内 Dropout；LOSO 数据量有限时不宜过大（原 0.3×2 + Transformer 0.5 易压任务信号）
        pc = float(p.get("Pc", p.get("dropout_rate", 0.25)))
        layers = int(p["depth"])
        nhead = int(p.get("nhead", 8))
        # Transformer FFN 宽度系数；=1 时 dim_feedforward==d_model，表达能力过弱
        ff_ratio = float(p.get("ff_ratio", 4.0))
        pt = float(p.get("Pt", 0.15))
        bn_momentum = float(p.get("bn_momentum", 0.01))
        bn_eps = float(p.get("bn_eps", 1e-3))

        patch = str(p.get("patch_embedding", "conv")).lower()
        if patch not in ("conv", "hybrid", "sinc-based"):
            patch = "conv"
        if patch != "conv":
            raise ValueError(
                "当前工程仅实现 MSVTNet patch_embedding='conv'（MSVTNet_TSConv）。"
                "请在 settings 的 params['MSVTNet'] 中设置 patch_embedding: 'conv'。"
            )

        self.mstsconv = nn.ModuleList(
            [
                nn.Sequential(
                    MSVTNet_TSConv(
                        n_channels,
                        f_list[b],
                        c1_list[b],
                        c2,
                        d,
                        p1,
                        p2,
                        pc,
                        bn_momentum=bn_momentum,
                        bn_eps=bn_eps,
                    ),
                    Rearrange("b d 1 t -> b t d"),
                )
                for b in range(len(f_list))
            ]
        )

        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, n_times)
            parts = [m(probe) for m in self.mstsconv]
            cat = torch.cat(parts, dim=2)
        seq_len, d_model = int(cat.shape[1]), int(cat.shape[2])
        if d_model % nhead != 0:
            raise ValueError(
                f"MSVTNet: d_model={d_model} 必须能被 nhead={nhead} 整除，请调整 F/D 或 nhead。"
            )

        self.transformer = MSVTNet_Transformer(
            seq_len, d_model, nhead, int(max(1, round(ff_ratio))), pt, layers
        )
        self.feat_dim = d_model
        self._init_conv_weights()

    def _init_conv_weights(self) -> None:
        for m in self.mstsconv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        xin = x.view(b, 1, c, t)
        parts = [m(xin) for m in self.mstsconv]
        h = torch.cat(parts, dim=2)
        h = self.transformer(h)
        return h.reshape(b, -1)
