"""
[IJCAI 2025] ID-RemovalNet 风格 IDS 近似实现（接口兼容版）:
- MDML: 时间域多尺度 + 频域多尺度分支融合
- Dec: 任务/身份支路去相关
- Sub: 从融合特征中减去身份分量
- LITFR: 按身份强度自适应回融（尽量保任务）
- AAFS: 注意力强调主导特征
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _best_gn_groups(ch: int, prefer: int = 4) -> int:
    """选一个能整除通道数的 GroupNorm groups（至少 1）。"""
    for g in (prefer, 8, 4, 2, 1):
        if g <= ch and ch % g == 0:
            return g
    return 1


class _MultiKernelBranch(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False) for k in kernels]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(out_ch * len(kernels), out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_best_gn_groups(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ys = [b(x) for b in self.branches]
        y = torch.cat(ys, dim=1)
        return self.mix(y)


class IdentityDecorrelationSeparation(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        bottleneck: int = 256,
        k_dim: int = 32,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        hidden = max(16, min(64, k_dim))

        # MDML: time-domain + frequency-domain multi-kernel branches
        self.time_branch = _MultiKernelBranch(
            in_ch=n_channels, out_ch=hidden, kernels=(3, 7, 15)
        )
        self.freq_branch = _MultiKernelBranch(
            in_ch=n_channels, out_ch=hidden, kernels=(3, 9, 17)
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden * 2, n_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_best_gn_groups(n_channels), n_channels),
            nn.ReLU(inplace=True),
        )

        # IDS heads
        self.task_proj = nn.Conv1d(n_channels, k_dim, kernel_size=1, bias=False)
        self.id_proj = nn.Conv1d(n_channels, k_dim, kernel_size=1, bias=False)
        self.id_decode = nn.Sequential(
            nn.Conv1d(k_dim, bottleneck // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck // 4, n_channels, kernel_size=1, bias=False),
        )

        # LITFR + AAFS
        self.attn = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=True), nn.Sigmoid()
        )
        self.sub_lambda = nn.Parameter(torch.tensor(0.55))
        self.refuse_gain = nn.Parameter(torch.tensor(0.25))
        self.dom_gain = nn.Parameter(torch.tensor(0.35))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def _pearson_corr_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a - a.mean(dim=0, keepdim=True)
        b = b - b.mean(dim=0, keepdim=True)
        sa = a.std(dim=0) + 1e-8
        sb = b.std(dim=0) + 1e-8
        rho = ((a * b).mean(dim=0) / (sa * sb)) ** 2
        return rho.mean()

    def _mdml_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        ft = self.time_branch(x)
        # 避免在部分 CUDA 环境触发 torch.fft 对 nvrtc-builtins 的依赖:
        # 用“去平滑残差”近似高频强度，作为频域分支输入。
        x_lp = F.avg_pool1d(x, kernel_size=9, stride=1, padding=4)
        xf = torch.abs(x - x_lp)
        ff = self.freq_branch(xf)
        return self.fuse(torch.cat([ft, ff], dim=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, T) 匿名器输出
        返回: x_clean (B,C,T), loss_dec (标量)
        """
        f = self._mdml_features(x)
        t_map = self.task_proj(f)
        i_map = self.id_proj(f)

        h_t = t_map.mean(dim=-1)
        h_i = i_map.mean(dim=-1)
        loss_dec = self._pearson_corr_sq(h_t, h_i)

        # Sub: remove identity component
        id_res = self.id_decode(i_map)
        lam = torch.sigmoid(self.sub_lambda) * 1.5
        f_task = f - lam * id_res

        # LITFR: identity-strength guided re-fusion
        id_energy = (h_i**2).mean(dim=1, keepdim=True)
        tau = torch.clamp(self.temp.abs(), min=0.25, max=4.0)
        w = torch.sigmoid(-id_energy / tau).unsqueeze(-1)
        f_refused = f_task + torch.tanh(self.refuse_gain) * w * id_res

        # AAFS-like dominant feature emphasis
        a = self.attn(f_refused)
        dom = a * f_refused
        weak = (1.0 - a) * f_refused
        f_enh = dom - 0.25 * weak

        # Residual clean output
        out = x + torch.tanh(self.dom_gain) * f_enh
        return out, loss_dec
