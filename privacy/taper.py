from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _upper_triangular_cov(x: torch.Tensor) -> torch.Tensor:
    b, c, t = x.shape
    xc = x - x.mean(dim=-1, keepdim=True)
    cov = torch.matmul(xc, xc.transpose(1, 2)) / max(1, t - 1)
    scale = cov.diagonal(dim1=1, dim2=2).mean(dim=1, keepdim=True).clamp_min(1e-6)
    cov = cov / scale.view(b, 1, 1)
    idx = torch.triu_indices(c, c, device=x.device)
    return cov[:, idx[0], idx[1]]


def _bandpower_proxy(x: torch.Tensor, chunks: int = 8) -> torch.Tensor:
    b, c, t = x.shape
    usable = max(chunks, (t // chunks) * chunks)
    if usable > t:
        pad = usable - t
        x = F.pad(x, (0, pad), mode="replicate")
    y = x[:, :, :usable].reshape(b, c, chunks, usable // chunks)
    local_power = torch.log1p(y.pow(2).mean(dim=-1))
    diff_power = torch.log1p((x[:, :, 1:] - x[:, :, :-1]).pow(2).mean(dim=-1, keepdim=True))
    return torch.cat([local_power.reshape(b, -1), diff_power.reshape(b, -1)], dim=1)


def _temporal_stats(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1)
    std = x.std(dim=-1).clamp_min(1e-6)
    peak = x.abs().amax(dim=-1)
    slope = (x[:, :, -1] - x[:, :, 0]) / max(1, x.shape[-1] - 1)
    return torch.cat([mean, std, peak, slope], dim=1)


class TAPERProbeBank(nn.Module):
    """
    TAPER: Task-Anchored Privacy Evidence Removal.

    The probes are trained on raw source EEG and frozen during anonymizer
    training. They expose identity evidence that transfers across simple
    covariance, band-power, and temporal-statistics views.
    """

    def __init__(self, n_channels: int, n_times: int, n_ids: int, n_task: int = 0):
        super().__init__()
        cov_dim = n_channels * (n_channels + 1) // 2
        bp_dim = n_channels * 8 + n_channels
        stat_dim = n_channels * 4
        self.n_ids = int(n_ids)
        self.n_task = int(n_task)
        self.cov_probe = nn.Linear(cov_dim, n_ids)
        self.band_probe = nn.Linear(bp_dim, n_ids)
        self.stat_probe = nn.Linear(stat_dim, n_ids)
        self.register_buffer("cov_template", torch.zeros(cov_dim))
        self.register_buffer("cov_template_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("task_cov_templates", torch.zeros(max(0, self.n_task), cov_dim))
        self.register_buffer("task_cov_template_ready", torch.zeros(max(0, self.n_task), dtype=torch.bool))

    @staticmethod
    def normalize_views(active_views: str | Sequence[str] | None) -> Tuple[str, ...]:
        if active_views is None:
            return ("cov", "band", "stat")
        if isinstance(active_views, str):
            raw = active_views.replace("+", ",").split(",")
            views = tuple(v.strip().lower() for v in raw if v.strip())
        else:
            views = tuple(str(v).strip().lower() for v in active_views if str(v).strip())
        aliases = {"all": ("cov", "band", "stat"), "full": ("cov", "band", "stat")}
        if len(views) == 1 and views[0] in aliases:
            return aliases[views[0]]
        allowed = {"cov", "band", "stat"}
        bad = [v for v in views if v not in allowed]
        if bad:
            raise ValueError(f"Unknown TAPER views: {bad}; allowed: cov,band,stat,all")
        return views or ("cov", "band", "stat")

    def features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _upper_triangular_cov(x), _bandpower_proxy(x), _temporal_stats(x)

    def logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cov, band, stat = self.features(x)
        return self.cov_probe(cov), self.band_probe(band), self.stat_probe(stat)

    def fit_cov_template(self, loader: Iterable, device: torch.device) -> None:
        total = None
        task_total = None
        task_count = None
        n = 0
        with torch.no_grad():
            for batch in loader:
                cov = _upper_triangular_cov(batch[0].to(device))
                batch_sum = cov.sum(dim=0)
                total = batch_sum if total is None else total + batch_sum
                n += cov.size(0)
                if self.n_task > 0:
                    y_task = batch[1].to(device).long()
                    if task_total is None:
                        task_total = torch.zeros(self.n_task, cov.size(1), device=device, dtype=cov.dtype)
                        task_count = torch.zeros(self.n_task, device=device, dtype=cov.dtype)
                    for cls in y_task.unique():
                        ci = int(cls.item())
                        if 0 <= ci < self.n_task:
                            mask = y_task == ci
                            task_total[ci].add_(cov[mask].sum(dim=0))
                            task_count[ci].add_(mask.sum().to(dtype=cov.dtype))
        if total is not None and n > 0:
            self.cov_template.copy_(total / float(n))
            self.cov_template_ready.fill_(True)
        if task_total is not None and task_count is not None:
            ready = task_count > 0
            if bool(ready.any().item()):
                denom = task_count.clamp_min(1.0).view(-1, 1)
                templates = task_total / denom
                self.task_cov_templates.copy_(templates.to(self.task_cov_templates.device))
                self.task_cov_template_ready.copy_(ready.to(self.task_cov_template_ready.device))

    def cov_alignment_loss(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.cov_template_ready.item()):
            return x.new_zeros(())
        cov = _upper_triangular_cov(x)
        target = self.cov_template.to(device=x.device, dtype=cov.dtype).view(1, -1)
        return F.smooth_l1_loss(cov, target.expand_as(cov))

    def task_cov_alignment_loss(self, x: torch.Tensor, y_task: torch.Tensor) -> torch.Tensor:
        if self.n_task <= 0 or self.task_cov_templates.numel() == 0:
            return self.cov_alignment_loss(x)
        cov = _upper_triangular_cov(x)
        y = y_task.long().clamp(0, self.n_task - 1)
        templates = self.task_cov_templates.to(device=x.device, dtype=cov.dtype)
        target = templates[y]
        ready = self.task_cov_template_ready.to(device=x.device)
        if bool(self.cov_template_ready.item()):
            fallback = self.cov_template.to(device=x.device, dtype=cov.dtype).view(1, -1).expand_as(target)
            target = torch.where(ready[y].view(-1, 1), target, fallback)
        return F.smooth_l1_loss(cov, target)

    def probe_loss(
        self,
        x: torch.Tensor,
        y_id: torch.Tensor,
        *,
        active_views: str | Sequence[str] | None = None,
    ) -> torch.Tensor:
        views = self.normalize_views(active_views)
        logits_by_view = dict(zip(("cov", "band", "stat"), self.logits(x)))
        return sum(F.cross_entropy(logits_by_view[v], y_id) for v in views) / float(len(views))

    def privacy_losses(
        self,
        x: torch.Tensor,
        y_task: torch.Tensor,
        task_logits: torch.Tensor,
        *,
        entropy_target_ratio: float = 0.98,
        active_views: str | Sequence[str] | None = None,
        privacy_mode: str = "hybrid",
        cov_align_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        views = self.normalize_views(active_views)
        privacy_mode = str(privacy_mode).strip().lower()
        if privacy_mode not in {"entropy", "cov_align", "task_cov_align", "hybrid", "task_hybrid"}:
            raise ValueError("privacy_mode must be entropy, cov_align, task_cov_align, hybrid, or task_hybrid")
        logits_by_view = dict(zip(("cov", "band", "stat"), self.logits(x)))
        probs = [F.softmax(logits_by_view[v], dim=1) for v in views]
        log_k = math.log(float(self.n_ids))
        entropies = [-(p * p.clamp_min(1e-8).log()).sum(dim=1) for p in probs]
        mean_entropy = torch.stack(entropies, dim=0).mean(dim=0)

        with torch.no_grad():
            task_prob = F.softmax(task_logits, dim=1)
            task_conf = task_prob.gather(1, y_task.view(-1, 1)).squeeze(1)
            identity_evidence = (1.0 - mean_entropy / log_k).clamp(0.0, 1.0)
            gate = (identity_evidence * (1.0 - task_conf).clamp(0.0, 1.0)).detach()

        target = float(entropy_target_ratio) * log_k
        entropy_deficit = F.relu(target - mean_entropy)
        loss_entropy = (gate * entropy_deficit).mean()

        loss_agreement = x.new_zeros(())
        pair_count = 0
        for i in range(len(probs)):
            for j in range(i + 1, len(probs)):
                loss_agreement = loss_agreement + (probs[i] * probs[j]).sum(dim=1).mean()
                pair_count += 1
        if pair_count:
            loss_agreement = loss_agreement / float(pair_count)

        loss_cov_align = self.cov_alignment_loss(x) if "cov" in views else x.new_zeros(())
        loss_task_cov_align = self.task_cov_alignment_loss(x, y_task) if "cov" in views else x.new_zeros(())
        if privacy_mode == "entropy":
            total = loss_entropy + loss_agreement
        elif privacy_mode == "cov_align":
            total = float(cov_align_weight) * loss_cov_align
        elif privacy_mode == "task_cov_align":
            total = float(cov_align_weight) * loss_task_cov_align
        elif privacy_mode == "task_hybrid":
            total = loss_entropy + loss_agreement + float(cov_align_weight) * loss_task_cov_align
        else:
            total = loss_entropy + loss_agreement + float(cov_align_weight) * loss_cov_align
        stats = {
            "taper_entropy": mean_entropy.mean().detach(),
            "taper_gate": gate.mean().detach(),
            "taper_loss_entropy": loss_entropy.detach(),
            "taper_loss_agreement": loss_agreement.detach(),
            "taper_loss_cov_align": loss_cov_align.detach(),
            "taper_loss_task_cov_align": loss_task_cov_align.detach(),
            "taper_active_views": ",".join(views),
            "taper_privacy_mode": privacy_mode,
        }
        return total, stats


@dataclass
class TAPERProbeReport:
    cov_acc: float
    band_acc: float
    stat_acc: float


def _probe_accuracy(bank: TAPERProbeBank, loader: Iterable, device: torch.device) -> TAPERProbeReport:
    bank.eval()
    correct = [0, 0, 0]
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y_id = batch[2].to(device)
            for i, logits in enumerate(bank.logits(x)):
                correct[i] += (logits.argmax(dim=1) == y_id).sum().item()
            total += x.size(0)
    denom = max(1, total)
    return TAPERProbeReport(*(100.0 * c / denom for c in correct))


def train_taper_probe_bank(
    bank: TAPERProbeBank,
    train_loader: Iterable,
    val_loader: Iterable,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    active_views: str | Sequence[str] | None = None,
) -> TAPERProbeReport:
    bank.to(device)
    bank.fit_cov_template(train_loader, device)
    opt = torch.optim.Adam(bank.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(max(0, int(epochs))):
        bank.train()
        for batch in train_loader:
            x = batch[0].to(device)
            y_id = batch[2].to(device)
            opt.zero_grad()
            loss = bank.probe_loss(x, y_id, active_views=active_views)
            loss.backward()
            opt.step()
    report = _probe_accuracy(bank, val_loader, device)
    for p in bank.parameters():
        p.requires_grad = False
    bank.eval()
    return report
