import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from settings import (
    dataset_name,
    preprocessing_params,
    params,
    seed,
    start_subject,
    end_subject,
    dataset,
    use_models,
)
from privacy.eeg_euclidean_align import apply_euclidean_alignment, fit_ea_from_trials


def get_preprocessed_data(*args, **kwargs):
    raise NotImplementedError(
        "The raw EEG preprocessing/data interface is intentionally not included in this method-code release. "
        "Please provide subject-wise preprocessed arrays matching the README input contract."
    )


def make_dataset_metadata(name: str, *, tmin: float = 0.0, tmax: float = 4.0):
    return SimpleNamespace(name=name, tmin=tmin, tmax=tmax)
from privacy.ids_module import IdentityDecorrelationSeparation
from privacy.taper import TAPERProbeBank, train_taper_probe_bank


def apply_dataset_override(args: argparse.Namespace) -> None:
    global dataset_name, dataset, start_subject, end_subject
    selected = str(getattr(args, "dataset_name", dataset_name)).strip()
    if selected in {"", "default", "settings"}:
        selected = dataset_name
    if selected == dataset_name:
        return
    if selected == "BCICIV_2a":
        dataset = make_dataset_metadata("BCICIV_2a", tmin=0, tmax=4.0)
    elif selected == "BCICIV_2b":
        dataset = make_dataset_metadata("BCICIV_2b", tmin=0, tmax=4.0)
    elif selected in {"OPENBMI_P300", "OpenBMI_P300", "openbmi_p300", "NEMAR_NM000323", "nm000323"}:
        dataset = SimpleNamespace(
            name="OPENBMI_P300",
            tmin=0.0,
            tmax=0.8,
            sample_rate=250,
            n_subjects=54,
            n_electrodes=64,
            n_classes=2,
            class_names=["NonTarget", "Target"],
            n_sessions_per_subject=1,
        )
    else:
        raise ValueError(f"Unknown dataset_name: {selected}")
    dataset_name = dataset.name
    start_subject = 1
    end_subject = int(getattr(dataset, "n_subjects", 9))
    params["DL_Training"]["dataset_name"] = dataset_name
    params["experiment"]["start_subject"] = start_subject
    params["experiment"]["end_subject"] = end_subject
    params["experiment"]["sessions_usage"]["train"] = [1]


def apply_backbone_override(args: argparse.Namespace) -> None:
    selected = str(getattr(args, "backbone", "auto")).strip()
    if selected == "auto":
        selected = get_selected_backbone()
    if selected not in {"EEGNet", "MSVTNet", "ShallowConvNet", "FBCNet"}:
        raise ValueError(f"Unknown backbone: {selected}")
    use_models["EEGNet"] = selected == "EEGNet"
    use_models["MSVTNet"] = selected == "MSVTNet"
    use_models["ShallowConvNet"] = selected == "ShallowConvNet"
    use_models["FBCNet"] = selected == "FBCNet"


def get_selected_backbone() -> str:
    if use_models.get("FBCNet", False):
        return "FBCNet"
    if use_models.get("ShallowConvNet", False):
        return "ShallowConvNet"
    if use_models.get("MSVTNet", False):
        return "MSVTNet"
    return "EEGNet"


def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _json_safe(value: Any) -> Any:
    """将运行时对象转为可 JSON 序列化的值。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def parse_seed_list(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_seed = args.seed if args.seed is not None else seed
    return [run_seed]


def score_seed_run(task_mean: float, id_mean: float, best_by: str) -> float:
    if best_by == "task":
        return task_mean
    if best_by == "id":
        return -id_mean
    return task_mean - 0.5 * id_mean


def collect_run_hyperparams(args: argparse.Namespace, run_seed: int) -> Dict[str, Any]:
    """汇总 CLI 参数与 settings.py 中的配置，便于日志复现。"""
    backbone = get_selected_backbone()
    dl_training = dict(params["DL_Training"])
    dl_training["device"] = str(dl_training.get("device", ""))
    return {
        "cli": {k: _json_safe(v) for k, v in sorted(vars(args).items())},
        "settings": {
            "seed": run_seed,
            "dataset_name": dataset_name,
            "dataset_resolved_name": dataset.name,
            "tmin": getattr(dataset, "tmin", None),
            "tmax": getattr(dataset, "tmax", None),
            "start_subject": start_subject,
            "end_subject": end_subject,
            "backbone": backbone,
            "use_models": _json_safe(dict(use_models)),
            "preprocessing": _json_safe(dict(preprocessing_params)),
            "model_params": _json_safe(dict(params[backbone])),
            "training_params": _json_safe(dl_training),
            "experiment": _json_safe(dict(params.get("experiment", {}))),
        },
    }


def print_hyperparams_report(hyperparams: Dict[str, Any]) -> None:
    print("\n================= All Hyperparameters =================")
    print(json.dumps(hyperparams, ensure_ascii=False, indent=2, sort_keys=True))
    print("=" * 55)


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GRLFn.apply(x, lambd)


class EEGFeatureExtractor(nn.Module):
    def __init__(self, n_channels: int, n_times: int):
        super().__init__()
        eeg = params["EEGNet"]
        f1 = int(eeg["F1"])
        d = int(eeg["D"])
        f2 = f1 * d
        k = int(eeg["kernel_length"])
        dconv2 = int(eeg["dconv2_size"])
        p1 = int(eeg["avg_pool1_size"])
        p2 = int(eeg["avg_pool2_size"])
        drop = float(eeg["dropout_rate"])

        self.temporal = nn.Conv2d(1, f1, kernel_size=(1, k), padding=(0, k // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1, momentum=0.01, eps=1e-3)
        self.spatial = nn.Conv2d(f1, f2, kernel_size=(n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2, momentum=0.01, eps=1e-3)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, p1))
        self.drop1 = nn.Dropout(drop)
        self.dconv = nn.Conv2d(
            f2, f2, kernel_size=(1, dconv2), padding=(0, dconv2 // 2), groups=f2, bias=False
        )
        self.pconv = nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2, momentum=0.01, eps=1e-3)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, p2))
        self.drop2 = nn.Dropout(drop)
        out_t = n_times // p1 // p2
        self.feat_dim = f2 * out_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        y = self.temporal(x.view(b, 1, c, t))
        y = self.bn1(y)
        y = self.spatial(y)
        y = self.bn2(y)
        y = self.elu1(y)
        y = self.pool1(y)
        y = self.drop1(y)
        y = self.dconv(y)
        y = self.pconv(y)
        y = self.bn3(y)
        y = self.elu2(y)
        y = self.pool2(y)
        y = self.drop2(y)
        return y.reshape(b, -1)


def build_backbone_extractor(n_channels: int, n_times: int) -> nn.Module:
    """由 settings.use_models 选择 EEGNet、MSVTNet 或 ShallowConvNet 主干。"""
    if use_models.get("FBCNet", False):
        from privacy.fbcnet_extractor import FBCNetFeatureExtractor

        return FBCNetFeatureExtractor(n_channels, n_times, dict(params["FBCNet"]))
    if use_models.get("ShallowConvNet", False):
        from privacy.shallowconvnet_extractor import ShallowConvNetFeatureExtractor

        return ShallowConvNetFeatureExtractor(n_channels, n_times, dict(params["ShallowConvNet"]))
    if use_models.get("MSVTNet", False):
        from privacy.msvtnet_extractor import MSVTNetFeatureExtractor

        return MSVTNetFeatureExtractor(n_channels, n_times, dict(params["MSVTNet"]))
    return EEGFeatureExtractor(n_channels, n_times)


class IDRemovalNet(nn.Module):
    """
    论文思想近似复现：
    - shared extractor
    - IDS（身份去相关 + subtraction + enhancement）
    - task cls + identity cls with GRL
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_task: int,
        n_ids: int,
        ids_bottleneck: int = 256,
        ids_k_dim: int = 32,
        raw_inject_scale: float = 1.0,
    ):
        super().__init__()
        self.ids = IdentityDecorrelationSeparation(
            n_channels=n_channels, n_times=n_times, bottleneck=ids_bottleneck, k_dim=ids_k_dim
        )
        self.extractor = build_backbone_extractor(n_channels, n_times)
        # 任务补偿分支：对原始输入先做实例归一化，再轻量通道适配，补回 IDS 造成的任务信息损失
        self.raw_inst_norm = nn.InstanceNorm1d(n_channels, affine=False, eps=1e-5)
        self.raw_adapter = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, n_channels),
            nn.ELU(inplace=True),
        )
        self.extractor_raw = build_backbone_extractor(n_channels, n_times)
        fd = self.extractor.feat_dim
        self.fuse_gate = nn.Sequential(
            nn.Linear(fd * 2, fd),
            nn.ReLU(inplace=True),
            nn.Linear(fd, 1),
            nn.Sigmoid(),
        )
        self.raw_inject_logit = nn.Parameter(torch.tensor(-0.8))
        self.raw_inject_scale = float(raw_inject_scale)
        self.attn = nn.Sequential(
            nn.Linear(fd, fd // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fd // 2, fd),
            nn.Sigmoid(),
        )
        self.task_head_total = nn.Linear(fd, n_task)
        sub_dim = max(128, (fd * 2) // 5)
        self.task_proj = nn.Sequential(
            nn.Linear(fd, sub_dim, bias=False),
            nn.LayerNorm(sub_dim),
            nn.GELU(),
            nn.Linear(sub_dim, sub_dim, bias=False),
            nn.LayerNorm(sub_dim),
            nn.GELU(),
        )
        self.task_head_sub = nn.Linear(sub_dim, n_task)
        self.id_head_sub = nn.Linear(sub_dim, n_ids)
        self.task_sub_mix = nn.Parameter(torch.tensor(0.0))
        self.task_proto = nn.Parameter(torch.empty(n_task, fd))
        nn.init.xavier_uniform_(self.task_proto)
        self.register_buffer("ema_proto", torch.zeros(n_task, fd))
        self.register_buffer("ema_proto_ready", torch.zeros(n_task))
        self.proto_scale = nn.Parameter(torch.tensor(10.0))
        self.logit_mix = nn.Parameter(torch.tensor(0.0))
        self.proto_ema_mix = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("release_basis", torch.empty(0, sub_dim))
        self.release_projector_strength = 0.0

    def _proto_logits(self, f: torch.Tensor) -> torch.Tensor:
        f_n = F.normalize(f, dim=1)
        p_param = F.normalize(self.task_proto, dim=1)
        p_ema = F.normalize(self.ema_proto + 1e-6, dim=1)
        w = torch.sigmoid(self.proto_ema_mix)
        p_n = F.normalize((1.0 - w) * p_param + w * p_ema, dim=1)
        return torch.matmul(f_n, p_n.t()) * self.proto_scale.clamp(1.0, 30.0)

    @torch.no_grad()
    def update_ema_proto(self, f_total: torch.Tensor, y: torch.Tensor, momentum: float = 0.97) -> None:
        f_n = F.normalize(f_total, dim=1)
        for c in y.unique():
            ci = int(c.item())
            m = y == ci
            if not torch.any(m):
                continue
            cls_mean = F.normalize(f_n[m].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            if self.ema_proto_ready[ci] < 0.5:
                self.ema_proto[ci].copy_(cls_mean)
                self.ema_proto_ready[ci].fill_(1.0)
            else:
                self.ema_proto[ci].mul_(momentum).add_((1.0 - momentum) * cls_mean)

    def forward(self, x: torch.Tensor, lambd: float = 0.0):
        x_clean, loss_dec = self.ids(x)
        f_clean = self.extractor(x_clean)

        x_rawn = self.raw_adapter(self.raw_inst_norm(x))
        f_raw = self.extractor_raw(x_rawn)

        g = self.fuse_gate(torch.cat([f_clean, f_raw], dim=1))
        beta = self.raw_inject_scale * torch.sigmoid(self.raw_inject_logit)
        # 受限注入：raw 仅做小幅任务补偿，避免身份特征大规模回流
        f = f_clean + beta * g * (f_raw - f_clean)

        a = self.attn(f)
        f_total = f + 0.5 * (a * f)
        z_task = self.task_proj(f_total)
        task_logits_linear = self.task_head_total(f_total)
        task_logits_proto = self._proto_logits(f_total)
        task_logits_sub = self.task_head_sub(z_task)
        m = torch.sigmoid(self.logit_mix)
        base_task_logits = (1.0 - m) * task_logits_linear + m * task_logits_proto
        ms = torch.sigmoid(self.task_sub_mix)
        task_logits = (1.0 - ms) * base_task_logits + ms * task_logits_sub
        id_logits_sub_adv = self.id_head_sub(grl(z_task, lambd))
        return (
            task_logits,
            id_logits_sub_adv,
            loss_dec,
            task_logits_linear,
            task_logits_proto,
            task_logits_sub,
            f_total,
            z_task,
        )

    def id_forward(self, x: torch.Tensor):
        with torch.no_grad():
            x_clean, _ = self.ids(x)
            f_clean = self.extractor(x_clean)
            x_rawn = self.raw_adapter(self.raw_inst_norm(x))
            f_raw = self.extractor_raw(x_rawn)
            g = self.fuse_gate(torch.cat([f_clean, f_raw], dim=1))
            beta = self.raw_inject_scale * torch.sigmoid(self.raw_inject_logit)
            f = f_clean + beta * g * (f_raw - f_clean)
            a = self.attn(f)
            f_total = f + 0.5 * (a * f)
            z_task = self.task_proj(f_total)
        return self.id_head_sub(z_task.detach())

    def configure_release_projector(self, basis: torch.Tensor, strength: float) -> None:
        if basis.numel() == 0 or strength <= 0.0:
            self.release_basis = torch.empty(
                0, self.task_head_sub.in_features, device=self.release_basis.device
            )
            self.release_projector_strength = 0.0
            return
        self.release_basis = F.normalize(basis.detach().to(self.release_basis.device), dim=1)
        self.release_projector_strength = float(strength)

    def apply_release_projector(self, z_task: torch.Tensor) -> torch.Tensor:
        if self.release_projector_strength <= 0.0 or self.release_basis.numel() == 0:
            return z_task
        basis = self.release_basis.to(device=z_task.device, dtype=z_task.dtype)
        coeff = torch.matmul(z_task, basis.t())
        projected = torch.matmul(coeff, basis)
        return z_task - float(self.release_projector_strength) * projected

    def extract(self, x: torch.Tensor, apply_release_projector: bool = True):
        x_clean, _ = self.ids(x)
        f_clean = self.extractor(x_clean)
        x_rawn = self.raw_adapter(self.raw_inst_norm(x))
        f_raw = self.extractor_raw(x_rawn)
        g = self.fuse_gate(torch.cat([f_clean, f_raw], dim=1))
        beta = self.raw_inject_scale * torch.sigmoid(self.raw_inject_logit)
        f = f_clean + beta * g * (f_raw - f_clean)
        a = self.attn(f)
        f_total = f + 0.5 * (a * f)
        z_task = self.task_proj(f_total)
        if apply_release_projector:
            z_task = self.apply_release_projector(z_task)
        # 与 forward 中任务分类所用表征一致，否则攻击器/任务评估不对齐
        return F.normalize(z_task, dim=1)

    def release_signal(self, x: torch.Tensor) -> torch.Tensor:
        """Return the EEG-like released signal for unseen architecture attacks."""
        x_clean, _ = self.ids(x)
        return x_clean


@dataclass
class FoldResult:
    test_subject: int
    anon_task_acc: float
    anon_id_acc: float
    anon_eval_mode: str = ""


def load_subject_data() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if dataset_name == "OPENBMI_P300":
        cache_path = os.environ.get("OPENBMI_P300_CACHE", "Data/openbmi_p300_ses1_cache.npz")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"OpenBMI P300 cache not found: {cache_path}. "
                "Provide a preprocessed cache matching the README input contract, or set OPENBMI_P300_CACHE."
            )
        cached = np.load(cache_path, allow_pickle=False)
        X = cached["X"].astype(np.float32)
        y_task = cached["y"].astype(np.int64)
        if X.ndim != 4:
            raise ValueError(f"Expected OpenBMI P300 X shape (subjects,trials,channels,time), got {X.shape}")
        if y_task.shape[:2] != X.shape[:2]:
            raise ValueError(f"Expected OpenBMI P300 y shape {X.shape[:2]}, got {y_task.shape}")
        return (
            [X[i] for i in range(X.shape[0])],
            [y_task[i] for i in range(y_task.shape[0])],
            [np.full((X.shape[1],), i, dtype=np.int64) for i in range(X.shape[0])],
        )

    experiment_type = "inter_subject"
    X_list, y_task_list, y_subject_list = [], [], []
    for subject_idx in range(start_subject - 1, end_subject):
        subject_name = str(subject_idx + 1)
        subject_session_name = subject_name + "s1"
        print(f"Loading subject {subject_session_name} ...")
        dataset.load([subject_idx + 1], session_list=[1])
        X_train_tmp, y_train_tmp, _, _, _ = get_preprocessed_data(
            experiment_type=experiment_type,
            dataset_name=dataset_name,
            subject_session_name=subject_session_name,
            subject_idx=subject_idx,
            dataset=dataset,
            params=params,
            preprocessing_params=preprocessing_params,
            X_train=[],
            info_list=[],
            Y_train=[],
            X_test=[],
            Y_test=[],
            seed=seed,
        )
        n_trials = X_train_tmp.shape[0]
        subject_ids = np.full((n_trials,), subject_idx, dtype=np.int64)
        X_list.append(X_train_tmp.astype(np.float32))
        y_task_list.append(y_train_tmp.astype(np.int64))
        y_subject_list.append(subject_ids)
    return X_list, y_task_list, y_subject_list


def anon_eval_logits(model_out: tuple, mode: str, *, temp: float = 1.0) -> torch.Tensor:
    """
    从 IDRemovalNet.forward 的完整返回值中取「任务评估用」logits。
    mean4：logit 空间平均；prob_mean4：softmax 概率平均再取 log（与 mean4 argmax 可能不同，常略好）。
    """
    (
        task_logits,
        _,
        _,
        task_logits_linear,
        task_logits_proto,
        task_logits_sub,
        _,
        _,
    ) = model_out
    m = (mode or "fused").lower()
    t = max(1e-4, float(temp))
    if m == "fused":
        return task_logits
    if m == "mean4":
        return (task_logits + task_logits_linear + task_logits_proto + task_logits_sub) * 0.25
    if m == "mean3":
        return (task_logits_linear + task_logits_proto + task_logits_sub) / 3.0
    if m == "prob_mean4":
        p = (
            F.softmax(task_logits / t, dim=1)
            + F.softmax(task_logits_linear / t, dim=1)
            + F.softmax(task_logits_proto / t, dim=1)
            + F.softmax(task_logits_sub / t, dim=1)
        ) * 0.25
        return torch.log(p.clamp_min(1e-8))
    if m == "prob_mean3":
        p = (
            F.softmax(task_logits_linear / t, dim=1)
            + F.softmax(task_logits_proto / t, dim=1)
            + F.softmax(task_logits_sub / t, dim=1)
        ) / 3.0
        return torch.log(p.clamp_min(1e-8))
    if m == "stacked":
        return (
            0.25 * task_logits
            + 0.35 * task_logits_linear
            + 0.2 * task_logits_proto
            + 0.2 * task_logits_sub
        )
    if m == "prob_stacked":
        p = (
            0.25 * F.softmax(task_logits / t, dim=1)
            + 0.35 * F.softmax(task_logits_linear / t, dim=1)
            + 0.2 * F.softmax(task_logits_proto / t, dim=1)
            + 0.2 * F.softmax(task_logits_sub / t, dim=1)
        )
        return torch.log(p.clamp_min(1e-8))
    raise ValueError(f"unknown eval_task_logits: {mode}")


def eval_logits_tensor(model_out: tuple, args) -> torch.Tensor:
    """与 eval_privacy_task_logits 单次前向一致（eval_auto_fallback + eval_ensemble_temp）。"""
    mode = getattr(args, "eval_auto_fallback", "prob_mean4")
    tau = float(getattr(args, "eval_ensemble_temp", 1.0))
    return anon_eval_logits(model_out, mode, temp=tau)


def apply_train_input_noise(
    x: torch.Tensor,
    args,
    *,
    noise_std: float,
    active: bool,
) -> torch.Tensor:
    """训练期对输入加高斯噪声（提升对扰动的鲁棒性）；验证/测试不加。"""
    if not active or noise_std <= 0.0:
        return x
    p = float(getattr(args, "train_input_noise_prob", 1.0))
    if p < 1.0 and torch.rand(1, device=x.device).item() > p:
        return x
    return x + float(noise_std) * torch.randn_like(x)


def eval_privacy_task_logits(
    model: nn.Module, x: torch.Tensor, args, logits_mode: Optional[str] = None
) -> torch.Tensor:
    """验证/测试用：可选 TTA；集成温度 eval_ensemble_temp 作用于 prob_mean*。logits_mode 非空时覆盖 args.eval_task_logits。"""
    mode = logits_mode if logits_mode is not None else getattr(args, "eval_task_logits", "prob_mean4")
    if mode == "auto":
        mode = getattr(args, "eval_auto_fallback", "prob_mean4")
    tau = float(getattr(args, "eval_ensemble_temp", 1.0))

    def _forward_once(xin: torch.Tensor) -> torch.Tensor:
        out = model(xin, lambd=0.0)
        return anon_eval_logits(out, mode, temp=tau)

    if int(getattr(args, "eval_tta", 0)) != 1:
        return _forward_once(x)
    n = max(1, int(getattr(args, "eval_tta_n", 5)))
    std = float(getattr(args, "eval_tta_std", 0.03))
    acc = None
    for i in range(n):
        xin = x if i == 0 else x + std * torch.randn_like(x)
        lg = _forward_once(xin)
        acc = lg if acc is None else acc + lg
    return acc / float(n)


def val_anon_task_acc_for_mode(
    model: nn.Module,
    loader: DataLoader,
    args,
    device: torch.device,
    logits_mode: str,
) -> float:
    """在验证集上算匿名任务准确率（用于 auto 选集成方式，不偷看测试集）。"""
    model.eval()
    c, tot = 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y_task = batch[1].to(device)
            logits = eval_privacy_task_logits(model, x, args, logits_mode=logits_mode)
            pred = logits.argmax(dim=1)
            c += (pred == y_task).sum().item()
            tot += x.size(0)
    return 100.0 * c / max(1, tot)


def pick_eval_task_logits_on_val(
    model: nn.Module,
    va_loader: DataLoader,
    args,
    device: torch.device,
) -> Tuple[str, float]:
    valid_modes = {
        "fused",
        "mean3",
        "mean4",
        "prob_mean3",
        "prob_mean4",
        "stacked",
        "prob_stacked",
    }
    candidates = [
        s.strip().lower()
        for s in str(getattr(args, "eval_auto_candidates", "")).split(",")
        if s.strip()
    ]
    candidates = [m for m in candidates if m in valid_modes]
    if not candidates:
        candidates = ["prob_stacked", "prob_mean4", "stacked", "mean4", "fused"]
    best_m, best_acc = candidates[0], -1.0
    for m in candidates:
        acc = val_anon_task_acc_for_mode(model, va_loader, args, device, m)
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m, best_acc


def train_attacker(
    extractor_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_ids: int,
    args,
    device: torch.device,
    id_inv_map: Optional[Dict[int, int]] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    sample_x, _ = next(iter(train_loader))
    with torch.no_grad():
        feat_dim = extractor_fn(sample_x[:1].to(device)).shape[1]
    attacker = nn.Linear(feat_dim, n_ids).to(device)
    opt = torch.optim.Adam(attacker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss().to(device)

    for _ in range(args.attacker_epochs):
        attacker.train()
        for x, y_id in train_loader:
            x = x.to(device)
            y_id = y_id.to(device)
            with torch.no_grad():
                f = extractor_fn(x)
            opt.zero_grad()
            loss = ce(attacker(f), y_id)
            loss.backward()
            opt.step()

    attacker.eval()
    y_true_chunks: List[np.ndarray] = []
    y_pred_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for x, y_id in val_loader:
            x = x.to(device)
            y_id = y_id.to(device)
            f = extractor_fn(x)
            pred = attacker(f).argmax(dim=1)
            y_true_chunks.append(y_id.cpu().numpy())
            y_pred_chunks.append(pred.cpu().numpy())
    y_true_r = np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
    y_pred_r = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)
    if id_inv_map is not None and y_true_r.size > 0:
        y_true = np.array([id_inv_map[int(v)] for v in y_true_r], dtype=np.int64)
        y_pred = np.array([id_inv_map[int(v)] for v in y_pred_r], dtype=np.int64)
    else:
        y_true, y_pred = y_true_r, y_pred_r
    acc = 100.0 * float((y_true_r == y_pred_r).mean()) if y_true_r.size > 0 else 0.0
    return acc, y_true, y_pred


def _collect_release_bank_split(
    model: IDRemovalNet,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    raw_chunks: List[np.ndarray] = []
    signal_chunks: List[np.ndarray] = []
    feature_chunks: List[np.ndarray] = []
    task_chunks: List[np.ndarray] = []
    id_chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y_task, y_id in loader:
            x_dev = x.to(device)
            raw_chunks.append(x.detach().cpu().numpy().astype(np.float32))
            signal_chunks.append(model.release_signal(x_dev).detach().cpu().numpy().astype(np.float32))
            feature_chunks.append(model.extract(x_dev).detach().cpu().numpy().astype(np.float32))
            task_chunks.append(y_task.detach().cpu().numpy().astype(np.int64))
            id_chunks.append(y_id.detach().cpu().numpy().astype(np.int64))
    return {
        "raw": np.concatenate(raw_chunks, axis=0),
        "released_signal": np.concatenate(signal_chunks, axis=0),
        "released_feature": np.concatenate(feature_chunks, axis=0),
        "y_task": np.concatenate(task_chunks, axis=0),
        "y_id": np.concatenate(id_chunks, axis=0),
    }


def save_release_bank(
    path: str,
    model: IDRemovalNet,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    *,
    run_seed: int,
    test_subject: int,
    id_map: Dict[int, int],
    uniq_ids: np.ndarray,
    release_projector_report: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tr = _collect_release_bank_split(model, tr_loader, device)
    va = _collect_release_bank_split(model, va_loader, device)
    metadata = {
        "description": (
            "Raw EA-aligned EEG, EEG-like TAPER-RP released signal, and vector "
            "released feature for unseen re-identification attacker evaluation."
        ),
        "run_seed": int(run_seed),
        "test_subject": int(test_subject),
        "dataset_name": dataset_name,
        "training_backbone": get_selected_backbone(),
        "id_map": {int(k): int(v) for k, v in id_map.items()},
        "uniq_train_subject_ids": [int(x) for x in uniq_ids.tolist()],
        "release_projector_report": _json_safe(release_projector_report),
        "use_ea": int(args.use_ea),
    }
    np.savez_compressed(
        path,
        X_raw_train=tr["raw"],
        X_raw_val=va["raw"],
        X_release_train=tr["released_signal"],
        X_release_val=va["released_signal"],
        Z_release_train=tr["released_feature"],
        Z_release_val=va["released_feature"],
        y_task_train=tr["y_task"],
        y_task_val=va["y_task"],
        y_id_train=tr["y_id"],
        y_id_val=va["y_id"],
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    return path


def fit_release_projector(
    model: IDRemovalNet,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    dim = max(0, int(getattr(args, "release_projector_dim", 0)))
    strength = float(getattr(args, "release_projector_strength", 0.0))
    if dim <= 0 or strength <= 0.0:
        model.configure_release_projector(torch.empty(0, model.task_head_sub.in_features), 0.0)
        return {"enabled": False, "dim": 0, "strength": 0.0}

    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x, _, y_id in loader:
            z = model.extract(x.to(device), apply_release_projector=False).detach().cpu()
            feats.append(z)
            labels.append(y_id.detach().cpu())
    if not feats:
        model.configure_release_projector(torch.empty(0, model.task_head_sub.in_features), 0.0)
        return {"enabled": False, "dim": 0, "strength": 0.0, "reason": "empty_loader"}

    z_all = torch.cat(feats, dim=0)
    y_all = torch.cat(labels, dim=0)
    centers = []
    for sid in y_all.unique(sorted=True):
        mask = y_all == sid
        if int(mask.sum().item()) > 0:
            centers.append(z_all[mask].mean(dim=0))
    if len(centers) < 2:
        model.configure_release_projector(torch.empty(0, z_all.shape[1]), 0.0)
        return {"enabled": False, "dim": 0, "strength": 0.0, "reason": "too_few_id_centers"}

    c = torch.stack(centers, dim=0)
    c = c - c.mean(dim=0, keepdim=True)
    try:
        _, s, vh = torch.linalg.svd(c, full_matrices=False)
    except RuntimeError:
        _, s, vh = torch.svd(c)
        vh = vh.t()
    use_dim = min(dim, vh.shape[0])
    basis = vh[:use_dim].contiguous()
    model.configure_release_projector(basis.to(device), strength)
    return {
        "enabled": True,
        "dim": int(use_dim),
        "requested_dim": int(dim),
        "strength": float(strength),
        "singular_values": s[:use_dim].detach().cpu().numpy(),
    }


def collect_task_predictions(
    model: IDRemovalNet,
    te_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    fold_eval_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_chunks: List[np.ndarray] = []
    y_pred_chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in te_loader:
            x = x.to(device)
            y = y.to(device)
            logits = eval_privacy_task_logits(model, x, args, logits_mode=fold_eval_mode)
            pred = logits.argmax(dim=1)
            y_true_chunks.append(y.cpu().numpy())
            y_pred_chunks.append(pred.cpu().numpy())
    if not y_true_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(y_true_chunks), np.concatenate(y_pred_chunks)


def save_fold_detail(path: str, detail: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(detail), f, ensure_ascii=False, indent=2)


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_repro_checkpoint(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def train_privacy(
    model: IDRemovalNet,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    args,
    device: torch.device,
    taper_bank: Optional[TAPERProbeBank] = None,
):
    main_params = (
        list(model.ids.parameters())
        + list(model.extractor.parameters())
        + list(model.raw_adapter.parameters())
        + list(model.extractor_raw.parameters())
        + list(model.fuse_gate.parameters())
        + list(model.attn.parameters())
        + list(model.task_proj.parameters())
        + list(model.task_head_total.parameters())
        + list(model.task_head_sub.parameters())
    )
    opt_main = torch.optim.Adam(main_params, lr=args.lr, weight_decay=args.weight_decay)
    # Keep the adversary aligned with the published privacy metric: the
    # evaluation attacker is trained on model.extract(), i.e. the task subspace.
    opt_id = torch.optim.Adam(model.id_head_sub.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_id = nn.CrossEntropyLoss().to(device)
    ts = float(getattr(args, "train_task_label_smoothing", 0.0))
    ce_task = (
        nn.CrossEntropyLoss(label_smoothing=ts).to(device)
        if ts > 0.0
        else ce_id
    )
    best = -1.0
    best_state = None
    noise_std_s1 = float(getattr(args, "train_input_noise_std", 0.0))

    tf = max(0, int(args.task_focus_epochs))
    for ep in range(1, args.epochs + 1):
        model.train()
        # 任务优先：前几轮不做 GRL 对抗、不乘 adv，去相关权重渐进，先稳住任务与 IDS 表征
        if tf > 0 and ep <= tf:
            lambd = 0.0
            adv_mult = 0.0
            dec_mult = float(args.decorr_weight) * (float(ep) / float(max(1, tf)))
        else:
            ep_adv = ep - tf if tf > 0 else ep
            lambd = args.lambda_max * min(1.0, ep_adv / max(1, args.warmup_epochs))
            adv_mult = 1.0
            dec_mult = float(args.decorr_weight)
        for x, y_task, y_id in tr_loader:
            x = x.to(device)
            y_task = y_task.to(device)
            y_id = y_id.to(device)
            x_in = apply_train_input_noise(x, args, noise_std=noise_std_s1, active=True)

            # identity classifier update（与测试一致：干净输入，便于攻击器与 forward 可比）
            opt_id.zero_grad()
            id_logits_sub = model.id_forward(x)
            loss_id = ce_id(id_logits_sub, y_id)
            loss_id.backward()
            opt_id.step()

            # extractor + IDS + task with GRL（学生路径可加输入噪声）
            opt_main.zero_grad()
            (
                task_logits,
                id_logits_sub_adv,
                loss_dec,
                _task_logits_linear,
                _task_logits_proto,
                _task_logits_sub,
                f_total,
                _z_task,
            ) = model(x_in, lambd=lambd)
            loss_task = ce_task(task_logits, y_task)
            loss_adv = ce_id(id_logits_sub_adv, y_id)
            loss_taper = task_logits.new_zeros(())
            if taper_bank is not None and float(args.taper_weight) > 0.0:
                x_taper, _ = model.ids(x_in)
                loss_taper, _ = taper_bank.privacy_losses(
                    x_taper,
                    y_task,
                    task_logits.detach(),
                    entropy_target_ratio=args.taper_entropy_target_ratio,
                    active_views=args.taper_views,
                    privacy_mode=args.taper_privacy_mode,
                    cov_align_weight=args.taper_cov_align_weight,
                )
            loss = (
                loss_task
                + adv_mult * args.adv_weight * loss_adv
                + dec_mult * loss_dec
                + args.taper_weight * loss_taper
            )
            loss.backward()
            opt_main.step()
            model.update_ema_proto(f_total.detach(), y_task, momentum=args.proto_ema_momentum)

        # val by task
        model.eval()
        c = 0
        t = 0
        with torch.no_grad():
            for x, y_task, _ in va_loader:
                x = x.to(device)
                y_task = y_task.to(device)
                logits = eval_privacy_task_logits(model, x, args)
                pred = logits.argmax(dim=1)
                c += (pred == y_task).sum().item()
                t += x.size(0)
        acc = 100.0 * c / max(1, t)
        if acc > best:
            best = acc
            best_state = copy.deepcopy(model.state_dict())
        if ep % max(1, args.log_every) == 0:
            print(
                f"  [Privacy ep {ep:03d}] lambda={lambd:.3f} adv_mult={adv_mult:.2f} "
                f"dec_mult={dec_mult:.4f} val_task={acc:.2f}%"
            )
    if best_state is not None:
        model.load_state_dict(best_state)

    # Stage-2: utility refit under privacy lock.
    if args.utility_refit_epochs > 0:
        for p in model.ids.parameters():
            p.requires_grad = False
        for p in model.extractor.parameters():
            p.requires_grad = False
        for p in model.raw_adapter.parameters():
            p.requires_grad = False
        for p in model.extractor_raw.parameters():
            p.requires_grad = False
        for p in model.fuse_gate.parameters():
            p.requires_grad = not (int(args.refit_train_fuse_gate) == 1)
        for p in model.id_head_sub.parameters():
            p.requires_grad = False

        refit_attn = int(args.refit_train_attn) == 1
        for p in model.attn.parameters():
            p.requires_grad = refit_attn

        refit_heads = list(model.task_head_total.parameters())
        refit_proto_extra: list = []
        refit_raw_inject_extra: list = []
        if int(args.refit_train_task_subspace) == 1:
            refit_proto_extra.extend(list(model.task_proj.parameters()))
            refit_proto_extra.extend(list(model.task_head_sub.parameters()))
            refit_proto_extra.append(model.task_proto)
            refit_proto_extra.extend(
                [model.proto_scale, model.logit_mix, model.task_sub_mix, model.proto_ema_mix]
            )
        if int(args.refit_train_raw_inject_logit) == 1:
            refit_raw_inject_extra.append(model.raw_inject_logit)
        refit_param_groups = [{"params": refit_heads, "lr": args.utility_refit_lr}]
        if refit_proto_extra:
            # raw 注入强度用更小 lr，避免 ID 突然反弹
            refit_param_groups.append(
                {
                    "params": refit_proto_extra,
                    "lr": args.refit_proto_lr,
                }
            )
        if refit_raw_inject_extra:
            refit_param_groups.append(
                {"params": refit_raw_inject_extra, "lr": args.refit_raw_inject_logit_lr}
            )
        if int(args.refit_train_fuse_gate) == 1:
            refit_param_groups.append(
                {"params": list(model.fuse_gate.parameters()), "lr": args.refit_fuse_gate_lr}
            )
        if refit_attn:
            refit_param_groups.append({"params": list(model.attn.parameters()), "lr": args.refit_attn_lr})
        opt_refit = torch.optim.Adam(refit_param_groups, weight_decay=args.weight_decay)
        refit_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_refit, T_max=max(1, args.utility_refit_epochs), eta_min=args.utility_refit_min_lr
        )
        ce = nn.CrossEntropyLoss(label_smoothing=args.refit_label_smoothing).to(device)
        best_refit = -1.0
        best_refit_state = None
        noise_std_r = float(getattr(args, "refit_input_noise_std", 0.0))

        for ep in range(1, args.utility_refit_epochs + 1):
            model.train()
            for x, y_task, y_id in tr_loader:
                x = x.to(device)
                y_task = y_task.to(device)
                y_id = y_id.to(device)
                x_in = apply_train_input_noise(x, args, noise_std=noise_std_r, active=True)
                opt_refit.zero_grad()
                model_out = model(x_in, lambd=0.0)
                (
                    task_logits,
                    _,
                    _,
                    _task_logits_linear,
                    _task_logits_proto,
                    _task_logits_sub,
                    f_total,
                    z_task,
                ) = model_out
                id_logits_sub_ref = model.id_head_sub(z_task)
                p_id = F.softmax(id_logits_sub_ref, dim=1)
                id_entropy = -(p_id * (p_id.clamp_min(1e-8)).log()).sum(dim=1).mean()
                loss_taper_refit = task_logits.new_zeros(())
                if taper_bank is not None and float(args.taper_refit_weight) > 0.0:
                    x_taper_refit, _ = model.ids(x_in)
                    loss_taper_refit, _ = taper_bank.privacy_losses(
                        x_taper_refit,
                        y_task,
                        task_logits.detach(),
                        entropy_target_ratio=args.taper_entropy_target_ratio,
                        active_views=args.taper_views,
                        privacy_mode=args.taper_privacy_mode,
                        cov_align_weight=args.taper_cov_align_weight,
                    )
                loss = (
                    ce(task_logits, y_task)
                    + args.refit_id_ent_deficit_weight
                    * F.relu(
                        args.refit_id_ent_target_ratio
                        * math.log(model.id_head_sub.out_features)
                        - id_entropy
                    )
                    + args.taper_refit_weight * loss_taper_refit
                )
                loss.backward()
                opt_refit.step()
                model.update_ema_proto(f_total.detach(), y_task, momentum=args.proto_ema_momentum)
            refit_sched.step()

            # val task only
            model.eval()
            c, t = 0, 0
            with torch.no_grad():
                for x, y_task, _ in va_loader:
                    x = x.to(device)
                    y_task = y_task.to(device)
                    logits = eval_privacy_task_logits(model, x, args)
                    pred = logits.argmax(dim=1)
                    c += (pred == y_task).sum().item()
                    t += x.size(0)
            acc = 100.0 * c / max(1, t)
            if acc > best_refit:
                best_refit = acc
                best_refit_state = copy.deepcopy(model.state_dict())
            if ep % max(1, args.log_every) == 0:
                print(f"  [Refit ep {ep:03d}] val_task={acc:.2f}%")

        if best_refit_state is not None:
            model.load_state_dict(best_refit_state)

    # Stage-3: task-only refit with the same privacy lock.
    if int(args.refit_task_only_epochs) > 0:
        for p in model.ids.parameters():
            p.requires_grad = False
        for p in model.extractor.parameters():
            p.requires_grad = False
        for p in model.raw_adapter.parameters():
            p.requires_grad = False
        for p in model.extractor_raw.parameters():
            p.requires_grad = False
        for p in model.fuse_gate.parameters():
            p.requires_grad = not (int(args.refit_train_fuse_gate) == 1)
        for p in model.id_head_sub.parameters():
            p.requires_grad = False

        refit_attn3 = int(args.refit_train_attn) == 1
        for p in model.attn.parameters():
            p.requires_grad = refit_attn3

        refit_heads3 = list(model.task_head_total.parameters())
        refit_proto_extra3: list = []
        refit_raw_inject_extra3: list = []
        if int(args.refit_train_task_subspace) == 1:
            refit_proto_extra3.extend(list(model.task_proj.parameters()))
            refit_proto_extra3.extend(list(model.task_head_sub.parameters()))
            refit_proto_extra3.append(model.task_proto)
            refit_proto_extra3.extend(
                [model.proto_scale, model.logit_mix, model.task_sub_mix, model.proto_ema_mix]
            )
        if int(args.refit_train_raw_inject_logit) == 1:
            refit_raw_inject_extra3.append(model.raw_inject_logit)
        groups3 = [{"params": refit_heads3, "lr": args.refit_task_only_lr}]
        if refit_proto_extra3:
            groups3.append({"params": refit_proto_extra3, "lr": args.refit_task_only_proto_lr})
        if refit_raw_inject_extra3:
            groups3.append({"params": refit_raw_inject_extra3, "lr": args.refit_task_only_raw_inject_lr})
        if int(args.refit_train_fuse_gate) == 1:
            groups3.append(
                {"params": list(model.fuse_gate.parameters()), "lr": args.refit_task_only_fuse_gate_lr}
            )
        if refit_attn3:
            groups3.append({"params": list(model.attn.parameters()), "lr": args.refit_task_only_attn_lr})
        opt_t = torch.optim.Adam(groups3, weight_decay=args.weight_decay)
        sched_t = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_t, T_max=max(1, args.refit_task_only_epochs), eta_min=args.refit_task_only_min_lr
        )
        ce_t = nn.CrossEntropyLoss(label_smoothing=args.refit_task_only_label_smoothing).to(device)
        best_t = -1.0
        best_state_t = None
        noise_std_t3 = float(getattr(args, "refit_input_noise_std", 0.0))

        for ep in range(1, args.refit_task_only_epochs + 1):
            model.train()
            for x, y_task, y_id in tr_loader:
                x = x.to(device)
                y_task = y_task.to(device)
                y_id = y_id.to(device)
                x_in = apply_train_input_noise(x, args, noise_std=noise_std_t3, active=True)
                opt_t.zero_grad()
                model_out_t = model(x_in, lambd=0.0)
                (
                    task_logits,
                    _,
                    _,
                    _task_logits_linear,
                    _task_logits_proto,
                    _task_logits_sub,
                    f_total,
                    z_task,
                ) = model_out_t
                w_ent3 = float(getattr(args, "refit_task_only_id_ent_weight", 0.0))
                if w_ent3 > 0.0:
                    id_logits_t3 = model.id_head_sub(z_task)
                    p3 = F.softmax(id_logits_t3, dim=1)
                    id_ent3 = -(p3 * (p3.clamp_min(1e-8)).log()).sum(dim=1).mean()
                    ratio3 = float(getattr(args, "refit_task_only_id_ent_target_ratio", 0.992))
                    loss_id_ent3 = w_ent3 * F.relu(
                        ratio3 * math.log(float(model.id_head_sub.out_features)) - id_ent3
                    )
                else:
                    loss_id_ent3 = task_logits.new_zeros(())
                loss_taper_t3 = task_logits.new_zeros(())
                if taper_bank is not None and float(args.taper_task_only_weight) > 0.0:
                    x_taper_t3, _ = model.ids(x_in)
                    loss_taper_t3, _ = taper_bank.privacy_losses(
                        x_taper_t3,
                        y_task,
                        task_logits.detach(),
                        entropy_target_ratio=args.taper_entropy_target_ratio,
                        active_views=args.taper_views,
                        privacy_mode=args.taper_privacy_mode,
                        cov_align_weight=args.taper_cov_align_weight,
                    )
                loss_to = (
                    ce_t(task_logits, y_task)
                    + loss_id_ent3
                    + args.taper_task_only_weight * loss_taper_t3
                )
                loss_to.backward()
                opt_t.step()
                model.update_ema_proto(f_total.detach(), y_task, momentum=args.proto_ema_momentum)
            sched_t.step()

            model.eval()
            c, t2 = 0, 0
            with torch.no_grad():
                for x, y_task, _ in va_loader:
                    x = x.to(device)
                    y_task = y_task.to(device)
                    logits = eval_privacy_task_logits(model, x, args)
                    pred = logits.argmax(dim=1)
                    c += (pred == y_task).sum().item()
                    t2 += x.size(0)
            acc = 100.0 * c / max(1, t2)
            if acc > best_t:
                best_t = acc
                best_state_t = copy.deepcopy(model.state_dict())
            if ep % max(1, args.log_every) == 0:
                print(f"  [TaskOnly ep {ep:03d}] val_task={acc:.2f}%")

        if best_state_t is not None:
            model.load_state_dict(best_state_t)


def run_loso_for_seed(
    run_seed: int,
    args: argparse.Namespace,
    X_list: List[np.ndarray],
    y_task_list: List[np.ndarray],
    y_subject_list: List[np.ndarray],
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(run_seed)
    print(f"\n#################### Seed run: {run_seed} ####################")
    S = len(X_list)
    folds: List[FoldResult] = []
    weight_paths: List[str] = []
    seed_out_dir = args.output_dir
    if len(parse_seed_list(args)) > 1:
        seed_out_dir = os.path.join(args.output_dir, f"seed_{run_seed}")
    os.makedirs(seed_out_dir, exist_ok=True)

    for test_idx in range(S):
        set_seed(run_seed + test_idx * 1000)
        print(f"\n========== LOSO Fold: test subject {test_idx + 1} (seed={run_seed}) ==========")
        X_test = X_list[test_idx]
        y_task_test = y_task_list[test_idx]
        X_train_all = np.concatenate([X_list[i] for i in range(S) if i != test_idx], axis=0)
        y_task_all = np.concatenate([y_task_list[i] for i in range(S) if i != test_idx], axis=0)
        y_id_all = np.concatenate([y_subject_list[i] for i in range(S) if i != test_idx], axis=0)

        uniq_ids = np.unique(y_id_all)
        id_map = {sid: i for i, sid in enumerate(uniq_ids.tolist())}
        y_id_all = np.array([id_map[s] for s in y_id_all], dtype=np.int64)

        full = TensorDataset(
            torch.from_numpy(X_train_all).float(),
            torch.from_numpy(y_task_all).long(),
            torch.from_numpy(y_id_all).long(),
        )
        n_valid = max(1, math.ceil(args.valid_ratio * len(full)))
        n_train = len(full) - n_valid
        tr_sub, va_sub = random_split(
            full, [n_train, n_valid], generator=torch.Generator().manual_seed(run_seed)
        )

        tr_idx = np.array(tr_sub.indices)
        va_idx = np.array(va_sub.indices)
        X_tr = X_train_all[tr_idx]
        X_va = X_train_all[va_idx]
        y_task_tr = y_task_all[tr_idx]
        y_task_va = y_task_all[va_idx]
        y_id_tr = y_id_all[tr_idx]
        y_id_va = y_id_all[va_idx]
        X_te = X_test
        r_inv = None

        if int(args.use_ea) == 1:
            _, r_inv = fit_ea_from_trials(X_tr)
            X_tr = apply_euclidean_alignment(X_tr, r_inv)
            X_va = apply_euclidean_alignment(X_va, r_inv)
            X_te = apply_euclidean_alignment(X_te, r_inv)

        tr_ds = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_task_tr).long(),
            torch.from_numpy(y_id_tr).long(),
        )
        va_ds = TensorDataset(
            torch.from_numpy(X_va).float(),
            torch.from_numpy(y_task_va).long(),
            torch.from_numpy(y_id_va).long(),
        )
        te_task_ds = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_task_test).long())
        va_id_ds = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_id_va).long())
        tr_id_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_id_tr).long())

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
        te_loader = DataLoader(te_task_ds, batch_size=args.batch_size, shuffle=False)
        tr_id_loader = DataLoader(tr_id_ds, batch_size=args.batch_size, shuffle=True)
        va_id_loader = DataLoader(va_id_ds, batch_size=args.batch_size, shuffle=False)

        n_channels = X_tr.shape[1]
        n_times = X_tr.shape[2]
        n_task = int(np.max(y_task_all) + 1)
        n_ids = len(uniq_ids)

        priv_model = IDRemovalNet(
            n_channels=n_channels,
            n_times=n_times,
            n_task=n_task,
            n_ids=n_ids,
            ids_bottleneck=args.ids_bottleneck,
            ids_k_dim=args.ids_k_dim,
            raw_inject_scale=args.raw_inject_scale,
        ).to(device)
        taper_bank = None
        taper_report = None
        if int(args.use_taper) == 1:
            taper_bank = TAPERProbeBank(
                n_channels=n_channels, n_times=n_times, n_ids=n_ids, n_task=n_task
            ).to(device)
            taper_report = train_taper_probe_bank(
                taper_bank,
                tr_loader,
                va_loader,
                epochs=args.taper_probe_epochs,
                lr=args.taper_probe_lr,
                weight_decay=args.weight_decay,
                device=device,
                active_views=args.taper_views,
            )
            print(
                f"[Fold {test_idx + 1}] TAPER probes val ID: "
                f"cov={taper_report.cov_acc:.2f}% band={taper_report.band_acc:.2f}% "
                f"stat={taper_report.stat_acc:.2f}%"
            )
        train_privacy(priv_model, tr_loader, va_loader, args, device, taper_bank=taper_bank)
        release_projector_report = fit_release_projector(priv_model, tr_loader, args, device)
        if release_projector_report.get("enabled", False):
            print(
                f"[Fold {test_idx + 1}] release projector: "
                f"dim={release_projector_report['dim']} "
                f"strength={release_projector_report['strength']:.2f}"
            )
        if int(getattr(args, "save_release_bank", 0)) == 1:
            bank_path = os.path.join(
                seed_out_dir,
                "release_bank",
                f"fold_{test_idx + 1}_seed_{run_seed}.npz",
            )
            save_release_bank(
                bank_path,
                priv_model,
                tr_loader,
                va_loader,
                run_seed=run_seed,
                test_subject=test_idx + 1,
                id_map=id_map,
                uniq_ids=uniq_ids,
                release_projector_report=release_projector_report,
                args=args,
                device=device,
            )
            print(f"[Fold {test_idx + 1}] release bank saved: {bank_path}")

        if args.eval_task_logits == "auto":
            fold_eval_mode, val_pick_acc = pick_eval_task_logits_on_val(
                priv_model, va_loader, args, device
            )
            print(
                f"[Fold {test_idx + 1}] eval_task_logits=auto -> val anon task={val_pick_acc:.2f}% "
                f"mode={fold_eval_mode}"
            )
        else:
            fold_eval_mode = args.eval_task_logits

        task_y_true, task_y_pred = collect_task_predictions(
            priv_model, te_loader, args, device, fold_eval_mode
        )
        anon_task_acc = (
            100.0 * float((task_y_true == task_y_pred).mean()) if task_y_true.size > 0 else 0.0
        )
        id_inv_map = {v: k for k, v in id_map.items()}
        anon_id_acc, id_y_true, id_y_pred = train_attacker(
            extractor_fn=priv_model.extract,
            train_loader=tr_id_loader,
            val_loader=va_id_loader,
            n_ids=n_ids,
            args=args,
            device=device,
            id_inv_map=id_inv_map,
        )

        weight_path = None
        if int(getattr(args, "save_weights", 0)) == 1:
            weight_path = os.path.join(
                seed_out_dir, "weights", f"fold_{test_idx + 1}_seed_{run_seed}_test_weights.pt"
            )
            save_repro_checkpoint(
                weight_path,
                {
                    "experiment_type": "idremovalnet",
                    "run_seed": int(run_seed),
                    "fold_seed": int(run_seed + test_idx * 1000),
                    "test_subject": int(test_idx + 1),
                    "backbone": get_selected_backbone(),
                    "model_state_dict": cpu_state_dict(priv_model),
                    "id_map": {int(k): int(v) for k, v in id_map.items()},
                    "uniq_train_subject_ids": [int(x) for x in uniq_ids.tolist()],
                    "train_indices": tr_idx.astype(int).tolist(),
                    "valid_indices": va_idx.astype(int).tolist(),
                    "use_ea": int(args.use_ea),
                    "ea_r_inv": r_inv.astype(np.float32) if int(args.use_ea) == 1 else None,
                    "eval_task_logits": fold_eval_mode,
                    "n_channels": int(n_channels),
                    "n_times": int(n_times),
                    "n_task": int(n_task),
                    "n_ids": int(n_ids),
                    "metrics": {
                        "anon_task_acc": float(anon_task_acc),
                        "anon_id_acc": float(anon_id_acc),
                    },
                    "hyperparameters": collect_run_hyperparams(args, run_seed),
                },
            )

        fold_detail = {
            "run_seed": run_seed,
            "test_subject": test_idx + 1,
            "anon_task_acc": float(anon_task_acc),
            "anon_id_acc": float(anon_id_acc),
            "anon_eval_mode": fold_eval_mode,
            "task_y_true": task_y_true,
            "task_y_pred": task_y_pred,
            "id_y_true": id_y_true,
            "id_y_pred": id_y_pred,
            "taper_probe_report": taper_report.__dict__ if taper_report is not None else None,
            "release_projector_report": release_projector_report,
            "weight_path": weight_path,
        }
        save_fold_detail(
            os.path.join(seed_out_dir, "fold_details", f"fold_{test_idx + 1}.json"),
            fold_detail,
        )
        if weight_path is not None:
            weight_paths.append(weight_path)

        folds.append(
            FoldResult(
                test_subject=test_idx + 1,
                anon_task_acc=anon_task_acc,
                anon_id_acc=anon_id_acc,
                anon_eval_mode=fold_eval_mode,
            )
        )
        print(
            f"[Fold {test_idx + 1}] "
            f"anon task={anon_task_acc:.2f}% | anon identity={anon_id_acc:.2f}%"
        )

    anon_task = np.array([x.anon_task_acc for x in folds], dtype=np.float32)
    anon_id = np.array([x.anon_id_acc for x in folds], dtype=np.float32)

    print(f"\n================= LOSO Summary (seed={run_seed}) =================")
    if args.eval_task_logits == "auto":
        print(
            f"(anon task) eval_task_logits=auto (per-fold mode in fold results) "
            f"temp={args.eval_ensemble_temp} tta={args.eval_tta}"
        )
    else:
        print(
            f"(anon task) eval_task_logits={args.eval_task_logits} temp={args.eval_ensemble_temp} "
            f"tta={args.eval_tta}"
        )
    print(f"Task accuracy (anonymized, LOSO):  {anon_task.mean():.2f}% ± {anon_task.std():.2f}%")
    print(f"Identity accuracy (anonymized):    {anon_id.mean():.2f}% ± {anon_id.std():.2f}%")

    all_hyperparams = collect_run_hyperparams(args, run_seed)
    print_hyperparams_report(all_hyperparams)

    out = {
        "run_seed": run_seed,
        "hyperparameters": all_hyperparams,
        "eval_task_logits": args.eval_task_logits,
        "eval_auto_candidates": getattr(args, "eval_auto_candidates", ""),
        "eval_auto_fallback": getattr(args, "eval_auto_fallback", "prob_mean4"),
        "train_input_noise_std": float(getattr(args, "train_input_noise_std", 0.0)),
        "train_input_noise_prob": float(getattr(args, "train_input_noise_prob", 1.0)),
        "refit_input_noise_std": float(getattr(args, "refit_input_noise_std", 0.0)),
        "train_task_label_smoothing": float(getattr(args, "train_task_label_smoothing", 0.0)),
        "refit_task_only_id_ent_weight": float(getattr(args, "refit_task_only_id_ent_weight", 0.0)),
        "refit_task_only_id_ent_target_ratio": float(
            getattr(args, "refit_task_only_id_ent_target_ratio", 0.992)
        ),
        "release_projector_dim": int(getattr(args, "release_projector_dim", 0)),
        "release_projector_strength": float(getattr(args, "release_projector_strength", 0.0)),
        "eval_ensemble_temp": float(args.eval_ensemble_temp),
        "eval_tta": int(args.eval_tta),
        "eval_tta_n": int(args.eval_tta_n),
        "eval_tta_std": float(args.eval_tta_std),
        "task_anon_mean": float(anon_task.mean()),
        "task_anon_std": float(anon_task.std()),
        "id_anon_mean": float(anon_id.mean()),
        "id_anon_std": float(anon_id.std()),
        "folds": [f.__dict__ for f in folds],
        "save_weights": int(getattr(args, "save_weights", 0)),
        "weight_paths": weight_paths,
    }
    with open(os.path.join(seed_out_dir, "summary_privacy_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="LOSO privacy ID-RemovalNet: anonymized task + identity metrics only"
    )
    parser.add_argument("--epochs", type=int, default=135)
    parser.add_argument("--batch_size", type=int, default=params["DL_Training"]["batch_size"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=params["DL_Training"]["weight_decay"])
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument(
        "--train_input_noise_std",
        type=float,
        default=0.014,
        help="阶段1：训练 batch 加高斯噪声标准差（0 关闭）",
    )
    parser.add_argument(
        "--train_input_noise_prob",
        type=float,
        default=0.55,
        help="阶段1~3：对 batch 施加 train/refit 噪声的概率（<1 可减少过强正则）",
    )
    parser.add_argument(
        "--refit_input_noise_std",
        type=float,
        default=0.007,
        help="阶段2/3：重拟合时噪声（通常小于阶段1）；0 关闭",
    )
    parser.add_argument(
        "--train_task_label_smoothing",
        type=float,
        default=0.018,
        help="阶段1：仅任务相关 CE 的标签平滑（身份 CE 仍为 0）",
    )
    parser.add_argument(
        "--lambda_max",
        type=float,
        default=1.03,
        help="GRL 梯度反传强度上限；过高易伤匿名任务，可与熵/门控/IDS 配合压 ID",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=28,
        help="GRL 强度 warmup，略长可减轻早期对任务表征的破坏",
    )
    parser.add_argument(
        "--task_focus_epochs",
        type=int,
        default=30,
        help="阶段1：前若干 epoch 关闭 GRL 对抗(adv_mult=0)，去相关权重按 epoch 线性 ramp；先拟合任务再压身份",
    )
    parser.add_argument("--adv_weight", type=float, default=0.705)
    parser.add_argument(
        "--raw_inject_scale",
        type=float,
        default=0.10,
        help="原始输入补偿分支注入上限缩放；0 关闭 raw 回流，<1 用于控制身份泄漏",
    )
    parser.add_argument("--decorr_weight", type=float, default=0.138)
    parser.add_argument(
        "--proto_ema_momentum",
        type=float,
        default=0.97,
        help="EMA 动态类中心动量",
    )
    parser.add_argument("--utility_refit_epochs", type=int, default=125)
    parser.add_argument("--utility_refit_lr", type=float, default=8e-4)
    parser.add_argument("--utility_refit_min_lr", type=float, default=1e-6)
    parser.add_argument(
        "--refit_train_attn",
        type=int,
        default=1,
        choices=[0, 1],
        help="重拟合时是否以 refit_attn_lr 微调 attention（IDS/提取器仍冻结）；1 通常能抬任务且不显著反弹 ID",
    )
    parser.add_argument(
        "--refit_attn_lr",
        type=float,
        default=2.5e-4,
        help="与 utility_refit_lr 配合；attention 用更小学习率避免破坏隐私表征",
    )
    parser.add_argument(
        "--refit_label_smoothing",
        type=float,
        default=0.03,
        help="重拟合阶段任务头 CE 的标签平滑，减轻过拟合、利于测试集任务准确率",
    )
    parser.add_argument(
        "--refit_id_ent_target_ratio",
        type=float,
        default=0.992,
        help="重拟合：只在 id_entropy 低于 target=ratio*log(n_ids) 时才继续压 ID（避免把任务继续压低）。",
    )
    parser.add_argument(
        "--refit_id_ent_deficit_weight",
        type=float,
        default=0.60,
        help="重拟合：当 id_entropy 低于 target 时的熵缺口惩罚权重。",
    )
    parser.add_argument(
        "--refit_train_task_subspace",
        type=int,
        default=1,
        choices=[0, 1],
        help="重拟合时是否训练 task_proj/task_proto/task_head_sub 等任务子空间参数",
    )
    parser.add_argument(
        "--refit_train_fuse_gate",
        type=int,
        default=1,
        choices=[0, 1],
        help="重拟合时是否解冻 fuse_gate（提升 clean/raw 融合带来的任务补偿），默认 1",
    )
    parser.add_argument(
        "--refit_train_raw_inject_logit",
        type=int,
        default=1,
        choices=[0, 1],
        help="重拟合时是否解冻 raw_inject_logit（控制 clean/raw 的注入强度），默认 1",
    )
    parser.add_argument(
        "--refit_raw_inject_logit_lr",
        type=float,
        default=5e-5,
        help="重拟合时 raw_inject_logit 学习率（建议更小，避免 ID 回升）",
    )
    parser.add_argument(
        "--refit_fuse_gate_lr",
        type=float,
        default=2e-4,
        help="重拟合时 fuse_gate 学习率（建议小于 utility_refit_lr）",
    )
    parser.add_argument(
        "--refit_proto_lr",
        type=float,
        default=5e-4,
        help="重拟合阶段原型/子空间参数学习率（通常略小于 utility_refit_lr）",
    )
    parser.add_argument(
        "--refit_task_only_epochs",
        type=int,
        default=115,
        help="阶段3：任务精调轮数；可选轻量 id 熵缺口",
    )
    parser.add_argument(
        "--refit_task_only_lr",
        type=float,
        default=9e-4,
        help="阶段3：任务头学习率",
    )
    parser.add_argument(
        "--refit_task_only_proto_lr",
        type=float,
        default=5e-4,
        help="阶段3：原型/子空间参数学习率",
    )
    parser.add_argument(
        "--refit_task_only_raw_inject_lr",
        type=float,
        default=5e-5,
        help="阶段3：raw_inject_logit 学习率",
    )
    parser.add_argument(
        "--refit_task_only_fuse_gate_lr",
        type=float,
        default=2e-4,
        help="阶段3：fuse_gate 学习率",
    )
    parser.add_argument(
        "--refit_task_only_attn_lr",
        type=float,
        default=2.5e-4,
        help="阶段3：attention 学习率",
    )
    parser.add_argument(
        "--refit_task_only_min_lr",
        type=float,
        default=1e-7,
        help="阶段3：Cosine 退火最小学习率",
    )
    parser.add_argument(
        "--refit_task_only_id_ent_weight",
        type=float,
        default=0.19,
        help="阶段3：id_head_sub(z_task) 熵缺口（身份头冻结）；过重易伤任务，与阶段2熵/门控配合压 ID",
    )
    parser.add_argument(
        "--refit_task_only_id_ent_target_ratio",
        type=float,
        default=0.992,
        help="阶段3：目标熵 ratio*log(n_ids)，与阶段2一致以强压 ID",
    )
    parser.add_argument(
        "--release_projector_dim",
        type=int,
        default=4,
        help="后处理发布表征投影：擦除训练集身份中心差异的前 k 个方向；0 关闭",
    )
    parser.add_argument(
        "--release_projector_strength",
        type=float,
        default=1.0,
        help="后处理发布表征投影强度；1=完全移除身份中心子空间，0=关闭",
    )
    parser.add_argument(
        "--refit_task_only_label_smoothing",
        type=float,
        default=0.008,
        help="阶段3：标签平滑（略低于 refit 阶段，利于任务准确率）",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=dataset_name,
        choices=["BCICIV_2a", "BCICIV_2b", "OPENBMI_P300"],
        help="覆盖 settings.py 中的数据集选择，便于同一代码分别跑 2a/2b/OpenBMI P300",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        choices=["auto", "EEGNet", "MSVTNet", "ShallowConvNet", "FBCNet"],
        help="覆盖 settings.py 中的 EEG decoder/backbone 选择，便于同一队列跑 EEGNet/MSVTNet/ShallowConvNet/FBCNet",
    )
    parser.add_argument(
        "--use_taper",
        type=int,
        default=1,
        choices=[0, 1],
        help="启用 TAPER：冻结多视图身份探针，压制可迁移身份判别证据",
    )
    parser.add_argument(
        "--taper_probe_epochs",
        type=int,
        default=25,
        help="每个 LOSO 折训练冻结身份探针的轮数；0 表示随机探针，不建议",
    )
    parser.add_argument(
        "--taper_probe_lr",
        type=float,
        default=8e-4,
        help="TAPER 身份探针学习率",
    )
    parser.add_argument(
        "--taper_views",
        type=str,
        default="all",
        help="TAPER 参与训练与隐私损失的视图，可选 all/cov/band/stat 或逗号组合",
    )
    parser.add_argument(
        "--taper_weight",
        type=float,
        default=0.16,
        help="阶段1 TAPER 隐私证据损失权重",
    )
    parser.add_argument(
        "--taper_refit_weight",
        type=float,
        default=0.06,
        help="阶段2 privacy lock refit 的 TAPER 损失权重，通常小于阶段1",
    )
    parser.add_argument(
        "--taper_task_only_weight",
        type=float,
        default=0.035,
        help="阶段3 任务精调时保留的轻量 TAPER 损失权重",
    )
    parser.add_argument(
        "--taper_entropy_target_ratio",
        type=float,
        default=0.985,
        help="TAPER 多探针身份熵目标：ratio*log(n_ids)",
    )
    parser.add_argument(
        "--taper_privacy_mode",
        type=str,
        default="hybrid",
        choices=["entropy", "cov_align", "task_cov_align", "hybrid", "task_hybrid"],
        help=(
            "TAPER 隐私目标：entropy=身份熵最大化，cov_align=共享模板对齐，"
            "task_cov_align=任务条件协方差模板对齐，hybrid/task_hybrid=与探针熵结合"
        ),
    )
    parser.add_argument(
        "--taper_cov_align_weight",
        type=float,
        default=1.0,
        help="TAPER 内部协方差模板对齐项权重；最终还会乘 taper_weight/refit_weight/task_only_weight",
    )
    _eval_logits_modes = [
        "fused",
        "mean3",
        "mean4",
        "prob_mean3",
        "prob_mean4",
        "stacked",
        "prob_stacked",
    ]
    parser.add_argument(
        "--eval_task_logits",
        type=str,
        default="auto",
        choices=_eval_logits_modes + ["auto"],
        help="匿名任务集成：auto=每折在验证集上选最优模式再测测试集（无泄露）；否则固定该模式",
    )
    parser.add_argument(
        "--eval_auto_candidates",
        type=str,
        default="prob_stacked,prob_mean4,prob_mean3,stacked,mean4,mean3,fused",
        help="eval_task_logits=auto 时在验证集上比较的候选（逗号分隔，须为 fused/mean*/prob_*/stacked）",
    )
    parser.add_argument(
        "--eval_auto_fallback",
        type=str,
        default="prob_mean4",
        choices=_eval_logits_modes,
        help="训练/验证阶段若 eval_task_logits=auto，则用该模式（与测试折上 auto 选择独立）",
    )
    parser.add_argument(
        "--eval_ensemble_temp",
        type=float,
        default=1.2,
        help="prob_mean* 中 softmax 温度；略>1 可略抬集成后的鲁棒性",
    )
    parser.add_argument(
        "--eval_tta",
        type=int,
        default=1,
        choices=[0, 1],
        help="验证/测试时 TTA：1=多次加噪前向平均 logits（匿名任务通常高 2~4pt，略慢）；0=单次前向",
    )
    parser.add_argument("--eval_tta_n", type=int, default=7)
    parser.add_argument(
        "--eval_tta_std",
        type=float,
        default=0.022,
        help="TTA 噪声标准差（与数据同量纲，宜小）",
    )
    parser.add_argument(
        "--attacker_epochs",
        type=int,
        default=30,
        help="LOSO 身份攻击器训练轮数（过长会抬高测得的 ID%%，与「压低匿名 ID」目标相反，一般保持 30 即可）",
    )
    parser.add_argument("--ids_bottleneck", type=int, default=256)
    parser.add_argument("--ids_k_dim", type=int, default=30)
    parser.add_argument("--use_ea", type=int, default=1, choices=[0, 1])
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="checkpoints_loso_idremovalnet")
    parser.add_argument(
        "--save_weights",
        type=int,
        default=0,
        choices=[0, 1],
        help="保存每个 LOSO fold 最终用于测试的模型权重和复现元数据（默认关闭）",
    )
    parser.add_argument(
        "--save_release_bank",
        type=int,
        default=0,
        choices=[0, 1],
        help="保存每个 fold 的 raw EEG、TAPER-RP released signal/released feature 与 ID 标签，用于未见攻击器并行评估",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="单次运行的随机种子（默认使用 settings.py 中的 seed）",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="多种子扫描，逗号分隔，如 1,2,3,4,5；固定超参数下比较不同初始化",
    )
    parser.add_argument(
        "--best_by",
        type=str,
        default="task",
        choices=["task", "id", "balanced"],
        help="多种子时选最优：task=匿名任务最高；id=匿名身份最低；balanced=task-0.5*id",
    )
    parser.add_argument(
        "--plot_confusion_matrix",
        type=int,
        default=1,
        choices=[0, 1],
        help="训练结束后根据 fold_details 画任务/ID 混淆矩阵（1=开启）",
    )
    parser.add_argument(
        "--merge_best_folds",
        type=int,
        default=1,
        choices=[0, 1],
        help="多种子时：按每被试最高任务准确率拼凑后再画混淆矩阵（1=开启，依赖 plot_confusion_matrix）",
    )
    args = parser.parse_args()
    apply_dataset_override(args)
    apply_backbone_override(args)
    if args.lr is None:
        args.lr = params[get_selected_backbone()]["learning_rate"]

    seed_list = parse_seed_list(args)
    device = params["DL_Training"]["device"]
    print(f"Device: {device}")
    print(f"Dataset: {dataset_name}")
    _bb = get_selected_backbone()
    print(f"Backbone (settings.use_models): {_bb}")
    if args.eval_task_logits == "auto":
        print(
            f"anon task eval: logits=auto candidates={args.eval_auto_candidates!r} "
            f"fallback={args.eval_auto_fallback} temp={args.eval_ensemble_temp} "
            f"tta={args.eval_tta} (n={args.eval_tta_n}, std={args.eval_tta_std})"
        )
    else:
        print(
            f"anon task eval: logits={args.eval_task_logits} temp={args.eval_ensemble_temp} "
            f"tta={args.eval_tta} (n={args.eval_tta_n}, std={args.eval_tta_std})"
        )
    print(
        f"train augment: noise_std={args.train_input_noise_std} noise_prob={args.train_input_noise_prob} "
        f"refit_noise_std={args.refit_input_noise_std} task_label_smoothing={args.train_task_label_smoothing}"
    )
    print(
        f"TAPER: use={args.use_taper} probe_epochs={args.taper_probe_epochs} "
        f"weights=({args.taper_weight}, {args.taper_refit_weight}, {args.taper_task_only_weight}) "
        f"entropy_target_ratio={args.taper_entropy_target_ratio} views={args.taper_views} "
        f"privacy_mode={args.taper_privacy_mode} cov_align_weight={args.taper_cov_align_weight}"
    )
    print(
        f"release projector: dim={args.release_projector_dim} "
        f"strength={args.release_projector_strength}"
    )
    print(f"Seed plan: {seed_list}" + (f" | best_by={args.best_by}" if len(seed_list) > 1 else ""))
    if int(args.eval_tta) == 0:
        print(
            "[提示] 本次未开启 TTA（--eval_tta 0），匿名任务通常比 --eval_tta 1 低约 2~4 个百分点；"
            "与历史结果对比时请固定同一套评估参数。"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    X_list, y_task_list, y_subject_list = load_subject_data()
    seed_runs: List[Dict[str, Any]] = []
    for run_seed in seed_list:
        seed_runs.append(
            run_loso_for_seed(run_seed, args, X_list, y_task_list, y_subject_list, device)
        )

    if len(seed_runs) > 1:
        ranked = sorted(
            seed_runs,
            key=lambda r: score_seed_run(r["task_anon_mean"], r["id_anon_mean"], args.best_by),
            reverse=True,
        )
        best = ranked[0]
        print("\n================= Seed Sweep Summary =================")
        print(f"best_by: {args.best_by}")
        for i, r in enumerate(ranked):
            mark = " <-- BEST" if i == 0 else ""
            print(
                f"  seed={r['run_seed']:>4} | task={r['task_anon_mean']:.2f}%±{r['task_anon_std']:.2f}% "
                f"| id={r['id_anon_mean']:.2f}%±{r['id_anon_std']:.2f}%{mark}"
            )
        print(
            f"\nBest seed: {best['run_seed']} | task={best['task_anon_mean']:.2f}% "
            f"| id={best['id_anon_mean']:.2f}%"
        )
        sweep_out = {
            "best_by": args.best_by,
            "seed_list": seed_list,
            "best_seed": best["run_seed"],
            "best_task_anon_mean": best["task_anon_mean"],
            "best_id_anon_mean": best["id_anon_mean"],
            "runs": [
                {
                    "run_seed": r["run_seed"],
                    "task_anon_mean": r["task_anon_mean"],
                    "task_anon_std": r["task_anon_std"],
                    "id_anon_mean": r["id_anon_mean"],
                    "id_anon_std": r["id_anon_std"],
                    "score": score_seed_run(r["task_anon_mean"], r["id_anon_mean"], args.best_by),
                }
                for r in ranked
            ],
        }
        with open(
            os.path.join(args.output_dir, "seed_sweep_summary.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(sweep_out, f, ensure_ascii=False, indent=2)
        print(f"Seed sweep summary saved: {os.path.join(args.output_dir, 'seed_sweep_summary.json')}")
        print("=" * 55)

    if int(args.plot_confusion_matrix) == 1:
        from merge_best_loso_folds import discover_run_dirs, merge_and_plot

        run_dirs = discover_run_dirs(args.output_dir)
        if not run_dirs:
            print(
                "[提示] 未找到 fold_details/fold_*.json，无法画混淆矩阵。"
                "请使用当前版本 train_loso_idremovalnet.py 重新训练。"
            )
        else:
            if len(seed_list) > 1 and int(args.merge_best_folds) == 1:
                cm_dir = os.path.join(args.output_dir, "best_merged")
            else:
                cm_dir = os.path.join(args.output_dir, "confusion_matrices")
            merge_and_plot(run_dirs, cm_dir, merge_multi_seed=len(seed_list) > 1)


if __name__ == "__main__":
    main()
