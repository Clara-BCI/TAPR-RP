"""
独立对比实验：LOSO + 骨干分类器（settings.use_models：EEGNet、MSVTNet 或 ShallowConvNet 特征提取 + 线性头）。

- 任务 LOSO：每折留 1 名被试，仅在其余「参与训练」的 8 人数据上划分 train/val 并训练，
  在留出被试上测运动想象等业务任务准确率。
- 隐私 ID 分类：只在当前折参与训练的那 8 名被试范围内做 multi-class「被试 ID」分类；
  在同等 train/val 划分下，在 **val（仅来源于上述 8 人）** 上报告隐私 ID 准确率。

运行：
  python train_loso_eegnet_baseline_only.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from types import SimpleNamespace
from typing import List, Tuple

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from privacy.eeg_euclidean_align import apply_euclidean_alignment, fit_ea_from_trials


def get_preprocessed_data(*args, **kwargs):
    raise NotImplementedError(
        "The raw EEG preprocessing/data interface is intentionally not included in this method-code release. "
        "Please provide subject-wise preprocessed arrays matching the README input contract."
    )


def make_dataset_metadata(name: str, *, tmin: float = 0.0, tmax: float = 4.0):
    return SimpleNamespace(name=name, tmin=tmin, tmax=tmax)
from settings import (
    dataset_name,
    end_subject,
    params,
    preprocessing_params,
    seed,
    start_subject,
    dataset,
    use_models,
)


def get_selected_backbone() -> str:
    if use_models.get("FBCNet", False):
        return "FBCNet"
    if use_models.get("ShallowConvNet", False):
        return "ShallowConvNet"
    if use_models.get("MSVTNet", False):
        return "MSVTNet"
    return "EEGNet"


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


def set_seed(seed_value: int) -> None:
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


class EEGFeatureExtractor(nn.Module):
    """与 train_loso_idremovalnet.BaselineTaskNet 内一致，对齐 settings['EEGNet']。"""

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


class TaskClassifierNet(nn.Module):
    """骨干（EEGNet、MSVTNet 或 ShallowConvNet）+ 线性分类头；用于任务或隐私 ID 监督。"""

    def __init__(self, n_channels: int, n_times: int, n_classes: int):
        super().__init__()
        self.extractor = build_backbone_extractor(n_channels, n_times)
        self.classifier = nn.Linear(self.extractor.feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extractor(x))


def load_subject_data() -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """与 train_loso_idremovalnet.load_subject_data 一致。"""
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


def collect_task_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_chunks: List[np.ndarray] = []
    y_pred_chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(x).argmax(dim=1)
            y_true_chunks.append(y.cpu().numpy())
            y_pred_chunks.append(pred.cpu().numpy())
    if not y_true_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(y_true_chunks), np.concatenate(y_pred_chunks)


def collect_id_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id_inv_map: dict,
) -> Tuple[float, np.ndarray, np.ndarray]:
    y_true_chunks: List[np.ndarray] = []
    y_pred_chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y_id in loader:
            x = x.to(device)
            y_id = y_id.to(device)
            pred = model(x).argmax(dim=1)
            y_true_chunks.append(y_id.cpu().numpy())
            y_pred_chunks.append(pred.cpu().numpy())
    y_true_r = np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
    y_pred_r = np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)
    if y_true_r.size > 0:
        y_true = np.array([id_inv_map[int(v)] for v in y_true_r], dtype=np.int64)
        y_pred = np.array([id_inv_map[int(v)] for v in y_pred_r], dtype=np.int64)
    else:
        y_true, y_pred = y_true_r, y_pred_r
    acc = 100.0 * float((y_true_r == y_pred_r).mean()) if y_true_r.size > 0 else 0.0
    return acc, y_true, y_pred


def save_fold_detail(path: str, detail: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)


def cpu_state_dict(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_repro_checkpoint(path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    return path


def train_eegnet(
    model: TaskClassifierNet,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    log_every: int,
    device: torch.device,
    log_prefix: str = "task",
    use_triplet_batches: bool = True,
) -> None:
    """
    use_triplet_batches=True：(x,y_task,y_id)，按 y_task 监督（任务 LOSO 头）。
    use_triplet_batches=False：(x,y_id)，按 y_id 监督（隐私 ID 头）。
    """

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss().to(device)
    best = -1.0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        if use_triplet_batches:
            for x, y_task, _ in tr_loader:
                x = x.to(device)
                y_task = y_task.to(device)
                opt.zero_grad()
                loss = ce(model(x), y_task)
                loss.backward()
                opt.step()
        else:
            for x, y_id in tr_loader:
                x = x.to(device)
                y_id = y_id.to(device)
                opt.zero_grad()
                loss = ce(model(x), y_id)
                loss.backward()
                opt.step()

        model.eval()
        c = t = 0
        with torch.no_grad():
            if use_triplet_batches:
                for x, y_task, _ in va_loader:
                    x = x.to(device)
                    y_task = y_task.to(device)
                    pred = model(x).argmax(dim=1)
                    c += (pred == y_task).sum().item()
                    t += x.size(0)
            else:
                for x, y_id in va_loader:
                    x = x.to(device)
                    y_id = y_id.to(device)
                    pred = model(x).argmax(dim=1)
                    c += (pred == y_id).sum().item()
                    t += x.size(0)
        acc = 100.0 * c / max(1, t)
        if acc > best:
            best = acc
            best_state = copy.deepcopy(model.state_dict())
        if ep % max(1, log_every) == 0:
            print(f"  [Cls-{log_prefix} ep {ep:03d}] val_{log_prefix}={acc:.2f}%")
    if best_state is not None:
        model.load_state_dict(best_state)


def main():
    parser = argparse.ArgumentParser(
        description="LOSO baseline: task + 8-subject privacy ID (use_models: EEGNet or MSVTNet backbone)"
    )
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=dataset_name,
        choices=["BCICIV_2a", "BCICIV_2b", "OPENBMI_P300"],
        help="覆盖 settings.py 中的数据集选择，便于同一代码分别跑 2a/2b/OpenBMI P300",
    )
    parser.add_argument(
        "--id_epochs",
        type=int,
        default=None,
        help="隐私分支训练轮数，默认与 --epochs 相同",
    )
    parser.add_argument("--batch_size", type=int, default=params["DL_Training"]["batch_size"])
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        choices=["auto", "EEGNet", "MSVTNet", "ShallowConvNet", "FBCNet"],
        help="覆盖 settings.py 中的 EEG decoder/backbone 选择，便于同一队列跑 EEGNet/MSVTNet/ShallowConvNet/FBCNet",
    )
    parser.add_argument("--weight_decay", type=float, default=params["DL_Training"]["weight_decay"])
    parser.add_argument("--valid_ratio", type=float, default=0.15)
    parser.add_argument("--use_ea", type=int, default=1, choices=[0, 1])
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="checkpoints_loso_eegnet_baseline_only")
    parser.add_argument(
        "--save_weights",
        type=int,
        default=0,
        choices=[0, 1],
        help="保存每个 LOSO fold 的 baseline 任务模型和身份模型权重（默认关闭）",
    )
    parser.add_argument(
        "--plot_confusion_matrix",
        type=int,
        default=1,
        choices=[0, 1],
        help="训练结束后根据 fold_details 画任务/ID 混淆矩阵（1=开启）",
    )
    args = parser.parse_args()
    apply_dataset_override(args)
    apply_backbone_override(args)
    if args.lr is None:
        args.lr = params[get_selected_backbone()]["learning_rate"]

    set_seed(seed)
    device = params["DL_Training"]["device"]
    id_epochs = args.epochs if args.id_epochs is None else args.id_epochs
    print(f"Device: {device}")
    _bb = get_selected_backbone()
    print(f"Backbone (settings.use_models): {_bb}")
    print(
        "说明：任务=在当前折留出被试上 LOSO；"
        "隐私 ID=仅在当期 8 名训练池被试上做 multi-class ID，校验集 val 来自这八人。"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    X_list, y_task_list, _y_subject_list = load_subject_data()
    S = len(X_list)
    fold_task: List[float] = []
    fold_privacy_id: List[float] = []
    weight_paths: List[str] = []

    for test_idx in range(S):
        print(f"\n========== LOSO Fold: test subject {test_idx + 1} ==========")
        X_test = X_list[test_idx]
        y_task_test = y_task_list[test_idx]
        X_train_all = np.concatenate([X_list[i] for i in range(S) if i != test_idx], axis=0)
        y_task_all = np.concatenate([y_task_list[i] for i in range(S) if i != test_idx], axis=0)
        y_id_all = np.concatenate(
            [_y_subject_list[i] for i in range(S) if i != test_idx], axis=0
        )

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
            full, [n_train, n_valid], generator=torch.Generator().manual_seed(seed)
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
        te_ds = TensorDataset(
            torch.from_numpy(X_te).float(),
            torch.from_numpy(y_task_test).long(),
        )

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

        n_channels = X_tr.shape[1]
        n_times = X_tr.shape[2]
        n_task_classes = int(np.max(y_task_all) + 1)
        n_id_classes = len(uniq_ids)

        task_model = TaskClassifierNet(n_channels, n_times, n_task_classes).to(device)
        train_eegnet(
            task_model,
            tr_loader,
            va_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            log_every=args.log_every,
            device=device,
            log_prefix="task",
            use_triplet_batches=True,
        )
        task_y_true, task_y_pred = collect_task_predictions(task_model, te_loader, device)
        task_test_acc = (
            100.0 * float((task_y_true == task_y_pred).mean()) if task_y_true.size > 0 else 0.0
        )
        fold_task.append(task_test_acc)

        tr_id_ds = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_id_tr).long(),
        )
        va_id_ds = TensorDataset(
            torch.from_numpy(X_va).float(),
            torch.from_numpy(y_id_va).long(),
        )
        tr_id_loader = DataLoader(tr_id_ds, batch_size=args.batch_size, shuffle=True)
        va_id_loader = DataLoader(va_id_ds, batch_size=args.batch_size, shuffle=False)

        id_model = TaskClassifierNet(n_channels, n_times, n_id_classes).to(device)
        train_eegnet(
            id_model,
            tr_id_loader,
            va_id_loader,
            epochs=id_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            log_every=args.log_every,
            device=device,
            log_prefix="privacy_id",
            use_triplet_batches=False,
        )
        id_inv_map = {v: k for k, v in id_map.items()}
        id_val_acc, id_y_true, id_y_pred = collect_id_predictions(
            id_model, va_id_loader, device, id_inv_map
        )
        fold_privacy_id.append(id_val_acc)

        weight_path = None
        if int(getattr(args, "save_weights", 0)) == 1:
            weight_path = os.path.join(
                args.output_dir, "weights", f"fold_{test_idx + 1}_seed_{seed}_baseline_weights.pt"
            )
            save_repro_checkpoint(
                weight_path,
                {
                    "experiment_type": "baseline",
                    "run_seed": int(seed),
                    "test_subject": int(test_idx + 1),
                    "backbone": get_selected_backbone(),
                    "task_model_state_dict": cpu_state_dict(task_model),
                    "id_model_state_dict": cpu_state_dict(id_model),
                    "id_map": {int(k): int(v) for k, v in id_map.items()},
                    "uniq_train_subject_ids": [int(x) for x in uniq_ids.tolist()],
                    "train_indices": tr_idx.astype(int).tolist(),
                    "valid_indices": va_idx.astype(int).tolist(),
                    "use_ea": int(args.use_ea),
                    "ea_r_inv": r_inv.astype(np.float32) if int(args.use_ea) == 1 else None,
                    "n_channels": int(n_channels),
                    "n_times": int(n_times),
                    "n_task_classes": int(n_task_classes),
                    "n_id_classes": int(n_id_classes),
                    "metrics": {
                        "task_loso_acc": float(task_test_acc),
                        "privacy_id_val_acc": float(id_val_acc),
                    },
                    "hyperparameters": {
                        "epochs_task": int(args.epochs),
                        "epochs_privacy_id": int(id_epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "valid_ratio": float(args.valid_ratio),
                        "use_ea": int(args.use_ea),
                    },
                },
            )
            weight_paths.append(weight_path)

        save_fold_detail(
            os.path.join(args.output_dir, "fold_details", f"fold_{test_idx + 1}.json"),
            {
                "run_seed": seed,
                "test_subject": test_idx + 1,
                "experiment_type": "baseline",
                "anon_task_acc": float(task_test_acc),
                "anon_id_acc": float(id_val_acc),
                "task_y_true": task_y_true.tolist(),
                "task_y_pred": task_y_pred.tolist(),
                "id_y_true": id_y_true.tolist(),
                "id_y_pred": id_y_pred.tolist(),
                "weight_path": weight_path,
            },
        )

        print(
            f"[Fold {test_idx + 1}] "
            f"task LOSO (held-out subj) = {task_test_acc:.2f}% | "
            f"privacy ID on 8-subj val = {id_val_acc:.2f}% (K={n_id_classes})"
        )

    task_scores = np.array(fold_task, dtype=np.float32)
    id_scores = np.array(fold_privacy_id, dtype=np.float32)
    print("\n================= LOSO Baseline Summary (9 folds) =================")
    print(f"Task (LOSO, held-out subject):     {task_scores.mean():.2f}% ± {task_scores.std():.2f}%")
    print(
        f"Privacy ID (train-pool only, val):  {id_scores.mean():.2f}% ± {id_scores.std():.2f}%"
    )
    print(f"Per-fold task: {task_scores}")
    print(f"Per-fold privacy ID (val): {id_scores}")

    out = {
        "description": "LOSO baseline task + privacy ID val; backbone from use_models (EEGNet or MSVTNet conv+Transformer)",
        "dataset_name": dataset_name,
        "epochs_task": args.epochs,
        "epochs_privacy_id": id_epochs,
        "use_ea": int(args.use_ea),
        "valid_ratio": args.valid_ratio,
        "task_loso_mean": float(task_scores.mean()),
        "task_loso_std": float(task_scores.std()),
        "per_fold_task_loso_acc": task_scores.tolist(),
        "privacy_id_val_mean": float(id_scores.mean()),
        "privacy_id_val_std": float(id_scores.std()),
        "per_fold_privacy_id_val_acc": id_scores.tolist(),
        "save_weights": int(getattr(args, "save_weights", 0)),
        "weight_paths": weight_paths,
    }
    path_summary = os.path.join(args.output_dir, "summary_loso_eegnet_only.json")
    with open(path_summary, "w", encoding="utf-8") as f_out:
        json.dump(out, f_out, ensure_ascii=False, indent=2)

    if int(args.plot_confusion_matrix) == 1:
        from merge_best_loso_folds import discover_run_dirs, merge_and_plot

        run_dirs = discover_run_dirs(args.output_dir)
        if not run_dirs:
            print("[提示] 未找到 fold_details，无法画混淆矩阵。")
        else:
            merge_and_plot(
                run_dirs,
                os.path.join(args.output_dir, "confusion_matrices"),
                merge_multi_seed=False,
            )


if __name__ == "__main__":
    main()
