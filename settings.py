import random
import numpy as np
import mne
import torch
from types import SimpleNamespace


def make_dataset_metadata(name, *, tmin=0.0, tmax=4.0):
    if name == "BCICIV_2a":
        return SimpleNamespace(name=name, tmin=tmin, tmax=tmax, sample_rate=250, n_subjects=9, n_electrodes=22, n_classes=4)
    if name == "BCICIV_2b":
        return SimpleNamespace(name=name, tmin=tmin, tmax=tmax, sample_rate=250, n_subjects=9, n_electrodes=3, n_classes=2)
    if name == "OPENBMI_P300":
        return SimpleNamespace(name=name, tmin=0.0, tmax=0.8, sample_rate=250, n_subjects=54, n_electrodes=64, n_classes=2)
    raise ValueError(f"Unknown dataset metadata: {name}")

# GPU/device configuration
use_CUDA = torch.cuda.is_available()
GPU_idx = 0  # GPU编号
# GPU_idx = 1  # GPU编号
device = torch.device('cuda:' + str(GPU_idx) if use_CUDA else 'cpu')


#本地没有GPU时
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集相关（LOSO 脚本均用「单 session」：load(..., session_list=[1])，与 2b 单段用法一致）
dataset_name = "BCICIV_2a"
# dataset_name = "BCICIV_2b"

# 预处理相关
preprocessing_params = {
    "use_CAR": True,
    # "use_CAR": False,
    "use_band_pass": True,
    # "use_band_pass": False,
    "moving_average_std": True,
    # "moving_average_std": False,
    "n_jobs": 2,
    # "n_jobs": 4,
    # "use_preprocessed_data_from_files": True,
    "use_preprocessed_data_from_files": False,
}
# 带通滤波参数设置
band_pass_params_eeg = {
    "iir_filter": True,
    # "filter_order": 1,
    # "filter_order": 2,
    "filter_order": 3,
    # "filter_order": 4,
    # "filter_order" : "auto",
    "filter_type": "butter",
    # "filter_type": "cheby2",
    "filter_rs": 4,
    "filter_rp": 0.5,
    "lower_bound_eeg": 0.5,
    "upper_bound_eeg": 50.,
}

# 把带通滤波的参数合并到预处理参数字典中
for item in band_pass_params_eeg:
    preprocessing_params[item] = band_pass_params_eeg[item]

# 数据增强相关
data_augment_params = {
    # "use_data_augment": True,
    "use_data_augment": False,
    "augment_times": 6,
    "method": "white_noise_data_augment",

}
# 添加噪声数据增强方法的参数
noise_adding_params = {
    "white_noise_std": 0.02,
    # "white_noise_std": 0.05, # MRP2023 subject2 卧姿
    "white_noise_snr": 1.,
    # "white_noise_snr": 0.8, # MRP2023 subject2 卧姿
    # "white_noise_snr": 4.,
}
# 把添加噪声的参数合并到数据增强的参数字典中
for item in noise_adding_params:
    data_augment_params[item] = noise_adding_params[item]

# 模型相关

# 隐私主干特征提取：三选一。优先级由训练脚本解析；命令行 --backbone 可覆盖。
use_models = {
    # 传统机器学习
    "ML:EEG_CSP+SVM": False,
    # 深度学习（IDRemovalNet 的 extractor / extractor_raw）
    "EEGNet": True,
    "MSVTNet": False,
    "ShallowConvNet": False,
    "FBCNet": False,
}

# 各个模型的参数（通过模型名称来索引）、通道选择参数、数据增强参数和训练参数
params = {
    "EEGNet": {
        "F1": 8,
        "D": 2,
        "kernel_length": 65,
        "avg_pool1_size": 8,
        "avg_pool2_size": 16,
        # "dconv2_size": 32,
        "dconv2_size": 33,
        "learning_rate": 1e-3,
        "dropout_rate": 0.5,
        "fc_max_norm": 0.25,
        "validation_stopping_patience": 200,  # 提前结束的容忍度（单位为迭代次数）
    },
    "MSVTNet": {
        "F": [9, 9, 9, 9],
        "C1": [15, 31, 63, 125],
        "kernel_length": 65,
        "depth": 2,
        # MSVTNet 主干更深、参数量更大，LOSO 下略降 lr 通常更稳（仍可用 --lr 覆盖）
        "learning_rate": 5e-4,
        "dropout_rate": 0.5,
        "validation_stopping_patience": 200,
        # 以下为 MSVTNet 主干（privacy/msvtnet_extractor.py）所需；仅实现 patch_embedding="conv"
        "patch_embedding": "conv",
        "C2": 15,
        "D": 2,
        "P1": 8,
        "P2": 7,
        "Pc": 0.25,
        "nhead": 8,
        "ff_ratio": 4,
        "Pt": 0.15,
        "bn_momentum": 0.01,
        "bn_eps": 1e-3,
        "fs": 250,
        # sinc/hybrid 时会用到；conv 分支不读取，仅占位
        "frequency_bands": [
            {"fmin": 4.0, "fmax": 8.0},
            {"fmin": 8.0, "fmax": 13.0},
            {"fmin": 13.0, "fmax": 30.0},
            {"fmin": 30.0, "fmax": 45.0},
        ],
    },
    "ShallowConvNet": {
        "F1": 40,
        "D": 1,
        "conv1_size": 25,
        "avg_pool_size": 75,
        "avg_pool_stride": 15,
        "max_norm_temporal_conv": 2.0,
        "max_norm_spatial_conv": 2.0,
        "max_norm_linear": 0.5,
        "learning_rate": 1e-3,
        "dropout_rate": 0.5,
        "validation_stopping_patience": 200,
        "bn_momentum": 0.1,
        "bn_eps": 1e-5,
        "log_eps": 1e-6,
    },
    "FBCNet": {
        "frequency_bands": [
            {"fmin": 4.0, "fmax": 8.0},
            {"fmin": 8.0, "fmax": 13.0},
            {"fmin": 13.0, "fmax": 30.0},
            {"fmin": 30.0, "fmax": 45.0},
        ],
        "fs": 250,
        "filterbank_kernel_size": 129,
        "temporal_filters": 8,
        "temporal_kernel_size": 25,
        "spatial_multiplier": 2,
        "stride": 4,
        "max_norm_spatial_conv": 2.0,
        "learning_rate": 1e-3,
        "dropout_rate": 0.5,
        "validation_stopping_patience": 200,
        "bn_momentum": 0.1,
        "bn_eps": 1e-5,
        "log_eps": 1e-6,
    },
    "FBCSP": {
        "reg": None,
        "n_components": 4,
        "n_frequency_bands": 9
    },
    "SVM": {
        "kernel": "linear",
        "svm_C": 1
    },
}

# 跨被试相关
inter_subject_params = {
    "subjects_usage": {  # 部分实验，根据要求预先安排好训练集、验证集或测试集。这里可以按需填被试编号（从1开始排）
        "train": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "validation": [13, 14],
        # "test": [15, 16, 17, 18],
    },
    "calibration_ratio": 0.1,
    # "calibration_ratio": 0,
    "calibration_with_label": True,
    "calibration_by_subjects": False,  # 对于跨被试的unseen场景，是否需要每个被试分别取校准集
    # "combine_train_calibration": True,  # 是否混合源域训练集和目标域校准集，作为整体构建训练集，训练模型。False表示训练阶段不含目标域数据，需单独做校准/微调
    "random_selection": False,  # 是否随机取校准集。为False则取前n个
    "stratified": True,  # （random_selection为True时生效）取校准集时，是否各个类别的样本都要取到（仅限分类任务）
}

# 训练相关
training_params = {
    # 深度学习训练相关
    "dataset_name": dataset_name,
    "use_CUDA": use_CUDA,
    "device": device,
    # "pretrain": True,
    "pretrain": False,
    "two_step": True,  # 是否训练2轮（第一轮区分训练集验证集，第二轮合并）
    # "two_step": False,
    "n_epochs": 400,  # fit的迭代次数,
    "n_epochs_2step": 400,
    # "n_epochs_2step": 10,
    "optimizer_type": "Adam",
    "momentum": 0.9,
    "weight_decay": 0,
    # "weight_decay": 0.1,
    "batch_size": 128,
    # "batch_size": 90,
    "early_stop": True,
    "early_stop_monitor": "val_loss",
    # "early_stop_by_first_phase": True,
    "early_stop_by_first_phase": False,
    # "early_stop_monitor": "val_accuracy",
    "checkpoint_monitor": "val_loss",
    # "checkpoint_monitor": "val_accuracy",
    # "num_workers": 2,
    "num_workers": 0,
}
params["DL_Training"] = training_params

# 场景
# scenario_name = "intra_subject_cross_validation"
# scenario_name = "intra_subject_unseen_evaluation"
scenario_name = "inter_subject_leave_one_subject_out"

K = 5
valid_size = 0.2
params["DL_Training"]["valid_size"] = valid_size


# 其他设置
seed = 1
n_jobs = 4
mne.set_log_level(verbose="WARNING")
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# Remove randomness (maybe slower on Tesla GPUs)
# https://pytorch.org/docs/stable/notes/randomness.html
if seed == 1:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 和数据集相关的一些初始化
if dataset_name == "BCICIV_2a":
    tmin = 0
    tmax = 4.
    dataset = make_dataset_metadata("BCICIV_2a", tmin=tmin, tmax=tmax)
elif dataset_name == "BCICIV_2b":
    dataset = make_dataset_metadata("BCICIV_2b", tmin=0, tmax=4)

dataset_name = dataset.name  # 统一名称


start_subject = 1
n_subjects = 9
end_subject = 9

target_subject = 4  # 只有1个受试者时可以指定想试验的受试者序号（从1开始排）


experiment_params = {
    "datetime_mark": "",
    "scenario_name": scenario_name,
    "start_subject": start_subject,
    "end_subject": end_subject,
    "sessions_usage": {
        "train": [1],  # 2a：仅 session 1（对应 A0xT.gdf）；与 LOSO 里 session_list=[1] 一致
        # "train": [1, 2, 3],  # 2b 多 session
        "valid": [],
        "test": [],
        "calibration": [],
    },
    "task_session": False, #判断当前是训练隐私分类头还是任务分类头
    "subject_session": True,
    "proposed_network": False
}

params["experiment"] = experiment_params
