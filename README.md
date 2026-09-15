# TAPR-RP: Task-Anchored EEG Privacy Removal with Release-Space Identity Subspace Projection

This repository contains the core method code for **TAPR-RP**, a task-anchored EEG representation release framework for reducing subject identity leakage while preserving task decoding performance.

The release is intentionally trimmed. It includes the privacy-removal method, decoder adapters, and reference LOSO training pipeline, but excludes raw datasets, preprocessing implementation, server scripts, experiment logs, trained weights, plotting assets, and paper drafts.

## Core Contributions

1. **Task-Anchored Privacy Evidence Removal** (`privacy/taper.py`)
   - trains frozen identity-evidence probes on raw EEG-derived views;
   - supports covariance, band-power proxy, and temporal-statistics privacy evidence;
   - applies entropy/agreement/covariance-alignment losses during representation release.

2. **Release-Space Identity Subspace Projection** (`train_loso_idremovalnet.py`)
   - estimates identity-dominant directions in the released task representation;
   - projects these directions away before downstream identity attack evaluation.

3. **EEG Release Module** (`privacy/ids_module.py`)
   - produces an EEG-like released signal before decoder-side representation learning;
   - combines task retention, identity suppression, and constrained raw-signal task compensation.

4. **Decoder-Agnostic Adapters**
   - EEGNet is implemented inside the training scripts;
   - MSVTNet, ShallowConvNet, and FBCNet adapters are under `privacy/*_extractor.py`;
   - all adapters expose the same `(B, C, T) -> (B, feat_dim)` feature interface.

## Repository Layout

```text
.
├── train_loso_idremovalnet.py          # Reference TAPR-RP LOSO training pipeline
├── train_loso_eegnet_baseline_only.py  # Reference baseline LOSO pipeline
├── settings.py                         # Model and training defaults
├── privacy/
│   ├── taper.py                        # Core TAPR privacy probes and losses
│   ├── ids_module.py                   # EEG release module
│   ├── eeg_euclidean_align.py          # Euclidean alignment utility
│   ├── msvtnet_extractor.py            # MSVTNet adapter
│   ├── shallowconvnet_extractor.py     # ShallowConvNet adapter
│   └── fbcnet_extractor.py             # FBCNet adapter
├── requirements.txt
└── environment.yml
```

## Input Contract

The full preprocessing/data-construction code is not included in this release. The training pipeline assumes that data have already been converted into per-subject EEG trial arrays:

- `X_list`: list of subject arrays, each shaped `(n_trials, n_channels, n_times)`;
- `y_task_list`: list of task-label arrays, each shaped `(n_trials,)`;
- `y_subject_list`: list of subject-ID arrays, each shaped `(n_trials,)`.

For OpenBMI ERP/P300 experiments, our internal preprocessing produced a balanced single-session cache equivalent to:

- `X`: `(n_subjects, n_trials, n_channels, n_times)` float32;
- `y`: `(n_subjects, n_trials)` int64 task labels;
- subject IDs are assigned by subject index.

## Preprocessing Description

For the P300 experiments, each subject used one session. The signal preprocessing followed this protocol:

1. select EEG channels and remove non-EEG auxiliary channels;
2. epoch each trial relative to stimulus onset using a post-stimulus P300 window;
3. remove per-channel DC offset;
4. apply a band-pass filter in the ERP-relevant range;
5. resample to a common sampling rate;
6. z-score each trial channel over time;
7. balance target and non-target trials within each subject using a fixed random seed;
8. store arrays in subject-major order for LOSO evaluation.

For BCI Competition IV 2a/2b experiments, preprocessing used the same general structure: single-session LOSO splits, CAR where applicable, band-pass filtering, trial-wise standardization, and optional Euclidean alignment fitted only on the current training pool.

## Reference Training Pipeline

`train_loso_idremovalnet.py` contains the reference implementation of the method pipeline:

1. construct a leave-one-subject-out split;
2. optionally fit Euclidean alignment on the training pool;
3. train the task/privacy release network;
4. train frozen TAPR privacy probes;
5. refit utility heads and task-only components;
6. estimate and apply release-space identity projection;
7. evaluate task accuracy and identity attack accuracy per held-out subject;
8. optionally save fold-level reproducibility checkpoints.

`train_loso_eegnet_baseline_only.py` contains the corresponding baseline structure used for task and identity-attack comparison.

## Dependencies

Use `environment.yml` or `requirements.txt` as a reference environment. PyTorch with CUDA is recommended for full LOSO experiments.

## Outputs

The reference scripts write summary JSON files and per-fold details when connected to a compatible data interface:

- `summary_loso_eegnet_only.json` for baseline runs;
- `summary_privacy_metrics.json` for TAPR-RP runs;
- `fold_details/fold_*.json` for per-subject metrics;
- `weights/` only when checkpoint saving is enabled.

## Release Notes

This repository is a method-code release, not a turnkey experiment package. Dataset download, preprocessing, cache construction, and experiment-launch scripts are intentionally omitted.

Please add the final citation and license before public release.
