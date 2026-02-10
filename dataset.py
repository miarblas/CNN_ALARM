from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

import utils
import constants as const


@dataclass
class PreprocessConfig: 
    """
    Configuration for dataset preprocessing.
    """
    window_sec: int = const.WINDOW_SIZE       # Only last N seconds of each record
    max_channels: int = const.MAX_CHANNELS    # Number of channels expected by the model


def _z_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-channel z-normalization (mean=0, std=1).
    Args:
        x: waveform array, shape (C, T)
        eps: small number to avoid division by zero
    Returns:
        Normalized waveform, shape (C, T)
    """
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + eps)


def _status_to_label(status: Any) -> int:
    """
    Convert H5 label/status to binary label for training.
    True alarm -> 1, False alarm -> 0
    """
    s = str(status).strip().lower()
    if "true" in s:
        return 1
    if "false" in s:
        return 0
    try:
        return int(float(status))
    except Exception as e:
        raise ValueError(f"Unrecognized status label: {status}") from e


class H5AlarmDataset(Dataset):
    """
    PyTorch Dataset for ICU alarm recordings from H5 files.

    Assumes waveform data is stored as (N, T, C) in the H5 file.
    Automatically converts it to (N, C, T) for CNN input.
    """

    def __init__(self, h5_path: str, pp: PreprocessConfig, require_labels: bool = True):
        self.h5_path = h5_path
        self.pp = pp

        # Load H5 data
        self.data = utils.loadh5(h5_path)
        if "waveform" not in self.data or "fs" not in self.data:
            raise ValueError("H5 file must contain 'waveform' and 'fs'.")

        # Extract waveform, sampling rates, and labels
        self.waveform = self.data["waveform"]    # shape: (N, T, C)
        self.fs = self.data["fs"]                # shape: (N,) or scalar
        self.status = self.data.get("status", None)

        if require_labels and self.status is None:
            raise ValueError("Labels ('status') are required but not found in this H5 file.")

        # Transpose waveform to (N, C, T) for CNN input
        self.wf_c_first = self.waveform.transpose(0, 2, 1).astype(np.float32)

        # Ensure fs_per_record is a 1D array of floats
        if np.ndim(self.fs) == 0:
            self.fs_per_record = np.full((self.wf_c_first.shape[0],), float(self.fs))
        else:
            self.fs_per_record = np.asarray(self.fs).astype(float).reshape(-1)

    def __len__(self) -> int:
        return self.wf_c_first.shape[0]

    def _crop_last_window(self, x_ct: np.ndarray) -> np.ndarray:
        """
        Keep only the last `window_sec` seconds of the record.
        Pads with zeros if record is shorter.
        Args:
            x_ct: waveform (C, T)
        Returns:
            Cropped waveform (C, win_samples)
        """
        T = x_ct.shape[1]
        win = int(self.pp.window_sec * self.fs_per_record[0])  # Using record fs for window length
        if T >= win:
            return x_ct[:, -win:]
        pad = win - T
        return np.pad(x_ct, ((0, 0), (pad, 0)), mode="constant")

    def _pad_channels(self, x_ct: np.ndarray) -> np.ndarray:
        """
        Pad or crop channels to match max_channels.
        Args:
            x_ct: waveform (C, T)
        Returns:
            x_ct with shape (max_channels, T)
        """
        C = x_ct.shape[0]
        if C > self.pp.max_channels:
            return x_ct[:self.pp.max_channels, :]
        if C < self.pp.max_channels:
            pad_c = self.pp.max_channels - C
            return np.pad(x_ct, ((0, pad_c), (0, 0)), mode="constant")
        return x_ct

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return preprocessed waveform and label (if available) as tensors.
        Steps:
        1) Load waveform for index
        2) Crop last window_sec seconds
        3) Pad or crop channels
        4) Z-normalize per channel
        5) Replace NaN/Inf with zeros
        6) Convert to torch tensor
        """
        x = self.wf_c_first[idx]                # (C, T)
        x = self._crop_last_window(x)
        x = self._pad_channels(x)
        x = _z_norm(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        x_t = torch.tensor(x, dtype=torch.float32)

        if self.status is None:
            return x_t, None

        y = _status_to_label(self.status[idx])
        y_t = torch.tensor(y, dtype=torch.float32)
        return x_t, y_t

if __name__ == "__main__":
    import os

    # Path to your H5 file (adjust if necessary)
    h5_path = os.path.join("data", "trainSet.h5")
    pp = PreprocessConfig()

    # Load dataset
    dataset = H5AlarmDataset(h5_path, pp, require_labels=True)
    print(f"Dataset loaded. Number of samples: {len(dataset)}")
    print(f"Waveform shape (N, T, C): {dataset.wf_c_first.shape}")

    # Inspect first 3 samples
    for i in range(3):
        x_t, y_t = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Tensor shape (C, window_samples): {x_t.shape}")
        print(f"  Label: {y_t.item() if y_t is not None else None}")
        print(f"  First 5 values of channel 0: {x_t[0, :5].numpy()}")

