from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VILSequenceDataset(Dataset):
    """Lazy sliding-window dataset for a [time, height, width] HDF5 array."""

    def __init__(self, path, key, input_frames, output_frames, start, stop, scale=1.0):
        self.path = Path(path)
        self.key = key
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.start = start
        self.stop = stop
        self.scale = scale
        self._file = None
        self._dataset = None

        if not self.path.is_file():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        with h5py.File(self.path, "r") as handle:
            if self.key not in handle:
                raise KeyError(f"HDF5 key not found: {self.key}")
            shape = handle[self.key].shape
        if len(shape) != 3:
            raise ValueError(f"Expected [time, height, width], got {shape}")
        if self.start < 0 or self.stop > shape[0] or self.start >= self.stop:
            raise ValueError(f"Invalid frame range: [{self.start}, {self.stop})")
        if len(self) < 1:
            raise ValueError("Dataset split is shorter than one sequence window")

    @property
    def window_size(self):
        return self.input_frames + self.output_frames

    def __len__(self):
        return self.stop - self.start - self.window_size + 1

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._dataset = self._file[self.key]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        self._open()
        begin = self.start + index
        frames = np.asarray(self._dataset[begin : begin + self.window_size], dtype=np.float32)
        if self.scale != 1.0:
            frames = frames / self.scale
        frames = torch.from_numpy(frames)
        return frames[: self.input_frames], frames[self.input_frames :]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_dataset"] = None
        return state

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dataset = None

    def __del__(self):
        self.close()


def create_datasets(config):
    with h5py.File(config.data_path, "r") as handle:
        if config.dataset_key not in handle:
            raise KeyError(f"HDF5 key not found: {config.dataset_key}")
        total_frames = int(handle[config.dataset_key].shape[0])

    train_stop = int(total_frames * config.train_ratio)
    val_stop = train_stop + int(total_frames * config.val_ratio)
    ranges = {
        "train": (0, train_stop),
        "val": (train_stop, val_stop),
        "test": (val_stop, total_frames),
    }
    return {
        name: VILSequenceDataset(
            config.data_path,
            config.dataset_key,
            config.input_frames,
            config.output_frames,
            start,
            stop,
            config.data_scale,
        )
        for name, (start, stop) in ranges.items()
    }


def create_dataloaders(config):
    datasets = create_datasets(config)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device.startswith("cuda"),
        "persistent_workers": config.num_workers > 0,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
