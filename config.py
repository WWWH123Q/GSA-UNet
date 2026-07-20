from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GSAConfig:
    project_name: str = "GSA-UNet"
    data_path: Path = Path("data/merged_data.h5")
    dataset_key: str = "vil"
    input_frames: int = 5
    output_frames: int = 3
    data_scale: float = 1.0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    channels: tuple[int, int, int, int] = (16, 32, 64, 128)
    bridge: bool = True
    batch_size: int = 2
    epochs: int = 60
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    min_learning_rate: float = 1e-5
    seed: int = 42
    amp: bool = False
    device: str = "cuda"
    output_dir: Path = Path("results/gsa_unet")
    thresholds: tuple[float, ...] = field(default_factory=lambda: (0.1, 0.3, 0.5, 0.7, 0.8))
    loss_log_batches: int = 5
    omega: float = 0.59
    strong_alpha: float = 0.25
    strong_beta: float = 1.0
    strong_power: float = 1.0
    smooth_weight: float = 0.001
    mass_weight: float = 0.04
    temporal_weight: float = 0.0003
    temporal_threshold: float = 0.5
    temporal_strong_weight: float = 0.0

    @property
    def window_size(self):
        return self.input_frames + self.output_frames

    def validate(self):
        if self.input_frames < 1 or self.output_frames < 1:
            raise ValueError("input_frames and output_frames must be positive")
        if self.data_scale <= 0:
            raise ValueError("data_scale must be positive")
        if not 0 < self.train_ratio < 1 or not 0 < self.val_ratio < 1:
            raise ValueError("train_ratio and val_ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")

    def model_kwargs(self):
        return {
            "num_classes": self.output_frames,
            "input_channels": self.input_frames,
            "channels": self.channels,
            "bridge": self.bridge,
        }

    def loss_kwargs(self):
        return {
            "omega": self.omega,
            "strong_alpha": self.strong_alpha,
            "strong_beta": self.strong_beta,
            "strong_power": self.strong_power,
            "smooth_weight": self.smooth_weight,
            "mass_weight": self.mass_weight,
            "temporal_weight": self.temporal_weight,
            "temporal_threshold": self.temporal_threshold,
            "temporal_strong_weight": self.temporal_strong_weight,
        }
