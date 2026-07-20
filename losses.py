import torch
import torch.nn.functional as F
from torch import nn


class GSALoss(nn.Module):
    """Asymmetric rain-rate loss with spatial, mass, and temporal constraints."""

    def __init__(
        self,
        omega=0.59,
        strong_alpha=0.25,
        strong_beta=1.0,
        strong_power=1.0,
        smooth_weight=0.001,
        mass_weight=0.04,
        temporal_weight=0.0003,
        temporal_threshold=0.5,
        temporal_strong_weight=0.0,
    ):
        super().__init__()
        self.omega = omega
        self.strong_alpha = strong_alpha
        self.strong_beta = strong_beta
        self.strong_power = strong_power
        self.smooth_weight = smooth_weight
        self.mass_weight = mass_weight
        self.temporal_weight = temporal_weight
        self.temporal_threshold = temporal_threshold
        self.temporal_strong_weight = temporal_strong_weight
        self.base_omega = 0.57
        self.last_components = {}

    @staticmethod
    def _laplacian(values):
        padded = F.pad(values, (1, 1, 1, 1))
        return (
            -4 * values
            + padded[..., 1:-1, 2:]
            + padded[..., 1:-1, :-2]
            + padded[..., 2:, 1:-1]
            + padded[..., :-2, 1:-1]
        )

    def _temporal_consistency(self, prediction, target):
        if prediction.ndim < 4 or prediction.size(1) < 2:
            return prediction.new_tensor(0.0)
        prediction_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        error = torch.abs(prediction_delta - target_delta)
        if self.temporal_strong_weight >= 1.0:
            return error.mean()
        pair_max = torch.maximum(target[:, 1:], target[:, :-1])
        weights = torch.where(
            pair_max >= self.temporal_threshold,
            torch.full_like(error, self.temporal_strong_weight),
            torch.ones_like(error),
        )
        return torch.sum(error * weights) / weights.sum().clamp_min(1.0)

    def forward(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
            )

        error = torch.abs(prediction - target)
        base_weights = torch.where(
            prediction >= target,
            1.0 - self.base_omega,
            self.base_omega,
        )
        base_loss = torch.mean(base_weights * error)

        bounded_target = target.clamp(0.0, 1.0)
        strong_weights = self.strong_alpha * torch.exp(
            self.strong_beta * bounded_target.pow(self.strong_power)
        )
        strong_asymmetry = torch.where(
            prediction >= target,
            1.0 - self.omega,
            self.omega,
        )
        strong_mask = target >= 0.7
        strong_loss = torch.mean(strong_asymmetry * strong_weights * error * strong_mask)
        data_loss = base_loss + strong_loss

        smooth_loss = torch.mean(torch.abs(self._laplacian(prediction)))
        mass_loss = torch.abs(prediction.mean() - target.mean())
        temporal_loss = self._temporal_consistency(prediction, target)
        total_loss = (
            data_loss
            + self.smooth_weight * smooth_loss
            + self.mass_weight * mass_loss
            + self.temporal_weight * temporal_loss
        )

        with torch.no_grad():
            constraint_loss = (
                self.smooth_weight * smooth_loss
                + self.mass_weight * mass_loss
                + self.temporal_weight * temporal_loss
            )
            self.last_components = {
                "base_loss": float(data_loss.detach().cpu()),
                "strong_loss": float(strong_loss.detach().cpu()),
                "smooth_loss": float(smooth_loss.detach().cpu()),
                "mass_loss": float(mass_loss.detach().cpu()),
                "temporal_loss": float(temporal_loss.detach().cpu()),
                "constraint_ratio": float(
                    (constraint_loss / data_loss.detach().abs().clamp_min(1e-12)).cpu()
                ),
                "total_loss": float(total_loss.detach().cpu()),
            }
        return total_loss
