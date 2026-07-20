import math

import torch


class ForecastMetrics:
    def __init__(self, thresholds):
        self.thresholds = tuple(thresholds)
        self.counts = {
            threshold: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for threshold in self.thresholds
        }
        self.squared_error = 0.0
        self.num_values = 0

    def update(self, prediction, target):
        prediction = prediction.detach()
        target = target.detach()
        self.squared_error += torch.sum((prediction - target) ** 2).item()
        self.num_values += target.numel()
        for threshold in self.thresholds:
            predicted_event = prediction >= threshold
            observed_event = target >= threshold
            values = self.counts[threshold]
            values["tp"] += torch.sum(predicted_event & observed_event).item()
            values["tn"] += torch.sum(~predicted_event & ~observed_event).item()
            values["fp"] += torch.sum(predicted_event & ~observed_event).item()
            values["fn"] += torch.sum(~predicted_event & observed_event).item()

    @staticmethod
    def _divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    def compute(self):
        result = {
            "rmse": math.sqrt(self.squared_error / self.num_values) if self.num_values else 0.0,
            "thresholds": {},
        }
        for threshold, values in self.counts.items():
            tp = values["tp"]
            tn = values["tn"]
            fp = values["fp"]
            fn = values["fn"]
            total = tp + tn + fp + fn
            hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
            result["thresholds"][str(threshold)] = {
                "accuracy": self._divide(tp + tn, total),
                "csi": self._divide(tp, tp + fp + fn),
                "hss": self._divide(2 * (tp * tn - fp * fn), hss_denominator),
                "pod": self._divide(tp, tp + fn),
                "far": self._divide(fp, tp + fp),
            }
        return result
