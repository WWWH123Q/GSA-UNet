from contextlib import nullcontext

import torch
from tqdm import tqdm

from metrics import ForecastMetrics
from utils import format_loss_components


def _autocast(device, enabled):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def train_epoch(
    loader, model, criterion, optimizer, device, epoch, logger, scaler=None, log_batches=0
):
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(loader, desc=f"Train {epoch}", leave=False)
    for batch_index, (inputs, targets) in enumerate(progress):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        with _autocast(device, amp_enabled):
            prediction = model(inputs)
            loss = criterion(prediction, targets)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.5f}")
        if batch_index < log_batches:
            details = format_loss_components(criterion)
            if details:
                logger.info("epoch=%d batch=%d %s", epoch, batch_index, details)
    return total_loss / max(total_samples, 1)


def evaluate(loader, model, criterion, device, thresholds, description="Evaluate"):
    model.eval()
    metrics = ForecastMetrics(thresholds)
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=description, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            metrics.update(prediction, targets)
    result = metrics.compute()
    result["loss"] = total_loss / max(total_samples, 1)
    return result


def validate(loader, model, criterion, device, thresholds):
    return evaluate(loader, model, criterion, device, thresholds, description="Validate")
