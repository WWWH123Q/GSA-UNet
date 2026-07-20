import argparse
from pathlib import Path

import torch

from config import GSAConfig
from datasets import create_dataloaders
from engine import evaluate, train_epoch, validate
from losses import GSALoss
from models import GSAUNet
from utils import (
    build_optimizer,
    build_scheduler,
    count_parameters,
    create_logger,
    load_model_state,
    read_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GSA-UNet")
    parser.add_argument("--data-path", type=Path, default=Path("data/merged_data.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/gsa_unet"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-scale", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main():
    args = parse_args()
    config = GSAConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        data_scale=args.data_scale,
        amp=args.amp,
    )
    config.validate()
    device = resolve_device(config.device)
    config.device = str(device)
    set_seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(config.output_dir / "train.log")
    logger.info("project=%s device=%s data=%s", config.project_name, device, config.data_path)

    loaders = create_dataloaders(config)
    model = GSAUNet(**config.model_kwargs()).to(device)
    criterion = GSALoss(**config.loss_kwargs()).to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    logger.info("trainable_parameters=%d", count_parameters(model))

    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        checkpoint = read_checkpoint(args.resume, device)
        load_model_state(model, checkpoint)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))
        logger.info("resumed=%s start_epoch=%d", args.resume, start_epoch)

    best_path = config.output_dir / "best_gsa_unet.pth"
    last_path = config.output_dir / "last_gsa_unet.pth"
    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_epoch(
            loaders["train"],
            model,
            criterion,
            optimizer,
            device,
            epoch,
            logger,
            scaler,
            config.loss_log_batches,
        )
        validation = validate(loaders["val"], model, criterion, device, config.thresholds)
        scheduler.step()
        validation_loss = validation["loss"]
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f lr=%.8f",
            epoch,
            train_loss,
            validation_loss,
            optimizer.param_groups[0]["lr"],
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_loss, config)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_loss, config)

    if not best_path.is_file():
        raise RuntimeError("Training finished without producing a checkpoint")
    load_model_state(model, read_checkpoint(best_path, device))
    test_result = evaluate(
        loaders["test"], model, criterion, device, config.thresholds, description="Test"
    )
    logger.info("test_result=%s", test_result)


if __name__ == "__main__":
    main()
