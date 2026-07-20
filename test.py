import argparse
import json
from pathlib import Path

import torch

from config import GSAConfig
from datasets import create_dataloaders
from engine import evaluate
from losses import GSALoss
from models import GSAUNet
from utils import load_model_state, read_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GSA-UNet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--data-scale", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def config_from_checkpoint(checkpoint):
    config = GSAConfig()
    for name, value in checkpoint.get("config", {}).items():
        if hasattr(config, name):
            setattr(config, name, value)
    return config


def main():
    args = parse_args()
    checkpoint = read_checkpoint(args.checkpoint, "cpu")
    config = config_from_checkpoint(checkpoint)
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.data_scale is not None:
        config.data_scale = args.data_scale
    if args.device is not None:
        config.device = args.device
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        config.device = "cpu"
    config.validate()
    set_seed(config.seed)

    device = torch.device(config.device)
    model = GSAUNet(**config.model_kwargs()).to(device)
    load_model_state(model, checkpoint)
    criterion = GSALoss(**config.loss_kwargs()).to(device)
    test_loader = create_dataloaders(config)["test"]
    result = evaluate(test_loader, model, criterion, device, config.thresholds, description="Test")
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
