"""Train the STGNN from per-video feature caches (this script is not run by dataset prep)"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from modules.dataset import Dataset
from modules.stgnn import STGNN


def _run_epoch(model, dataset, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for history, target in dataset:
            history = history.to(device)
            target = target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(history)
            loss = F.smooth_l1_loss(prediction, target)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError(
            f"the {dataset.split} split has no aligned windows; prepare more clips "
            "or adjust --window-size"
        )
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="dataset_bundle/local_data/prepared")
    parser.add_argument("--output", default="dataset_bundle/local_data/stgnn_checkpoint.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--graph-radius",
        type=float,
        required=True,
        help="neighbor radius in normalized image-coordinate units",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("--epochs must be positive and --learning-rate must be positive")

    torch.manual_seed(args.seed)
    train_data = Dataset(args.features_dir, "train", args.window_size, seed=args.seed)
    validation_data = Dataset(args.features_dir, "validation", args.window_size, seed=args.seed)
    device = torch.device(args.device)
    model = STGNN(args.graph_radius, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_validation_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_data, device, optimizer)
        validation_loss = _run_epoch(model, validation_data, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "graph_radius": args.graph_radius,
                        "hidden_dim": args.hidden_dim,
                    },
                    "window_size": args.window_size,
                    "feature_count": 13,
                    "coordinate_units": "normalized image fractions",
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "seed": args.seed,
                },
                output_path,
            )

    print(f"Best checkpoint saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
