import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from Dataloaders.DataloaderECG import ECGDataset
from Forecast.PatchForecast import EncoderDecoderPatchTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train patch encoder-decoder Transformer")
    parser.add_argument("--data-dir", type=str, default="Preprocess/PTBXL")
    parser.add_argument("--norm", type=str, default="zscore")
    parser.add_argument("--channel", type=int, default=0)

    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--lookback", type=int, default=192)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/patch_transformer.pt")
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_extra_dims(x: Tensor) -> Tensor:
    while x.ndim > 3:
        x = x.reshape(-1, *x.shape[2:])
    return x


def _to_time_last(x: Tensor, expected_length: int) -> Tensor:
    if x.ndim == 2:
        return x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError(f"Expected a tensor with 2 or 3 dims, got shape {tuple(x.shape)}")

    if x.shape[1] == expected_length and x.shape[2] != expected_length:
        return x
    if x.shape[2] == expected_length:
        return x.transpose(1, 2)
    if x.shape[1] == expected_length and x.shape[2] == expected_length:
        return x

    raise ValueError(
        f"Cannot infer time axis for expected length {expected_length} from shape {tuple(x.shape)}"
    )


def prepare_batch(x: Tensor, y: Tensor, lookback: int, horizon: int, device: str) -> tuple[Tensor, Tensor]:
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    x = _flatten_extra_dims(x)
    y = _flatten_extra_dims(y)

    x = _to_time_last(x, expected_length=lookback)
    y = _to_time_last(y, expected_length=horizon)

    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model: EncoderDecoderPatchTransformer,
    loader: DataLoader,
    criterion: nn.Module,
    lookback: int,
    horizon: int,
    device: str,
) -> tuple[float, float]:
    model.eval()
    teacher_forcing_loss = 0.0
    autoregressive_loss = 0.0
    n_batches = 0

    for x, y in loader:
        src, tgt = prepare_batch(x, y, lookback=lookback, horizon=horizon, device=device)
        pred_tf = model(src, tgt)
        pred_ar = model(src)

        teacher_forcing_loss += criterion(pred_tf, tgt).item()
        autoregressive_loss += criterion(pred_ar, tgt).item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0

    return teacher_forcing_loss / n_batches, autoregressive_loss / n_batches


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.lookback % args.patch_len != 0:
        raise ValueError("lookback must be divisible by patch-len")

    set_seed(args.seed)

    train_dataset = ECGDataset(
        dir=args.data_dir,
        dataset="train",
        norm=args.norm,
        lookback=args.lookback,
        horizon=args.horizon,
        channel=args.channel,
    )
    val_dataset = ECGDataset(
        dir=args.data_dir,
        dataset="validation",
        norm=args.norm,
        lookback=args.lookback,
        horizon=args.horizon,
        channel=args.channel,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    sample_x, sample_y = next(iter(train_loader))
    sample_x, sample_y = prepare_batch(
        sample_x,
        sample_y,
        lookback=args.lookback,
        horizon=args.horizon,
        device=args.device,
    )
    input_dim = sample_x.shape[-1]
    target_dim = sample_y.shape[-1]

    model = EncoderDecoderPatchTransformer(
        input_dim=input_dim,
        target_dim=target_dim,
        num_layers=args.num_layers,
        horizon=args.horizon,
        lookback=args.lookback,
        patch_len=args.patch_len,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(args.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_tf = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            src, tgt = prepare_batch(
                x,
                y,
                lookback=args.lookback,
                horizon=args.horizon,
                device=args.device,
            )

            optimizer.zero_grad()
            pred = model(src, tgt)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_tf_loss, val_ar_loss = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            lookback=args.lookback,
            horizon=args.horizon,
            device=args.device,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_tf_mse={train_loss:.6f} | "
            f"val_tf_mse={val_tf_loss:.6f} | "
            f"val_ar_mse={val_ar_loss:.6f}"
        )

        if val_tf_loss < best_val_tf:
            best_val_tf = val_tf_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_tf_mse": val_tf_loss,
                    "val_ar_mse": val_ar_loss,
                    "config": vars(args),
                    "input_dim": input_dim,
                    "target_dim": target_dim,
                },
                save_path,
            )
            print(f"Saved best checkpoint to {save_path}")


if __name__ == "__main__":
    main()
