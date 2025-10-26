import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from src.models.inverse_flow import InverseFlowDecoder, InverseFlowEncoder


@dataclass
class FlowInverseConfig:
    epochs: int
    lr: float
    latent_dim: int
    hidden_dim: int
    conditioning_dim: Optional[int]
    geometry_weight: float
    flow_weight: float
    latent_weight: float
    checkpoint_dir: Optional[str]
    resume_path: Optional[str]
    log_every: int
    reynolds_number: Optional[float]
    scheduler_step: Optional[int]
    scheduler_gamma: float


class FlowCompositeLoss(torch.nn.Module):
    """Composite loss that exposes each component for logging."""

    def __init__(self, geometry_weight: float, flow_weight: float, latent_weight: float) -> None:
        super().__init__()
        self.geometry_weight = geometry_weight
        self.flow_weight = flow_weight
        self.latent_weight = latent_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        latent: torch.Tensor,
        encoder_outputs: Dict[str, torch.Tensor],
        flow_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        geometry_loss = F.mse_loss(predictions, targets)

        if flow_tensor is not None and "flow_embedding" in encoder_outputs:
            flow_embedding = encoder_outputs["flow_embedding"]
            flow_mean = flow_tensor.view(flow_tensor.shape[0], -1).mean(dim=-1, keepdim=True)
            flow_mean_expanded = flow_mean.expand_as(flow_embedding)
            flow_consistency = F.mse_loss(flow_embedding, flow_mean_expanded)
        else:
            flow_consistency = predictions.new_tensor(0.0)

        latent_regularization = latent.pow(2).mean()

        total = (
            self.geometry_weight * geometry_loss
            + self.flow_weight * flow_consistency
            + self.latent_weight * latent_regularization
        )

        return {
            "total": total,
            "geometry": geometry_loss.detach(),
            "flow_consistency": flow_consistency.detach(),
            "latent_regularization": latent_regularization.detach(),
        }


class FlowInverseTrainer:
    """Trainer responsible for inverse flow supervision."""

    def __init__(
        self,
        config: FlowInverseConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        log_fn: Callable[[str], None],
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log = log_fn

        self.encoder = InverseFlowEncoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            conditioning_dim=config.conditioning_dim,
        ).to(device)
        self.decoder = InverseFlowDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=None,
            conditioning_dim=config.conditioning_dim,
        ).to(device)
        self.loss_fn = FlowCompositeLoss(
            geometry_weight=config.geometry_weight,
            flow_weight=config.flow_weight,
            latent_weight=config.latent_weight,
        )

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=config.lr)
        if config.scheduler_step is not None and config.scheduler_step > 0:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
            )
        else:
            self.scheduler = None

        self.current_epoch = 0
        self.best_val = None

        if config.checkpoint_dir:
            os.makedirs(config.checkpoint_dir, exist_ok=True)

        if config.resume_path:
            self._restore(config.resume_path)

    def fit(self) -> None:
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch + 1
            train_metrics = self._run_epoch(self.train_loader, training=True)
            val_metrics = self._run_epoch(self.val_loader, training=False)

            self.log(
                f"[FlowInverse] Epoch {self.current_epoch}/{self.config.epochs} "
                f"train_total={train_metrics['total']:.4f} val_total={val_metrics['total']:.4f}"
            )
            self.log(
                f"           Components -> "
                f"train_geom={train_metrics['geometry']:.4f} | "
                f"train_flow={train_metrics['flow_consistency']:.4f} | "
                f"train_latent={train_metrics['latent_regularization']:.4f}"
            )
            self.log(
                f"                       "
                f"val_geom={val_metrics['geometry']:.4f} | "
                f"val_flow={val_metrics['flow_consistency']:.4f} | "
                f"val_latent={val_metrics['latent_regularization']:.4f}"
            )

            if self.scheduler is not None:
                self.scheduler.step()

            improved = self.best_val is None or val_metrics["total"] < self.best_val
            if improved:
                self.best_val = val_metrics["total"]
            if self.config.checkpoint_dir:
                self._checkpoint(epoch=self.current_epoch, is_best=improved)

    def _checkpoint(self, epoch: int, is_best: bool) -> None:
        if not self.config.checkpoint_dir:
            return
        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        filename = os.path.join(self.config.checkpoint_dir, f"flow_inverse_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, filename)
        if is_best:
            best_filename = os.path.join(self.config.checkpoint_dir, "flow_inverse_best.pt")
            torch.save(checkpoint, best_filename)

    def _restore(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.log(f"Restored flow inverse trainer from {checkpoint_path} at epoch {self.current_epoch}.")

    def _run_epoch(self, loader: DataLoader, training: bool) -> Dict[str, float]:
        if training:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        total_losses = {"total": 0.0, "geometry": 0.0, "flow_consistency": 0.0, "latent_regularization": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            flow_tensor, geometry_targets, conditioning = self._prepare_batch(batch)

            with torch.set_grad_enabled(training):
                losses = self._forward_loss(flow_tensor, geometry_targets, conditioning)

            if training:
                self.optimizer.zero_grad()
                losses["total"].backward()
                self.optimizer.step()

            for key in total_losses:
                total_losses[key] += losses[key].item()
            num_batches += 1

            if training and (batch_idx + 1) % self.config.log_every == 0:
                self.log(
                    f"    [FlowInverse][Batch {batch_idx+1}/{len(loader)}] "
                    f"loss={losses['total'].item():.4f} "
                    f"geom={losses['geometry'].item():.4f} "
                    f"flow={losses['flow_consistency'].item():.4f} "
                    f"latent={losses['latent_regularization'].item():.4f}"
                )

        for key in total_losses:
            total_losses[key] /= max(1, num_batches)
        return total_losses

    def _prepare_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        flow_tensor = self._extract_tensor(batch, ["flow", "flow_tensor", "flow_fields"])
        geometry_targets = self._extract_tensor(batch, ["geometry", "geometry_target", "geometry_targets"])
        conditioning = None

        if "conditioning" in batch:
            conditioning = batch["conditioning"].to(self.device)
        elif "reynolds" in batch:
            conditioning = batch["reynolds"].to(self.device)
        elif self.config.reynolds_number is not None:
            batch_size = flow_tensor.shape[0]
            conditioning = torch.full(
                (batch_size, 1), self.config.reynolds_number, device=self.device, dtype=flow_tensor.dtype
            )

        flow_tensor = flow_tensor.to(self.device)
        geometry_targets = geometry_targets.to(self.device)
        if conditioning is not None:
            conditioning = conditioning.to(self.device)
        return flow_tensor, geometry_targets, conditioning

    def _extract_tensor(self, batch: Dict[str, torch.Tensor], keys: Iterable[str]) -> torch.Tensor:
        for key in keys:
            if key in batch:
                tensor = batch[key]
                if not torch.is_tensor(tensor):
                    tensor = torch.as_tensor(tensor)
                return tensor
        raise KeyError(f"Flow inverse trainer expected one of keys {keys} in the batch.")

    def _forward_loss(
        self,
        flow_tensor: torch.Tensor,
        geometry_targets: torch.Tensor,
        conditioning: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        geometry_flat = geometry_targets.view(geometry_targets.shape[0], -1)
        if self.decoder.output_dim is None:
            self.decoder.set_output_dim(geometry_flat.shape[-1])

        encoder_outputs = self.encoder(flow_tensor, conditioning)
        latent = encoder_outputs["latent"]
        predictions = self.decoder(latent, conditioning)

        losses = self.loss_fn(predictions, geometry_flat, latent, encoder_outputs, flow_tensor)
        return losses


def build_flow_inverse_dataloaders(
    dataset: Dataset,
    batch_size: int,
    val_split: float,
    collate_fn: Optional[Callable[[Iterable], Dict[str, torch.Tensor]]] = None,
) -> Tuple[DataLoader, DataLoader]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    val_size = int(len(dataset) * val_split)
    if val_size == 0 and len(dataset) > 1:
        val_size = 1
    train_size = len(dataset) - val_size
    if train_size == 0 and len(dataset) > 0:
        train_size = len(dataset) - 1
        val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def run_flow_inverse_training(args, log_fn: Callable[[str], None], device: torch.device) -> None:
    if not getattr(args, "flow_dataset", None):
        log_fn("Flow inverse training requires --flow_dataset pointing to a prepared dataset file or identifier.")
        return

    try:
        from src.data_processing.flow_dataset import FlowInverseDataset, flow_inverse_collate
    except ImportError as exc:  # pragma: no cover - optional dependency
        log_fn(
            "Flow inverse dataset utilities are missing. Please implement "
            "src.data_processing.flow_dataset.FlowInverseDataset before running flow inverse training."
        )
        log_fn(str(exc))
        return

    dataset = FlowInverseDataset(args.flow_dataset)
    train_loader, val_loader = build_flow_inverse_dataloaders(
        dataset=dataset,
        batch_size=args.flow_batch_size,
        val_split=args.flow_val_split,
        collate_fn=flow_inverse_collate,
    )

    checkpoint_dir = args.flow_checkpoint_dir
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    config = FlowInverseConfig(
        epochs=args.flow_epochs,
        lr=args.flow_lr,
        latent_dim=args.flow_latent_dim,
        hidden_dim=args.flow_hidden_dim,
        conditioning_dim=args.flow_condition_dim,
        geometry_weight=args.flow_loss_geometry_weight,
        flow_weight=args.flow_loss_flow_weight,
        latent_weight=args.flow_loss_latent_weight,
        checkpoint_dir=checkpoint_dir,
        resume_path=args.flow_resume_path,
        log_every=args.flow_log_every,
        reynolds_number=args.reynolds_number,
        scheduler_step=args.flow_scheduler_step,
        scheduler_gamma=args.flow_scheduler_gamma,
    )

    trainer = FlowInverseTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_fn=log_fn,
    )
    trainer.fit()
