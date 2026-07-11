from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback


class GradientNormLogger(Callback):
    def __init__(self, log_every_n_steps: int = 10, max_records: int = 2_000):
        super().__init__()
        self.log_every_n_steps = int(log_every_n_steps)
        self.max_records = int(max_records)
        self.records: list[dict[str, float | int]] = []

    def on_after_backward(self, trainer, pl_module) -> None:
        step = int(trainer.global_step)

        if step % self.log_every_n_steps != 0 or len(self.records) >= self.max_records:
            return

        grad_norms = []
        nonfinite_grads = 0

        for parameter in pl_module.parameters():
            if parameter.grad is None:
                continue

            grad = parameter.grad.detach()
            norm = float(grad.norm(2).cpu())

            if np.isfinite(norm):
                grad_norms.append(norm)
            else:
                nonfinite_grads += 1

        if not grad_norms:
            return

        total_norm = float(np.sqrt(np.sum(np.square(grad_norms))))

        self.records.append(
            {
                "step": step,
                "epoch": int(trainer.current_epoch),
                "grad_total_norm": total_norm,
                "grad_mean_norm": float(np.mean(grad_norms)),
                "grad_max_norm": float(np.max(grad_norms)),
                "grad_parameter_count": int(len(grad_norms)),
                "grad_nonfinite_count": int(nonfinite_grads),
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def summary(self) -> dict[str, float]:
        df = self.to_frame()

        if df.empty:
            return {
                "grad_records": 0.0,
                "grad_total_norm_mean": 0.0,
                "grad_total_norm_max": 0.0,
                "grad_total_norm_last": 0.0,
                "grad_max_norm_max": 0.0,
                "grad_nonfinite_count_sum": 0.0,
            }

        return {
            "grad_records": float(len(df)),
            "grad_total_norm_mean": float(df["grad_total_norm"].mean()),
            "grad_total_norm_max": float(df["grad_total_norm"].max()),
            "grad_total_norm_last": float(df["grad_total_norm"].iloc[-1]),
            "grad_max_norm_max": float(df["grad_max_norm"].max()),
            "grad_nonfinite_count_sum": float(df["grad_nonfinite_count"].sum()),
        }


def log_gradient_diagnostics(
    callback: GradientNormLogger,
    model_name: str,
    run_label: str,
    mlflow_module,
    wandb_module=None,
    artifact_dir: str | Path = "reports/gradient_logs",
) -> dict[str, float]:

    metrics = callback.summary()

    mlflow_module.log_metrics(metrics)

    if wandb_module is not None:
        wandb_module.log(metrics)

    df = callback.to_frame()

    if not df.empty:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        path = artifact_dir / f"{model_name.lower()}_{run_label}_gradients.csv"

        df.to_csv(path, index=False)

        mlflow_module.log_artifact(str(path))

        if wandb_module is not None:
            wandb_module.log(
                {
                    f"{model_name.lower()}_gradient_history":
                    wandb_module.Table(dataframe=df)
                }
            )

    return metrics


def strip_neuralforecast_callbacks(model) -> None:
    """
    Remove training-only callbacks before saving NeuralForecast models.

    GradientNormLogger is useful during training but cannot be serialized
    inside MLflow/PyTorch Lightning hparams because it is a Python object.
    """

    # Remove from trainer kwargs
    if hasattr(model, "trainer_kwargs"):
        callbacks = model.trainer_kwargs.get("callbacks")

        if callbacks is not None:
            model.trainer_kwargs["callbacks"] = [
                cb
                for cb in callbacks
                if not isinstance(cb, GradientNormLogger)
            ]

    # Remove from Lightning hyperparameters
    hparams = getattr(model, "hparams", None)

    if hparams is not None and "callbacks" in hparams:
        hparams["callbacks"] = []

    # Remove from initial hyperparameters
    hparams_initial = getattr(model, "_hparams_initial", None)

    if hparams_initial is not None and "callbacks" in hparams_initial:
        hparams_initial["callbacks"] = []