import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from datamodule import MNISTDataModule
from lit_model import MNISTLightningModule

DEFAULT_CONFIG_PATH = "configs/exp001.yaml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training config YAML file.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(config["run"]["seed"])

    datamodule = MNISTDataModule(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        persistent_workers=config["data"]["persistent_workers"],
    )

    model = MNISTLightningModule(
        lr=config["model"]["lr"],
        hidden_dim=config["model"]["hidden_dim"],
    )

    checkpoint_config = config["checkpoint"]
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        monitor=checkpoint_config["monitor"],
        mode=checkpoint_config["mode"],
        save_top_k=checkpoint_config["save_top_k"],
        filename=checkpoint_config["filename"],
    )

    mlflow_config = config["mlflow"]
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_config["experiment_name"],
        run_name=mlflow_config["run_name"],
        tracking_uri=f"sqlite:///{output_dir / 'mlflow.db'}",
        artifact_location=str(output_dir / "mlruns"),
        log_model=mlflow_config["log_model"],
        tags=mlflow_config.get("tags"),
    )

    trainer_config = config["trainer"]
    trainer = L.Trainer(
        max_epochs=trainer_config["max_epochs"],
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        callbacks=[checkpoint_callback],
        logger=mlflow_logger,
        log_every_n_steps=trainer_config["log_every_n_steps"],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
