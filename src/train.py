from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from datamodule import MNISTDataModule
from lit_model import MNISTLightningModule

OUTPUT_DIR = "outputs"


def main():
    L.seed_everything(42)

    datamodule = MNISTDataModule(
        data_dir="data",
        batch_size=128,
        num_workers=4,
    )

    model = MNISTLightningModule(
        lr=1e-3,
        hidden_dim=32,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{OUTPUT_DIR}/checkpoints",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_acc:.4f}",
    )

    mlflow_logger = MLFlowLogger(
        experiment_name="mnist-classification",
        run_name="mnist-mlp",
        tracking_uri=f"sqlite:///{OUTPUT_DIR}/mlflow.db",
        artifact_location=f"{OUTPUT_DIR}/mlruns",
        log_model=True,
        tags={
            "model": "MLP",
            "dataset": "MNIST",
        },
    )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=mlflow_logger,
        log_every_n_steps=50,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
