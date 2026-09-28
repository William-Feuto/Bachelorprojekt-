import hydra
from omegaconf import DictConfig
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from lightning_lstm import LightningLSTM
from dataset import EEGDataSet


import numpy as np
import os
import joblib


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(configuration: DictConfig):

    print(
        f"Training model: hidden={configuration.model.hidden_size}, "
        f"layers={configuration.model.n_layers}, lr={configuration.model.lr}, "
        f"path={configuration.data.path}"
    )

    data_path = configuration.data.path.lower()

    if "beta" in data_path:
        scaler_name = "global_scaler_beta_std.pkl"
    elif "alpha" in data_path:
        scaler_name = "global_scaler_alpha_std.pkl"
    elif "theta" in data_path:
        scaler_name = "global_scaler_theta_std.pkl"

    scaler_path = f"LSTM/{scaler_name}"
    scaler = joblib.load(scaler_path)

    dataset = EEGDataSet(configuration.data.path, scaler=scaler)
    dataset_length = len(dataset)

    indices = np.arange(dataset_length)
    np.random.shuffle(indices)

    train_end = int(0.7 * dataset_length)
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:]

    model = LightningLSTM(
        input_dim=32,
        hidden_dim=configuration.model.hidden_size,
        num_layers=configuration.model.n_layers,
        learning_rate=configuration.model.lr,
        bidirectional=configuration.model.bidirectional,
        batch_size=256,
        data_path=configuration.data.path,
        scaler=scaler,
        entry_length=configuration.model.entry_length,
        train_indices=train_indices,
        val_indices=validation_indices
    )

    log_directory = os.getcwd()

    logger = CSVLogger(
        save_dir=log_directory + "/LSTM",
        name=f"{os.path.basename(configuration.data.path)}_"
             f"{configuration.model.hidden_size}_"
             f"{configuration.model.n_layers}_"
             f"{configuration.model.lr}"
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="lstm_best"
    )

    trainer = pl.Trainer(
        max_epochs=configuration.model.sweep_epochs,
        accelerator="cpu",
        devices=1,
        strategy="auto",
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            checkpoint
        ]
    )

    trainer.fit(model)
    return trainer.callback_metrics.get("val_loss")


if __name__ == "__main__":
    main()
