import torch
from torch import nn
import lightning as pl
from torch.utils.data import DataLoader, RandomSampler
from dataset import EEGDataSet


class LightningLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 batch_size, learning_rate=1e-3,
                 bidirectional=False, scaler=None, data_path=None,
                 entry_length=10,
                 train_indices=None,
                 val_indices=None):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.scaler = scaler

        self.train_indices = train_indices
        self.val_indices = val_indices

        self.lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.model = nn.LSTM(input_dim, hidden_dim, num_layers,
                             batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(self.lstm_output_dim, input_dim)
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        lstm_out, _ = self.model(input_seq)
        output = self.fc(lstm_out)

        loss = self.criterion(
            output[:, self.hparams.entry_length:],
            target_seq[:, self.hparams.entry_length:]
        )
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        lstm_out, _ = self.model(input_seq)
        output = self.fc(lstm_out)

        loss = self.criterion(
            output[:, self.hparams.entry_length:],
            target_seq[:, self.hparams.entry_length:]
        )
        self.log("val_loss", loss,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=3
        )
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
    
    def forward(self, x):
        y, _ = self.model(x)
        y = self.fc(y)
        return y

    def train_dataloader(self):
        dataset = EEGDataSet(self.hparams.data_path, scaler=self.scaler,
                             indices=self.train_indices)

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True   
            
        )

    def val_dataloader(self):
        dataset = EEGDataSet(self.hparams.data_path, scaler=self.scaler,
                             indices=self.val_indices)

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False
        )
