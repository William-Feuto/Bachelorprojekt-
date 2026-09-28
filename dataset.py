from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
import joblib




class EEGDataSet(Dataset):
    def __init__(self, root, scaler=None, indices=None):



        self.files = [str(f) for f in Path(root).glob("*.npy")]
        self.scaler = scaler

        self.window_size = 200
        self.overlap = 100
        self.step = self.window_size - self.overlap

        
        full_index_map = []
        for file_idx, file in enumerate(self.files):
            eeg = np.load(file).astype(np.float32)
            n_channels, duration = eeg.shape

            num_windows = max(0, (duration - self.window_size) // self.step + 1)

            for window in range(num_windows):
                full_index_map.append((file_idx, window))

        # Use only selected indices
        if indices is None:
            self.index_map = full_index_map
        else:
            self.index_map = [full_index_map[i] for i in indices]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, window_idx = self.index_map[idx]

        eeg = np.load(self.files[file_idx]).astype(np.float32)

        start = window_idx * self.step
        end = start + self.window_size
        window = eeg[:, start:end]

        if self.scaler is not None:
            window = self.scaler.transform(window.T).T

        window = window.T  # (200, C)

        input_seq = window[:-1, :]
        target_seq = window[1:, :]

        return (
            torch.tensor(input_seq, dtype=torch.float32),
            torch.tensor(target_seq, dtype=torch.float32)
        )
