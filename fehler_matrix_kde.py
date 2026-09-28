import numpy as np
import torch
from pathlib import Path
import joblib
import hydra
from omegaconf import DictConfig
from lightning_lstm import LightningLSTM


def compute_error(
    model,
    eeg_folder,
    scaler,
    window_size,
    step_size,
    channel_count,
    start_index,
    output_folder,
    model_name,
):
    eeg_folder = Path(eeg_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    eeg_files = sorted(eeg_folder.glob("*.npy"))
    print(f"Found {len(eeg_files)} EEG files.")

    all_mse_matrices = []

    for eeg_path in eeg_files:
        eeg_id = eeg_path.stem
        print(f"\nProcessing: {eeg_id}")

        eeg_data = np.load(eeg_path).astype(np.float32)
        channels, total_length = eeg_data.shape

        window_count = (total_length - start_index) // window_size
        if window_count == 0:
            print(f"Skipped (too short): {eeg_id}")
            continue

        segment = eeg_data[:, start_index:start_index + window_count * window_size]
        segment = scaler.transform(segment.T)

        x_values = segment[:-1, :]
        y_values = segment[1:, :]

        x_tensor = torch.tensor(x_values, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = model(x_tensor).squeeze(0)

        prediction = prediction.cpu().numpy()
        target = y_tensor.squeeze(0).cpu().numpy()

        skip_windows = 200 // window_size
        mse_matrix = np.zeros((window_count - skip_windows, channel_count))

        for channel in range(channel_count):
            for window_index in range(skip_windows, window_count):
                start = window_index * step_size
                end = start + window_size

                true_window = target[start:end, channel]
                pred_window = prediction[start:end, channel]

                mse_value = np.mean((true_window - pred_window) ** 2)
                mse_matrix[window_index - skip_windows, channel] = mse_value

        all_mse_matrices.append(mse_matrix)

    all_mse = np.concatenate(all_mse_matrices, axis=0)

    save_path = (
        Path("LSTM")
        / model_name
        / f"window_size_{window_size}"
        / "all_mse.npy"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, all_mse)

    print(f"MSE matrix saved to: {save_path}")
    return all_mse


@hydra.main(version_base=None, config_path="config", config_name="predict_train")
def main(configuration: DictConfig):
    models_folder = Path(configuration.predict.models_root)
    eeg_folder = configuration.predict.eeg_root
    output_folder = configuration.predict.output_root_mse

    model_name = configuration.predict.model_name
    checkpoint_name = configuration.predict.model_ckpt
    scaler_path = configuration.predict.scaler_path

    window_size = configuration.predict.window_size
    step_size = configuration.predict.step
    channel_count = configuration.predict.num_channels
    start_index = configuration.predict.start_point

    scaler = joblib.load(scaler_path)

    checkpoint_path = (
        models_folder
        / model_name
        / "version_0"
        / "checkpoints"
        / checkpoint_name
    )

    print(f"Loading model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")

    model = LightningLSTM.load_from_checkpoint(str(checkpoint_path), weights_only=False)
    model.eval()
    model.model_name = model_name

    compute_error(
        model=model,
        eeg_folder=eeg_folder,
        scaler=scaler,
        window_size=window_size,
        step_size=step_size,
        channel_count=channel_count,
        start_index=start_index,
        output_folder=output_folder,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()

