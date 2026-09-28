import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import hydra
from omegaconf import DictConfig
from lightning_lstm import LightningLSTM


def plot_single_window_for_model(
    model,
    eeg_file,
    scaler,
    window_size,
    step,
    channel,
    start_point,
    window_index,
    model_name,
):
    eeg_file = Path(eeg_file)
    #output_root = Path(output_root)
    #output_root.mkdir(parents=True, exist_ok=True)

    eeg_name = eeg_file.stem
    print(f"\n[SINGLE] Processing EEG: {eeg_name}")

    # Load EEG (C, T)
    eeg = np.load(eeg_file).astype(np.float32)
    C, T = eeg.shape

    # Extract enough data for multiple windows
    num_windows = 50  # same as your original code
    segment = eeg[:, start_point:start_point + num_windows * window_size]

    
    segment_scaled = scaler.transform(segment.T)

    
    x = segment_scaled[:-1, :]
    y_true = segment_scaled[1:, :]

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    
    with torch.no_grad():
        y_pred = model(x_t).squeeze(0).cpu().numpy()

    
    w_start = window_index * step
    w_end = w_start + window_size

    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true[w_start:w_end, channel], label="True", color="blue")
    ax.plot(y_pred[w_start:w_end, channel], label="Predicted", color="red")

    ax.set_title(
        f"Channel {channel} "
    )
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude in 10 microvolts")

    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    #save_path = output_root / "real_vs_predicted.pdf"
    plt.show()
    

    plt.close(fig)


@hydra.main(version_base=None, config_path="config", config_name="predict_train")
def main(cfg: DictConfig):
    models_root = Path(cfg.predict.models_root)
    eeg_file = cfg.predict.single_eeg_file
    #output_root = cfg.predict.output_root_windows

    model_name = cfg.predict.model_name
    model_ckpt = cfg.predict.model_ckpt
    scaler_path = cfg.predict.scaler_path

    window_size = cfg.predict.window_size
    step = cfg.predict.step
    start_point = cfg.predict.start_point
    channel = cfg.predict.single_channel
    window_index = cfg.predict.single_window_index

    scaler = joblib.load(scaler_path)

    ckpt_path = models_root / model_name / "version_0/checkpoints" / model_ckpt
    print(f"Loading model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")

    model = LightningLSTM.load_from_checkpoint(str(ckpt_path), weights_only=False)
    model.eval()
    model.model_name = model_name

    plot_single_window_for_model(
        model=model,
        eeg_file=eeg_file,
        scaler=scaler,
        window_size=window_size,
        step=step,
        channel=channel,
        start_point=start_point,
        window_index=window_index,
        model_name=model_name
    )


if __name__ == "__main__":
    main()

