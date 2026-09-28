import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import joblib
import torch
import hydra
from omegaconf import DictConfig
from lightning_lstm import LightningLSTM


def compute_mse_for_file(eeg_path, model, scaler, window_size, step, num_channels, start_point):
    eeg = np.load(eeg_path).astype(np.float32)
    n_channels, duration= eeg.shape

    num_windows = (duration - start_point) // window_size
    if num_windows == 0:
        print(f"Skipped (too short): {eeg_path.name}")
        return None

    segment = eeg[:, start_point:start_point + num_windows * window_size]
    segment = scaler.transform(segment.T)

    x = segment[:-1, :]
    y = segment[1:, :]

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        y_pred = model(x_t).squeeze(0).cpu().numpy()

    y_true = y_t.squeeze(0).cpu().numpy()

    mse_per_window = np.zeros((num_windows, num_channels))
    for ch in range(num_channels):
        for i in range(num_windows):
            w_start = i * step
            w_end = w_start + window_size
            mse_per_window[i, ch] = np.mean((y_true[w_start:w_end, ch] -
                                             y_pred[w_start:w_end, ch]) ** 2)
    return mse_per_window


@hydra.main(version_base=None, config_path="config", config_name="predict_anomaly")
def main(cfg: DictConfig):

    model_root = Path(cfg.predict_anomaly.models_root) / cfg.predict_anomaly.model_name
    eeg_root = Path(cfg.predict_anomaly.eeg_root)

    window_size = cfg.predict_anomaly.window_size

    root = Path(cfg.predict_anomaly.models_root) / cfg.predict_anomaly.model_name / f"window_size_{window_size}"


    save_root = root 
    save_root.mkdir(exist_ok=True)
    
    scaler = joblib.load(cfg.predict_anomaly.scaler_path)

    kde = joblib.load(root / "kde_model.pkl")
    thresh = float(np.load(root / "kde_threshold.npy")[0])

    ckpt_path = model_root / "version_0/checkpoints" / cfg.predict_anomaly.model_ckpt
    model = LightningLSTM.load_from_checkpoint(str(ckpt_path), weights_only=False)
    model.eval()

    eeg_files = list(eeg_root.glob("*.npy"))
    print(f"Found {len(eeg_files)} EEG files.")

    percentages=[]

    for eeg_path in eeg_files:    #[31:] for val
        eeg_name = eeg_path.stem
        print(f"\nProcessing {eeg_name}")

        error_matrix = compute_mse_for_file(
            eeg_path=eeg_path,
            model=model,
            scaler=scaler,
            window_size=cfg.predict_anomaly.window_size,
            step=cfg.predict_anomaly.step,
            num_channels=cfg.predict_anomaly.num_channels,
            start_point=cfg.predict_anomaly.start_point,
        )

        if error_matrix is None:
            continue


        log_dens = kde.score_samples(error_matrix)
        dens = np.exp(log_dens)

        anomalies = np.where(dens < thresh)[0]
        percentage_anomalies = len(anomalies) * 100 / len(dens)

        percentages.append(percentage_anomalies)


        np.save(root / f"{eeg_name}_anomalies_indexes.npy", anomalies)
        np.save(root / f"{eeg_name}_densities.npy", dens)
        np.save( root / f"{eeg_name}_errors.npy", error_matrix)

     

        print(f"{len(anomalies)} anomalies out of {len(dens)} windows, percentage:{percentage_anomalies}%")

       
    np.save(root / "percentages_anomalies_TS.npy", percentages)




if __name__ == "__main__":
    main()