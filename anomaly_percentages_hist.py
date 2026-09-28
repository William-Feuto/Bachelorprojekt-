import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig


def extract_band_and_size(model_name: str):
    name = model_name.lower()

    if "alpha" in name:
        band = "alpha"
    elif "beta" in name:
        band = "beta"
    elif "theta" in name:
        band = "theta"
    else:
        band = "unknown"

    size = "64" if "64" in name else "128" if "128" in name else "32" if "32" in name else "6" if "16" in name else "unknown"
    return band, size


def plot_percentages(model_dir: Path, band: str, size: str):
    anomalies_file = f"percentages_anomalies_TS_{band}_{size}.pdf"

    anomalies_array_path = model_dir / "percentages_anomalies_TS.npy"
    

    save_dir = Path("LSTM")
    save_dir.mkdir(parents=True, exist_ok=True)

    if anomalies_array_path.exists():
        anomalies = np.load(anomalies_array_path)

        plt.figure(figsize=(14, 6))
        plt.bar(range(len(anomalies)), anomalies, color="red", alpha=0.7)
        plt.title(f"Anomaly Percentage — {model_dir.name}")
        plt.xlabel("Patient Index")
        plt.ylabel("Anomaly Percentage")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / anomalies_file, dpi=300)
        plt.close()
    else:
        print(f"Missing anomalies file in {model_dir}")







@hydra.main(version_base=None, config_path="config", config_name="predict_anomaly")
def main(cfg: DictConfig):

    models_root = Path(cfg.predict_anomaly.models_root)
    model_name = cfg.predict_anomaly.model_name
    window_size= cfg.predict_anomaly.window_size

    
    model_dir = models_root / model_name / f"window_size_{window_size}"

    print(f"\nProcessing model: {model_dir}")

    band, size = extract_band_and_size(model_name)

    plot_percentages(model_dir, band, size)
    


if __name__ == "__main__":
    main()
