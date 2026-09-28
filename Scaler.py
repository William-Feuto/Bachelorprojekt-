import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIRECTORY = "eeg_data/train"

BANDS = {
    "beta":  f"{BASE_DIRECTORY}/HC_Preprocessed_beta",
    "alpha": f"{BASE_DIRECTORY}/HC_Preprocessed_alpha",
    "theta": f"{BASE_DIRECTORY}/HC_Preprocessed_theta"
}

OUTPUT_DIRECTORY = "LSTM"
Path(OUTPUT_DIRECTORY).mkdir(exist_ok=True)


def build_global_scaler(input_folder, output_filename):
    input_files = sorted(Path(input_folder).glob("*.npy"))
    scaler = StandardScaler()
    all_samples = []

    for file_path in input_files:
        array = np.load(file_path).astype(np.float32)
        all_samples.append(array.T)

    combined_samples = np.concatenate(all_samples, axis=0)
    scaler.fit(combined_samples)

    output_path = f"{OUTPUT_DIRECTORY}/{output_filename}"

    joblib.dump(scaler, output_path)


    try:
        print("Mean per channel:", scaler.mean_)
        print("Std per channel:", scaler.std_)
    except:
        print("Error")


build_global_scaler(BANDS["beta"],  "global_scaler_beta_std.pkl")
build_global_scaler(BANDS["alpha"], "global_scaler_alpha_std.pkl")
build_global_scaler(BANDS["theta"], "global_scaler_theta_std.pkl")
