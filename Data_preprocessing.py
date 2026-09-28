import os
import mne
import numpy as np
from pathlib import Path

BASE_DIRECTORY = Path("eeg_data")

PATIENT_GROUPS = ["HC", "TS"]

OUTPUT_DIRECTORIES = {
    "HC": {
        "train": {
            "theta": BASE_DIRECTORY / "train/HC_preprocessed_theta",
            "alpha": BASE_DIRECTORY / "train/HC_preprocessed_alpha",
            "beta":  BASE_DIRECTORY / "train/HC_preprocessed_beta",
        },
        "validation": {
            "theta": BASE_DIRECTORY / "validation/HC_preprocessed_theta",
            "alpha": BASE_DIRECTORY / "validation/HC_preprocessed_alpha",
            "beta":  BASE_DIRECTORY / "validation/HC_preprocessed_beta",
        }
    },
    "TS": {
        "theta": BASE_DIRECTORY / "TS_preprocessed_theta",
        "alpha": BASE_DIRECTORY / "TS_preprocessed_alpha",
        "beta":  BASE_DIRECTORY / "TS_preprocessed_beta",
    }
}


for group in OUTPUT_DIRECTORIES:
    for key in OUTPUT_DIRECTORIES[group]:
        if isinstance(OUTPUT_DIRECTORIES[group][key], dict):
            for band in OUTPUT_DIRECTORIES[group][key]:
                OUTPUT_DIRECTORIES[group][key][band].mkdir(parents=True, exist_ok=True)
        else:
            OUTPUT_DIRECTORIES[group][key].mkdir(parents=True, exist_ok=True)


def preprocess_eeg(raw_data, n_components=0.999999):
    raw_data.notch_filter(np.arange(50, 245, 100), filter_length='auto', phase='zero')
    raw_data = raw_data.filter(l_freq=1, h_freq=45)

    try:
        bad_channels = mne.preprocessing.find_bad_channels_lof(raw_data)
        raw_data.info['bads'] = bad_channels
        raw_data.interpolate_bads()
        raw_data.set_eeg_reference('average')
    except:
        raw_data.set_eeg_reference('average')

    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter='auto')
    ica.fit(raw_data)

    muscle_components = ica.find_bads_muscle(raw_data)[0]

    try:
        eye_components = ica.find_bads_eog(raw_data)[0]
        ica.exclude = muscle_components + eye_components
    except:
        ica.exclude = muscle_components

    raw_data.load_data()
    ica.apply(raw_data)
    return raw_data


def save_frequency_band(clean_data, low_freq, high_freq, output_folder, eeg_name):
    filtered_data = clean_data.copy().filter(l_freq=low_freq, h_freq=high_freq)
    filtered_data.resample(100)
    numpy_array = filtered_data.get_data(return_times=False)
    output_path = output_folder / f"{eeg_name}_{low_freq}_{high_freq}.npy"
    np.save(output_path, numpy_array)


def process_patient_group(group_name):
    group_folder = BASE_DIRECTORY / group_name
    vhdr_files = sorted([f for f in os.listdir(group_folder) if f.endswith(".vhdr")])

    successful_files = []
    failed_files = []



    if group_name == "HC":
        np.random.seed(42) 
        shuffled = np.random.permutation(vhdr_files)

        split_index = int(0.7 * len(shuffled))
        train_files = shuffled[:split_index]
        validation_files = shuffled[split_index:]

        file_splits = {
            "train": train_files,
            "validation": validation_files
        }


    else:
        file_splits = {"all": vhdr_files}


    for split_name, files in file_splits.items():
        for vhdr_file in files:
            eeg_name = vhdr_file[:-5]
            full_path = group_folder / vhdr_file

            try:
                raw_data = mne.io.read_raw_brainvision(full_path, preload=True)
            except:
                failed_files.append(vhdr_file)
                continue

            successful_files.append(vhdr_file)
            cleaned_data = preprocess_eeg(raw_data)

            # HC uses split directories
            if group_name == "HC":
                save_frequency_band(cleaned_data, 4, 8,
                                    OUTPUT_DIRECTORIES["HC"][split_name]["theta"], eeg_name)
                save_frequency_band(cleaned_data, 8, 13,
                                    OUTPUT_DIRECTORIES["HC"][split_name]["alpha"], eeg_name)
                save_frequency_band(cleaned_data, 13, 30,
                                    OUTPUT_DIRECTORIES["HC"][split_name]["beta"], eeg_name)

            
            else:
                save_frequency_band(cleaned_data, 4, 8,
                                    OUTPUT_DIRECTORIES[group_name]["theta"], eeg_name)
                save_frequency_band(cleaned_data, 8, 13,
                                    OUTPUT_DIRECTORIES[group_name]["alpha"], eeg_name)
                save_frequency_band(cleaned_data, 13, 30,
                                    OUTPUT_DIRECTORIES[group_name]["beta"], eeg_name)

 
    with open(f"{group_name}_successful.txt", "w") as file:
        for name in successful_files:
            file.write(name + "\n")

    with open(f"{group_name}_failed.txt", "w") as file:
        for name in failed_files:
            file.write(name + "\n")



for group_name in PATIENT_GROUPS:
    process_patient_group(group_name)

