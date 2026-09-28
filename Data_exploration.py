import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

def display_metadata(eeg_raw):
    print(eeg_raw)
    print(eeg_raw.info)

def display_amplitude_range(eeg_raw):
    eeg_values = eeg_raw.get_data()
    min_microvolt = eeg_values.min() * 1e6
    max_microvolt = eeg_values.max() * 1e6
    print("min in microvolt:", min_microvolt, "max in microvolt:", max_microvolt)

def plot_power_spectral_density(eeg_raw, max_frequency):
    eeg_raw.compute_psd(fmax=max_frequency + 15).plot(picks="data", amplitude=False)
    plt.show()

def recording_duration(eeg_raw):
    return eeg_raw.duration - 0.01

def plot_timeseries(eeg_array, sampling_rate, channel_names,
                    channels_to_plot, pdf_filename,
                    start_time, end_time,
                    rows=8, columns=4):

    total_channels = min(channels_to_plot, eeg_array.shape[0])
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)
    selected_values = eeg_array[:total_channels, start_sample:end_sample]
    total_samples = selected_values.shape[1]
    time_axis = np.arange(total_samples) / sampling_rate + start_time

    figure, axes = plt.subplots(rows, columns, figsize=(11, 14),
                                sharex=True, sharey=False)
    axes = axes.flatten()

    for channel_index in range(total_channels):
        axis = axes[channel_index]
        axis.plot(time_axis, selected_values[channel_index],
                  color="black", linewidth=0.5)
        axis.text(0.02, 0.9, channel_names[channel_index],
                  transform=axis.transAxes, fontsize=6)
        if channel_index % columns == 0:
            axis.set_ylabel("Amplitude (µV)", fontsize=8)
        if channel_index >= (rows - 1) * columns:
            axis.set_xlabel("Time (s)", fontsize=8)

    for empty_axis_index in range(total_channels, len(axes)):
        figure.delaxes(axes[empty_axis_index])

    figure.tight_layout(pad=0.2)
    figure.savefig(pdf_filename, dpi=300, bbox_inches="tight")
    plt.close(figure)



def plot_spectrogram(eeg_raw, channels_to_plot, pdf_filename):
    channel_names = eeg_raw.ch_names
    eeg_values = eeg_raw.get_data()
    sampling_rate = eeg_raw.info["sfreq"]
    total_channels = min(channels_to_plot, eeg_values.shape[0])
    rows = int(np.ceil(total_channels / 2))
    columns = 2
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for channel_index in range(total_channels):
        axis = axes[channel_index]
        axis.specgram(eeg_values[channel_index], Fs=sampling_rate)
        axis.text(0.02, 0.9, channel_names[channel_index], transform=axis.transAxes, fontsize=10)
        axis.set_ylabel("Frequency (Hz)", fontsize=12)
        axis.set_xlabel("Time (s)", fontsize=12)

    for empty_axis_index in range(total_channels, len(axes)):
        figure.delaxes(axes[empty_axis_index])

    figure.tight_layout()
    figure.savefig(pdf_filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print("Saved PDF:", pdf_filename)


def plot_phase_spectrum(eeg_raw, channels_to_plot, pdf_filename):
    channel_names = eeg_raw.ch_names
    eeg_values = eeg_raw.get_data()
    sampling_rate = eeg_raw.info["sfreq"]
    total_channels = min(channels_to_plot, eeg_values.shape[0])
    rows = int(np.ceil(total_channels / 2))
    columns = 2
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    for channel_index in range(total_channels):
        axis = axes[channel_index]
        axis.phase_spectrum(eeg_values[channel_index], Fs=sampling_rate)
        axis.text(0.02, 0.9, channel_names[channel_index], transform=axis.transAxes, fontsize=10)
        axis.set_xlabel("Frequency (Hz)", fontsize=12)
        axis.set_ylabel("Phase (rad)", fontsize=12)
    for empty_axis_index in range(total_channels, len(axes)):
        figure.delaxes(axes[empty_axis_index])
    figure.tight_layout()
    figure.savefig(pdf_filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print("Saved PDF:", pdf_filename)




def plot_fft(eeg_raw, channels_to_plot, pdf_filename):
    channel_names = eeg_raw.ch_names
    sampling_rate = eeg_raw.info["sfreq"]

    seconds = eeg_raw.n_times / sampling_rate

    spectrum = eeg_raw.compute_psd(
        method="welch",
        n_fft=int(500 * seconds),
        n_overlap=0,
        n_per_seg=int(500 * seconds),
        fmin=1,
        fmax=100,
        window="boxcar",
        verbose=False
    )

    psds, freqs = spectrum.get_data(return_freqs=True)

    total_channels = min(channels_to_plot, psds.shape[0])
    rows = int(np.ceil(total_channels / 2))
    columns = 2

    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), sharex=True)
    axes = axes.flatten()

    for ch in range(total_channels):
        ax = axes[ch]
        ax.plot(freqs, psds[ch, :], color="black", linewidth=0.8)
        ax.text(0.02, 0.9, channel_names[ch], transform=ax.transAxes, fontsize=10)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Power (V²/Hz)", fontsize=12)
        ax.grid(True)

    for empty_axis_index in range(total_channels, len(axes)):
        fig.delaxes(axes[empty_axis_index])

    fig.tight_layout()
    fig.savefig(pdf_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved PDF:", pdf_filename)





eeg_raw = mne.io.read_raw_brainvision(
    "eeg_data/HC/GraphiTics_HC_PDG_EEG_140626_CL06LE01.vhdr",
    preload=True
)

'''
eeg_raw = mne.io.read_raw_brainvision(
    "eeg_data/TS/GraphiTics_TS_PDG_060326_SO12LO71.vhdr",
    preload=True
)
'''

eeg_array = eeg_raw.get_data() * 1e6
sampling_rate = eeg_raw.info["sfreq"]
channel_names = eeg_raw.ch_names

plot_timeseries(
    eeg_array=eeg_array,
    sampling_rate=sampling_rate,
    channel_names=channel_names,
    channels_to_plot=32,
    start_time= 100,
    end_time= 103,
    pdf_filename="timeseries_plot_ts.pdf"
)


plot_spectrogram(
    eeg_raw=eeg_raw,
    channels_to_plot=4,
    pdf_filename="spectrogram_plot_ts.pdf"
)

plot_fft(
    eeg_raw=eeg_raw,
    channels_to_plot=4,
    pdf_filename="fft_plot_ts.pdf"
)

plot_phase_spectrum(
    eeg_raw=eeg_raw,
    channels_to_plot=4,
    pdf_filename="phase_spectrum_plot_ts.pdf"
)



#To plot for all files in HC or TS folder

'''
def process_folder(folder_path, label, channels_to_plot,
                   start_time, end_time):

    folder_path = Path(folder_path)
    vhdr_files = sorted(folder_path.glob("*.vhdr"))

    for vhdr_file in vhdr_files:
        eeg_raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
        eeg_array = eeg_raw.get_data() * 1e6
        sampling_rate = eeg_raw.info["sfreq"]
        channel_names = eeg_raw.ch_names

        base_name = vhdr_file.stem

        out_ts = Path(f"data_exploration/time_series/{label}/{base_name}.pdf")
        out_spec = Path(f"data_exploration/spectrogram/{label}/{base_name}.pdf")
        out_phase = Path(f"data_exploration/phase_spectrum/{label}/{base_name}.pdf")
        out_fft = Path(f"data_exploration/fft/{label}/{base_name}.pdf")

        out_ts.parent.mkdir(parents=True, exist_ok=True)
        out_spec.parent.mkdir(parents=True, exist_ok=True)
        out_phase.parent.mkdir(parents=True, exist_ok=True)
        out_fft.parent.mkdir(parents=True, exist_ok=True)

        plot_timeseries(eeg_array, sampling_rate, channel_names,
                        channels_to_plot, out_ts,
                        start_time, end_time)

        plot_spectrogram(eeg_raw, channels_to_plot, out_spec)
        plot_phase_spectrum(eeg_raw, channels_to_plot, out_phase)
        plot_fft(eeg_raw, channels_to_plot, out_fft)



process_folder("eeg_data/TS", "TS", 32, 0, 10)
process_folder("eeg_data/HC", "HC", 32, 0, 10)
'''