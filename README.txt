 It is assumed that the raw eeg data are saved under the eeg_data folder within subfolders: HC for Healthy EEG, TS for Tourette patients EEG.

Pipeline:
0) Data_preprocessing :Transfroms the raw eeg data into preprocesssed npy files for Theta, aplha and beta bands

1) Scaler :Creates a scaler for each eeg band group(only from healthy train group)

2) lstm_sweep: Model training

3) fehler_matrix_kde: Create a file containing all windows error for healthy train files which will be used for KDE (set error window size, eeg_root and model name in predict_train config file)

4) bandwidth_optimized: Compute the optimal bandwidth with the same configuration as fehler_matrix_kde (set error window size and model name in kde config file)

5) validation_anomaly_detection: Computes the anomalies in the validation set for KDE (set error window size, eeg_root and model name in predict_validation config file)
  
  i) anomaly_percentages_hist_validation (Plots the histogram of anomaly percentages in the validation set) 

6) anomaly_detection: Computes the anomalies in the test set(TS) (set error window size, eeg_root and model name in predict_anomaly config file)

  i) anomaly_percentages_hist (Plots the histogram of anomaly percentages in the test set) 

Other:
1) Data_exploration contains functions to plot the timeseries, fast fourrier transform, spectogram and phase spectrogram from the raw files

2)the config folder contains the Hydra files

3) One EEG preprocessed file each for a theta band model train set, validation set and tourette, a scaler and a kde_model for the samwe theta model with the corresponfing additional kde files are already available.
  
