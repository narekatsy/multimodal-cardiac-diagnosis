import os
import glob
import numpy as np
import wfdb
from tqdm import tqdm

input_dir = "split_data/ECG/training/"
output_dir = "processed_data/ecg/training/signals/"
os.makedirs(output_dir, exist_ok=True)

def high_pass_filter(signal, sampling_rate, cutoff=0.5):
    """ Apply a high-pass filter to remove baseline wander. """
    from scipy.signal import butter, filtfilt
    b, a = butter(1, cutoff / (sampling_rate / 2), btype='highpass')
    return filtfilt(b, a, signal, axis=0)

def notch_filter(signal, freq=50, sampling_rate=1000):
    """ Apply a notch filter to remove powerline interference (50/60Hz). """
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(freq / (sampling_rate / 2), Q=30)
    return filtfilt(b, a, signal, axis=0)

def low_pass_filter(signal, sampling_rate, cutoff=100):
    """ Apply a low-pass filter to remove high-frequency noise. """
    from scipy.signal import butter, filtfilt
    b, a = butter(1, cutoff / (sampling_rate / 2), btype='lowpass')
    return filtfilt(b, a, signal, axis=0)

def normalize_signal(signal):
    """ Normalize the ECG signal using z-score normalization. """
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

def process_ecg_files():
    """ Process all ECG files found in patient subdirectories. """
    dat_files = glob.glob(os.path.join(input_dir, "**/*.dat"), recursive=True)

    if not dat_files:
        print("❌ No .dat files found!")
        return

    for dat_path in tqdm(dat_files, desc="Processing ECG Files"):
        try:
            # Check if corresponding .hea file exists
            hea_path = dat_path.replace(".dat", ".hea")
            if not os.path.exists(hea_path):
                print(f"⚠️ Skipping {dat_path}: Missing corresponding .hea file.")
                continue

            # Use wfdb to read the .dat and .hea files directly (ignoring .xyz files)
            record_path = dat_path[:-4]  # Remove .dat extension
            record = wfdb.rdsamp(record_path, channels=[0])  # Only read the first channel (if available)
            ecg_signal = record[0]  # Get the signal (e.g., numpy array)
            fs = record[1]['fs']  # Sampling rate from the header file

            # Filter and normalize ECG signal
            filtered = high_pass_filter(ecg_signal, fs)
            filtered = notch_filter(filtered, freq=50, sampling_rate=fs)
            filtered = low_pass_filter(filtered, fs)
            normalized = normalize_signal(filtered)

            # Save the processed signal as .npy
            patient_id = os.path.basename(os.path.dirname(dat_path))
            record_id = os.path.splitext(os.path.basename(dat_path))[0]
            filename = f"{patient_id}_{record_id}.npy"
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, normalized)

            print(f"✅ Saved: {save_path}")

        except Exception as e:
            print(f"❌ Error processing {dat_path}: {e}")

if __name__ == "__main__":
    process_ecg_files()
