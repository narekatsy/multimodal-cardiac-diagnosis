import os
import glob
import numpy as np
import wfdb
from tqdm import tqdm

RAW_DATA_PATH = "data/ECG/"
PROCESSED_DATA_PATH = "processed_data/ECG/"

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def high_pass_filter(signal, sampling_rate, cutoff=0.5):
    """Apply a high-pass filter to remove baseline wander."""
    from scipy.signal import butter, filtfilt
    b, a = butter(1, cutoff / (sampling_rate / 2), btype='highpass')
    return filtfilt(b, a, signal, axis=0)

def notch_filter(signal, freq=50, sampling_rate=1000):
    """Apply a notch filter to remove powerline interference (50/60Hz)."""
    from scipy.signal import iirnotch, filtfilt
    b, a = iirnotch(freq / (sampling_rate / 2), Q=30)
    return filtfilt(b, a, signal, axis=0)

def low_pass_filter(signal, sampling_rate, cutoff=100):
    """Apply a low-pass filter to remove high-frequency noise."""
    from scipy.signal import butter, filtfilt
    b, a = butter(1, cutoff / (sampling_rate / 2), btype='lowpass')
    return filtfilt(b, a, signal, axis=0)

def normalize_signal(signal):
    """Normalize the ECG signal using z-score normalization."""
    return (signal - np.mean(signal)) / np.std(signal)

def process_ecg_files():
    """Process all ECG files found in patient subdirectories."""
    files = glob.glob(os.path.join(RAW_DATA_PATH, "**/*.dat"), recursive=True)

    if not files:
        print("❌ No .dat files found! Check the folder structure.")
        return

    for file in tqdm(files, desc="Processing ECG Files"):
        try:
            record_path = file[:-4]  # Remove .dat extension
            record = wfdb.rdsamp(record_path)

            ecg_signal, meta = record
            sampling_rate = meta['fs']

            filtered_ecg = high_pass_filter(ecg_signal, sampling_rate)
            filtered_ecg = notch_filter(filtered_ecg, freq=50, sampling_rate=sampling_rate)
            filtered_ecg = low_pass_filter(filtered_ecg, sampling_rate)
            normalized_ecg = normalize_signal(filtered_ecg)

            filename = os.path.basename(file).replace(".dat", ".npy")
            save_path = os.path.join(PROCESSED_DATA_PATH, filename)
            np.save(save_path, normalized_ecg)

            print(f"✅ Processed and saved: {save_path}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

if __name__ == "__main__":
    process_ecg_files()
