import os
import pandas as pd

from torch.utils.data import DataLoader, Dataset

import torch

import numpy as np
import torch
from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt

class VibrationPipeline:
    def __init__(self, harmonics=10, points_per_harmonic=30, smoothing_steps=32, smoothing_param=0.1):
        """
        Pipeline to process vibration data with low-pass smoothing, FFT, detailed harmonic scaling, and normalization.

        Args:
        - harmonics (int): Number of harmonics to scale the FFT result to (e.g., 10 harmonics of sync_freq).
        - points_per_harmonic (int): Number of interpolation points per harmonic.
        - smoothing_steps (int): Number of times to apply the low-pass filter.
        - smoothing_param (float): Low-pass filter cutoff frequency as a fraction of the Nyquist frequency.
        """
        self.harmonics = harmonics
        self.points_per_harmonic = points_per_harmonic
        self.smoothing_steps = smoothing_steps
        self.smoothing_param = smoothing_param

    def smooth_data(self, data, sampling_rate):
        """
        Apply low-pass filtering to the data multiple times while maintaining the same length.

        Args:
        - data (torch.Tensor): Input data of shape (channels, time_steps).
        - sampling_rate (float): Sampling rate of the data.

        Returns:
        - torch.Tensor: Smoothed data of the same shape.
        """
        data_np = data.numpy()
        nyquist = 0.5 * sampling_rate  # Nyquist frequency
        cutoff = self.smoothing_param * nyquist  # Cutoff frequency for low-pass filter

        # Design low-pass filter
        b, a = butter(4, cutoff / nyquist, btype='low')

        # Apply filter multiple times
        for _ in range(self.smoothing_steps):
            data_np = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=1, arr=data_np)

        return torch.tensor(data_np, dtype=torch.float32)

    def scale_to_detailed_harmonics(self, magnitude, freqs, sync_freq):
        """
        Scale FFT magnitude to detailed harmonics of sync_freq with interpolation.

        Args:
        - magnitude (np.ndarray): FFT magnitude of shape (channels, freq_bins).
        - freqs (np.ndarray): Corresponding frequency values.
        - sync_freq (float): Rotational synchronous frequency.

        Returns:
        - np.ndarray: Scaled FFT magnitude with detailed harmonic bins.
        - np.ndarray: Frequencies corresponding to the detailed harmonic bins.
        """
        # Define harmonic range starting at 0.5 * sync_freq
        detailed_bins = []
        for i in range(0, self.harmonics):
            harmonic_start = (i + 0.5) * sync_freq
            harmonic_end = (i + 1) * sync_freq
            detailed_bins.append(np.linspace(harmonic_start, harmonic_end, self.points_per_harmonic))

        # Flatten harmonic bins
        detailed_bins = np.concatenate(detailed_bins)

        # Interpolate magnitude to match detailed harmonic bins
        resampled_magnitude = np.array([
            np.interp(detailed_bins, freqs, magnitude[channel])
            for channel in range(magnitude.shape[0])
        ])
        return resampled_magnitude, detailed_bins

    def __call__(self, data, sampling_rate, sync_freq):
        """
        Process a single tensor of vibration data.

        Args:
        - data (torch.Tensor): Input data of shape (channels, time_steps).
        - sampling_rate (float): Sampling rate of the data in Hz.
        - sync_freq (float): Rotational synchronous frequency in Hz.

        Returns:
        - torch.Tensor: Processed data of shape (channels, harmonics * points_per_harmonic).
        - np.ndarray: Frequencies corresponding to the processed data.
        """
        # Step 1: Low-pass smoothing
        smoothed_data = self.smooth_data(data, sampling_rate)

        # Step 2: FFT
        fft_result = np.fft.rfft(smoothed_data.numpy(), axis=1)
        magnitude = np.abs(fft_result)

        # Step 3: Frequency bins
        freqs = np.fft.rfftfreq(data.shape[1], 1 / sampling_rate)

        # Step 4: Scale to detailed harmonics
        detailed_magnitude, detailed_bins = self.scale_to_detailed_harmonics(magnitude, freqs, sync_freq)

        # Step 5: Normalize
        norm_magnitude = detailed_magnitude / np.sum(detailed_magnitude, axis=1, keepdims=True)

        return torch.tensor(norm_magnitude, dtype=torch.float32), detailed_bins
    
class VibrationDataset(Dataset):
    def __init__(self, data_root, target_class, target_ch, transform=None):
        """
        Dataset for vibration data.
        
        Args:
        - data_root (str): Root directory containing vibration data files.
        - target_ch (list): List of target channel names to extract from CSV files.
        - transform (callable): Optional transform to apply on the data.
        """
        self.data_path_list = [
            os.path.join(data_root, dataset_name, class_name, file_name)
            for dataset_name in os.listdir(data_root)
            for class_name in os.listdir(os.path.join(data_root,dataset_name))
            if class_name in target_class
            for file_name in os.listdir(os.path.join(data_root, dataset_name, class_name))
        ]
        self.target_ch = target_ch
        self.transform = transform

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index):
        file_path = self.data_path_list[index]
        
        # Load metadata
        file_name = os.path.basename(file_path)
        class_name = file_name.split('_')[0]
        sampling_rate = file_name.split('_')[1]
        sampling_rate = float(sampling_rate.replace('kHz',''))*1000
        rotating_speed = float(file_name.split('_')[2])
        sync_freq = rotating_speed/60

        # Load vibration data
        file_pd = pd.read_csv(file_path)  # Skip metadata row
        file_pd = file_pd[self.target_ch]
        data = torch.tensor(file_pd.to_numpy().transpose(), dtype=torch.float32)  # (channels, time_steps)

        # Apply transform if available
        if self.transform:
            data, freq = self.transform(data, sampling_rate, sync_freq)

        meta_data = {
            'freq' : freq,
            'sync_freq' : sync_freq,
            'class_name' : class_name
        }
        
        return data, meta_data
    
if __name__ == '__main__':
    # Example parameters
    target_channels = ['motor_x', 'motor_y']  # Example channel names
    dxai_root = os.path.join(os.getcwd(), 'new_dataset')  # Path to vibration data

    pipeline = VibrationPipeline(
        harmonics=8,  # Number of harmonics to extract
        points_per_harmonic=32,  # 30 points per harmonic
        smoothing_steps=1,  # Apply low-pass filter 3 times
        smoothing_param=0.1  # Lowpass cutoff frequency as 0.2 * Nyquist
    )
    dataset = VibrationDataset(dxai_root, target_ch=target_channels, target_class=['normal', 'unbalance'], transform=pipeline)
    sample_data, meta_data = dataset[4000]
    
    print(f'data_shape : {sample_data.shape}')
    
    plt.title(meta_data['class_name'])
    plt.plot(meta_data['freq'], sample_data[0])
    plt.plot(meta_data['freq'], sample_data[1])
    plt.show()