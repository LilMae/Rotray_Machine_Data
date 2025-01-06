from scipy.fft import fft
import numpy as np

def extract_frequency_domain_features(signal, sampling_rate=4096):
    """
    주파수 도메인 특징 추출
    :param signal: numpy array (1xN 형태)
    :param sampling_rate: 샘플링 주파수 (Hz)
    :return: dict 형태의 주파수 도메인 특징
    """
    N = len(signal)
    freq = np.fft.fftfreq(N, d=1/sampling_rate)[:N//2]
    fft_values = np.abs(fft(signal))[:N//2]
    
    total_power = np.sum(fft_values**2)
    max_frequency = freq[np.argmax(fft_values)]
    mean_frequency = np.sum(freq * fft_values) / np.sum(fft_values)
    median_frequency = freq[np.cumsum(fft_values) >= np.sum(fft_values) / 2][0]
    spectral_skewness = np.mean((freq - mean_frequency)**3 * fft_values) / (np.std(freq) + 1e-10)**3
    spectral_kurtosis = np.mean((freq - mean_frequency)**4 * fft_values) / (np.std(freq) + 1e-10)**4
    peak_amplitude = np.max(fft_values)
    band_energy = np.sum(fft_values[(freq >= 0.1) & (freq <= 1.0)]**2)
    dominant_frequency_power = fft_values[np.argmax(fft_values)]**2
    spectral_entropy = -np.sum((fft_values / np.sum(fft_values)) * np.log2(fft_values / np.sum(fft_values) + 1e-10))
    rms_frequency = np.sqrt(np.mean(fft_values**2))
    variance_frequency = np.var(fft_values)
    
    features = {
        "total_power": total_power,
        "max_frequency": max_frequency,
        "mean_frequency": mean_frequency,
        "median_frequency": median_frequency,
        "spectral_skewness": spectral_skewness,
        "spectral_kurtosis": spectral_kurtosis,
        "peak_amplitude": peak_amplitude,
        "band_energy_0.1_1Hz": band_energy,
        "dominant_frequency_power": dominant_frequency_power,
        "spectral_entropy": spectral_entropy,
        "rms_frequency": rms_frequency,
        "variance_frequency": variance_frequency
    }
    
    return features

def extract_time_domain_features(signal):
    """
    시간 도메인 특징 추출
    :param signal: numpy array (1xN 형태)
    :return: dict 형태의 시간 도메인 특징
    """
    features = {}
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    signal_max = np.max(signal)
    signal_min = np.min(signal)
    signal_rms = np.sqrt(np.mean(signal**2))
    signal_skew = np.mean((signal - signal_mean)**3) / (signal_std**3 + 1e-10)
    signal_kurt = np.mean((signal - signal_mean)**4) / (signal_std**4 + 1e-10)
    signal_peak = np.max(np.abs(signal))
    signal_ppv = signal_max - signal_min  # Peak-to-Peak Value
    signal_crest = signal_peak / (signal_rms + 1e-10)
    signal_impulse = signal_peak / (np.mean(np.abs(signal)) + 1e-10)
    signal_shape = signal_rms / (np.mean(np.abs(signal)) + 1e-10)
    
    features.update({
        "mean": signal_mean,
        "std": signal_std,
        "max": signal_max,
        "min": signal_min,
        "rms": signal_rms,
        "skewness": signal_skew,
        "kurtosis": signal_kurt,
        "peak": signal_peak,
        "ppv": signal_ppv,
        "crest_factor": signal_crest,
        "impulse_factor": signal_impulse,
        "shape_factor": signal_shape
    })
    
    return features

def compute_fft(signal: np.ndarray, sampling_rate: float):
    """
    입력 신호에 대해 FFT 변환을 수행하고 magnitude, phase, freq를 계산.
    
    :param signal: np.ndarray, 입력 신호 (float32)
    :param sampling_rate: float, 샘플링 레이트 (Hz, float32)
    :return: tuple (magnitude, phase, freq)
        - magnitude: np.ndarray, 진폭 스펙트럼
        - phase: np.ndarray, 위상 스펙트럼
        - freq: np.ndarray, 주파수 축
    """
    # 신호 길이
    n = len(signal)
    
    # FFT 계산
    fft_result = np.fft.fft(signal)
    
    # 주파수 계산
    freq = np.fft.fftfreq(n, d=1 / sampling_rate)
    
    # 진폭 및 위상 계산
    magnitude = np.abs(fft_result)  # FFT의 크기 (magnitude)
    phase = np.angle(fft_result)   # FFT의 위상 (phase)
    
    # 주파수는 대칭이므로 양수 주파수 부분만 반환
    half_n = n // 2
    return magnitude[:half_n], phase[:half_n], freq[:half_n]