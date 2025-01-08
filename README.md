# 1. Rotray_Machine_Data
EDA and integrate various rotary machine dataset  
최대한 다양한 데이터를 모으려고 하고 있음, 대부분의 데이터가 베어링 이상인데 가능하면 베어링 외에 회전체에서 발생가능한 데이터셋을 찾는 것을 주 목적으로 함

- 아래의 데이터 개수는 다음 세팅을 기준으로 데이터를 분할한 경우의 결과임

| Parameter       | Value   |
|-----------------|---------|
| Target Sampling(Hz) | 20000   |
| Window Size(point 개수)     | 2048    |
| Hop Size(point 개수)        | 1024    |

## 1.1 FaultDXAI (Fault Diagnosis using eXplainable AI)
paper : https://www.sciencedirect.com/science/article/pii/S0957417423013623  
dataset link : https://data.mendeley.com/datasets/zx8pfhdtnb/3?utm_source=chatgpt.com  


| Metric          | Value                                                                 |
|------------------|----------------------------------------------------------------------|
| Data Length      | 142800                                                              |
| Classes          | {'Looseness': 35700, 'Unbalance': 35700, 'Normal Condition': 35700, 'Misalignment': 35700} |


## 1.2 Machine Learning-Based Unbalance Detection of a Rotating Shaft Using Vibration Data
paper : https://ieeexplore.ieee.org/document/9212000  
dataset link : https://fordatis.fraunhofer.de/handle/fordatis/151.3

| Metric          | Value                                                                 |
|------------------|----------------------------------------------------------------------|
| Data Length      | 170                                                                  |
| Classes          | {'unbalance60.7': 34, 'unbalance75.5': 34, 'unbalance152.1': 34, 'unbalance45.9': 34, 'normalNone': 34} |


## 1.3 Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Operating Conditions for Fault Diagnosis
> I call it VAT-MCD dataset
paper : https://www.sciencedirect.com/science/article/pii/S2352340923001671  
dataset link : https://data.mendeley.com/datasets/ztmf3m7h5x/6?utm_source=chatgpt.com  

| Metric          | Value                                                                 |
|------------------|----------------------------------------------------------------------|
| Data Length      | 765                                                                  |
| Classes          | {'Misalign03': 51, 'normal': 51, 'BPFI10': 51, 'Unbalance0583mg': 34, 'Misalign01': 51, 'BPFO30': 51, 'Unbalance1169mg': 34, 'Misalign05': 51, 'BPFI03': 51, 'Unbalalnce1751mg': 17, 'Unbalance2239mg': 34, 'BPFO03': 51, 'BPFI30': 51, 'Unbalance3318mg': 34, 'BPFO10': 51, 'Unbalalnce3318mg': 17, 'Unbalalnce2239mg': 17, 'Unbalalnce1169mg': 17, 'Unbalance1751mg': 34, 'Unbalalnce0583mg': 17} |


## 1.4 VBL-V001
paper : https://link.springer.com/article/10.1007/s42417-023-00959-9  
dataset link : https://github.com/bagustris/VBL-VA001?tab=readme-ov-file (github) https://zenodo.org/records/7006575#.Y3W9lzPP2og (data link)

| Metric          | Value                                                                 |
|------------------|----------------------------------------------------------------------|
| Data Length      | 68000                                                                |
| Classes          | {'normal': 17000, 'misalignment': 17000, 'bearing': 17000, 'unbalance': 17000} |


# 2. 사용하고자 하는 특징

## 2.1 시간 도메인 특징
| 번호 | 특징 이름        | 설명                                                         | 수식                                                                                              |
|------|------------------|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| 1    | Mean             | 신호의 평균값                                                | ![mean](https://latex.codecogs.com/svg.latex?%5Cmu%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5ENx_i) |
| 2    | Standard Deviation | 신호의 분산 정도를 측정                                       | ![std](https://latex.codecogs.com/svg.latex?%5Csigma%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%28x_i-%5Cmu%29%5E2%7D) |
| 3    | Maximum          | 신호의 최대 진폭 값                                          | ![max](https://latex.codecogs.com/svg.latex?x_%7Bmax%7D%20%3D%20%5Cmax%28x%29)                   |
| 4    | Minimum          | 신호의 최소 진폭 값                                          | ![min](https://latex.codecogs.com/svg.latex?x_%7Bmin%7D%20%3D%20%5Cmin%28x%29)                   |
| 5    | RMS              | 신호의 에너지 크기를 측정                                     | ![rms](https://latex.codecogs.com/svg.latex?RMS%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5ENx_i%5E2%7D) |
| 6    | Skewness         | 신호의 비대칭성을 나타냄                                     | ![skewness](https://latex.codecogs.com/svg.latex?Skewness%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%5Cfrac%7B%28x_i-%5Cmu%29%5E3%7D%7B%5Csigma%5E3%7D) |
| 7    | Kurtosis         | 신호의 피크 정도를 나타냄                                     | ![kurtosis](https://latex.codecogs.com/svg.latex?Kurtosis%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%5Cfrac%7B%28x_i-%5Cmu%29%5E4%7D%7B%5Csigma%5E4%7D) |
| 8    | Peak             | 신호에서 절대값으로 가장 큰 진폭                              | ![peak](https://latex.codecogs.com/svg.latex?Peak%20%3D%20%5Cmax%28%7C%5Cvec%7Bx%7D%7C%29)        |
| 9    | PPV              | 최대 진폭과 최소 진폭 간의 차이                               | ![ppv](https://latex.codecogs.com/svg.latex?PPV%20%3D%20x_%7Bmax%7D-x_%7Bmin%7D)                |
| 10   | Crest Factor     | 피크 진폭과 RMS의 비율                                        | ![crest](https://latex.codecogs.com/svg.latex?Crest%20Factor%20%3D%20%5Cfrac%7BPeak%7D%7BRMS%7D) |
| 11   | Impulse Factor   | 피크 진폭과 신호 평균의 비율                                  | ![impulse](https://latex.codecogs.com/svg.latex?Impulse%20Factor%20%3D%20%5Cfrac%7BPeak%7D%7B%5Cmu%7D) |
| 12   | Shape Factor     | RMS와 신호 평균의 비율                                        | ![shape](https://latex.codecogs.com/svg.latex?Shape%20Factor%20%3D%20%5Cfrac%7BRMS%7D%7B%5Cmu%7D) |

## 2.2 주파수 도메인 특징
| 번호 | 특징 이름                | 설명                                                       | 수식                                                                                              |
|------|--------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| 1    | Total Power             | 주파수 스펙트럼의 에너지 총합                               | ![power](https://latex.codecogs.com/svg.latex?P_%7Btotal%7D%20%3D%20%5Csum_%7Bk%3D1%7D%5EN%7CFFT%28x_k%29%7C%5E2) |
| 2    | Max Frequency           | 가장 큰 진폭을 가진 주파수                                  | ![max_freq](https://latex.codecogs.com/svg.latex?f_%7Bmax%7D%20%3D%20f_%7Bargmax%7D)             |
| 3    | Mean Frequency          | 주파수와 진폭의 가중 평균                                   | ![mean_freq](https://latex.codecogs.com/svg.latex?f_%7Bmean%7D%20%3D%20%5Cfrac%7B%5Csum%28f%5Ccdot%7CFFT%28x%29%7C%29%7D%7B%5Csum%7CFFT%28x%29%7C%7D) |
| 4    | Median Frequency        | 총 에너지의 절반이 위치하는 주파수                          | ![median_freq](https://latex.codecogs.com/svg.latex?f_%7Bmedian%7D%20%3D%20f%28%5Csum%7CFFT%28x%29%7C%20%3E%20%5Cfrac%7BP_%7Btotal%7D%7D%7B2%7D%29) |
| 5    | Spectral Skewness       | 스펙트럼의 비대칭성                                        | ![spectral_skew](https://latex.codecogs.com/svg.latex?Skewness%20%3D%20%5Cfrac%7B%5Csum%28f-Mean%29%5E3%20%7CFFT%7C%7D%7BStd%5E3%7D) |
| 6    | Spectral Kurtosis       | 스펙트럼의 피크 정도                                        | ![spectral_kurt](https://latex.codecogs.com/svg.latex?Kurtosis%20%3D%20%5Cfrac%7B%5Csum%28f-Mean%29%5E4%20%7CFFT%7C%7D%7BStd%5E4%7D) |
| 7    | Peak Amplitude          | 스펙트럼에서 가장 큰 진폭 값                                | ![peak_amp](https://latex.codecogs.com/svg.latex?Peak%20Amplitude%20%3D%20%5Cmax%28%7CFFT%28x%29%7C%29) |
| 8    | Band Energy (0.1~1Hz)   | 특정 대역(0.1~1Hz)의 에너지 합                              | ![band_energy](https://latex.codecogs.com/svg.latex?E_%7Bband%7D%20%3D%20%5Csum%28f%3D0.1%29%5E1%20%7CFFT%28x%29%7C%5E2) |
| 9    | Dominant Frequency Power | 가장 큰 진폭의 주파수 성분 에너지                          | ![dominant_power](https://latex.codecogs.com/svg.latex?Power_%7Bdominant%7D%20%3D%20%7CFFT%28f_%7Bmax%7D%29%7C%5E2) |
| 10   | Spectral Entropy        | 스펙트럼의 무질서도                                         | ![entropy](https://latex.codecogs.com/svg.latex?Entropy%20%3D%20-%5Csum%20P_%7Bf%7D%20log%28P_%7Bf%7D%29) |
| 11   | RMS Frequency           | 주파수 스펙트럼의 RMS 값                                    | ![rms_freq](https://latex.codecogs.com/svg.latex?RMS_%7Bfreq%7D%20%3D%20%5Csqrt%7B%5Cfrac%7B1%7D%7BN%7D%5Csum%7CFFT%28x%29%7C%5E2%7D) |
| 12   | Variance Frequency      | 주파수 스펙트럼의 분산                                      | ![variance](https://latex.codecogs.com/svg.latex?Variance%20%3D%20Var%28%7CFFT%28x%29%7C%29)     |