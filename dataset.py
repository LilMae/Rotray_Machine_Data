import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
import scipy.io
from dataclasses import dataclass
from collections import Counter
@dataclass
class raw_dataset:
    file_name : list = None,
    file_path : list = None,
    class_name : list = None,
    severity : list = None,
    sampling_rate : list = None,
    speed : int = None

@dataclass
class VibrationSet:
    x_motor_np: np.ndarray = None
    y_motor_np: np.ndarray = None
    x_disk_np: np.ndarray = None
    y_disk_np: np.ndarray = None
    speed: np.ndarray = None  # fg dataset은 파일을 열어야 속도를 알 수 있다

    def slicing(self, start_idx, end_idx):
        # 슬라이싱된 데이터 생성
        return VibrationSet(
            x_motor_np=self.x_motor_np[start_idx:end_idx] if self.x_motor_np is not None else None,
            y_motor_np=self.y_motor_np[start_idx:end_idx] if self.y_motor_np is not None else None,
            x_disk_np=self.x_disk_np[start_idx:end_idx] if self.x_disk_np is not None else None,
            y_disk_np=self.y_disk_np[start_idx:end_idx] if self.y_disk_np is not None else None,
            speed=self.speed[start_idx:end_idx] if self.speed is not None else None
        )
        

def load_dxai(fault_dxai_root):
    sampling_rate = 20*1000 # 20kHz
    motor_speed = 1238 # 3000 RPM fixed

    file_name_list = []
    file_path_list = []
    class_name_list = []

    for class_name in os.listdir(fault_dxai_root):
        # 압축 풀 때 폴더가 하나 더생기더라;;;
        class_dir = os.path.join(fault_dxai_root, class_name, class_name)
        
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)

            # 클래스가 unbalance 인 경우에는 하부 클래스 정보가 추가로 필요하다.
            class_name =class_name.split('_')[-1]
            
            file_name_list.append(file_name)
            file_path_list.append(file_path)
            class_name_list.append(class_name)

    dxai_dataset = raw_dataset()
    dxai_dataset.file_name = file_name_list
    dxai_dataset.file_path = file_path_list
    dxai_dataset.class_name = class_name_list
    sampling_rate_list = [sampling_rate for _ in range(len(file_name_list))]
    dxai_dataset.sampling_rate = sampling_rate_list
    speed_list = [motor_speed for _ in range(len(file_name_list))]
    dxai_dataset.speed = speed_list
    
    return dxai_dataset

def load_fg(fg_root):
    
    sampling_rate = 20*1000 # 20kHz
    
    file_name_list = []
    file_path_list = []
    class_name_list = []
    severity_list = []
    
    severity_dict = {
        '0' : None,
        '1' : 45.9,
        '2' : 60.7,
        '3' : 75.5,
        '4' : 152.1
    }

    files = [f for f in os.listdir(fg_root) if f.endswith('.csv')]
    for file_name in files:
        
        file_path = os.path.join(fg_root, file_name)
        
        class_name = 'normal' if file_name[0]=='0' else 'unbalance'
        severity = severity_dict[file_name[0]]
        
        file_name_list.append(file_name)
        file_path_list.append(file_path)
        class_name_list.append(class_name)
        severity_list.append(str(severity))

    fg_dataset = raw_dataset()
    fg_dataset.file_name = file_name_list
    fg_dataset.file_path = file_path_list
    fg_dataset.class_name = class_name_list
    fg_dataset.severity = severity_list
    sampling_rate_list = [sampling_rate for _ in range(len(file_name_list))]
    fg_dataset.sampling_rate = sampling_rate_list

    return fg_dataset

def load_vat(vat_root):

    sampling_rate = 25.6 * 1000 # 25.6 kHz

    file_name_list = []
    file_path_list = []
    class_name_list = []
    severity_list = []

    for file_name in os.listdir(vat_root):
        file_path = os.path.join(vat_root, file_name)
        
        file_name_list.append(file_name)
        file_path_list.append(file_path)
        
        file_info = file_name.split('_')
        severity = None
        if len(file_info) ==2:
            class_name = 'normal'
            load = file_info[0]
        elif len(file_info) ==3:
            load = file_info[0] 
            class_name = file_info[1]
            severity = file_info[2].split('.')[0]
        else:
            print('err : {file_name}')
            
            
        class_name_list.append(class_name)
        severity_list.append(severity)
    
    
    vat_dataset = raw_dataset()
    vat_dataset.file_name = file_name_list
    vat_dataset.file_path = file_path_list
    vat_dataset.class_name = class_name_list
    vat_dataset.severity = severity_list
    sampling_rate_list = [sampling_rate for _ in range(len(file_name_list))]
    vat_dataset.sampling_rate = sampling_rate_list
    
    return vat_dataset

def load_vbl(vbl_root):

    sampling_rate = 20*1000 # 20kHz
    motor_speed = 3000 # 3000 RPM fixed

    file_name_list = []
    file_path_list = []
    class_name_list = []
    severity_list = []

    for class_name in os.listdir(vbl_root):
        class_dir = os.path.join(vbl_root, class_name)
        
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)

            severity = None
            # 클래스가 unbalance 인 경우에는 하부 클래스 정보가 추가로 필요하다.
            if class_name == 'unbalance':
                unbalance_scale = file_name.split('_')[0]
                if unbalance_scale =='ub':
                    unbalance_scale = file_name.split('_')[1]
                else:
                    unbalance_scale = unbalance_scale[-2:]
                severity = unbalance_scale
            
            file_name_list.append(file_name)
            file_path_list.append(file_path)
            class_name_list.append(class_name)
            severity_list.append(severity)

    vbl_dataset = raw_dataset()
    vbl_dataset.file_name = file_name_list
    vbl_dataset.file_path = file_path_list
    vbl_dataset.class_name = class_name_list
    sampling_rate_list = [sampling_rate for _ in range(len(file_name_list))]
    vbl_dataset.sampling_rate = sampling_rate_list
    speed_list = [motor_speed for _ in range(len(file_name_list))]
    vbl_dataset.speed = speed_list

    return vbl_dataset

# 1. open_dxai_file
def open_dxai_file(file_path):
    data_np = np.load(file_path)

    y_pulley_np = data_np[0]
    x_pulley_np = data_np[1]
    y_disk_np = data_np[2]
    x_disk_np = data_np[3]
    
    vibration = VibrationSet()
    
    vibration.x_motor_np = x_pulley_np
    vibration.y_motor_np = y_pulley_np
    vibration.x_disk_np = x_disk_np
    vibration.y_disk_np = y_disk_np

    return vibration

# 2. open_vbl_file 
def open_fg_file(file_path):
    
    data_pd = pd.read_csv(file_path)

    # 1. 'Measured_RPM' 칼럼을 int 형으로 변경
    data_pd = data_pd.dropna(subset=['Measured_RPM'])
    data_pd['Measured_RPM'] = data_pd['Measured_RPM'].astype(int)
    # 2. 'Measured_RPM'이 100 이하인 행 제거
    data_pd = data_pd[data_pd['Measured_RPM'] > 100]
    
    x_disk_np = data_pd['Vibration_1'].to_numpy('float32')
    y_disk_np = data_pd['Vibration_2'].to_numpy('float32')
    y_pulley_np = data_pd['Vibration_3'].to_numpy('float32')
    motor_speed = data_pd['Measured_RPM'].to_numpy('float32')
    
    vibration = VibrationSet()
    
    vibration.y_motor_np = y_pulley_np
    vibration.x_disk_np = x_disk_np
    vibration.y_disk_np = y_disk_np
    vibration.speed = motor_speed
    
    return vibration

# 3. open_vat_file 
def open_vat_file(file_path):
    try:
        vib_data = scipy.io.loadmat(file_path)
    except:
        print(file_path)
        exit()
    signal_data = vib_data['Signal']

    sensor_data = signal_data[0][0][1][0][0][0]
    sensor_data = sensor_data.transpose()

    x_disk_np = sensor_data[0].astype('float32')
    y_disk_np = sensor_data[1].astype('float32')
    x_pulley_np = sensor_data[2].astype('float32')
    y_pulley_np = sensor_data[3].astype('float32')
    
    vibration = VibrationSet()
    
    vibration.x_motor_np = x_pulley_np
    vibration.y_motor_np = y_pulley_np
    vibration.x_disk_np = x_disk_np
    vibration.y_disk_np = y_disk_np
    
    return vibration

# 4. open_vbl_file
def open_vbl_file(file_path):
    data_pd = pd.read_csv(file_path, header=None)
    data_pd.columns = ['time', 'x', 'y', 'z']

    time_np = data_pd['time'].to_numpy(dtype='float32')
    x_np = data_pd['x'].to_numpy(dtype='float32')
    y_np = data_pd['y'].to_numpy(dtype='float32')
    z_np = data_pd['z'].to_numpy(dtype='float32')
    
    vibration = VibrationSet()
    vibration.x_motor_np = x_np
    vibration.y_motor_np = y_np
    
    return vibration

# integrted open file
def open_raw_file(dataset_name, file_path):
    
    if dataset_name == 'DXAI':
        vibration = open_dxai_file(file_path)
    elif dataset_name == 'FG':
        vibration = open_fg_file(file_path)
    elif dataset_name == 'VAT':
        vibration = open_vat_file(file_path)
    elif dataset_name == 'VBL':
        vibration = open_vbl_file(file_path)
    else:
        print('Error : wrong dataset_name : {dataset_name}')
    
    return vibration

def interpol_total_signal(VibrationSet, sampling_rate, target_sampling):
    
    if VibrationSet.x_motor_np is not None:
        x_motor_np = VibrationSet.x_motor_np
        VibrationSet.x_motor_np = interpol_single_signal(x_motor_np, sampling_rate, target_sampling)
    if VibrationSet.y_motor_np is not None:
        y_motor_np = VibrationSet.y_motor_np
        VibrationSet.y_motor_np = interpol_single_signal(y_motor_np, sampling_rate, target_sampling)
    if VibrationSet.x_disk_np is not None:
        x_disk_np = VibrationSet.x_disk_np
        VibrationSet.x_disk_np = interpol_single_signal(x_disk_np, sampling_rate, target_sampling)
    if VibrationSet.y_disk_np is not None:
        y_disk_np = VibrationSet.y_disk_np
        VibrationSet.y_disk_np = interpol_single_signal(y_disk_np, sampling_rate, target_sampling)

    return VibrationSet
        
def interpol_single_signal(signal_np, sampling_rate, target_sampling):

    time_np = np.arange(len(signal_np))/sampling_rate
    target_time = np.linspace(0, time_np[-1], target_sampling, endpoint=False) # target sampling rate에 맞도록 시간축 생성

    interpolator = interp1d(time_np, signal_np, kind='linear')
    interpolated_signal = interpolator(target_time)
    
    return interpolated_signal


class VibrationDataset(Dataset):
    def __init__(self, dataset_name, raw_dataset, target_sampling, window_size, hop_size):

        self.raw_dataset = raw_dataset
        x_list = []
        y_list = []
        global_indices = []

        for file_idx, (file_path, class_name, sampling_rate) in enumerate(zip(raw_dataset.file_path, raw_dataset.class_name, raw_dataset.sampling_rate)):
            
            vibration = open_raw_file(dataset_name=dataset_name, file_path=file_path)
            vibration = interpol_total_signal(vibration, sampling_rate, target_sampling)
            

            data_len = len(vibration.y_motor_np)
            max_len = (data_len - window_size) // hop_size * hop_size + window_size
            num_sample = (data_len - window_size) // hop_size

            vibration = vibration.slicing(0,max_len)

            x_list.append(vibration)

            if raw_dataset.severity[0] is not None:
                severity = raw_dataset.severity[file_idx]
                y_list.append([class_name, severity])
            else:
                y_list.append(class_name)

            # 전역 인덱스 생성
            global_indices.extend([(file_idx, sample_idx) for sample_idx in range(num_sample)])

        self.x_list = x_list
        self.y_list = y_list
        self.global_indices = global_indices
        self.window_size = window_size
        self.hop_size = hop_size

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        # 전역 인덱스를 활용해 파일 인덱스와 샘플 인덱스를 가져옴
        file_idx, sample_idx = self.global_indices[idx]

        x_set = self.x_list[file_idx]
        

        # 슬라이싱 범위 계산
        start_idx = sample_idx * self.hop_size
        end_idx = start_idx + self.window_size

        vibration = x_set.slicing(start_idx,end_idx)
        
        # 클래스 이름을 인덱스로 변환
        if self.raw_dataset.severity[0] is not None:
            class_name = self.y_list[file_idx][0]
            severity = self.y_list[file_idx][1]
        else:
            class_name = self.y_list[file_idx]
            severity = None

        return vibration, class_name, severity
        
    
    def get_minimum_window(self, num_cycles = 10):

        rotation_frequency = self.motor_speed / 60

        # 윈도우 크기 (샘플 수)
        window_size = int(sampling_rate * num_cycles / rotation_frequency)
        window_time = window_size/sampling_rate

        print(f"회전 주파수: {rotation_frequency} Hz")
        print(f"윈도우 크기: {window_size} 샘플, {window_time} 초")

        return window_size

if __name__ == '__main__':
    
    target_sampling = 20*1000
    window_size = 2048
    hop_size=1024
    
    print(f"""<data split setting>
            target_sampling = {target_sampling}
            window_size = {window_size}
            hop_size={hop_size}
          """)
    

    fault_dxai_root = os.path.join(os.getcwd(), 'dataset', 'FaultDXAI')
    dxai_data = load_dxai(fault_dxai_root)
    dxai_dataset = VibrationDataset(dataset_name='DXAI',
                                raw_dataset=dxai_data, 
                                target_sampling=target_sampling, 
                                window_size=window_size, 
                                hop_size=hop_size)
    
    class_name_list = []
    for vibration, class_name, severity in dxai_dataset:        
        if severity is not None:
            class_name += severity
        class_name_list.append(class_name)
        
    counter = Counter(class_name_list)
    print(f"""
          dxai_dataset describe : 
          data_len : {len(dxai_dataset)}
          classes : {dict(counter)}
          """)
    
    fault_fg_root = os.path.join(os.getcwd(), 'dataset', 'FG')
    fg_data = load_fg(fault_fg_root)
    fg_dataset = VibrationDataset(dataset_name='FG',
                                raw_dataset=fg_data, 
                                target_sampling=target_sampling, 
                                window_size=window_size, 
                                hop_size=hop_size)
    
    class_name_list = []
    for vibration, class_name, severity in fg_dataset:    
        
        if severity is not None:
            class_name += severity
        class_name_list.append(class_name)
    counter = Counter(class_name_list)
    print(f"""
          fg_dataset describe : 
          data_len : {len(fg_dataset)}
          classes : {dict(counter)}
          """)

    
    
    vat_root = os.path.join(os.getcwd(), 'dataset', 'VAT-MCD', 'vibration')
    vat_data = load_vat(vat_root)
    vat_dataset = VibrationDataset(dataset_name='VAT',
                                raw_dataset=vat_data, 
                                target_sampling=target_sampling, 
                                window_size=window_size, 
                                hop_size=hop_size)
    
    class_name_list = []
    for vibration, class_name, severity in vat_dataset:        
        if severity is not None:
            class_name += severity
        class_name_list.append(class_name)
    
    counter = Counter(class_name_list)
    print(f"""
          vat_dataset describe : 
          data_len : {len(vat_dataset)}
          classes : {dict(counter)}
          """)

    
    
    vbl_root = os.path.join(os.getcwd(), 'dataset', 'VBL-VA001')
    vbl_data = load_vbl(vbl_root)
    vbl_dataset = VibrationDataset(dataset_name='VBL',
                                raw_dataset=vbl_data, 
                                target_sampling=target_sampling, 
                                window_size=window_size, 
                                hop_size=hop_size)
    
    class_name_list = []
    for vibration, class_name, severity in vbl_dataset:        
        if severity is not None:
            class_name += severity
        class_name_list.append(class_name)
    
    counter = Counter(class_name_list)
    print(f"""
          vbl_dataset describe : 
          data_len : {len(vbl_dataset)}
          classes : {dict(counter)}
          """)
    print('Check Finish')