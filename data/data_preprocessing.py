import os
import pandas as pd
import numpy as np
import shutil
import glob
from scipy import io
from tqdm import tqdm

def dxai(data_root, new_data_root):
    dxai_root = os.path.join(data_root, '1_FaultDXAI')

    rpm = 1238
    sampling_rate_str = '25kHz'
    load_condition = 'unknwon'
    severity = 'none'

    new_dxai_root = os.path.join(new_data_root, 'dxai')
    if os.path.isdir(new_dxai_root):
        shutil.rmtree(new_dxai_root)
    os.mkdir(new_dxai_root)

    for test_name in os.listdir(dxai_root):
        test_folder = os.path.join(dxai_root, test_name, test_name)
        
        class_name = test_name.split('_')[-1].split(' ')[0].lower()
        test_folder_dist = os.path.join(new_dxai_root, class_name)
        if not os.path.exists(test_folder_dist):
            os.mkdir(test_folder_dist)
        
        for test_file in os.listdir(test_folder):
            file_path = os.path.join(test_folder, test_file)
            file_np = np.load(file_path)
            motor_y = file_np[0]
            motor_x = file_np[1]
            disk_y = file_np[2]
            disk_x = file_np[3]
            
            data_pd = pd.DataFrame({'motor_x' : motor_x, 
                                    'motor_y' : motor_y, 
                                    'disk_x' : disk_x, 
                                    'disk_y' : disk_y})
            data_pd.to_csv(os.path.join(test_folder_dist, f'{class_name}_{sampling_rate_str}_{rpm}_{severity}_{load_condition}_{len(os.listdir(test_folder_dist))+1}.csv'))

def iis(data_root, new_data_root):

    iis_root = os.path.join(data_root, '2_IIS')

    # Parameters
    sampling_rate = 4096  # 4096 samples per second
    sampling_rate_str = '4.096kHz'
    window_size = 1.0  # 1 second
    stride = 0.5  # 0.5 second
    load_condition = 'unknwon'

    new_iss_root = os.path.join(new_data_root, 'iis')
    if os.path.isdir(new_iss_root):
        shutil.rmtree(new_iss_root)
    os.mkdir(new_iss_root)

    for test_path in glob.glob(os.path.join(iis_root,'*.csv')):
        
        test_name = os.path.split(test_path)[-1]
        if int(test_name[0]) == 0:
            class_name = 'normal'
            severity = 0
        else:
            class_name = 'unbalance'
            severity = [0, 459, 607, 755, 1521]
            severity = severity[int(test_name[0])]
        
        test_folder_dist = os.path.join(new_iss_root, class_name)
        if not os.path.exists(test_folder_dist):
            os.mkdir(test_folder_dist)
        
        
        start = 0
        data_sec = 0.0
        cnt = 0
        is_first_segment = True  # Flag to skip the first segment
        
        exp_pd = pd.read_csv(test_path)
        
        speed_df = exp_pd['Measured_RPM']
        
        change_indices = [0]  # Start with the first index
        change_indices += speed_df.index[
            (speed_df.diff().abs() > 10)
        ].tolist()
        
        for change_index in change_indices[1:]:
            end = change_index
            
            static_speed = exp_pd.iloc()[start:end]
            
            disk_x = static_speed['Vibration_1']
            disk_y = static_speed['Vibration_2']
            motor_y = static_speed['Vibration_3']
            
            data_pd = pd.DataFrame({'motor_x' : None, 
                                    'motor_y' : motor_y, 
                                    'disk_x' : disk_x, 
                                    'disk_y' : disk_y})
            
            mean_rpm = int(static_speed['Measured_RPM'].mean())
            
            cnt += 1
            start = end
            
            # Skip the first segment
            if is_first_segment:
                is_first_segment = False
                continue
            
            # Slice data_pd by 1-second window with 0.5-second stride
            window_samples = int(window_size * sampling_rate)
            stride_samples = int(stride * sampling_rate)
            num_rows = len(data_pd)
            
            for slice_start in range(0, num_rows - window_samples + 1, stride_samples):
                slice_end = slice_start + window_samples
                sliced_data = data_pd.iloc[slice_start:slice_end]
                
                cnt += 1
                
                # Save sliced data to a CSV file
                output_file = os.path.join(test_folder_dist, f'{class_name}_{sampling_rate_str}_{mean_rpm}_{severity/10}mmg_{load_condition}_{cnt}.csv')
                sliced_data.to_csv(output_file, index=False)
            


def vat(data_root, new_data_root):
    vat_root = os.path.join(data_root, '3_VAT_mat')

    sampling_rate = 25600 # 25.6kHz
    sampling_rate_str = '25.6kHz'
    window_size = 1.0  # 1 second
    stride = 0.5  # 0.5 second
    rpm = 3010

    new_vat_root = os.path.join(new_data_root, 'vat')
    if os.path.isdir(new_vat_root):
        shutil.rmtree(new_vat_root)
    os.mkdir(new_vat_root)

    for file_name in os.listdir(vat_root):
        
        load_condition = file_name.split('.')[0].split('_')[0]
        class_name = file_name.split('.')[0].split('_')[1].lower()
        if class_name != 'normal':
            severity = file_name.split('.')[0].split('_')[2]
        else:
            severity = 'normal'
        if class_name == 'unbalalnce':
            class_name = 'unbalance'
        if class_name =='bpfo':
            class_name = 'bearing-bpfo'
        if class_name =='bpfi':
            class_name = 'bearing-bpfi'
        
        
        test_folder_dist = os.path.join(new_vat_root, class_name)
        if not os.path.exists(test_folder_dist):
            os.mkdir(test_folder_dist)
        
        file_path = os.path.join(vat_root, file_name)
        
        mat_file = io.loadmat(file_path)
        signal = mat_file['Signal'][0][0][1][0][0][0]
        data_pd = pd.DataFrame(signal, columns=['motor_x','motor_y','disk_x','disk_y'])
        
        # Slice data_pd by 1-second window with 0.5-second stride
        window_samples = int(window_size * sampling_rate)
        stride_samples = int(stride * sampling_rate)
        num_rows = len(data_pd)
        
        cnt = 0
        for slice_start in range(0, num_rows - window_samples + 1, stride_samples):
            slice_end = slice_start + window_samples
            sliced_data = data_pd.iloc[slice_start:slice_end]
            
            cnt += 1
            
            # Save sliced data to a CSV file
            output_file = os.path.join(test_folder_dist, f'{class_name}_{sampling_rate_str}_{rpm}_{severity}_{load_condition}_{cnt}.csv')
            sliced_data.to_csv(output_file, index=False)


def vbl(data_root, new_data_root):

    vbl_root = os.path.join(data_root, '4_VBL-VA001')

    sampling_rate = 20000 # 20kHz
    sampling_rate_str = '20kHz'
    window_size = 1.0  # 1 second
    stride = 0.5  # 0.5 second
    rpm = 3000
    load_condition = 'unknown'

    new_vbl_root = os.path.join(new_data_root, 'vbl')
    if os.path.isdir(new_vbl_root):
        shutil.rmtree(new_vbl_root)
    os.mkdir(new_vbl_root)


    for class_name in os.listdir(vbl_root):
        class_folder = os.path.join(vbl_root, class_name)
        
        if class_name == 'bpfo':
            class_name = 'bearing-bpfo'
        
        test_folder_dist = os.path.join(new_vbl_root, class_name)
        if not os.path.exists(test_folder_dist):
            os.mkdir(test_folder_dist)
        
        cnt = 0
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            data_pd = pd.read_csv(file_path, header=None)
            data_pd.columns = ['time', 'motor_x', 'motor_y', 'motor_z']
            data_pd = data_pd.drop(labels='time', axis=1)
            
            if class_name == 'unbalance':
                severity = file_name.split('_')[1]
                
                if severity == 'z':
                    severity = file_name.split('_')[0][-2:]
            else:
                severity = 'none'
            
            # Slice data_pd by 1-second window with 0.5-second stride
            window_samples = int(window_size * sampling_rate)
            stride_samples = int(stride * sampling_rate)
            num_rows = len(data_pd)
            
            
            for slice_start in range(0, num_rows - window_samples + 1, stride_samples):
                slice_end = slice_start + window_samples
                sliced_data = data_pd.iloc[slice_start:slice_end]
                
                cnt += 1
                
                # Save sliced data to a CSV file
                output_file = os.path.join(test_folder_dist, f'{class_name}_{sampling_rate_str}_{rpm}_{severity}_{load_condition}_{cnt}.csv')
                sliced_data.to_csv(output_file, index=False)
            

def mfd_file_read_save(data_pd, folder_dist , class_name, severity):
    sampling_rate_str = '50kHz'
    
    load_condition = 'unknown'
    # Hyper Params
    sampling_rate = 50000
    mfd_columns = ['tachometer', 'motor_z', 'motor_y', 'motor_x', 'disk_z', 'disk_y', 'disk_x', 'microphone']
    window_size = 1.0  # 1 second
    stride = 0.5  # 0.5 second
    
    if not os.path.exists(folder_dist):
        os.mkdir(folder_dist)
    
    data_pd.columns = mfd_columns

    tachometer_np = data_pd['tachometer'].to_numpy()
    tachometer_np = np.convolve(tachometer_np, np.ones(5) / 5, mode='same')
    binary_np = (tachometer_np >3).astype(int)
    rising_edges = np.where(np.diff(binary_np) > 0)[0]
    pulse_intervals = np.diff(rising_edges) / sampling_rate 

    avg_time_per_revolution = np.mean(pulse_intervals)

    rpm = int((1 / avg_time_per_revolution) * 60)


    # Slice data_pd by 1-second window with 0.5-second stride
    window_samples = int(window_size * sampling_rate)
    stride_samples = int(stride * sampling_rate)
    num_rows = len(data_pd)

    for slice_start in range(0, num_rows - window_samples + 1, stride_samples):
        slice_end = slice_start + window_samples
        sliced_data = data_pd.iloc[slice_start:slice_end]
        
        
        # Save sliced data to a CSV file
        output_file = os.path.join(folder_dist, f'{class_name}_{sampling_rate_str}_{rpm}_{severity}_{load_condition}_{len(os.listdir(folder_dist))+1}.csv')
        sliced_data.to_csv(output_file, index=False)

def mfd(data_root, new_data_root):

    mfd_root = os.path.join(data_root, '6_MaFaulDa')

    sampling_rate = 50000
    window_size = 1.0
    stride = 0.5

    new_mfd_root = os.path.join(new_data_root, 'mfd')
    if os.path.isdir(new_mfd_root):
        shutil.rmtree(new_mfd_root)
    os.mkdir(new_mfd_root)

    for class_name in os.listdir(mfd_root):
        
        class_dir = os.path.join(mfd_root, class_name)

        if class_name in ['horizontal-misalignment', 'imbalance', 'vertical-misalignment']:
            for severity in os.listdir(class_dir):
                severity_dir = os.path.join(class_dir, severity)
                for file_name in os.listdir(severity_dir):
                    
                    file_path = os.path.join(severity_dir, file_name)
                    
                    data_pd = pd.read_csv(file_path)
                    
                    if class_name == 'horizontal-misalignment':
                        class_name = 'misalignment-horizontal'
                    if class_name == 'vertical-misalignment':
                        class_name = 'misalignment-vertical'
                    
                    dis_folder = os.path.join(new_mfd_root, class_name)

                    mfd_file_read_save(data_pd, dis_folder, class_name, severity)
                    
                
        elif class_name in ['overhang', 'underhang']:
            for specific_class in os.listdir(class_dir):
                specific_class_dir = os.path.join(class_dir, specific_class)
                
                for severity in os.listdir(specific_class_dir):
                    severity_dir = os.path.join(specific_class_dir, severity)
                    
                    for file_name in os.listdir(severity_dir):
                        file_path = os.path.join(severity_dir, file_name)
                        data_pd = pd.read_csv(file_path)
                        
                        dis_folder = os.path.join(new_mfd_root, class_name)

                        specific_class_name = class_name + specific_class.split('_')[0]
                        mfd_file_read_save(data_pd, dis_folder, specific_class_name, severity)
        
        elif class_name == 'normal':

            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                data_pd = pd.read_csv(file_path)
                
                
                dis_folder = os.path.join(new_mfd_root, class_name)

                mfd_file_read_save(data_pd, dis_folder, class_name, 'none')
                
        else:
            print(f'Wrong Class Name : {class_name}')

def snu(data_root, new_data_root):

    snu_root = os.path.join(data_root, '5_SNU')

    window_size = 1.0
    stride = 0.5

    rotary_class_dict = {'H' : 'normal', 
                        'L' : 'losseness', 
                        'U1' : 'unbalance',    # 2g
                        'U2' : 'unbalance',    # 3g
                        'U3' : 'unbalance',    # 5g
                        'M1' : 'misalignment', # 0.6mm
                        'M2' : 'misalignment', # 0.8mm
                        'M3' : 'misalignment'  # 1.0mm
                        }
    bearing_class_dict = {'H' : 'normal', 
                        'B' : 'encompass', 
                        'IR': 'bpfi', 
                        'OR': 'bpfo'}
    unbalance_severity_list = ['2g', '3g', '5g']
    misalignment_severity_list = ['0.6mm', '0.8mm', '1mm']
    load_condition = 'unknown'

    new_snu_root = os.path.join(new_data_root, 'snu')
    if os.path.isdir(new_snu_root):
        shutil.rmtree(new_snu_root)
    os.mkdir(new_snu_root)

    for bearing_type in os.listdir(snu_root):
        bearing_dir = os.path.join(snu_root, bearing_type)
        
        for sampling_rate in os.listdir(bearing_dir):
            sampling_rate_dir = os.path.join(bearing_dir, sampling_rate)
            sampling_rate_str = f'{int(sampling_rate.split("_")[-1])/1000}kHz'
            sampling_rate = int(sampling_rate.split('_')[-1])
            
            for speed in os.listdir(sampling_rate_dir):
                speed_dir = os.path.join(sampling_rate_dir, speed)
                rpm = speed.split('_')[-1]
                
                print(f'Processing : {bearing_type}/{sampling_rate}/{speed}')
                for file_name in tqdm(os.listdir(speed_dir)):
                    file_path = os.path.join(speed_dir, file_name)
                    mat_file = io.loadmat(file_path)
                    disk_y = mat_file['Data']
                    
                    rotary_class_query = file_name.split('_')[0]
                    rotary_class = rotary_class_dict[rotary_class_query]
                    if rotary_class == 'unbalance':
                        severity = unbalance_severity_list[int(rotary_class_query[-1])-1]
                    elif rotary_class == 'misalignment':
                        severity = misalignment_severity_list[int(rotary_class_query[-1])-1]
                    else:
                        severity = 'none'
                    bearing_class_query = file_name.split('_')[1]
                    bearing_class = bearing_class_dict[bearing_class_query]
                    
                    class_name = rotary_class
                    folder_dist = os.path.join(new_snu_root, class_name)
                    if not os.path.exists(folder_dist):
                        os.mkdir(folder_dist)
                        
                    # Slice data_pd by 1-second window with 0.5-second stride
                    window_samples = int(window_size * sampling_rate)
                    stride_samples = int(stride * sampling_rate)
                    num_rows = len(disk_y)

                    for slice_start in range(0, num_rows - window_samples + 1, stride_samples):
                        slice_end = slice_start + window_samples
                        sliced_data = disk_y[slice_start:slice_end]
                        
                        # Save sliced data to a CSV file
                        output_file = os.path.join(folder_dist, f'{rotary_class}+{bearing_class}_{sampling_rate_str}_{rpm}_{severity}_{load_condition}_{len(os.listdir(folder_dist))+1}.csv')
                        data_pd = pd.DataFrame(sliced_data, columns=['disk_y'])
                        
                        data_pd.to_csv(output_file, index=False)

if __name__ == '__main__':
    
    data_root = os.path.join(os.getcwd(), 'dataset')
    new_data_root = os.path.join(os.getcwd(), 'snu')
    if os.path.isdir(new_data_root):
        shutil.rmtree(new_data_root)
    os.mkdir(new_data_root)

    header = ['motor_x', 'motor_y', 'disk_x', 'disk_y']
    
    # print('Processing dxai')
    # dxai(data_root, new_data_root)
    # print('Processing iis')
    # iis(data_root, new_data_root)
    # print('Processing vat')
    # vat(data_root, new_data_root)
    # print('Processing vbl')
    # vbl(data_root, new_data_root)
    # print('Processing mfd')
    # mfd(data_root, new_data_root)
    
    snu(data_root, new_data_root)