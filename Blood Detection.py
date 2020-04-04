import numpy as np
import pandas as pd
import json
from os.path import join
import matplotlib.pyplot as plt
from scipy import signal
%matplotlib inline
input_path = join('..', 'input', 'Cuff-less Non-invasive Blood Pressure Estimation Data Set') 
num_patients = 26
with open(join(input_path, '1.json'), 'r') as f:
    data = json.load(f)
data_keys = ['data_PPG', 'data_ECG', 'data_PCG', 'data_FSR', 'data_BP']
{k: len(data[k]) for k in data_keys}
data['data_BP']
plt.figure(figsize=(14, 6))
data_FSR = -np.array(data['data_FSR'])
plt.plot(data_FSR);
plt.figure(figsize=(14, 6))
max_diff = 50
data_FSR_clear = np.array(data_FSR, dtype=np.float)
data_FSR_outliers = np.abs(data_FSR[1:] - data_FSR[:-1]) > max_diff
data_FSR_outliers = np.append(data_FSR_outliers, False)
data_FSR_clear[data_FSR_outliers] = np.nan
plt.plot(data_FSR_clear);
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    mean_window = 10

data_FSR_roll_mean = np.nanmean(rolling_window(data_FSR_clear, mean_window), axis=-1)

data_FSR_clear[np.isnan(data_FSR_clear)] = \
    data_FSR_roll_mean[np.isnan(data_FSR_clear)[:1-mean_window]]
    
assert np.isnan(data_FSR_clear).sum() == 0
plt.figure(figsize=(14, 6))
plt.plot(data_FSR_clear);
plt.figure(figsize=(14, 6))
data_FSR_smooth = signal.savgol_filter(data_FSR_clear, 51, 0)
plt.plot(data_FSR_smooth);
plt.figure(figsize=(14, 6))
diff_n = 1000
roll_window = 21
data_FSR_diff = data_FSR_smooth[diff_n:] - data_FSR_smooth[:-diff_n]
data_FSR_diff_roll = rolling_window(data_FSR_diff, roll_window).mean(axis=-1)
plt.title('FSR slope')
plt.plot(data_FSR_diff_roll);
num_mins = len(data['data_BP'])
min_window = 15000

def find_mins(a, num_mins, window):
    found_mins = []
    amax = a.max()
    
    hwindow = window // 2
    
    a = np.array(a)

    for i in range(num_mins):
        found_min = np.argmin(a)
        found_mins.append(found_min)
        a[found_min-hwindow:found_min+hwindow] = amax
     del a
return sorted(found_mins)
        
data_FSR_mins = find_mins(data_FSR_diff_roll, num_mins, min_window)
plt.figure(figsize=(14, 6))
plt.plot(data_FSR_smooth, label='Smoothed FSR')

data_FSR_max, data_FSR_min = data_FSR_smooth.max(), data_FSR_smooth.min()
for m in data_FSR_mins:
    plt.vlines(m + diff_n/2, data_FSR_min, data_FSR_max, color='red')
plt.legend()
plt.title('BP measures points');
