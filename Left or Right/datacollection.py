import trials
import pylsl as lsl
import numpy as np
from tqdm import tqdm
from time import sleep
from datetime import datetime

channels = 8  # Num of headset channels
samples = 375  # 25 samples ~= 1 second
output_folder = 'right'  # Name of folder with desired action
trial = trials.left_right()  # Trial to view when collecting data.

stream = lsl.resolve_stream('type', 'EEG')  # Start LSL Data stream with FFT graph in OpenBCI GUI 
inlet = lsl.stream_inlet(stream[0])

print('Starting in: 3')
sleep(1)
print('Starting in: 2')
sleep(1)
print('Starting in: 1')
sleep(1)
print('Starting!')

data = []
for i in tqdm(range(samples)):
    round_data = []
    for ii in range(channels):
        sample, timestamp = inlet.pull_sample()
        round_data.append(sample)
    data.append(round_data)

data = np.array(data)
data = data[:,:,:60]  # Trimms to 60hz
current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
np.save(f'data/{output_folder}/{current_time}.npy', data)