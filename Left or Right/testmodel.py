import trials
import pylsl as lsl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import sleep


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(3456, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = x.view(batch, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

batch = 1
device = torch.device('cpu')
net = Net()
net = net.double()
net.load_state_dict(torch.load('models/60.79-2020-11-15 01-36-29.pt', map_location=device))

channels = 8  # Num of headset channels
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

while True:
    round_data = []
    for i in range(channels):
        sample, timestamp = inlet.pull_sample()
        round_data.append(sample)
    round_data = torch.from_numpy(np.array(round_data))
    round_data = round_data[:,:60]  # Trimms to 60hz
    output = net(round_data.view(1, 1, 8, 60))
    output = int(torch.argmax(output))
    if output == 0:
        trial.move(-1)
    elif output == 2:
        trial.move(1)
    trial.update()