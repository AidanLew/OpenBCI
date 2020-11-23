import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
from datetime import datetime
from tqdm import tqdm

learning_rate=0.001
epochs = 2
batch_size=64
percent_test = 0.10
save_model = True


class data_preperation():
    def __init__(self, classes):
        self.classes = classes
        self.x = []
        self.y = []

    def import_data(self):
        class_number = 0
        for name in self.classes:
            delta_x = len(self.x)
            for array in os.listdir(f'data/{name}/'):
                try:
                    self.x = np.concatenate((self.x, np.load(f'data/{name}/{array}')), axis=0)
                except Exception:
                    self.x = np.load(f'data/{name}/{array}')
            delta_x = len(self.x) - delta_x
            for i in range(delta_x):
                self.y.append(np.eye(len(classes))[class_number])
            class_number += 1
        self.y = np.array(self.y)

    def shuffle(self):
        data = list(zip(self.x, self.y))
        random.shuffle(data)
        self.x, self.y = zip(*data)
        self.x = np.array(self.x)
        self.y = np.array(self.y)


classes = ['left', 'nothing', 'right']
data = data_preperation(classes)
data.import_data()
data.shuffle()

data.x = data.x / np.amax(data.x)
num_train = int(len(data.x) * (1 - percent_test))
num_test = int(len(data.x) * percent_test)

train_x = torch.from_numpy(data.x[:num_train]).double().view(-1, 1, 8, 60)
train_y = torch.from_numpy(data.y[:num_train])
test_x = torch.from_numpy(data.x[num_train:]).double().view(-1, 1, 8, 60)
test_y = torch.from_numpy(data.y[num_train:])


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

device = torch.device('cpu')
net = Net().to(device)
net = net.double()

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

for epoch in range(epochs):
    batch = batch_size
    for i in tqdm(range(0, len(train_x), batch)):
        batch_x = train_x[i:i+batch_size]
        batch_y = train_y[i:i+batch_size]
        batch = len(batch_x)

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        net.zero_grad()

        optimizer.zero_grad()
        outputs = net(batch_x)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1}. Loss: {loss}")

correct = 0
total = 0

with torch.no_grad():
    batch = 1
    for i in tqdm(range(len(test_x))):
        real_class = torch.argmax(test_y[i]).to(device)
        net_out = net(test_x[i].view(-1, 1, 8, 60).to(device))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1
accuracy = round(((correct/total)*100), 3)
print("Accuracy: ", round(((correct/total)*100), 3), '%')

if save_model:
    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    torch.save(net.state_dict(), f'models/{accuracy}-{current_time}.pt')