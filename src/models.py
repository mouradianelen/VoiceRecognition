import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, freq_bins, time_steps):
        super(Net, self).__init__()
        self.freq_bins = freq_bins
        self.time_steps = time_steps

        self.conv1 = nn.Conv2d(1, 60, 5)
        self.bn1 = nn.BatchNorm2d(60)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(60, 160, 5)
        self.bn2 = nn.BatchNorm2d(160)
        self.conv3 = nn.Conv2d(160, 320, 3)  
        self.bn3 = nn.BatchNorm2d(320)

        self.fc1_input_size = self._get_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 120)
        self.dropout1 = nn.Dropout(p=0.2)  
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(84, 10)

    def _get_fc1_input_size(self):
        with torch.no_grad():
            
            x = torch.zeros(1, 1, self.freq_bins, self.time_steps)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))  
            return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.dropout1(x)))
        x = self.fc3(self.dropout2(x))
        return x
