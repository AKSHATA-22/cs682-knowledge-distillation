'''
Block1  (Conv1(64,(3,3)) -> Conv2(64,(3,3)) -> Max Pooling(2,(2,2)))
Block2  (Conv1(128,(3,3)) -> Conv2(128,(3,3)) -> Max Pooling(2,(2,2)))
Last Block (Flatten -> FC1) 
'''

import torch.nn as nn
import torch.nn.functional as F
from utils.layers import *

class StudentCNN(nn.Module):

    def __init__(self, alpha, temperature, num_of_classes):
        super(StudentCNN, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),

            nn.AdaptiveAvgPool2d((14,14)),

            nn.Flatten(),
            nn.Linear(50176, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_of_classes)
        )  

    def forward(self, x):
        return self.network(x)


    def risk(self, Y, ta_preds, output):
        return loss_l2(Y, output) + loss_kl_divergence(output, Y, ta_preds, alpha=self.alpha, temperature=self.temperature)

    