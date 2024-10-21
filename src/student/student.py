'''
Block1  (Conv1(64,(3,3)) -> Conv2(64,(3,3)) -> Max Pooling(2,(2,2)))
Block2  (Conv1(128,(3,3)) -> Conv2(128,(3,3)) -> Max Pooling(2,(2,2)))
Last Block (Flatten -> FC1) 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
import seaborn as sns
from utils.layers import *

class StudentCNN(nn.Module):

    def __init__(self):
        super(StudentCNN, self).__init__()
        
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.activation_fn1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.activation_fn1_2 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.activation_fn2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.activation_fn2_2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.activation_fn4 = nn.ReLU()
        self.flatten_layer = nn.Flatten()
        self.fc1 = nn.Linear(8192, 1000)  

    def forward(self, X_data):
        x = self.activation_fn1_1(self.conv1_1(x))
        x = self.activation_fn1_2(self.conv1_2(x))
        x = self.maxPool1(x)

        x = self.activation_fn2_1(self.conv2_1(x))
        x = self.activation_fn2_2(self.conv2_2(x))
        x = self.maxPool2(x)
        
        x = self.flatten_layer(self.activation_fn4(x))
        x = self.fc1(x)
        
        return x


    def risk(self, Y, teacher_preds, output):
        return loss_l2(Y, output) + loss_kl_divergence(output, Y, teacher_preds, alpha=self.alpha, temperature=self.temperature)

    