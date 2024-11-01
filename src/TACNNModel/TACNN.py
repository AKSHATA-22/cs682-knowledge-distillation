import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import keras
from utils.layers import *

## We should take the num of blocks as a hyper parameter, get a graph of the accuracies vs the num of blocks used for the TA - This will help us decide the size of TA


# Define the CNN model
class TACNN(nn.Module):
    def __init__(self, alpha, temperature):
        super(TACNN, self).__init__()
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
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.activation_fn3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.activation_fn3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.maxPool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        self.flatten_layer = nn.Flatten()
        self.fc1 = nn.Linear(2058, 1024)  
        self.activation_fn4 = nn.ReLU()# assuming input images are 32x32
        self.fc2 = nn.Linear(1024, 1000)
        self.classifier_activation = nn.Softmax()  # assuming 10 classes
        
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, x):
        x = self.activation_fn1_1(self.conv1_1(x))
        x = self.activation_fn1_2(self.conv1_2(x))
        x = self.maxPool1(x)

        x = self.activation_fn2_1(self.conv2_1(x))
        x = self.activation_fn2_2(self.conv2_2(x))
        x = self.maxPool2(x)

        x = self.activation_fn3_1(self.conv3_1(x))
        x = self.activation_fn3_2(self.conv3_2(x))
        x = self.activation_fn3_2(self.conv3_3(x))
        x = self.maxPool3(x)  

        x = self.flatten_layer(x)

        x = self.fc1(x)
        x = self.activation_fn4(x)

        x = self.fc2(x)
        x = self.classifier_activation(x)
        
        return x
    
    def risk(self, Y, teacher_preds, output):
        return loss_l2(Y, output) + loss_kl_divergence(output, Y, teacher_preds, alpha=self.alpha, temperature=self.temperature)