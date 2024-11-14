import torch.nn as nn

from utils.HWGQQuantizer import quantize

class VGG_CMTKD_Teacher(nn.Module):
    def __init__(self, alpha, temperature, bit_width):
        super(VGG_CMTKD_Teacher, self).__init__()
        
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout = nn.Dropout(0.5)
                
        # input - 64, 64, 3
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        
        # input - 32, 32, 64
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        
        # input - 16, 16, 128
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        
        # input - 8, 8, 256
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        
        # input - 4, 4, 512
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        
        # input - 2, 2, 512
        self.flatten_layer = nn.Flatten()
        
        # input - 2048,
        self.fc1 = nn.Linear(2048, 1024)  
        self.fc2 = nn.Linear(1024, 1024)  
        self.fc3 = nn.Linear(1024, 1000)
        
        self.classifier_activation = nn.Softmax() 
        
        self.alpha = alpha
        self.temperature = temperature
        self.bit_width = bit_width
        self.cache = {}

    def forward(self, x):
        x = self.activation(self.conv1_1(x))
        x = quantize(x, self.bit_width)
        self.cache["conv1_1"] = x
        x = self.activation(self.conv1_2(x))
        x = quantize(x, self.bit_width)
        self.cache["conv1_2"] = x
        x = self.maxPool(x)
        
        x = self.activation(self.conv2_1(x))
        x = quantize(x, self.bit_width)
        self.cache["conv2_1"] = x
        x = self.activation(self.conv2_2(x))
        x = quantize(x, self.bit_width)
        self.cache["conv2_2"] = x
        x = self.maxPool(x)
        
        x = self.activation(self.conv3_1(x))
        x = quantize(x, self.bit_width)
        self.cache["conv3_1"] = x
        x = self.activation(self.conv3_2(x))
        x = quantize(x, self.bit_width)
        self.cache["conv3_2"] = x
        x = self.maxPool(x)

        x = self.activation(self.conv4_1(x))
        x = quantize(x, self.bit_width)
        self.cache["conv4_1"] = x
        x = self.activation(self.conv4_2(x))
        x = quantize(x, self.bit_width)
        self.cache["conv4_2"] = x
        x = self.activation(self.conv4_3(x))
        x = quantize(x, self.bit_width)
        self.cache["conv4_3"] = x
        x = self.maxPool(x)
        
        x = self.activation(self.conv5_1(x))
        x = quantize(x, self.bit_width)
        self.cache["conv5_1"] = x
        x = self.activation(self.conv5_2(x))
        x = quantize(x, self.bit_width)
        self.cache["conv5_2"] = x
        x = self.activation(self.conv5_3(x))
        x = quantize(x, self.bit_width)
        self.cache["conv5_3"] = x
        x = self.maxPool(x)
        
        x = self.flatten_layer(x)
        x = self.dropout(self.activation(self.fc1(x)))
        x = quantize(x, self.bit_width)
        self.cache["fc1"] = x
        x = self.dropout(self.activation(self.fc2(x)))
        x = quantize(x, self.bit_width)
        self.cache["fc2"] = x
        x = self.fc3(x)
        x = quantize(x, self.bit_width)
        
        x = self.classifier_activation(x)
        self.cache["logit"] = x
        
        return x