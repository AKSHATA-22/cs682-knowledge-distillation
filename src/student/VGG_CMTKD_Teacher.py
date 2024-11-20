import torch.nn as nn
from utils.HWGQQuantizer import quantize

class VGG_CMTKD_Teacher(nn.Module):
    def __init__(self, bit_width, num_of_classes, teacher_idx):
        super(VGG_CMTKD_Teacher, self).__init__()
        
        # input - 64, 64, 3
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.activation1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(64)
        self.activation1_2 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 32, 32, 64
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.activation2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(128)
        self.activation2_2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 16, 16, 128
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm3_1 = nn.BatchNorm2d(256)
        self.activation3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm3_2 = nn.BatchNorm2d(256)
        self.activation3_2 = nn.ReLU()
        self.maxPool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 8, 8, 256
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm4_1 = nn.BatchNorm2d(512)
        self.activation4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm4_2 = nn.BatchNorm2d(512)
        self.activation4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm4_3 = nn.BatchNorm2d(512)
        self.activation4_3 = nn.ReLU()
        self.maxPool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 4, 4, 512
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm5_1 = nn.BatchNorm2d(512)
        self.activation5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm5_2 = nn.BatchNorm2d(512)
        self.activation5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm5_3 = nn.BatchNorm2d(512)
        self.activation5_3 = nn.ReLU()
        self.maxPool5 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 2, 2, 512
        self.flatten_layer = nn.Flatten()
        
        # input - 2048,
        self.fc1 = nn.Linear(25088, 4096)  
        self.activation_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)  
        self.activation_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, num_of_classes)
        
        
        # self.alpha = alpha
        # self.temperature = temperature
        self.bit_width = bit_width
        self.teacher_idx = teacher_idx
        self.cache = {}

    def forward(self, x):
        x = self.activation1_1(self.conv1_1(x))
        x = quantize(self.batchnorm1_1(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_1_1.pt")
        self.cache["conv1_1"] = x
        x = self.activation1_2(self.conv1_2(x))
        x = quantize(self.batchnorm1_2(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_1_2.pt")
        self.cache["conv1_2"] = x
        x = self.maxPool1(x)
        
        x = self.activation2_1(self.conv2_1(x))
        x = quantize(self.batchnorm2_1(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_2_1.pt")
        self.cache["conv2_1"] = x
        x = self.activation2_2(self.conv2_2(x))
        x = quantize(self.batchnorm2_2(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_2_2.pt")
        self.cache["conv2_2"] = x
        x = self.maxPool2(x)
        
        x = self.activation3_1(self.conv3_1(x))
        x = quantize(self.batchnorm3_1(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_3_1.pt")
        self.cache["conv3_1"] = x
        x = self.activation3_2(self.conv3_2(x))
        x = quantize(self.batchnorm3_2(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_3_2.pt")
        self.cache["conv3_2"] = x
        x = self.maxPool3(x)

        x = self.activation4_1(self.conv4_1(x))
        x = quantize(self.batchnorm4_1(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_1.pt")
        self.cache["conv4_1"] = x
        x = self.activation4_2(self.conv4_2(x))
        x = quantize(self.batchnorm4_2(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_2.pt")
        self.cache["conv4_2"] = x
        x = self.activation4_3(self.conv4_3(x))
        x = quantize(self.batchnorm4_3(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_3.pt")
        self.cache["conv4_3"] = x
        x = self.maxPool4(x)
        
        x = self.activation5_1(self.conv5_1(x))
        x = quantize(self.batchnorm5_1(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_1.pt")
        self.cache["conv5_1"] = x
        x = self.activation5_2(self.conv5_2(x))
        x = quantize(self.batchnorm5_2(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_2.pt")
        self.cache["conv5_2"] = x
        x = self.activation5_3(self.conv5_3(x))
        x = quantize(self.batchnorm5_3(x), self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_3.pt")
        self.cache["conv5_3"] = x
        x = self.maxPool5(x)
        
        x = self.flatten_layer(x)
        x = self.activation_fc1(self.fc1(x))
        x = quantize(x, self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/fc1.pt")
        self.cache["fc1"] = x
        x = self.activation_fc2(self.fc2(x))
        x = quantize(x, self.bit_width)
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/fc2.pt")
        self.cache["fc2"] = x
        x = self.fc3(x)
        x = quantize(x, self.bit_width)
        
        # x = self.classifier_activation(x)
        # self.cache["logit"] = x
        
        return x