import torch.nn as nn
import torch 
from student.QuantizationLayer import QuantizationLayer
from utils.HWGQQuantizer import quantize
from student.QuantizeConv2D import QuantizeConv2D

class VGG_CMTKD_Teacher(nn.Module):
    def __init__(self, bit_width, num_of_classes, teacher_idx):
        super(VGG_CMTKD_Teacher, self).__init__()
        
        # input - 64, 64, 3
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.activation1_1 = nn.ReLU()
        self.quantize1_1 = QuantizationLayer(bit_width=bit_width)
        self.conv1_2 = QuantizeConv2D(bit_width=bit_width, in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(64)
        self.activation1_2 = nn.ReLU()
        self.quantize1_2 = QuantizationLayer(bit_width=bit_width)
        self.maxPool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 32, 32, 64
        self.conv2_1 = QuantizeConv2D(bit_width=bit_width, in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.activation2_1 = nn.ReLU()
        self.quantize2_1 = QuantizationLayer(bit_width=bit_width)
        self.conv2_2 = QuantizeConv2D(bit_width=bit_width, in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(128)
        self.activation2_2 = nn.ReLU()
        self.quantize2_2 = QuantizationLayer(bit_width=bit_width)
        self.maxPool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 16, 16, 128
        self.conv3_1 = QuantizeConv2D(bit_width=bit_width, in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm3_1 = nn.BatchNorm2d(256)
        self.activation3_1 = nn.ReLU()
        self.quantize3_1 = QuantizationLayer(bit_width=bit_width)
        self.conv3_2 = QuantizeConv2D(bit_width=bit_width, in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm3_2 = nn.BatchNorm2d(256)
        self.activation3_2 = nn.ReLU()
        self.quantize3_2 = QuantizationLayer(bit_width=bit_width)
        self.maxPool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 8, 8, 256
        self.conv4_1 = QuantizeConv2D(bit_width=bit_width, in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm4_1 = nn.BatchNorm2d(512)
        self.activation4_1 = nn.ReLU()
        self.quantize4_1 = QuantizationLayer(bit_width=bit_width)
        self.conv4_2 = QuantizeConv2D(bit_width=bit_width, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm4_2 = nn.BatchNorm2d(512)
        self.activation4_2 = nn.ReLU()
        self.quantize4_2 = QuantizationLayer(bit_width=bit_width)
        self.conv4_3 = QuantizeConv2D(bit_width=bit_width, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm4_3 = nn.BatchNorm2d(512)
        self.activation4_3 = nn.ReLU()
        self.quantize4_3 = QuantizationLayer(bit_width=bit_width)
        self.maxPool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 4, 4, 512
        self.conv5_1 = QuantizeConv2D(bit_width=bit_width, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm5_1 = nn.BatchNorm2d(512)
        self.activation5_1 = nn.ReLU()
        self.quantize5_1 = QuantizationLayer(bit_width=bit_width)
        self.conv5_2 = QuantizeConv2D(bit_width=bit_width, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm5_2 = nn.BatchNorm2d(512)
        self.activation5_2 = nn.ReLU()
        self.quantize5_2 = QuantizationLayer(bit_width=bit_width)
        self.conv5_3 = QuantizeConv2D(bit_width=bit_width, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm5_3 = nn.BatchNorm2d(512)
        self.activation5_3 = nn.ReLU()
        self.quantize5_3 = QuantizationLayer(bit_width=bit_width)
        self.maxPool5 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 2, 2, 512
        self.flatten_layer = nn.Flatten()
        
        # input - 2048,
        self.fc1 = nn.Linear(25088, 4096)  
        self.batcnorm_fc1 = nn.BatchNorm1d(4096)
        self.activation_fc1 = nn.ReLU()
        self.quantize_fc1 = QuantizationLayer(bit_width=bit_width)
        self.fc2 = nn.Linear(4096, 4096)  
        self.batcnorm_fc2 = nn.BatchNorm1d(4096)
        self.activation_fc2 = nn.ReLU()
        self.quantize_fc2 = QuantizationLayer(bit_width=bit_width)
        self.fc3 = nn.Linear(4096, num_of_classes)
        self.quantize_fc3 = QuantizationLayer(bit_width=bit_width)
        
        
        # self.alpha = alpha
        # self.temperature = temperature
        self.bit_width = bit_width
        self.teacher_idx = teacher_idx
        
        

    def forward(self, input):
        if self.training:
            x, epoch, batch_idx = input
        else:
            x = input
        # cache = {}
        x = self.batchnorm1_1(self.conv1_1(x))
        x = self.quantize1_1(self.activation1_1(x))
        # torch.save(x, f"self.cache/Teacher{self.teacher_idx}/conv_1_1.pt")
        # print(self.training)
        # print(cache.keys())
        # if self.training:
        #     cache["conv1_1"] = x
        x = self.batchnorm1_2(self.conv1_2(x))
        x = self.quantize1_2(self.activation1_2(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_1_2.pt")
        # if self.training:
        #     cache["conv1_2"] = x
        #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_1.pth')
        #     cache.clear()
            
        x = self.maxPool1(x)
        
        x = self.batchnorm2_1(self.conv2_1(x))
        x = self.quantize2_1(self.activation2_1(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_2_1.pt")
        # if self.training:
        #     cache["conv2_1"] = x
        x = self.batchnorm2_2(self.conv2_2(x))
        x = self.quantize2_2(self.activation2_2(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_2_2.pt")
        # if self.training:
        #     cache["conv2_2"] = x
        #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_2.pth')
        #     cache.clear()
        x = self.maxPool2(x)
        
        x = self.batchnorm3_1(self.conv3_1(x))
        x = self.quantize3_1(self.activation3_1(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_3_1.pt")
        # if self.training:
        #     cache["conv3_1"] = x
        x = self.batchnorm3_2(self.conv3_2(x))
        x = self.quantize3_2(self.activation3_2(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_3_2.pt")
        # if self.training:
        #     cache["conv3_2"] = x
        #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_3.pth')
        #     cache.clear()
        x = self.maxPool3(x)

        x = self.batchnorm4_1(self.conv4_1(x))
        x = self.quantize4_1(self.activation4_1(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_1.pt")
        # if self.training:
        #     cache["conv4_1"] = x
        x = self.batchnorm4_2(self.conv4_2(x))
        x = self.quantize4_2(self.activation4_2(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_2.pt")
        # if self.training:
        #     cache["conv4_2"] = x
        # print(x)
        x = self.batchnorm4_3(self.conv4_3(x))
        # print("-----")
        # print(x)
        x = self.quantize4_3(self.activation4_3(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_3.pt")
        # if self.training:
        #     cache["conv4_3"] = x
        #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_4.pth')
        #     cache.clear()
        x = self.maxPool4(x)
        
        x = self.batchnorm5_1(self.conv5_1(x))
        x = self.quantize5_1(self.activation5_1(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_1.pt")
        # if self.training:
        #     cache["conv5_1"] = x
        x = self.batchnorm5_2(self.conv5_2(x))
        x = self.quantize5_2(self.activation5_2(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_2.pt")
        # if self.training:
        #     cache["conv5_2"] = x
        x = self.batchnorm5_3(self.conv5_3(x))
        x = self.quantize5_3(self.activation5_3(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_3.pt")
        # if self.training:
        #     cache["conv5_3"] = x
        #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_5.pth')
        #     cache.clear()
        x = self.maxPool5(x)
        
        x = self.flatten_layer(x)
        x = self.batcnorm_fc1(self.fc1(x))
        x = self.quantize_fc1(self.activation_fc1(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/fc1.pt")
        # if self.training:
        #     cache["fc1"] = x
        x = self.batcnorm_fc2(self.fc2(x))
        x = self.quantize_fc2(self.activation_fc2(x))
        # torch.save(x, f"cache/Teacher{self.teacher_idx}/fc2.pt")
        # if self.training:
        #     cache["fc2"] = x
        #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_6.pth')
        #     cache.clear()
        x = self.fc3(x)
        # x = self.quantize_fc3(x)
        # print(cache.keys())
        # torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}.pth')
        # x = self.classifier_activation(x)
        # cache["logit"] = x
        # del cache
        return x