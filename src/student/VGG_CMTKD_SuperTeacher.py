import torch.nn as nn
import torch 
from student.QuantizationLayer import QuantizationLayer
from utils.HWGQQuantizer import quantize
from student.QuantizeConv2D import QuantizeConv2D
from utils.layers import *

class VGG_CMTKD_SuperTeacher(nn.Module):
    
    def __init__(self, bit_width1, bit_width2, num_of_classes, pi1, pi2):
        super(VGG_CMTKD_SuperTeacher, self).__init__()
        
        # input - 64, 64, 3
        self.T1_block1 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width1, in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            QuantizeConv2D(bit_width=bit_width1, in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.T1_block2 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width1, in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            QuantizeConv2D(bit_width=bit_width1, in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.T1_block3 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width1, in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            QuantizeConv2D(bit_width=bit_width1, in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.T1_block4 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width1, in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            QuantizeConv2D(bit_width=bit_width1, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            QuantizeConv2D(bit_width=bit_width1, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.T1_block5 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width1, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            QuantizeConv2D(bit_width=bit_width1, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            QuantizeConv2D(bit_width=bit_width1, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.T1_block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096) , 
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            nn.Linear(4096, 4096)  ,
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width1),
            nn.Linear(4096, num_of_classes),
            QuantizationLayer(bit_width=bit_width1)
        )
        
        self.T2_block1 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width2, in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            QuantizeConv2D(bit_width=bit_width2, in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.T2_block2 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width2, in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            QuantizeConv2D(bit_width=bit_width2, in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.T2_block3 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width2, in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            QuantizeConv2D(bit_width=bit_width2, in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.T2_block4 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width2, in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            QuantizeConv2D(bit_width=bit_width2, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            QuantizeConv2D(bit_width=bit_width2, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.T2_block5 = nn.Sequential(
            QuantizeConv2D(bit_width=bit_width2, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            QuantizeConv2D(bit_width=bit_width2, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            QuantizeConv2D(bit_width=bit_width2, in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        self.T2_block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096) , 
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            nn.Linear(4096, 4096)  ,
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            QuantizationLayer(bit_width=bit_width2),
            nn.Linear(4096, num_of_classes),
            QuantizationLayer(bit_width=bit_width2)
        )
        # input - 2, 2, 512
        
        # self.alpha = alpha
        # self.temperature = temperature
        # self.bit_width = bit_width
        # self.teacher_idx = teacher_idx
        self.pi1 = pi1
        self.pi2 = pi2
        
        
    def forward(self, input):
        if self.training:
            x, epoch, batch_idx = input
        else:
            x = input
        x = get_combined_teacher_output(self.T1_block1(x), self.T2_block1(x), self.pi1, self.pi2)
        
        x = get_combined_teacher_output(self.T1_block2(x), self.T2_block2(x), self.pi1, self.pi2)
        x = get_combined_teacher_output(self.T1_block3(x), self.T2_block3(x), self.pi1, self.pi2)
        x = get_combined_teacher_output(self.T1_block4(x), self.T2_block4(x), self.pi1, self.pi2)
        x = get_combined_teacher_output(self.T1_block5(x), self.T2_block5(x), self.pi1, self.pi2)
        x = get_combined_teacher_output(self.T1_block6(x), self.T2_block6(x), self.pi1, self.pi2)
        return x
        # cache = {}
        # x = self.batchnorm1_1(self.conv1_1(x))
        # x = self.quantize1_1(self.activation1_1(x))
        # # torch.save(x, f"self.cache/Teacher{self.teacher_idx}/conv_1_1.pt")
        # # print(self.training)
        # # print(cache.keys())
        # # if self.training:
        # #     cache["conv1_1"] = x
        # x = self.batchnorm1_2(self.conv1_2(x))
        # x = self.quantize1_2(self.activation1_2(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_1_2.pt")
        # # if self.training:
        # #     cache["conv1_2"] = x
        # #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_1.pth')
        # #     cache.clear()
            
        # x = self.maxPool1(x)
        
        # x = self.batchnorm2_1(self.conv2_1(x))
        # x = self.quantize2_1(self.activation2_1(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_2_1.pt")
        # # if self.training:
        # #     cache["conv2_1"] = x
        # x = self.batchnorm2_2(self.conv2_2(x))
        # x = self.quantize2_2(self.activation2_2(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_2_2.pt")
        # # if self.training:
        # #     cache["conv2_2"] = x
        # #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_2.pth')
        # #     cache.clear()
        # x = self.maxPool2(x)
        
        # x = self.batchnorm3_1(self.conv3_1(x))
        # x = self.quantize3_1(self.activation3_1(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_3_1.pt")
        # # if self.training:
        # #     cache["conv3_1"] = x
        # x = self.batchnorm3_2(self.conv3_2(x))
        # x = self.quantize3_2(self.activation3_2(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_3_2.pt")
        # # if self.training:
        # #     cache["conv3_2"] = x
        # #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_3.pth')
        # #     cache.clear()
        # x = self.maxPool3(x)

        # x = self.batchnorm4_1(self.conv4_1(x))
        # x = self.quantize4_1(self.activation4_1(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_1.pt")
        # # if self.training:
        # #     cache["conv4_1"] = x
        # x = self.batchnorm4_2(self.conv4_2(x))
        # x = self.quantize4_2(self.activation4_2(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_2.pt")
        # # if self.training:
        # #     cache["conv4_2"] = x
        # # print(x)
        # x = self.batchnorm4_3(self.conv4_3(x))
        # # print("-----")
        # # print(x)
        # x = self.quantize4_3(self.activation4_3(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_4_3.pt")
        # # if self.training:
        # #     cache["conv4_3"] = x
        # #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_4.pth')
        # #     cache.clear()
        # x = self.maxPool4(x)
        
        # x = self.batchnorm5_1(self.conv5_1(x))
        # x = self.quantize5_1(self.activation5_1(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_1.pt")
        # # if self.training:
        # #     cache["conv5_1"] = x
        # x = self.batchnorm5_2(self.conv5_2(x))
        # x = self.quantize5_2(self.activation5_2(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_2.pt")
        # # if self.training:
        # #     cache["conv5_2"] = x
        # x = self.batchnorm5_3(self.conv5_3(x))
        # x = self.quantize5_3(self.activation5_3(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/conv_5_3.pt")
        # # if self.training:
        # #     cache["conv5_3"] = x
        # #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_5.pth')
        # #     cache.clear()
        # x = self.maxPool5(x)
        
        # x = self.flatten_layer(x)
        # x = self.batcnorm_fc1(self.fc1(x))
        # x = self.quantize_fc1(self.activation_fc1(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/fc1.pt")
        # # if self.training:
        # #     cache["fc1"] = x
        # x = self.batcnorm_fc2(self.fc2(x))
        # x = self.quantize_fc2(self.activation_fc2(x))
        # # torch.save(x, f"cache/Teacher{self.teacher_idx}/fc2.pt")
        # # if self.training:
        # #     cache["fc2"] = x
        # #     torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}_6.pth')
        # #     cache.clear()
        # x = self.fc3(x)
        # x = self.quantize_fc3(x)
        # # print(cache.keys())
        # # torch.save(cache, f'cache/Teacher1/cache_{epoch}_{batch_idx}.pth')
        # # x = self.classifier_activation(x)
        # # cache["logit"] = x
        # # del cache
        # return x