import torch.nn as nn

from utils.HWGQQuantizer import quantize
from utils.layers import *

class VGG_CMTKD_Student(nn.Module):
    def __init__(self, alpha, beta, temperature, bit_width, pi1, pi2):
        super(VGG_CMTKD_Student, self).__init__()
        
        # input - 64, 64, 3
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm1_1 = nn.BatchNorm2d(64)
        self.activation1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm1_2 = nn.BatchNorm2d(64)
        self.activation1_2 = nn.ReLU()
        self.maxPool1_1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 32, 32, 64
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm2_1 = nn.BatchNorm2d(128)
        self.activation2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm2_2 = nn.BatchNorm2d(128)
        self.activation2_2 = nn.ReLU()
        self.maxPool2_1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 16, 16, 128
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm3_1 = nn.BatchNorm2d(256)
        self.activation3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm3_2 = nn.BatchNorm2d(256)
        self.activation3_2 = nn.ReLU()
        self.maxPool3_1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
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
        self.maxPool4_1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
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
        self.maxPool5_1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        
        # input - 2, 2, 512
        self.flatten_layer = nn.Flatten()
        
        # input - 2048,
        self.fc1 = nn.Linear(2048, 1024)  
        self.activation_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)  
        self.activation_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1000)
        
        
        self.alpha = alpha
        self.temperature = temperature
        self.bit_width = bit_width
        self.cache = {}
        
        self.classifier_activation = nn.Softmax() 
        
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.bit_width = bit_width
        # self.cache_T1 = cache_T1
        # self.cache_T2 = cache_T2
        self.pi1 = pi1
        self.pi2 = pi2
        self.inter_loss = 0
        

    def forward(self, x):
        pi1, pi2 = self.pi1,self.pi2
        cache1, cache2 = self.cache_T1, self.cache_T2
        
        x = self.activation1_1(self.conv1_1(x))
        x = quantize(self.batchnorm1_1(x), self.bit_width)

        cache1 = torch.load(f"cache/Teacher1/conv_1_1.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_1_1.pt")
        F_T_11 = pi1 * cache1 + pi2 * cache2
        l_feat = nn.functional.normalize(x - F_T_11)

        x = self.activation1_2(self.conv1_2(x))
        x = quantize(self.batchnorm1_2(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_1_2.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_1_2.pt")
        F_T_12 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_12)

        x = self.maxPool1_1(x)
        
        x = self.activation2_1(self.conv2_1(x))
        x = quantize(self.batchnorm2_1(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_2_1.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_2_1.pt")
        F_T_21 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_21)

        x = self.activation2_2(self.conv2_2(x))
        x = quantize(self.batchnorm2_2(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_2_2.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_2_2.pt")
        F_T_22 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_22)

        x = self.maxPool2_1(x)
        
        x = self.activation3_1(self.conv3_1(x))
        x = quantize(self.batchnorm3_1(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_3_1.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_3_1.pt")
        F_T_31 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_31)

        x = self.activation3_2(self.conv3_2(x))
        x = quantize(self.batchnorm3_2(x), self.bit_width)

        cache1 = torch.load(f"cache/Teacher1/conv_3_2.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_3_2.pt")
        F_T_32 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_32)

        x = self.maxPool3_1(x)

        x = self.activation4_1(self.conv4_1(x))
        x = quantize(self.batchnorm4_1(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_4_1.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_4_1.pt")
        F_T_41 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_41)


        x = self.activation4_2(self.conv4_2(x))
        x = quantize(self.batchnorm4_2(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_4_2.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_4_2.pt")
        F_T_42 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_42)

        x = self.activation4_3(self.conv4_3(x))
        x = quantize(self.batchnorm4_3(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_4_3.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_4_3.pt")
        F_T_43 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_43)


        x = self.maxPool4_1(x)
        
        x = self.activation5_1(self.conv5_1(x))
        x = quantize(self.batchnorm5_1(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_5_1.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_5_1.pt")
        F_T_51 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_51)


        x = self.activation5_2(self.conv5_2(x))
        x = quantize(self.batchnorm5_2(x), self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/conv_5_2.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_5_2.pt")
        F_T_52 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_52)

        x = self.activation5_3(self.conv5_3(x))
        x = quantize(self.batchnorm5_3(x), self.bit_width)
       
        cache1 = torch.load(f"cache/Teacher1/conv_5_3.pt")
        cache2 = torch.load(f"cache/Teacher2/conv_5_3.pt")
        F_T_53 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_53)

        x = self.maxPool5_1(x)
        
        x = self.flatten_layer(x)
        x = self.activation_fc1(self.fc1(x))
        x = quantize(x, self.bit_width)
        
        cache1 = torch.load(f"cache/Teacher1/fc1.pt")
        cache2 = torch.load(f"cache/Teacher2/fc1.pt")
        F_T_fc1 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_fc1)

        x = self.activation_fc2(self.fc2(x))
        x = quantize(x, self.bit_width)

        cache1 = torch.load(f"cache/Teacher1/fc2.pt")
        cache2 = torch.load(f"cache/Teacher2/fc2.pt")
        F_T_fc2 = pi1 * cache1 + pi2 * cache2
        l_feat += nn.functional.normalize(x - F_T_fc2)

        x = self.fc3(x)
        x = quantize(x, self.bit_width)
        
        x = self.classifier_activation(x)
        
        self.inter_loss += l_feat

        del F_T_11, F_T_12, F_T_21, F_T_22, F_T_31, F_T_32, F_T_41, F_T_42, F_T_43, F_T_51, F_T_52, F_T_53, F_T_fc1, F_T_fc2
        del cache1, cache2
        return x
    
    def loss(self, labels, teacher_1_output, teacher_2_output, student_output):
        combined_teacher_output = get_combined_teacher_output(teacher_1_output, teacher_2_output, self.pi1, self.pi2)
        return self.alpha*(loss_l2(combined_teacher_output, labels) + loss_l2(student_output, labels)) + self.beta*(loss_kl_divergence_with_logits(student_output, labels, combined_teacher_output, self.temperature)) + self.inter_loss