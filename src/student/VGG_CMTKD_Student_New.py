import torch.nn as nn

from utils.HWGQQuantizer import quantize
from utils.layers import *

class VGG_CMTKD_Student_New(nn.Module):
    def __init__(self, alpha, beta, temperature, num_of_classes):
        super(VGG_CMTKD_Student_New, self).__init__()
        
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.block6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_of_classes)
        )
        # self.forwardCov1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1))
        # self.forwardCov2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        # self.forwardCov3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1))
        # self.forwardCov4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1))
        # self.forwardCov5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 1))
        # self.forwardCov6 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        # self.bit_width = bit_width
        
        # self.classifier_activation = nn.Softmax() 
        
        # self.alpha = alpha
        # self.beta = beta
        # self.temperature = temperature
        # self.bit_width = bit_width
        # # self.cache_T1 = cache_T1
        # # self.cache_T2 = cache_T2
        # self.pi1 = pi1
        # self.pi2 = pi2
        self.l_feat = 0
        

    def forward(self, input):
        if self.training:
            x, teacher_cache = input
            self.l_feat=0
        else:
            x = input
            
        x = self.block1(x)
        if self.training:
            # teacher_block1 = 
            # x = self.forwardCov1(x)
            self.l_feat += torch.norm(F.normalize(teacher_cache["block1"] - x).detach())
            # print(teacher_block1[0])
            # print(x[0])
            # print(x)
            # print(torch.norm(teacher_block1 - self.forwardCov1(x)).detach())
            print(self.l_feat)
            # print("--------------------------------------------")
        
        x = self.block2(x)
        if self.training:
            # teacher_block2 = 
            self.l_feat += torch.norm(F.normalize(teacher_cache["block2"] - x).detach())
            # print(x)
            # print(torch.norm(teacher_block2 - self.forwardCov2(x)).detach())
            print(self.l_feat)
            # print("--------------------------------------------")
        
        x = self.block3(x)
        
        if self.training:
            # teacher_block3 = 
            self.l_feat += torch.norm(F.normalize(teacher_cache["block3"] - x).detach())
            # print(x)
            # print(torch.norm(teacher_block3 - self.forwardCov3(x)).detach())
            print(self.l_feat)
            # print("--------------------------------------------")
        
        x = self.block4(x)
        
        if self.training:
            # teacher_block4 =        
            self.l_feat += torch.norm(F.normalize(teacher_cache["block4"] - x).detach())
            # print(x)
            # print(torch.norm(teacher_block4 - self.forwardCov4(x)).detach() )
            print(self.l_feat)
            # print("--------------------------------------------")
        
        x = self.block5(x)
        
        if self.training:
            # teacher_block5 =         
            self.l_feat += torch.norm(F.normalize(teacher_cache["block5"] - x).detach()) 
            # print(x)
            # print(torch.norm(teacher_block5 - self.forwardCov5(x)).detach() )
            print(self.l_feat)
            # print("--------------------------------------------")
        
        x = self.block6(x)
        
        if self.training:
            # teacher_block6 =        
            self.l_feat += torch.norm(F.normalize(teacher_cache["block6"] - x).detach())
            # print(torch.norm(teacher_block6.unsqueeze(1).unsqueeze(3) - self.forwardCov6(x.unsqueeze(1).unsqueeze(3))).detach()  )
            # print(teacher_block1)
            print(self.l_feat)
            # print("--------------------------------------------")
            
        # pi1, pi2 = self.pi1,self.pi2
        # # cache1, cache2 = self.cache_T1, self.cache_T2
        
        # x = self.activation1_1(self.conv1_1(x))
        # x = quantize(self.batchnorm1_1(x), self.bit_width)

        # cache1 = torch.load(f"cache/Teacher1/conv_1_1.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_1_1.pt")
        # F_T_11 = pi1 * cache1 + pi2 * cache2
        # # print(F_T_11.shape)
        # # print(x.shape)
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_11))
        # # print(l_feat.shape)

        # x = self.activation1_2(self.conv1_2(x))
        # x = quantize(self.batchnorm1_2(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_1_2.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_1_2.pt")
        # F_T_12 = pi1 * cache1 + pi2 * cache2
        # # print(F_T_12.shape)
        # # print(x.shape)
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_12))
        # # l_feat += nn.functional.normalize(x - F_T_12)

        # x = self.maxPool1_1(x)
        
        # x = self.activation2_1(self.conv2_1(x))
        # x = quantize(self.batchnorm2_1(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_2_1.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_2_1.pt")
        # F_T_21 = pi1 * cache1 + pi2 * cache2
        # # print(F_T_21)
        # # print(x.shape)
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_21))
        # # l_feat += nn.functional.normalize(x - F_T_21)

        # x = self.activation2_2(self.conv2_2(x))
        # x = quantize(self.batchnorm2_2(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_2_2.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_2_2.pt")
        # F_T_22 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_22))
        # # l_feat += nn.functional.normalize(x - F_T_22)

        # x = self.maxPool2_1(x)
        
        # x = self.activation3_1(self.conv3_1(x))
        # x = quantize(self.batchnorm3_1(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_3_1.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_3_1.pt")
        # F_T_31 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_31))
        # # l_feat += nn.functional.normalize(x - F_T_31)

        # x = self.activation3_2(self.conv3_2(x))
        # x = quantize(self.batchnorm3_2(x), self.bit_width)

        # cache1 = torch.load(f"cache/Teacher1/conv_3_2.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_3_2.pt")
        # F_T_32 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_32))
        # # l_feat += nn.functional.normalize(x - F_T_32)

        # x = self.maxPool3_1(x)

        # x = self.activation4_1(self.conv4_1(x))
        # x = quantize(self.batchnorm4_1(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_4_1.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_4_1.pt")
        # F_T_41 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_41))
        # # l_feat += nn.functional.normalize(x - F_T_41)


        # x = self.activation4_2(self.conv4_2(x))
        # x = quantize(self.batchnorm4_2(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_4_2.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_4_2.pt")
        # F_T_42 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_42))
        # # l_feat += nn.functional.normalize(x - F_T_42)

        # x = self.activation4_3(self.conv4_3(x))
        # x = quantize(self.batchnorm4_3(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_4_3.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_4_3.pt")
        # F_T_43 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_43))
        # # l_feat += nn.functional.normalize(x - F_T_43)


        # x = self.maxPool4_1(x)
        
        # x = self.activation5_1(self.conv5_1(x))
        # x = quantize(self.batchnorm5_1(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_5_1.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_5_1.pt")
        # F_T_51 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_51))
        # # l_feat += nn.functional.normalize(x - F_T_51)


        # x = self.activation5_2(self.conv5_2(x))
        # x = quantize(self.batchnorm5_2(x), self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/conv_5_2.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_5_2.pt")
        # F_T_52 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_52))
        # # l_feat += nn.functional.normalize(x - F_T_52)

        # x = self.activation5_3(self.conv5_3(x))
        # x = quantize(self.batchnorm5_3(x), self.bit_width)
       
        # cache1 = torch.load(f"cache/Teacher1/conv_5_3.pt")
        # cache2 = torch.load(f"cache/Teacher2/conv_5_3.pt")
        # F_T_53 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_53))
        # # l_feat += nn.functional.normalize(x - F_T_53)

        # x = self.maxPool5_1(x)
        
        # x = self.flatten_layer(x)
        # x = self.activation_fc1(self.fc1(x))
        # x = quantize(x, self.bit_width)
        
        # cache1 = torch.load(f"cache/Teacher1/fc1.pt")
        # cache2 = torch.load(f"cache/Teacher2/fc1.pt")
        # F_T_fc1 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_fc1))
        # # l_feat += nn.functional.normalize(x - F_T_fc1)

        # x = self.activation_fc2(self.fc2(x))
        # x = quantize(x, self.bit_width)

        # cache1 = torch.load(f"cache/Teacher1/fc2.pt")
        # cache2 = torch.load(f"cache/Teacher2/fc2.pt")
        # F_T_fc2 = pi1 * cache1 + pi2 * cache2
        # l_feat = torch.sum(nn.functional.normalize(x - F_T_fc2))
        # # l_feat += nn.functional.normalize(x - F_T_fc2)

        # x = self.fc3(x)
        # x = quantize(x, self.bit_width)
        
        # x = self.classifier_activation(x)
        

        # del F_T_11, F_T_12, F_T_21, F_T_22, F_T_31, F_T_32, F_T_41, F_T_42, F_T_43, F_T_51, F_T_52, F_T_53, F_T_fc1, F_T_fc2
        if self.training: 
            # self.l_feat += l_feat.detach()
            del teacher_cache
            
        return x
    
    def loss(self, labels, teacher_output, student_output):
        # combined_teacher_output = get_combined_teacher_output(teacher_1_output, teacher_2_output, self.pi1, self.pi2)
        print("----------------------Losses here--------------------------")

        print(self.l_feat)
        
        print('Teacher L2 loss')
        print(loss_l2(teacher_output, labels))
        print('Student L2 loss')
        print(loss_l2(student_output, labels))
        # comp1 = 
        # print(comp1)
        # comp2 = 
        # print(comp2)
        
        return self.alpha*(loss_l2(teacher_output, labels) + loss_l2(student_output, labels)) + self.beta*(loss_kl_divergence_with_logits(student_output, labels, teacher_output, self.temperature)) + self.l_feat