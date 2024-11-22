import torch.nn as nn

from utils.layers import *

class Custom_Lite_TACNN(nn.Module):
    def __init__(self, alpha, temperature, num_of_classes):
        super(Custom_Lite_TACNN, self).__init__()
        
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        # )
        # self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            
        # )
        # self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        # )
        
        # self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(),
            nn.Linear(50176, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            # nn.Linear(4096, 4096),
            # nn.BatchNorm1d(4096),
            # nn.ReLU(),
            nn.Linear(4096, num_of_classes)
        )
        
        self.alpha = alpha
        # self.beta = beta
        self.temperature = temperature
        # self.l_feat = 0
        
    def forward(self, x):
        return self.block1(x)
    
    def risk(self, Y, teacher_preds, output):
        return loss_kl_divergence(output, Y, teacher_preds, alpha=self.alpha, temperature=self.temperature)
        # if self.training:
        #     x, teacher_cache = input
        #     self.l_feat=0
        # else:
        #     x = input
            
        # x = self.block1(x)
        # if self.training:
        #     div = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        #     self.l_feat += (torch.norm(teacher_cache["block1"] - x) / div)
        #     # print(self.l_feat)
        
        # x = self.block2(x)
        # if self.training:
        #     div = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        #     self.l_feat += (torch.norm(teacher_cache["block2"] - x) / div)
        #     # print(self.l_feat)
        #     # print("--------------------------------------------")
        
        # x = self.block3(x)
        
        # if self.training:
        #     div = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        #     self.l_feat += (torch.norm(teacher_cache["block3"] - x) / div)
        #     # print(self.l_feat)
        
        # x = self.block4(x)
        
        # if self.training:
        #     div = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        #     self.l_feat += (torch.norm(teacher_cache["block4"] - x) / div)
        #     # print(self.l_feat)
        
        # x = self.block5(x)
        
        # if self.training:      
        #     div = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        #     self.l_feat += (torch.norm(teacher_cache["block5"] - x) / div) 
        #     # print(self.l_feat)
        
        # x = self.block6(x)
        
        # if self.training:     
        #     div = x.shape[0] * x.shape[1] 
        #     self.l_feat += (torch.norm(teacher_cache["block6"] - x) / div)
        #     # print(self.l_feat)
            
        # if self.training: 
        #     del teacher_cache
            
        # return x
    
    # def loss(self, labels, teacher_output, student_output):
    #     # combined_teacher_output = get_combined_teacher_output(teacher_1_output, teacher_2_output, self.pi1, self.pi2)
    #     # print("----------------------L_feat here--------------------------")
    #     # print(self.l_feat)
        
    #     # print('Teacher L2 loss')
    #     # print(loss_l2(teacher_output, labels))
    #     # print('Student L2 loss')
    #     # print(loss_l2(student_output, labels))
        
    #     return self.alpha*(loss_l2(teacher_output, labels) + loss_l2(student_output, labels)) + self.beta*(loss_kl_divergence_with_logits(student_output, labels, teacher_output, self.temperature)) + self.l_feat