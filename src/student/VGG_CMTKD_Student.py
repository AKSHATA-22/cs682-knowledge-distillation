import torch.nn as nn

from utils.HWGQQuantizer import quantize

class VGG_CMTKD_Student(nn.Module):
    def __init__(self, alpha, beta, temperature, bit_width, cache_T1, cache_T2, pi1, pi2):
        super(VGG_CMTKD_Student, self).__init__()
        
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
        self.beta = beta
        self.temperature = temperature
        self.bit_width = bit_width
        self.cache_T1 = cache_T1
        self.cache_T2 = cache_T2
        self.pi1 = pi1
        self.pi2 = pi2
        

    def forward(self, x):
        pi1, pi2 = self.pi1,self.pi2
        cache1, cache2 = self.cache_T1, self.cache_T2
        
        x = self.activation(self.conv1_1(x))
        x = quantize(x, self.bit_width)

        F_T_11 = pi1 * cache1["conv1_1"] + pi2 * cache2["conv1_1"]
        l_feat = nn.functional.normalize(x - F_T_11)

        x = self.activation(self.conv1_2(x))
        x = quantize(x, self.bit_width)
        
        F_T_12 = pi1 * cache1["conv1_2"] + pi2 * cache2["conv1_2"]
        l_feat += nn.functional.normalize(x - F_T_12)

        x = self.maxPool(x)
        
        x = self.activation(self.conv2_1(x))
        x = quantize(x, self.bit_width)
        
        F_T_21 = pi1 * cache1["conv2_1"] + pi2 * cache2["conv2_1"]
        l_feat += nn.functional.normalize(x - F_T_21)

        x = self.activation(self.conv2_2(x))
        x = quantize(x, self.bit_width)
        
        F_T_22 = pi1 * cache1["conv2_2"] + pi2 * cache2["conv2_2"]
        l_feat += nn.functional.normalize(x - F_T_22)

        x = self.maxPool(x)
        
        x = self.activation(self.conv3_1(x))
        x = quantize(x, self.bit_width)
        
        F_T_31 = pi1 * cache1["conv3_1"] + pi2 * cache2["conv3_1"]
        l_feat += nn.functional.normalize(x - F_T_31)

        x = self.activation(self.conv3_2(x))
        x = quantize(x, self.bit_width)

        F_T_32 = pi1 * cache1["conv3_2"] + pi2 * cache2["conv3_2"]
        l_feat += nn.functional.normalize(x - F_T_32)

        x = self.maxPool(x)

        x = self.activation(self.conv4_1(x))
        x = quantize(x, self.bit_width)
        
        F_T_41 = pi1 * cache1["conv4_1"] + pi2 * cache2["conv4_1"]
        l_feat += nn.functional.normalize(x - F_T_41)


        x = self.activation(self.conv4_2(x))
        x = quantize(x, self.bit_width)
        
        F_T_42 = pi1 * cache1["conv4_2"] + pi2 * cache2["conv4_2"]
        l_feat += nn.functional.normalize(x - F_T_42)

        x = self.activation(self.conv4_2(x))
        x = quantize(x, self.bit_width)
        
        F_T_43 = pi1 * cache1["conv4_3"] + pi2 * cache2["conv4_3"]
        l_feat += nn.functional.normalize(x - F_T_43)


        x = self.maxPool(x)
        
        x = self.activation(self.conv5_1(x))
        x = quantize(x, self.bit_width)
        
        F_T_51 = pi1 * cache1["conv5_1"] + pi2 * cache2["conv5_1"]
        l_feat += nn.functional.normalize(x - F_T_51)


        x = self.activation(self.conv5_2(x))
        x = quantize(x, self.bit_width)
        
        F_T_52 = pi1 * cache1["conv5_2"] + pi2 * cache2["conv5_2"]
        l_feat += nn.functional.normalize(x - F_T_52)

        x = self.activation(self.conv5_2(x))
        x = quantize(x, self.bit_width)
       
        F_T_53 = pi1 * cache1["conv5_3"] + pi2 * cache2["conv5_3"]
        l_feat += nn.functional.normalize(x - F_T_53)

        x = self.maxPool(x)
        
        x = self.flatten_layer(x)
        x = self.dropout(self.activation(self.fc1(x)))
        
        F_T_fc1 = pi1 * cache1["fc1"] + pi2 * cache2["fc1"]
        l_feat += nn.functional.normalize(x - F_T_fc1)

        x = self.dropout(self.activation(self.fc2(x)))
        x = quantize(x, self.bit_width)

        F_T_fc2 = pi1 * cache1["fc2"] + pi2 * cache2["fc2"]
        l_feat += nn.functional.normalize(x - F_T_fc2)

        x = self.fc3(x)
        x = quantize(x, self.bit_width)
        
        x = self.classifier_activation(x)
        
        
        return x