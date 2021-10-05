import math
import torch 
import torch.nn as nn
import torch.functional as F
import torchvision

class Basic_CNN(nn.Module):
    def __init__(self):
        super(Basic_CNN, self).__init__()
        """
        Image size (256, 256)
        Input convolution (3, 256, 256)
        Conv1 output (8, 128, 128)
        Conv2 output (8, 64, 64)
        Conv3 output (16, 32, 32)
        Conv4 output (16, 8, 8)
        """
        self.conv1 = nn.Conv2d(3, 8, 3, padding= 1)
        self.conv2 = nn.Conv2d(8, 8, 5, padding= 2)
        self.conv3 = nn.Conv2d(8, 16, 5, padding= 2)
        self.conv4 = nn.Conv2d(16, 16, 5, padding= 2)

        self.act_conv = nn.ReLU(inplace= True)
        self.act_fc = nn.LeakyReLU(0.1)

        self.bn8 = nn.BatchNorm2d(8)
        self.bn16 = nn.BatchNorm2d(16)

        self.mp2 = nn.MaxPool2d(kernel_size= (2, 2))
        self.mp4 = nn.MaxPool2d(kernel_size= (4, 4))

        self.drop = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(16*8*8, 16)
        self.fc2 = nn.Linear(16, 2)

        # self.log = nn.Sigmoid()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.act_conv(x)
        x = self.bn8(x)
        x = self.mp2(x)

        x = self.conv2(x)
        x = self.act_conv(x)
        x = self.bn8(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.act_conv(x)
        x = self.bn16(x)
        x = self.mp2(x)

        x = self.conv4(x)
        x = self.act_conv(x)
        x = self.bn16(x)
        x = self.mp4(x)

        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act_fc(x)
        
        x = self.drop(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
    
        return x

