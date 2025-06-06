import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCNN(nn.Module):
    """Simple fully connected neural network for MNIST"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)
        self.dp = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.dp(x)
        x = F.relu(self.fc2(x)); x = self.dp(x)
        x = F.relu(self.fc3(x)); x = self.dp(x)
        x = F.relu(self.fc4(x)); x = self.dp(x)
        x = F.relu(self.fc5(x))
        return self.fc6(x)

def _conv_bn_relu(inp: int, out: int, pool: bool = False, kernel_size: int = 3, 
                  padding: int = 1, stride: int = 1) -> nn.Sequential:
    layers = [nn.Conv2d(inp, out, kernel_size, stride, padding, bias=False), 
              nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    """ResNet9 architecture for MNIST"""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.c1 = _conv_bn_relu(in_channels, 64)
        self.c2 = _conv_bn_relu(64, 128, pool=True)
        self.r1 = nn.Sequential(_conv_bn_relu(128, 128), _conv_bn_relu(128, 128))
        self.c3 = _conv_bn_relu(128, 256, pool=True)
        self.c4 = _conv_bn_relu(256, 512, pool=True)
        self.r2 = nn.Sequential(_conv_bn_relu(512, 512), _conv_bn_relu(512, 512))
        self.classifier_head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), 
                                           nn.Dropout(0.2), nn.Linear(512, num_classes))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x); x = self.c2(x); x = self.r1(x) + x
        x = self.c3(x); x = self.c4(x); x = self.r2(x) + x
        return self.classifier_head(x)

class SimpleFCNNCompressed(nn.Module):
    """Compressed version of SimpleFCNN with reduced hidden layers"""
    def __init__(self, k2:int, k3:int, k4:int, k5:int, p_dropout:float=0.2):
        super().__init__()
        self.k2, self.k3, self.k4, self.k5 = k2, k3, k4, k5
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, k2)
        self.fc3 = nn.Linear(k2, k3)
        self.fc4 = nn.Linear(k3, k4)
        self.fc5 = nn.Linear(k4, k5)
        self.fc6 = nn.Linear(k5, 10)
        self.dp = nn.Dropout(p_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = self.dp(x)
        x = F.relu(self.fc2(x)); x = self.dp(x)
        x = F.relu(self.fc3(x)); x = self.dp(x)
        x = F.relu(self.fc4(x)); x = self.dp(x)
        x = F.relu(self.fc5(x))
        return self.fc6(x)