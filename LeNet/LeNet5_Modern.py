import torch
import torch.nn as nn
import torch.nn.functional as F

print("\n" + "=" * 70)
print("LeNet-5 现代改进版")
print("=" * 70)

class LeNet5_Modern(nn.Module):
    """
    LeNet-5 现代改进版
    
    改进:
    1. 使用ReLU代替tanh (收敛更快)
    2. 使用MaxPool代替AvgPool (效果更好)
    3. 添加Dropout防止过拟合
    4. 使用BatchNorm加速训练
    """
    
    def __init__(self, num_classes=10):
        super(LeNet5_Modern, self).__init__()
        
        self.features = nn.Sequential(
            # C1: Conv 1→6
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6),
            
            # S2: MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # C3: Conv 6→16
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            # S4: MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            # C5: FC 400→120
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # F6: FC 120→84
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # Output: FC 84→10
            nn.Linear(84, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 创建现代版模型
modern_model = LeNet5_Modern()

print("\n现代改进版模型:")
print(modern_model)

print("\n改进对比:")
print(f"{'特性':<20} {'原版':<20} {'现代版':<20}")
print("-" * 60)
print(f"{'激活函数':<20} {'Tanh':<20} {'ReLU':<20}")
print(f"{'池化方式':<20} {'AvgPool':<20} {'MaxPool':<20}")
print(f"{'正则化':<20} {'无':<20} {'Dropout + BatchNorm':<20}")
print(f"{'收敛速度':<20} {'较慢':<20} {'更快':<20}")