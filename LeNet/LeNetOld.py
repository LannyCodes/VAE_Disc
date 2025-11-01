import torch
import torch.nn as nn
import torch.nn.functional as F

print("\n" + "=" * 70)
print("LeNet-5 PyTorch实现")
print("=" * 70)

class LeNet5(nn.Module):
    """
    LeNet-5 网络
    
    论文原版使用sigmoid/tanh激活函数
    现代版本通常用ReLU
    """
    
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # Layer 1: 卷积层 C1
        # 输入: 1通道, 输出: 6通道, 卷积核: 5×5
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=0
        )
        
        # Layer 2: 池化层 S2
        # 原版用平均池化，现代用最大池化
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 3: 卷积层 C3
        # 输入: 6通道, 输出: 16通道, 卷积核: 5×5
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0
        )
        
        # Layer 4: 池化层 S4
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Layer 5: 全连接层 C5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # Layer 6: 全连接层 F6
        self.fc2 = nn.Linear(120, 84)
        
        # Layer 7: 输出层
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 (batch, 1, 32, 32)
        
        Returns:
            out: 输出logits (batch, 10)
        """
        # 输入: (batch, 1, 32, 32)
        
        # C1: 卷积 + 激活
        x = self.conv1(x)           # → (batch, 6, 28, 28)
        x = torch.tanh(x)           # 原版用tanh，也可用ReLU
        
        # S2: 池化
        x = self.pool1(x)           # → (batch, 6, 14, 14)
        
        # C3: 卷积 + 激活
        x = self.conv2(x)           # → (batch, 16, 10, 10)
        x = torch.tanh(x)
        
        # S4: 池化
        x = self.pool2(x)           # → (batch, 16, 5, 5)
        
        # Flatten: 展平
        x = x.view(x.size(0), -1)   # → (batch, 400)
        
        # C5: 全连接 + 激活
        x = self.fc1(x)             # → (batch, 120)
        x = torch.tanh(x)
        
        # F6: 全连接 + 激活
        x = self.fc2(x)             # → (batch, 84)
        x = torch.tanh(x)
        
        # OUTPUT: 输出层
        x = self.fc3(x)             # → (batch, 10)
        
        return x

# 创建模型
model = LeNet5()

print("\n模型结构:")
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")

print("\n" + "=" * 70)
print("数据流动详细跟踪")
print("=" * 70)

# 创建示例输入
x = torch.randn(1, 1, 32, 32)  # batch=1, channels=1, H=32, W=32

print(f"\n输入形状: {x.shape}")
print(f"  ↓")

# Layer 1: Conv1
x = model.conv1(x)
print(f"C1 卷积后: {x.shape}")
print(f"  说明: 32-5+1 = 28, 6个filter")
x = torch.tanh(x)
print(f"  激活后: {x.shape}")
print(f"  ↓")

# Layer 2: Pool1
x = model.pool1(x)
print(f"S2 池化后: {x.shape}")
print(f"  说明: 28/2 = 14 (下采样2倍)")
print(f"  ↓")

# Layer 3: Conv2
x = model.conv2(x)
print(f"C3 卷积后: {x.shape}")
print(f"  说明: 14-5+1 = 10, 16个filter")
x = torch.tanh(x)
print(f"  激活后: {x.shape}")
print(f"  ↓")

# Layer 4: Pool2
x = model.pool2(x)
print(f"S4 池化后: {x.shape}")
print(f"  说明: 10/2 = 5 (下采样2倍)")
print(f"  ↓")

# Flatten
x = x.view(x.size(0), -1)
print(f"Flatten: {x.shape}")
print(f"  说明: 16×5×5 = 400")
print(f"  ↓")

# Layer 5: FC1
x = model.fc1(x)
print(f"C5 全连接: {x.shape}")
x = torch.tanh(x)
print(f"  ↓")

# Layer 6: FC2
x = model.fc2(x)
print(f"F6 全连接: {x.shape}")
x = torch.tanh(x)
print(f"  ↓")

# Layer 7: Output
x = model.fc3(x)
print(f"输出层: {x.shape}")
print(f"  说明: 10个类别的logits")