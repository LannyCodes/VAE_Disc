import torch
import torch.nn as nn

print("\n" + "=" * 70)
print("Dense Layer 结构")
print("=" * 70)

class DenseLayer(nn.Module):
    """
    单个Dense层的结构
    
    组合函数: BN → ReLU → Conv
    """
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, 
                     kernel_size=3, padding=1, bias=False)
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (可能是多个特征图的拼接)
        
        Returns:
            new_features: 新生成的特征 (growth_rate个通道)
        """
        new_features = self.layer(x)
        return new_features

# 示例
layer = DenseLayer(in_channels=64, growth_rate=12)
x = torch.randn(1, 64, 32, 32)
out = layer(x)

print(f"\n输入形状: {x.shape}")
print(f"输出形状: {out.shape}")
print(f"说明: 输出 {out.shape[1]} 个通道 (growth_rate=12)")

print("\n" + "=" * 70)
print("Dense Block 结构")
print("=" * 70)

class DenseBlock(nn.Module):
    """
    Dense Block: 多个Dense Layer的堆叠
    
    每层的输出都拼接到后续层的输入
    """
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 当前层的输入通道数
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(DenseLayer(layer_in_channels, growth_rate))
    
    def forward(self, x):
        """
        前向传播: 逐层拼接特征
        
        Args:
            x: 初始输入
        
        Returns:
            拼接后的所有特征
        """
        features = [x]  # 保存所有层的输出
        
        for layer in self.layers:
            # 拼接所有之前的特征
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        
        # 返回所有特征的拼接
        return torch.cat(features, dim=1)

# 示例
block = DenseBlock(num_layers=4, in_channels=64, growth_rate=12)
x = torch.randn(1, 64, 32, 32)
out = block(x)

print(f"\n输入形状: {x.shape}")
print(f"输出形状: {out.shape}")
print(f"说明: 64 + 4×12 = {out.shape[1]} 通道")

print("\n通道数增长:")
for i in range(4):
    channels = 64 + (i+1) * 12
    print(f"  Layer {i+1}后: {channels} 通道")

