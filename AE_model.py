import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """标准自编码器 (AE)"""
    
    def __init__(self, latent_dim=20):
        super(Autoencoder, self).__init__()
        
        # ============================================
        # Encoder (编码器)
        # ============================================
        
        # 卷积层 1: 1→32 channels, 28×28→14×14
        self.enc_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # 卷积层 2: 32→64 channels, 14×14→7×7
        self.enc_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        
        # 全连接层 1: 3136→128
        self.enc_fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # 全连接层 2 (瓶颈层): 128→20
        self.enc_fc2 = nn.Linear(128, latent_dim)
        
        # ============================================
        # Decoder (解码器)
        # ============================================
        
        # 全连接层 1: 20→128
        self.dec_fc1 = nn.Linear(latent_dim, 128)
        
        # 全连接层 2: 128→3136
        self.dec_fc2 = nn.Linear(128, 64 * 7 * 7)
        
        # 转置卷积 1: 64→32 channels, 7×7→14×14
        self.dec_conv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        
        # 转置卷积 2: 32→1 channels, 14×14→28×28
        self.dec_conv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
    
    def encode(self, x):
        """
        编码器: 图像 → 潜在向量
        
        Args:
            x: 输入图像 (batch, 1, 28, 28)
            
        Returns:
            z: 潜在向量 (batch, latent_dim)
        """
        # x: (batch, 1, 28, 28)
        
        h = F.relu(self.enc_conv1(x))   # → (batch, 32, 14, 14)
        h = F.relu(self.enc_conv2(h))   # → (batch, 64, 7, 7)
        h = h.view(h.size(0), -1)       # → (batch, 3136)  Flatten
        h = F.relu(self.enc_fc1(h))     # → (batch, 128)
        z = self.enc_fc2(h)             # → (batch, 20)  瓶颈层
        
        # 注意：这里没有激活函数，z 可以是任意实数
        return z
    
    def decode(self, z):
        """
        解码器: 潜在向量 → 重建图像
        
        Args:
            z: 潜在向量 (batch, latent_dim)
            
        Returns:
            x_recon: 重建图像 (batch, 1, 28, 28)
        """
        # z: (batch, 20)
        
        h = F.relu(self.dec_fc1(z))     # → (batch, 128)
        h = F.relu(self.dec_fc2(h))     # → (batch, 3136)
        h = h.view(h.size(0), 64, 7, 7) # → (batch, 64, 7, 7)  Reshape
        h = F.relu(self.dec_conv1(h))   # → (batch, 32, 14, 14)
        x_recon = torch.sigmoid(self.dec_conv2(h))  # → (batch, 1, 28, 28)
        
        # Sigmoid 将输出限制在 [0, 1]，表示像素强度
        return x_recon
    
    def forward(self, x):
        """
        前向传播: 编码 → 解码
        
        Args:
            x: 输入图像 (batch, 1, 28, 28)
            
        Returns:
            x_recon: 重建图像 (batch, 1, 28, 28)
            z: 潜在向量 (batch, latent_dim)
        """
        z = self.encode(x)          # 编码
        x_recon = self.decode(z)    # 解码
        return x_recon, z


# ============================================
# 损失函数
# ============================================

def ae_loss(x_recon, x):
    """
    AE 损失 = 重建损失 (只有一个！)
    
    Args:
        x_recon: 重建图像 (batch, 1, 28, 28)
        x: 原始图像 (batch, 1, 28, 28)
        
    Returns:
        loss: 重建损失
    """
    # 方法 1: Binary Cross Entropy (BCE)
    # 适用于像素值在 [0, 1] 范围内
    loss_bce = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # 方法 2: Mean Squared Error (MSE)
    # 也常用于重建任务
    loss_mse = F.mse_loss(x_recon, x, reduction='sum')
    
    # 通常使用 BCE，因为图像像素是 [0, 1]
    return loss_bce


# ============================================
# 使用示例
# ============================================

# 创建模型
model = Autoencoder(latent_dim=20)

# 打印模型结构
print("=" * 80)
print("Autoencoder 模型结构")
print("=" * 80)
print(model)
print()

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
print()

# 输入图像 (batch_size=8, channels=1, height=28, width=28)
# 注意：图像数据需要在 [0, 1] 范围内
x = torch.rand(8, 1, 28, 28)  # 使用 rand 生成 [0, 1] 范围的数据

# 前向传播
x_recon, z = model(x)

# 计算损失
loss = ae_loss(x_recon, x)

print("=" * 80)
print("数据流动")
print("=" * 80)
print(f"输入图像形状:   {x.shape}")          # (8, 1, 28, 28)
print(f"潜在向量形状:   {z.shape}")          # (8, 20)
print(f"重建图像形状:   {x_recon.shape}")    # (8, 1, 28, 28)
print(f"重建损失:       {loss.item():.2f}")
print()

# 查看压缩比例
original_size = 28 * 28  # 784
latent_size = 20
compression_ratio = original_size / latent_size
print("=" * 80)
print("压缩信息")
print("=" * 80)
print(f"原始维度:   {original_size}")
print(f"潜在维度:   {latent_size}")
print(f"压缩比例:   {compression_ratio:.1f}:1")
print(f"压缩率:     {(1 - latent_size/original_size)*100:.1f}%")