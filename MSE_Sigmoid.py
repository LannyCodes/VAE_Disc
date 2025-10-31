import torch
import numpy as np
import matplotlib.pyplot as plt

# 简单逻辑回归: y = sigmoid(w*x)
x = torch.tensor([1.0])  # 输入
y_true = torch.tensor([1.0])  # 真实标签

# 测试不同参数值
w_range = torch.linspace(-10, 10, 300)

mse_losses = []
bce_losses = []

for w in w_range:
    y_pred = torch.sigmoid(w * x)
    
    # MSE损失
    mse = ((y_pred - y_true) ** 2).item()
    mse_losses.append(mse)
    
    # BCE损失
    bce = -(y_true * torch.log(y_pred + 1e-10)).item()
    bce_losses.append(bce)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MSE损失曲线
axes[0].plot(w_range.numpy(), mse_losses, 'r-', linewidth=2.5)
axes[0].set_xlabel('参数 w', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('MSE + Sigmoid: 非凸函数\n有平坦区域，难以优化', fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# BCE损失曲线
axes[1].plot(w_range.numpy(), bce_losses, 'b-', linewidth=2.5)
axes[1].set_xlabel('参数 w', fontsize=12)
axes[1].set_ylabel('BCE Loss', fontsize=12)
axes[1].set_title('BCE: 凸函数\n单调递减，全局最优', fontsize=13)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.savefig('./mse_sigmoid.png', dpi=150)
print("图像已保存")

print("=" * 60)
print("观察损失函数形状:")
print("=" * 60)
print("MSE: 在 w < 0 时有平坦区域（损失≈1），梯度接近0")
print("BCE: 单调递减，始终有明确的下降方向")