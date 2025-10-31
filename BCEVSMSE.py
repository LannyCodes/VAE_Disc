import torch
import numpy as np
import matplotlib.pyplot as plt

# 逻辑回归模型: y = sigmoid(w*x + b)
# 简化版: y = sigmoid(w*x), 只有一个参数 w

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# 真实数据点
x = torch.tensor([2.0])  # 输入
y_true = torch.tensor([1.0])  # 真实标签

# 测试不同的 w 值
w_values = torch.linspace(-5, 5, 200)

# 计算两种损失
mse_losses = []
bce_losses = []

for w in w_values:
    # 预测值
    y_pred = sigmoid(w * x)
    
    # MSE 损失
    mse = ((y_pred - y_true) ** 2).item()
    mse_losses.append(mse)
    
    # BCE 损失（负对数似然）
    bce = -(y_true * torch.log(y_pred + 1e-8)).item()
    bce_losses.append(bce)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(w_values.numpy(), mse_losses, 'r-', linewidth=2)
plt.xlabel('参数 w')
plt.ylabel('MSE Loss')
plt.title('MSE + Sigmoid: 非凸！多个局部极值')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(w_values.numpy(), bce_losses, 'b-', linewidth=2)
plt.xlabel('参数 w')
plt.ylabel('BCE Loss')
plt.title('BCE (负对数似然): 凸函数！单一最优解')
plt.grid(True)

plt.tight_layout()
plt.savefig('/tmp/convexity_comparison.png', dpi=150)
print("图像已保存")

# 检查凸性（二阶导数）
print("\n=== 凸性分析 ===")
w_test = torch.tensor([0.5], requires_grad=True)
y_pred = sigmoid(w_test * x)

# MSE的二阶导数
mse_loss = (y_pred - y_true) ** 2
mse_grad = torch.autograd.grad(mse_loss, w_test, create_graph=True)[0]
mse_hessian = torch.autograd.grad(mse_grad, w_test)[0]
print(f"MSE 二阶导数 (w=0.5): {mse_hessian.item():.6f}")

# BCE的二阶导数
w_test = torch.tensor([0.5], requires_grad=True)
y_pred = sigmoid(w_test * x)
bce_loss = -(y_true * torch.log(y_pred))
bce_grad = torch.autograd.grad(bce_loss, w_test, create_graph=True)[0]
bce_hessian = torch.autograd.grad(bce_grad, w_test)[0]
print(f"BCE 二阶导数 (w=0.5): {bce_hessian.item():.6f}")