import torch

# 计算梯度公式
print("=" * 60)
print("梯度公式对比")
print("=" * 60)

# 设定: y = sigmoid(w*x), 真实标签 = 1
x = torch.tensor([2.0])
y_true = torch.tensor([1.0])

# 测试3个场景
test_cases = [
    (-5.0, "预测很差"),
    (0.0, "不确定"),
    (5.0, "预测很好")
]

for w_val, desc in test_cases:
    print(f"\n{desc}: w = {w_val}")
    print("-" * 60)
    
    # MSE梯度
    w = torch.tensor([w_val], requires_grad=True)
    y_pred = torch.sigmoid(w * x)
    mse_loss = (y_pred - y_true) ** 2
    mse_loss.backward()
    mse_grad = w.grad.item()
    
    # BCE梯度
    w = torch.tensor([w_val], requires_grad=True)
    y_pred = torch.sigmoid(w * x)
    bce_loss = -(y_true * torch.log(y_pred))
    bce_loss.backward()
    bce_grad = w.grad.item()
    
    # Sigmoid导数
    sigma_val = torch.sigmoid(torch.tensor([w_val * x.item()])).item()
    sigma_derivative = sigma_val * (1 - sigma_val)
    
    print(f"  y_pred = {sigma_val:.6f}")
    print(f"  σ'(wx) = {sigma_derivative:.6f}")
    print(f"  MSE梯度 = {mse_grad:.6f}  ← 包含 σ'(wx)，可能很小！")
    print(f"  BCE梯度 = {bce_grad:.6f}  ← 不包含 σ'(wx)，稳定！")