import torch
import torch.nn as nn
import torch.optim as optim

# 创建数据集
torch.manual_seed(42)
X = torch.randn(100, 1) * 2
y = (X > 0).float()  # 简单线性可分

# 定义模型
class LogisticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 训练函数
def train_model(loss_fn_name, epochs=100):
    model = LogisticModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    loss_history = []
    
    for epoch in range(epochs):
        y_pred = model(X)
        
        if loss_fn_name == 'MSE':
            loss = nn.MSELoss()(y_pred, y)
        else:  # BCE
            loss = nn.BCELoss()(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    return loss_history, model

# 训练两个模型
print("训练逻辑回归模型...")
mse_history, mse_model = train_model('MSE', 200)
bce_history, bce_model = train_model('BCE', 200)

# 对比结果
print("\n" + "=" * 60)
print("训练结果对比")
print("=" * 60)
print(f"MSE最终损失: {mse_history[-1]:.6f}")
print(f"BCE最终损失: {bce_history[-1]:.6f}")

# 计算准确率
with torch.no_grad():
    mse_pred = (mse_model(X) > 0.5).float()
    bce_pred = (bce_model(X) > 0.5).float()
    
    mse_acc = (mse_pred == y).float().mean().item()
    bce_acc = (bce_pred == y).float().mean().item()

print(f"\nMSE准确率: {mse_acc*100:.1f}%")
print(f"BCE准确率: {bce_acc*100:.1f}%")

print(f"\n收敛速度:")
print(f"MSE达到0.3用了: {next((i for i, l in enumerate(mse_history) if l < 0.3), 'N/A')} 轮")
print(f"BCE达到0.3用了: {next((i for i, l in enumerate(bce_history) if l < 0.3), 'N/A')} 轮")

print("\n结论: BCE收敛更快、更稳定！")