"""
逻辑回归的本质：建模条件概率

P(y=1|x) = σ(w^T x)
P(y=0|x) = 1 - σ(w^T x)

统一形式:
P(y|x) = σ(w^T x)^y · [1 - σ(w^T x)]^(1-y)
"""

import torch

def explain_likelihood():
    print("=" * 60)
    print("从最大似然到BCE")
    print("=" * 60)
    
    # 示例数据
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[0.0], [1.0], [1.0]])
    
    # 假设参数
    w = torch.tensor([[1.5]])
    
    # 预测概率
    y_pred = torch.sigmoid(x @ w)
    
    print("\n数据点:")
    for i in range(3):
        print(f"  x={x[i].item():.1f}, y={y[i].item():.0f}, "
              f"P(y|x)={y_pred[i].item():.4f}")
    
    # 似然函数 (连乘)
    likelihood = 1.0
    for i in range(3):
        p = y_pred[i].item()
        y_val = y[i].item()
        prob = (p ** y_val) * ((1-p) ** (1-y_val))
        likelihood *= prob
        print(f"\n  样本{i+1}: P(y={y_val:.0f}|x) = {prob:.6f}")
    
    print(f"\n似然函数 L(w) = {likelihood:.10f}")
    
    # 对数似然
    log_likelihood = torch.sum(
        y * torch.log(y_pred) + (1-y) * torch.log(1-y_pred)
    ).item()
    print(f"对数似然 log L(w) = {log_likelihood:.6f}")
    
    # 负对数似然 = BCE
    nll = -log_likelihood
    bce = torch.nn.functional.binary_cross_entropy(y_pred, y, reduction='sum').item()
    
    print(f"\n负对数似然 = {nll:.6f}")
    print(f"BCE损失     = {bce:.6f}")
    print(f"两者相等！   ✓")
    
    print("\n" + "=" * 60)
    print("结论: BCE = 负对数似然 = 最大似然估计的损失函数")
    print("这是有坚实统计学理论基础的！")
    print("=" * 60)

explain_likelihood()