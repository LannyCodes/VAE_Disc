"""
KL散度的直观理解：

假设有两个概率分布:
- P(x): 真实分布
- Q(x): 近似分布

KL散度回答：
"如果我用Q来近似P，会损失多少信息？"

数学定义:
KL(P || Q) = Σ P(x) · log(P(x) / Q(x))
           = Σ P(x) · [log P(x) - log Q(x)]

或者连续形式:
KL(P || Q) = ∫ p(x) · log(p(x) / q(x)) dx
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("KL散度：信息论的核心概念")
print("=" * 70)

print("""
核心思想:
- 如果 P 和 Q 完全相同 → KL = 0
- P 和 Q 差异越大 → KL 越大
- KL散度不对称: KL(P||Q) ≠ KL(Q||P)
- KL散度总是 ≥ 0
""")

print("\n" + "=" * 70)
print("示例1: 离散概率分布的KL散度")
print("=" * 70)

# 掷骰子的例子
print("\n场景: 两个骰子的概率分布")

# 真实分布P: 公平骰子
P = torch.tensor([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

# 近似分布Q1: 稍微不公平
Q1 = torch.tensor([0.15, 0.15, 0.18, 0.18, 0.17, 0.17])

# 近似分布Q2: 很不公平
Q2 = torch.tensor([0.4, 0.1, 0.1, 0.1, 0.1, 0.2])

def kl_divergence_discrete(p, q):
    """计算离散KL散度"""
    return torch.sum(p * torch.log(p / q)).item()

kl_pq1 = kl_divergence_discrete(P, Q1)
kl_pq2 = kl_divergence_discrete(P, Q2)

print(f"\n真实分布 P: {P.tolist()}")
print(f"近似分布 Q1: {Q1.tolist()}")
print(f"KL(P||Q1) = {kl_pq1:.6f}  ← Q1 接近 P，KL 小")

print(f"\n近似分布 Q2: {Q2.tolist()}")
print(f"KL(P||Q2) = {kl_pq2:.6f}  ← Q2 远离 P，KL 大")

print("\n解释: KL散度越小，两个分布越接近！")

print("\n" + "=" * 70)
print("VAE中KL散度的代码实现")
print("=" * 70)

def vae_kl_loss(mu, log_var):
    """
    计算VAE的KL散度损失
    
    Args:
        mu: 均值 (batch, latent_dim)
        log_var: 对数方差 (batch, latent_dim)
    
    Returns:
        kl_loss: KL散度
    """
    # 公式: KL = 0.5 * Σ[μ² + σ² - log(σ²) - 1]
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl

# 示例
batch_size = 4
latent_dim = 3

mu = torch.randn(batch_size, latent_dim)
log_var = torch.randn(batch_size, latent_dim)

kl_loss = vae_kl_loss(mu, log_var)

print(f"\n示例计算:")
print(f"batch_size = {batch_size}, latent_dim = {latent_dim}")
print(f"\nμ (均值):")
print(mu)
print(f"\nlog_var (对数方差):")
print(log_var)
print(f"\nKL散度损失 = {kl_loss.item():.4f}")

# 逐项分解
print("\n" + "=" * 70)
print("公式分解")
print("=" * 70)

mu_squared = mu.pow(2)
var = log_var.exp()

print(f"\nμ² 项:")
print(mu_squared)

print(f"\nσ² = exp(log_var) 项:")
print(var)

print(f"\nlog_var 项:")
print(log_var)

# 完整公式
kl_manual = -0.5 * torch.sum(1 + log_var - mu_squared - var)
print(f"\n手动计算 KL = {kl_manual.item():.4f}")
print(f"函数计算 KL = {kl_loss.item():.4f}")
print(f"误差 = {abs(kl_manual.item() - kl_loss.item()):.2e}")