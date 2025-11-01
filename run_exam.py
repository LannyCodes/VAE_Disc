import torch
import numpy as np

print("\n" + "=" * 70)
print("线性回归 → 逻辑回归 的演变")
print("=" * 70)

# 示例数据
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 线性回归
w_linear = torch.tensor([2.0])
b_linear = torch.tensor([1.0])
y_linear = w_linear * x + b_linear

print("\n【线性回归】")
print(f"公式: y = wx + b")
print(f"输出: {y_linear.tolist()}")
print(f"范围: (-∞, +∞)")
print(f"用途: 预测连续值（如房价）")
print(f"任务: 回归 ✓")

# 逻辑回归
w_logistic = torch.tensor([1.0])
b_logistic = torch.tensor([-2.5])
z = w_logistic * x + b_logistic
y_logistic = torch.sigmoid(z)

print("\n【逻辑回归】")
print(f"公式: y = σ(wx + b)")
print(f"线性部分 z = wx + b: {z.tolist()}")
print(f"Sigmoid后 y: {[f'{v:.3f}' for v in y_logistic.tolist()]}")
print(f"范围: (0, 1)")
print(f"用途: 预测概率/类别（如是否点击）")
print(f"任务: 分类 ✓")

print("\n" + "=" * 70)
print("回归 vs 分类")
print("=" * 70)

comparison = [
    ("任务目标", "预测连续值", "预测类别/概率"),
    ("输出范围", "(-∞, +∞)", "[0, 1]"),
    ("损失函数", "MSE", "BCE/交叉熵"),
    ("输出解释", "具体数值", "属于某类的概率"),
    ("例子", "预测房价3.5万", "预测点击概率0.8"),
]

print(f"\n{'维度':<12} {'回归任务':<20} {'分类任务':<20}")
print("-" * 70)
for aspect, regression, classification in comparison:
    print(f"{aspect:<12} {regression:<20} {classification:<20}")

print("\n逻辑回归属于: 分类任务 ✓")

print("\n" + "=" * 70)
print("逻辑回归的标准定义")
print("=" * 70)

print("""
名称: Logistic Regression (逻辑回归)
别名: 
  - 对数几率回归
  - Logit回归
  
模型:
  P(y=1|x) = σ(w^T x + b) = 1 / (1 + e^(-(w^T x + b)))
  
损失函数:
  BCE = -[y·log(p) + (1-y)·log(1-p)]
  
输出:
  - 概率值 p ∈ [0, 1]
  - 决策: y = 1 if p > 0.5 else 0
  
任务类型: 二分类 (Binary Classification)
  
应用场景:
  ✓ 垃圾邮件检测 (是/否)
  ✓ 疾病诊断 (患病/健康)
  ✓ 点击率预测 (点击/不点击)
  ✓ 信用评分 (违约/不违约)
  ✗ 预测房价 (这是回归任务，用线性回归)
""")

