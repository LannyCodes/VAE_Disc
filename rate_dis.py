import torch

print("=" * 60)
print("BCE vs MSE: 语义对比")
print("=" * 60)

x = torch.tensor([1.0])
y_true = torch.tensor([1.0])  # 真实标签: 正类

# 三种预测
predictions = [
    (0.99, "非常确定是正类"),
    (0.51, "勉强认为是正类"),
    (0.49, "勉强认为是负类")
]

for y_pred_val, desc in predictions:
    y_pred = torch.tensor([y_pred_val])
    
    mse = ((y_pred - y_true) ** 2).item()
    bce = -(y_true * torch.log(y_pred)).item()
    
    print(f"\n{desc}: y_pred = {y_pred_val}")
    print(f"  MSE = {mse:.6f}  (距离的平方)")
    print(f"  BCE = {bce:.6f}  (信息量/惊讶度)")

print("\n" + "=" * 60)
print("观察:")
print("  0.51 vs 0.49: 都接近0.5，但分类结果相反！")
print("  MSE: 两者损失几乎一样 (0.24 vs 0.26)")
print("  BCE: 能区分细微差别，符合概率解释")
print("=" * 60)