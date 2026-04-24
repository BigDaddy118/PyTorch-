import torch
import matplotlib.pyplot as plt

# 准备数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 权重初始化
w = torch.tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# 用于记录每个 epoch 的平均损失
epoch_losses = []

print("predict (before training)", 4, forward(4).item())

# 训练
for epoch in range(100):
    total_loss = 0.0
    for x, y in zip(x_data, y_data):
        l = loss(x, y)          # 前向传播并计算损失
        l.backward()            # 反向传播求梯度
        # 更新权重
        w.data = w.data - 0.01 * w.grad.data
        # 梯度清零
        w.grad.data.zero_()
        
        total_loss += l.item()
    
    avg_loss = total_loss / len(x_data)
    epoch_losses.append(avg_loss)
    print(f'progress: {epoch} avg_loss: {avg_loss:.6f}')

print("predict (after training)", 4, forward(4).item())

# ---------- 画图 ----------
plt.figure(figsize=(12, 4))

# 图1：损失下降曲线
plt.subplot(1, 2, 1)
plt.plot(range(len(epoch_losses)), epoch_losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Curve')
plt.grid(True)

# 图2：数据点与拟合直线
plt.subplot(1, 2, 2)
# 原始数据散点
plt.scatter(x_data, y_data, color='red', label='Training data')
# 用训练好的 w 画出拟合直线
x_line = torch.linspace(0, 4, 100)
y_line = forward(x_line).detach().numpy()
plt.plot(x_line.numpy(), y_line, 'g-', label=f'Fitted line: y = {w.item():.3f}x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Fit')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()