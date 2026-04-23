import matplotlib.pyplot as plt

# 数据
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [2.0, 4.0, 6.0, 8.0, 10.0]

w = 1.0

def forward(x):
    return w * x

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

print('Predict (before training)', 4, forward(4))

# 用于绘图的数据记录
epoch_list = []          # epoch 序号
w_list = []              # 每个 epoch 结束时的 w
avg_loss_list = []       # 每个 epoch 的平均损失

for epoch in range(100):
    epoch_loss_sum = 0   # 累计当前 epoch 所有样本的损失
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01 * grad
        l = loss(x, y)
        epoch_loss_sum += l
        # print("\tgrad:", x, y, grad)  # 若需要可取消注释

    avg_loss = epoch_loss_sum / len(x_data)
    epoch_list.append(epoch)
    w_list.append(w)
    avg_loss_list.append(avg_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}: w = {w:.4f}, avg_loss = {avg_loss:.6f}")

print('Predict (after training)', 4, forward(4))

# ------------------ 绘图 ------------------
plt.figure(figsize=(12, 4))

# 子图1：平均损失曲线
plt.subplot(1, 3, 1)
plt.plot(epoch_list, avg_loss_list, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Loss Curve (SGD)')
plt.grid(True)

# 子图2：参数 w 的变化曲线
plt.subplot(1, 3, 2)
plt.plot(epoch_list, w_list, 'r-')
plt.axhline(y=2.0, color='gray', linestyle='--', label='True w = 2.0')
plt.xlabel('Epoch')
plt.ylabel('w')
plt.title('Parameter w Convergence')
plt.legend()
plt.grid(True)

# 子图3：拟合直线对比
plt.subplot(1, 3, 3)
plt.scatter(x_data, y_data, color='red', label='True Data')
# 训练前的线 (w=1.0)
y_pred_before = [1.0 * x for x in x_data]
plt.plot(x_data, y_pred_before, 'g--', label='Before training: w=1.0')
# 训练后的线 (当前 w)
y_pred_after = [w * x for x in x_data]
plt.plot(x_data, y_pred_after, 'b-', label=f'After training: w={w:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Fit Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
