# This Python file uses the following encoding: gbk
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据
data_train = pd.read_csv('LSTM-Multivariate_pollution.csv')
data_test = pd.read_csv('pollution_test_data1.csv')

columns = (['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd','snow', 'rain'])
train_scaled = data_train.copy()
test_scaled = data_test.copy()

# Define the mapping dictionary
mapping = {'NE': 0, 'SE': 1, 'NW': 2, 'cv': 3}

# Replace the string values with numerical values
train_scaled['wnd_dir'] = train_scaled['wnd_dir'].map(mapping)
test_scaled['wnd_dir'] = test_scaled['wnd_dir'].map(mapping)
train_scaled['date'] = pd.to_datetime(train_scaled['date'])
# Resetting the index
train_scaled.set_index('date', inplace=True)
test_scaled = test_scaled[columns]

train_scaled = np.array(train_scaled)
test_scaled = np.array(test_scaled)

X = []
y = []
n_future = 1
n_past = 10

#  训练集
for i in range(n_past, len(train_scaled) - n_future+1):
    X.append(train_scaled[i - n_past:i, 0:train_scaled.shape[1]])
    y.append(train_scaled[i + n_future - 1:i + n_future, 0])
X_train, y_train = np.array(X), np.array(y)

#  测试集
X = []
y = []
for i in range(n_past, len(test_scaled) - n_future+1):
    X.append(test_scaled[i - n_past:i, 0:test_scaled.shape[1]])
    y.append(test_scaled[i + n_future - 1:i + n_future, 0])
X_test, y_test = np.array(X), np.array(y)

# 数据归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

batch_size = 128

print("Train size : " , X_train.shape , y_train.shape,"\n ------- \n"
      "Test Size : ",X_test.shape , y_test.shape)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32).reshape(-1,1))
y_test = torch.from_numpy(y_test.astype(np.float32).reshape(-1,1))

#定义数据集
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出
        return out

input_size = 10 # 特征数量
hidden_size = 32  # LSTM 隐藏层维度
num_layers = 5  # LSTM 层数
output_size = 1  # 输出维度

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

train_losses = []  # To store train losses
test_losses = []   # To store test losses

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        inputs = inputs.reshape(-1, 8, 10)
        targets = targets.reshape(-1, 1)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backward and optimize
        loss.backward()
        optimizer.step()
        # Accumulate train loss
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    # 评估模型
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in test_loader:
            inputs = inputs.reshape(-1, 8, 10)
            targets = targets.reshape(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    # Reduce learning rate if test loss does not improve
    reduce_lr.step(test_loss)


plt.figure(figsize=(10, 5))
# 绘制训练和测试损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Losses')
plt.legend()
plt.show()

# 进行预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test.reshape(-1, 8, 10))

y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1).numpy())
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1,1).numpy())

plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='True Values')
plt.plot(y_pred_actual, label='Predicted Values')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.title('True vs Predicted Pollution Levels')
plt.legend()
plt.show()

# 计算评估指标
mse = mean_squared_error(y_test_actual, y_pred_actual)
mae = mean_absolute_error(y_test_actual, y_pred_actual)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")