# This Python file uses the following encoding: gbk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# 从文件中读取数据
data = pd.read_excel('C:\\Users\huawei\\Desktop\\PRML\\assignment1\\data.xlsx').values

# 提取训练集和测试集
x_train = data[:, 0]  # 第一列作为训练集的输入
y_train = data[:, 1]  # 第二列作为训练集的输出
x_test = data[:, 2]  # 第三列作为测试集的输入
y_test = data[:, 3]  # 第四列作为测试集的输出

# OLS
def Ordinary_least_squares(x, y):
    X = np.hstack([x, np.ones_like(x)])
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    theta0, theta1 = theta[1], theta[0]
    return theta0, theta1

# Gradient Descent
def Gradient_descent(x, y, lr=0.001, epochs=10000):
    n = len(y)
    theta0, theta1 = 0, 0  # Initial parameters
    
    for _ in range(epochs):
        y_pred = theta0 + theta1 * x
        loss = (1/n) * np.sum((y - y_pred)**2)
        
        dtheta0 = -(2/n) * np.sum(y - y_pred)
        dtheta1 = -(2/n) * np.sum(x * (y - y_pred))
        
        theta0 = theta0 - lr * dtheta0
        theta1 = theta1 - lr * dtheta1
        
    return theta0, theta1

# Newton-Raphson Method
def Newton_Raphson(x, y, lr=0.001, epochs=5000):
    n = len(y)
    theta0, theta1 = 0, 0  # Initial parameters
    
    for _ in range(epochs):
        y_pred = theta0 + theta1 * x
        loss = (1/n) * np.sum((y - y_pred)**2)
        
        dtheta0 = -(2/n) * np.sum(y - y_pred)
        dtheta1 = -(2/n) * np.sum(x * (y - y_pred))
        
        H00 = 2
        H01 = 2 * np.sum(x)
        H10 = 2 * np.sum(x)
        H11 = 2 * np.sum(x**2)
        
        H = np.array([[H00, H01], [H10, H11]])
        dtheta = np.array([dtheta0, dtheta1])
        theta = np.array([theta0, theta1])
        
        theta = theta - lr * np.linalg.inv(H) @ dtheta
        theta0, theta1 = theta[0], theta[1]
        
    return theta0, theta1

# fourier series fit
def fourier_series_fit(x, a0, a1, b1, a2, b2, a3, b3, a4, b4):
    return a0 + a1 * np.cos(x) + b1 * np.sin(x) + a2 * np.cos(2 * x) + b2 * np.sin(2 * x) + a3 * np.cos(3 * x) + b3 * np.sin(3 * x) + a4 * np.cos(4 * x) + b4 * np.sin(4 * x)

#高次多项式拟合
def polynomial_fit(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    return a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5 + a6 * x**6 + a7 * x**7 + a8 * x**8 + a9 * x**9 + a10 * x**10

# 训练模型
theta0_ols, theta1_ols = Ordinary_least_squares(x_train.reshape(-1, 1), y_train)
theta0_gd, theta1_gd = Gradient_descent(x_train, y_train)
theta0_nr, theta1_nr = Newton_Raphson(x_train, y_train)
initial_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
initial_params1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
params, cov = curve_fit(fourier_series_fit, x_train, y_train, p0=initial_params)
params1, cov1 = curve_fit(polynomial_fit, x_train, y_train, p0=initial_params1)
a0, a1, b1, a2, b2, a3, b3, a4, b4 = params
a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10 = params1

# 计算训练集的均方误差
y_pred_ols_train = theta0_ols + theta1_ols * x_train
y_pred_gd_train = theta0_gd + theta1_gd * x_train
y_pred_nr_train = theta0_nr + theta1_nr * x_train
y_pred_fs_train = fourier_series_fit(x_train, a0, a1, b1, a2, b2, a3, b3, a4, b4)
y_pred_poly_train = polynomial_fit(x_train, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10)
mse_ols_train = np.mean((y_train - y_pred_ols_train)**2)
mse_gd_train = np.mean((y_train - y_pred_gd_train)**2)
mse_nr_train = np.mean((y_train - y_pred_nr_train)**2)
mse_fs_train = np.mean((y_train - y_pred_fs_train)**2)
mse_poly_train = np.mean((y_train - y_pred_poly_train)**2)

# 测试模型
y_pred_ols = theta0_ols + theta1_ols * x_test
y_pred_gd = theta0_gd + theta1_gd * x_test
y_pred_nr = theta0_nr + theta1_nr * x_test
y_pred_fs = fourier_series_fit(x_test, a0, a1, b1, a2, b2, a3, b3, a4, b4)
y_pred_poly = polynomial_fit(x_test, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10)

# 计算测试集的均方误差
mse_ols = np.mean((y_test - y_pred_ols)**2)
mse_gd = np.mean((y_test - y_pred_gd)**2)
mse_nr = np.mean((y_test - y_pred_nr)**2)
mse_fs = np.mean((y_test - y_pred_fs)**2)
mse_poly = np.mean((y_test - y_pred_poly)**2)

# 绘制图像
plt.figure(figsize=(10, 5))
plt.plot(x_train, y_train, 'ro', label='Train data')
plt.plot(x_test, y_test, 'ro', label='Test data')
plt.plot(x_test, y_pred_ols, 'b*', label='OLS')
plt.plot(x_test, y_pred_gd, 'g-', label='GD')
plt.plot(x_test, y_pred_nr, 'y-', label='NR')
plt.plot(x_test, y_pred_fs, 'b-', label='FS')
plt.plot(x_test, y_pred_poly, 'c-', label='Poly')
plt.plot(x_train, y_pred_ols_train, 'b*', label='OLS')
plt.plot(x_train, y_pred_gd_train, 'g-', label='GD')
plt.plot(x_train, y_pred_nr_train, 'y-', label='NR')
plt.plot(x_train, y_pred_fs_train, 'b-', label='FS')
plt.plot(x_train, y_pred_poly_train, 'c-', label='Poly')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

# 打印均方误差
print('MSE of OLS (train):', mse_ols_train)
print('MSE of OLS:', mse_ols)
print('MSE of GD (train):', mse_gd_train)
print('MSE of GD:', mse_gd)
print('MSE of NR (train):', mse_nr_train)
print('MSE of NR:', mse_nr)
print('MSE of FS (train):', mse_fs_train)
print('MSE of FS:', mse_fs)
print('MSE of Poly (train):', mse_poly_train)
print('MSE of Poly:', mse_poly)
