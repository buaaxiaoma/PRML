# This Python file uses the following encoding: gbk

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

# 决策树分类器
def decision_tree_classifier(X, label, X_test, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, label)
    # 预测测试集
    y_pred = clf.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def adboost_decision_tree_classifier(X, label, X_test, y_test):
    ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),  # 使用决策树桩作为基础估计器
    n_estimators=200,  # 设置弱学习器的数量
    learning_rate=1,  # 设置学习率
    algorithm='SAMME',  # 使用SAMME算法
    random_state=42)
    ada_clf.fit(X, label)
    y_pred = ada_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def SVM_classifier(X, label, X_test, y_test, kernel):
    clf = svm.SVC(kernel=kernel, C=1.0, random_state=42)
    clf.fit(X, label)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    accuracy1 = 0
    accuracy2 = 0
    accuracy3 = 0
    accuracy4 = 0
    accuracy5 = 0
    accuracy6 = 0
    for _ in range(10):
    # Generate the data (1000 training points)(500 test points)
        X, label = make_moons_3d(n_samples=1000, noise=0.2)
        X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)
        
        accuracy1 += decision_tree_classifier(X, label, X_test, y_test)
        accuracy2 += adboost_decision_tree_classifier(X, label, X_test, y_test)
        accuracy3 += SVM_classifier(X, label, X_test, y_test, kernel='linear')
        accuracy4 += SVM_classifier(X, label, X_test, y_test, kernel='rbf')
        accuracy5 += SVM_classifier(X, label, X_test, y_test, kernel='poly')
        accuracy6 += SVM_classifier(X, label, X_test, y_test, kernel='sigmoid')

    print(f"决策树分类器的准确率: {accuracy1 / 10:.4f}")
    print(f"AdaBoost+决策树分类器的准确率: {accuracy2 / 10:.4f}")
    print(f"SVM(线性核)分类器的准确率: {accuracy3 / 10:.4f}")
    print(f"SVM(RBF核)分类器的准确率: {accuracy4 / 10:.4f}")
    print(f"SVM(多项式核)分类器的准确率: {accuracy5 / 10:.4f}")
    print(f"SVM(sigmoid核)分类器的准确率: {accuracy6 / 10:.4f}")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制训练数据
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)

# 绘制决策边界
# 创建一个3D网格来预测
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50),
                         np.linspace(z_min, z_max, 50))

# 将网格点转换为模型输入格式
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
clf = svm.SVC(kernel='rbf', C=1.0, random_state=42)  # 使用RBF核
clf.fit(X, label)
grid_predictions = clf.predict(grid_points)
grid_predictions = grid_predictions.reshape(xx.shape)

# 绘制预测结果
ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(), c=grid_predictions, cmap='viridis', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Make Moons with SVM Classification')
plt.show()