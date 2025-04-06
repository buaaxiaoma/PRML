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

# ������������
def decision_tree_classifier(X, label, X_test, y_test):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, label)
    # Ԥ����Լ�
    y_pred = clf.predict(X_test)
    # ����׼ȷ��
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def adboost_decision_tree_classifier(X, label, X_test, y_test):
    ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),  # ʹ�þ�����׮��Ϊ����������
    n_estimators=200,  # ������ѧϰ��������
    learning_rate=1,  # ����ѧϰ��
    algorithm='SAMME',  # ʹ��SAMME�㷨
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

    print(f"��������������׼ȷ��: {accuracy1 / 10:.4f}")
    print(f"AdaBoost+��������������׼ȷ��: {accuracy2 / 10:.4f}")
    print(f"SVM(���Ժ�)��������׼ȷ��: {accuracy3 / 10:.4f}")
    print(f"SVM(RBF��)��������׼ȷ��: {accuracy4 / 10:.4f}")
    print(f"SVM(����ʽ��)��������׼ȷ��: {accuracy5 / 10:.4f}")
    print(f"SVM(sigmoid��)��������׼ȷ��: {accuracy6 / 10:.4f}")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# ����ѵ������
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label, cmap='viridis', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)

# ���ƾ��߽߱�
# ����һ��3D������Ԥ��
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50),
                         np.linspace(z_min, z_max, 50))

# �������ת��Ϊģ�������ʽ
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
clf = svm.SVC(kernel='rbf', C=1.0, random_state=42)  # ʹ��RBF��
clf.fit(X, label)
grid_predictions = clf.predict(grid_points)
grid_predictions = grid_predictions.reshape(xx.shape)

# ����Ԥ����
ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(), c=grid_predictions, cmap='viridis', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Make Moons with SVM Classification')
plt.show()