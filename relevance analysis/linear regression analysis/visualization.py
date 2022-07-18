import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error as mse, mean_absolute_error as mae


def delete_zeros(y, X):
    arr = []
    for i in range(y.shape[0]):
        if not y[i].any():
            arr.append(i)
    arr = np.array(arr)
    X = np.delete(X, arr, axis=0)
    y = np.delete(y, arr, axis=0)
    return y, X


"""r2_score与区域边长的关系(包含delete_zeros和普通两条线)"""
x = np.arange(1, 6)
r2_scores = []
r2_scores_ = []
for i in range(1, 6):
    X = np.load('poi_data(region=%d%d).npy' % (i, i))
    y = np.load('taxi_data(region=%d%d).npy' % (i, i))
    model = LinearRegression(normalize=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2_scores.append(r2_score(y, y_pred))
for i in range(1, 6):
    X = np.load('poi_data(region=%d%d).npy' % (i, i))
    y = np.load('taxi_data(region=%d%d).npy' % (i, i))
    y, X = delete_zeros(y, X)
    model = LinearRegression(normalize=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2_scores_.append(r2_score(y, y_pred))
plt.figure(num=3, figsize=(8, 5))
l1, = plt.plot(x, r2_scores)
l2, = plt.plot(x, r2_scores_)
plt.legend(handles=[l1, l2], labels=['normal', 'delete regions without taxi'], loc='best')
plt.xticks(np.arange(5)+1)
plt.xlabel('side length')
plt.ylabel('r2_score(linear model)')
plt.show()



"""不同时段拟合效果比较"""
for i in range(1, 6):
    r2_scores = []
    X = np.load('poi_data(region=%d%d).npy' % (i, i))
    y = np.load('taxi_data(region=%d%d).npy' % (i, i))
    for j in range(24):
        model = LinearRegression(normalize=True)
        model.fit(X, y[:, j])
        y_pred = model.predict(X)
        r2_scores.append(r2_score(y[:, j], y_pred))
    x = np.arange(24)
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, r2_scores, c='red')
    plt.scatter(x, r2_scores)
    plt.xticks(np.arange(24))
    plt.xlabel('dimension of taxi_vector')
    plt.ylabel('r2_score(linear model)')
    plt.show()