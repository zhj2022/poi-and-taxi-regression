import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

"""对linear model拟合效果不好的side length=1, 2的情况进行多项式拟合"""
for i in range(1, 4):
    pf = PolynomialFeatures(degree=i)
    for j in range(1, 3):
        y = np.load(f'taxi_data(region={j}{j}).npy')
        X = np.load(f'poi_data(region={j}{j}).npy')
        X = pf.fit_transform(X)
        model = LinearRegression(normalize=True)
        model.fit(X, y)
        y_pred = model.predict(X)
        print(f"degree为{i},side length为{j}时R^2的值为{r2_score(y, y_pred)}")

"""对linear model拟合效果不好的taxi_data4, 5, 6, 7列进行多项式拟合(以side length=4为例)"""
for i in range(1, 4):
    pf = PolynomialFeatures(degree=i)
    for j in range(4, 8):
        y = np.load('taxi_data(region=44).npy')[:, j]
        X = np.load('poi_data(region=44).npy')
        X = pf.fit_transform(X)
        model = LinearRegression(normalize=True)
        model.fit(X, y)
        y_pred = model.predict(X)
        print(f"degree={i},dimension={j}时R^2的值为{r2_score(y, y_pred)}")