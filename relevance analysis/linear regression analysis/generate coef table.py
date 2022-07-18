import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json


def delete_zeros(y, X):
    arr = []
    for i in range(y.shape[0]):
        if not y[i].any():
            arr.append(i)
    arr = np.array(arr)
    X = np.delete(X, arr, axis=0)
    y = np.delete(y, arr, axis=0)
    return y, X


def linear_regression(y, X):
    model = LinearRegression(normalize=True)
    model.fit(X, y)
    coef.append(model.coef_)
    y_pred = model.predict(X)
    print(r2_score(y, y_pred))


def linear_regression_delete_zeros(y, X):
    y, X = delete_zeros(y, X)
    model = LinearRegression(normalize=True)
    model.fit(X, y)
    coef.append(model.coef_)
    y_pred = model.predict(X)
    print(r2_score(y, y_pred))

coef = []

y = np.load('taxi_data(region=55).npy')
X = np.load('poi_data(region=55).npy')
for i in range(24):
    linear_regression(y[:, i], X)
coef = np.array(coef)
coef_dict = {}
with open('tagz_and_num_match(region=11).txt') as f:
    tagz_and_num_match = json.loads(f.read())
for poi_name in tagz_and_num_match.keys():
    coef_dict[poi_name] = coef[:, tagz_and_num_match[poi_name]]
coef_table = pd.DataFrame(coef_dict)
coef_table.to_excel('coef_table.xlsx')
