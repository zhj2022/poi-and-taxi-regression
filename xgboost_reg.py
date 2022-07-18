import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


n = 4   # use data in f"data{n}"
taxi_data = np.load(f"data/data{n}/taxi_data(region={n}{n}).npy")
poi_data = np.load(f"data/data{n}/poi_data(region={n}{n}).npy")

# delete zero taxi-vectors and poi-vectors correlated
zero_row = []
for i in range(np.shape(taxi_data)[0]):
    if np.all(taxi_data[i,:] == 0):
        zero_row.append(i)
taxi_data = np.delete(taxi_data, zero_row, 0)
poi_data = np.delete(poi_data, zero_row, 0)

r2_array = []
mse_array = []
y_average = []
test_size = 0.3

if not os.path.exists(f"xgboost/n={n},test_size={test_size}"):
    os.makedirs(f"xgboost/n={n},test_size={test_size}")
if not os.path.exists(f"xgboost/n={n},test_size={test_size}/importance_figs"):
    os.makedirs(f"xgboost/n={n},test_size={test_size}/importance_figs")

for tag_index in range(24):
    xtrain,xtest,ytrain,ytest = train_test_split(poi_data,taxi_data[:,tag_index],test_size=test_size,random_state=10)

    print(f"tag_index = {tag_index}")

    reg_model = xgb.XGBRegressor(objective='reg:squarederror',verbosity=0, nthread=15)

    reg_model.fit(xtrain,ytrain,verbose=False)
    ypred = reg_model.predict(xtest)
    importance = reg_model.feature_importances_
    
    r2 = r2_score(ytest, ypred)
    r2_array.append(r2)
    mse = MSE(ytest, ypred)
    print("r2 = %f, MSE = %f, y_average = %f" % (r2, mse, ytest.mean()))
    mse_array.append(mse)
    y_average.append(ytest.mean())

    # print(importance)
    # print("\n")
    plt.figure()
    plt.bar([i for i in range(25)], importance)
    plt.savefig(f"xgboost/n={n},test_size={test_size}/importance_figs/importance_to_{tag_index}.jpg")

df = pd.DataFrame(list(zip(r2_array,mse_array,y_average)),columns=['r2','mse','y_average'])
df.to_csv(f"xgboost/n={n},test_size={test_size}/xgboost(n={n},test_size={test_size}).csv")

