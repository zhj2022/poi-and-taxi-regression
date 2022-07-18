import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


n = 2   # use data in f"data{n}"
taxi_data = np.load(f"data/data{n}/taxi_data(region={n}{n}).npy")
poi_data = np.load(f"data/data{n}/poi_data(region={n}{n}).npy")

# delete zero taxi-vectors and poi-vectors correlated
zero_row = []
for i in range(np.shape(taxi_data)[0]):
    if np.all(taxi_data[i,:] == 0):
        zero_row.append(i)
taxi_data = np.delete(taxi_data, zero_row, 0)
poi_data = np.delete(poi_data, zero_row, 0)

# print(np.shape(taxi_data),np.shape(poi_data))

r2_array = []
mse_array = []
y_average = []
test_size = 0.3

if not os.path.exists(f"linear/n={n},test_size={test_size}"):
    os.makedirs(f"linear/n={n},test_size={test_size}")
if not os.path.exists(f"linear/n={n},test_size={test_size}/importance_figs"):
    os.makedirs(f"linear/n={n},test_size={test_size}/importance_figs")

for tag_index in range(24):
    xtrain,xtest,ytrain,ytest = train_test_split(poi_data,taxi_data[:,tag_index],test_size=test_size,random_state=10)
    reg = LR().fit(xtrain,ytrain)
    ypred = reg.predict(xtest)
    # Linear Model:
    coef = reg.coef_    
    intercept = reg.intercept_
    # print(coef,intercept)     

    mse = MSE(ytest, ypred)
    r2 = r2_score(ytest, ypred)
    
    print("tag_index = %d, r2 = %f, mse = %f, y_average = %f" % (tag_index, r2, mse, ytest.mean()))
    r2_array.append(r2)
    mse_array.append(mse)
    y_average.append(ytest.mean())

    plt.figure()
    plt.bar([i for i in range(25)], coef)
    plt.savefig(f"linear/n={n},test_size={test_size}/importance_figs/importance_to_{tag_index}.jpg")

df = pd.DataFrame(list(zip(r2_array,mse_array,y_average)),columns=['r2','mse','y_average'])
# print(df)
df.to_csv(f"linear/n={n},test_size={test_size}/linear(n={n},test_size={test_size}).csv")
