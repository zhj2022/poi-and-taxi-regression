import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


n = 3   # use data in f"data{n}"
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

for tag_index in range(24):
    xtrain,xtest,ytrain,ytest = train_test_split(poi_data,taxi_data[:,tag_index],test_size=0.3,random_state=10)
    reg = LR().fit(xtrain,ytrain)
    ypred = reg.predict(xtest)
    # Linear Model:
    coef = reg.coef_    
    intercept = reg.intercept_
    # print(coef,intercept)     

    mse = MSE(ytest, ypred)
    r2 = r2_score(ytest, ypred)
    r2_array.append(r2)
    print("tag_index = %d, r2 = %f, mse = %f, y_average = %f" % (tag_index, r2, mse, ytest.mean()))

print(r2_array)
