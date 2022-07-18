# poi-and-taxi-regression
## data processing
- get the data in arrive_point_array.npy and shanghai_poi.xlsx.  
- divide Shanghai map into several regions which have the same area.  

- for each region, construct two vectors: the first vector have 25 dimensions corresponding to 25 types of poi. For any dimension, the number of that dimension refers to the number of that kind of poi in this region;  the second vector has 24 dimensions corresponding to the 24 hours of a day. For any dimension, the number of that dimension refers to the number of taxis reaching this region during this certain period of time.  

- We integrate all the "first vectors" to form the matrix poi_data, and integrate all the "second vectors" to form the matrix taxi_data.  

- (Note: the indices of poi_data and taxi_data correspond to each other, i.e. poi_data[i] and taxi_data[i] are from the same region. Therefore, poi_data.shape[0] == taxi_data[0])  
All the data we use are in the files poi_data.npy, taxi_data.npy(for regression) and tagz_and_num_match.txt(for further analysis).  

## regression
- We tried linear model and XGBoost model. 
- Both in linear model and XGBoost model, we found that when the side length of each region is 3 km, the model fits best.
- We measured the importance of each dimension of poi_vector to judge which kind of poi influence the taxi_data more. 

