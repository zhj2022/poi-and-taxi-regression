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
- Both in linear model and XGBoost model, we found that when the side length of each region is 3 km, the model predicts best.
- We measured the importance of each dimension of poi_vector to judge which kind of poi influence the taxi_data more. 

## relevance analysis  
- In this part, we don't care about prediction ability of the model. Thus, we no longer divide the dataset into training set and test set.  
- We use linear model and necessarily, polynomial model to analyze how poi_data determines taxi_data.
- We've found that: 1.The bigger the region, the linear model fits better.  2.Considering regions without any taxi or not has little influence on this part.  3.The linear model fits worse on the 4th, 5th, 6th and 7th dimension of taxi_data, compared with other dimensions. We can use polynomial model to solve this problem.
