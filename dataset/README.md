# Datasets
## Dataset Overview
In this work, three publicly available datasets are used to examine the performance of the proposed PT4Rec. 
Here, we provide the links for downloading these datasets for readers to retrieve:

Douban Movie Review Dataset:
Link: https://pan.baidu.com/s/1hrJP6rq#list/path=%2F
Description: The Douban movie review dataset contains user ratings and reviews for movies, which is used to build a user-movie interaction model.

MOvielens Dataset:
Link: https://grouplens.org/datasets/movielens
Description: The ML-1M movie rating dataset contains user ratings for movies, which is used to evaluate the performance of recommendation systems.

Gowalla Dataset:
Link: https://github.com/kuandeng/LightGCN/tree/master/Data/gowalla
Description: The Gowalla location check-in dataset contains user check-in information at locations in the Gowalla social network, which is used for location recommendation and social network analysis.


## Dataset Usage Instructions
After downloading, please preprocess and load the datasets according to the dataset’s documentation.
When using the datasets for experiments, please ensure compliance with the dataset’s license agreement.

Our process code has been uploaded to the following dir:
```
dataset/
├── douban
│   ├── split.py
├── gowalla
│   ├── process.py
├── ml-1M
│   ├── split.py
```

