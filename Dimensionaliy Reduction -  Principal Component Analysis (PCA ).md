<h1>Principal component analysis (PCA)</h1>:<br/> Is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation, increasing the interpretability of data while preserving the maximum amount of information, and enabling the visualization of multidimensional data. Formally, PCA is a statistical technique for reducing the dimensionality of a dataset. - <a href="https://en.wikipedia.org/wiki/Principal_component_analysis">Principal component analysis</a>

<b>PCA Demo:</b> <a href= "https://setosa.io/ev/principal-component-analysis/">Link</a>

![image-2.png](attachment:image-2.png)

![image.png](attachment:image.png)

 For more on the topic of PCA: <a href='https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d'> Towards DataScience</a>


```python
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# This filters ALL warnings, but you can also filter by category
import warnings
warnings.filterwarnings("ignore")

# Filtering by category:
warnings.filterwarnings("ignore",category=DeprecationWarning)
```

<h3><b>Data source:</b></h3>
<a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"> Machine Learning Group ULB</a>  on Kaggle.


```python
# read the dataset
data = pd.read_csv("Bank Customer Churn Prediction.csv")
```

<h3><b>Explore EDA:</b></h3>


```python
# examine the first 10 rows
data = data.copy()
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>credit_score</th>
      <th>country</th>
      <th>gender</th>
      <th>age</th>
      <th>tenure</th>
      <th>balance</th>
      <th>products_number</th>
      <th>credit_card</th>
      <th>active_member</th>
      <th>estimated_salary</th>
      <th>churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15634602</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15647311</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15619304</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15701354</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15737888</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# customer id has no predictive power
data = data.drop(['customer_id', 'gender'], axis=1)
```


```python
# check unique value of each columns
data.nunique()
```




    credit_score         460
    country                3
    age                   70
    tenure                11
    balance             6382
    products_number        4
    credit_card            2
    active_member          2
    estimated_salary    9999
    churn                  2
    dtype: int64




```python
# check for missing values
data.isna().sum()
```




    credit_score        0
    country             0
    age                 0
    tenure              0
    balance             0
    products_number     0
    credit_card         0
    active_member       0
    estimated_salary    0
    churn               0
    dtype: int64




```python
# check data types
data.dtypes
```




    credit_score          int64
    country              object
    age                   int64
    tenure                int64
    balance             float64
    products_number       int64
    credit_card           int64
    active_member         int64
    estimated_salary    float64
    churn                 int64
    dtype: object




```python
# create a copu of the data
data = data.copy()
data.columns
```




    Index(['credit_score', 'country', 'age', 'tenure', 'balance',
           'products_number', 'credit_card', 'active_member', 'estimated_salary',
           'churn'],
          dtype='object')



<h3><b>Preprocess with sklearn StandardScaler:</b></h3>


```python
# import standard scaler to 
from sklearn.preprocessing import StandardScaler
```


```python
# instatiate the standard scaler
scaler = StandardScaler()
```


```python
# oneHot encode the values
data_dummies = pd.get_dummies(data, drop_first=True)
```


```python
# fit the dataset
scaler.fit(data_dummies)
```




    StandardScaler()




```python
# transform the dataset
scaled_data = scaler.transform(data_dummies)
```

<h3><b>PCA:</b></h3>


```python
# PCA
from sklearn.decomposition import PCA
```


```python
# instatiate the pca and number of components
pca = PCA(n_components= 2)
```


```python
# fit the dataset to the pca
pca.fit(data_dummies)
```




    PCA(n_components=2)




```python
# transform the dataset
x_pca = pca.transform(scaled_data)
```


```python
# shape of the dataset
scaled_data.shape
```




    (10000, 11)




```python
# after pca
x_pca.shape
```




    (10000, 2)




```python
# plot the components
plt.figure(figsize=(15,10))
plt.scatter(x_pca[:,0], x_pca[:,1],c=data['churn'],cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Principal Component Analysis');
```


    
![png](output_28_0.png)
    



```python
# examine the values
pca.components_
```




    array([[ 9.51816562e-06,  4.65281356e-06, -5.39944291e-07,
             9.96979873e-01, -2.81486667e-06, -1.13309082e-07,
            -8.69970340e-08,  7.76603729e-02,  7.67689358e-07,
             2.78116284e-06, -9.32788075e-07],
           [ 3.21058039e-06,  1.74611085e-06, -4.42665725e-07,
             7.76603728e-02, -4.02880589e-07,  6.86475509e-08,
             9.16735326e-08, -9.96979873e-01, -1.45324519e-08,
             1.77587451e-07, -3.68449116e-08]])




```python
# create a dataframe with dummies column names assigned 
df_comp = pd.DataFrame(pca.components_, columns = list(data_dummies.columns))
```


```python
# plot the correlation of the features to the churn class 0, 1 
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')
plt.title('Correlation of the Churn');
```


    
![png](output_31_0.png)
    

