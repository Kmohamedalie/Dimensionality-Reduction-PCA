#!/usr/bin/env python
# coding: utf-8

# <h1>Principal component analysis (PCA)</h1>:<br/> Is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation, increasing the interpretability of data while preserving the maximum amount of information, and enabling the visualization of multidimensional data. Formally, PCA is a statistical technique for reducing the dimensionality of a dataset. - <a href="https://en.wikipedia.org/wiki/Principal_component_analysis">Principal component analysis</a>

# <b>PCA Demo:</b> <a href= "https://setosa.io/ev/principal-component-analysis/">Link</a>

# ![image-2.png](attachment:image-2.png)

# ![image.png](attachment:image.png)

#  For more on the topic of PCA: <a href='https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d'> Towards DataScience</a>

# In[1]:


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


# <h3><b>Data source:</b></h3>
# <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"> Machine Learning Group ULB</a>  on Kaggle.

# In[2]:


# read the dataset
data = pd.read_csv("Bank Customer Churn Prediction.csv")


# <h3><b>Explore EDA:</b></h3>

# In[3]:


# examine the first 10 rows
data = data.copy()
data.head()


# In[4]:


# customer id has no predictive power
data = data.drop(['customer_id', 'gender'], axis=1)


# In[5]:


# check unique value of each columns
data.nunique()


# In[6]:


# check for missing values
data.isna().sum()


# In[7]:


# check data types
data.dtypes


# In[8]:


# create a copu of the data
data = data.copy()
data.columns


# <h3><b>Preprocess with sklearn StandardScaler:</b></h3>

# In[9]:


# import standard scaler to 
from sklearn.preprocessing import StandardScaler


# In[10]:


# instatiate the standard scaler
scaler = StandardScaler()


# In[11]:


# oneHot encode the values
data_dummies = pd.get_dummies(data, drop_first=True)


# In[12]:


# fit the dataset
scaler.fit(data_dummies)


# In[13]:


# transform the dataset
scaled_data = scaler.transform(data_dummies)


# <h3><b>PCA:</b></h3>

# In[27]:


# PCA
from sklearn.decomposition import PCA


# In[28]:


# instatiate the pca and number of components
pca = PCA(n_components= 2)


# In[29]:


# fit the dataset to the pca
pca.fit(data_dummies)


# In[30]:


# transform the dataset
x_pca = pca.transform(scaled_data)


# In[31]:


# shape of the dataset
scaled_data.shape


# In[32]:


# after pca
x_pca.shape


# In[33]:


# plot the components
plt.figure(figsize=(15,10))
plt.scatter(x_pca[:,0], x_pca[:,1],c=data['churn'],cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Principal Component Analysis');


# In[34]:


# examine the values
pca.components_


# In[35]:


# create a dataframe with dummies column names assigned 
df_comp = pd.DataFrame(pca.components_, columns = list(data_dummies.columns))


# In[36]:


# plot the correlation of the features to the churn class 0, 1 
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap='plasma')
plt.title('Correlation of the Churn');

