#!/usr/bin/env python
# coding: utf-8

# 1. Work for predicting faulty device using Crisp DM model (Supervised classification techniques)

# In[ ]:


# import pandas as pd
import numpy as np
import sklearn
import scipy
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import seaborn as sns
LABELS = ["Normal", "Faulty"]
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import utils
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (10,6)


# In[33]:


data_df = pd.read_csv('C:/Users/Dipendra Singh/Desktop/DataSet_CRISP_DM/secom_mod.txt', index_col=0)


# In[34]:


data_df.head(n=10)


# In[35]:


data_df.describe()


# In[36]:


data_df.isnull().any()


# In[37]:


#Filling Nan with  mean imputation,
data_df.fillna(data_df.mean(),inplace=True)


# In[38]:


data_df.drop(columns=['timestamp'],inplace=True)


# In[39]:


data_df.dtypes


# In[40]:


data_df.astype(float)


# In[42]:


data_df.isnull().sum().sum()


# In[43]:


#Check shape of data
data_df.shape


# In[44]:


#Outlier detection and removal using 3 standard deviation
upper_limit = data_df.mean() + 3*data_df.std()
upper_limit


# In[45]:


lower_limit = data_df.mean() -3*data_df.std()
lower_limit


# In[46]:


data_df[(data_df>upper_limit) | (data_df<lower_limit)]


# In[47]:


#Now remove these outliers and generate new dataframe
df_no_outlier_std_dev = data_df[(data_df<upper_limit) & (data_df>lower_limit)]
df_no_outlier_std_dev


# In[58]:


#Filling Nan with  mean imputation,
df_no_outlier_std_dev.fillna(df_no_outlier_std_dev.mean(),inplace=True)


# In[59]:


df_no_outlier_std_dev.describe()


# In[60]:


df_no_outlier_std_dev.isnull().sum().sum()


# In[66]:


#drop columns of Nan values
df.dropna(axis=1)
df


# In[68]:


df


# In[ ]:




