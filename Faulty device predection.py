#!/usr/bin/env python
# coding: utf-8

# 1. Work for predicting faulty device using Crisp DM model (Supervised classification techniques)

# In[37]:


import pandas as pd
import numpy as np
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
LABELS = ["Normal", "Faulty"]


# In[20]:


data_df = pd.read_csv('C:/Users/Dipendra Singh/Desktop/DataSet_CRISP_DM/secom_mod.txt')


# In[21]:


data_df


# In[22]:


data_df.head()


# In[39]:


data_df.describe().T


# In[24]:


data_df.dtypes


# In[25]:


data_df = data_df.drop(columns="Unnamed: 0")


# In[26]:


data_df.describe()


# In[27]:


data_df.isnull().any()


# In[38]:


count_classes = pd.value_counts(data_df['class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[35]:



## Get the Fraud and the normal dataset 

faulty = data_df[data_df['class']==1]

normal = data_df[data_df['class']==0]


# In[36]:


print(faulty.shape,normal.shape)


# In[ ]:




