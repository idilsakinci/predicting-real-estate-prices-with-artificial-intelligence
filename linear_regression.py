#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("multilinearregression.csv", sep = ";")


# In[2]:


df


# In[3]:


reg = linear_model.LinearRegression()
# Define independent variables first, then dependent variables
# The dependent variable will be created by evaluating the independent variables
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])

# Make a new data entry according to the independent variables, request a dependent variable(fiyat)
reg.predict([[275, 3, 11]])


# In[4]:


# Multiple data
reg.predict([[230, 4, 10], [230, 6, 0], [350, 4, 2]])


# In[ ]:




