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


# In[5]:


# Multiple linear regression formula:
# y = a + b1x1 + b2x2 + b3x3 + ...


# In[6]:


# Calculate coefficients
reg.coef_


# In[7]:


# Calculate constant
reg.intercept_


# In[8]:


# Examine formula

a = reg.intercept_
b1 = reg.coef_[0]
b2 = reg.coef_[1]
b3 = reg.coef_[2]

x1 = 275
x2 = 3
x3 = 11
y = a + b1*x1 + b2*x2 + b3*x3

y


# In[ ]:




