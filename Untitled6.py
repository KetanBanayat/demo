#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb


# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[4]:


df1 = pd.read_csv(r'C:\Users\ketan\Downloads\User_Data.csv')


# In[5]:


df1.info()


# In[6]:


df1.shape


# In[7]:


df1.isnull().sum()


# In[10]:


df1.duplicated().sum()


# In[35]:


df1.describe


# In[11]:


df1.columns


# In[12]:


df1['Gender']=df1['Gender'].map({'Female':0,'Male':1})


# In[37]:


df1.head()


# In[13]:


del df1['User ID']


# In[14]:


df1.head()


# In[16]:


X=df1.iloc[:,:3]
print(X)


# In[17]:


Y=df1.iloc[:,3]
print(Y)


# In[28]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=55)


# In[29]:


Classify=LogisticRegression()


# In[30]:


Classify.fit(X_train,Y_train)


# In[31]:


Y_predict=Classify.predict(X_test)


# In[32]:


Y_test


# In[33]:


Y_predict


# In[34]:


from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(Y_test,Y_predict))


# In[ ]:




