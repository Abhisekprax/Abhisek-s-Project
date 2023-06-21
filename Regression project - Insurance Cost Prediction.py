#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[4]:


#Data collection and Analysis
df = pd.read_csv('insurance Regression.csv')


# In[5]:


#View first few rows of the dataset
df.head()


# In[7]:


#to find the rows and the columns 
df.shape


# In[9]:


#getting some information about the dataset
#categorical features are sex, smoker and region
df.info()


# In[10]:


#checking for missing values
df.isnull().sum


# In[12]:


#Analysis of data
#Statistical measures of the dataset
df.describe()


# In[16]:


#distribution of age value
sns.set
plt.figure(figsize=(6,6))
sns.distplot(df['age'])
plt.title('Age Distribution')
plt.show()


# In[17]:


#GenderColumn
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=df)
plt.title('Sex Distribution')
plt.show()


# In[18]:


#distribution of bmi value
sns.set
plt.figure(figsize=(6,6))
sns.distplot(df['bmi'])
plt.title('BMI Distribution')
plt.show()


# In[19]:


#ChildrenColumn
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=df)
plt.title('Children Distribution')
plt.show()


# In[21]:


#distribution of smoker
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=df)
plt.title('Smoker Distribution')
plt.show()


# In[22]:


#distribution of region
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=df)
plt.title('region Distribution')
plt.show()


# In[23]:


#distribution of charges
sns.set
plt.figure(figsize=(6,6))
sns.distplot(df['charges'])
plt.title('Charges Distribution')
plt.show()


# In[25]:


#Data Preprocessing
# encoding sex column
df.replace({'sex':{'male':0,'female':1}}, inplace=True)

3 # encoding 'smoker' column
df.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
df.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# In[27]:


X = df.drop(columns='charges', axis=1)
Y = df['charges']


# In[28]:


print(X)


# In[30]:


print(Y)


# In[31]:


#Splitting the data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[32]:


print(X.shape, X_train.shape, X_test.shape)


# In[33]:


#Model Training
#Linear Regression
# loading the Linear Regression model
regressor = LinearRegression()


# In[34]:


regressor.fit(X_train, Y_train)


# In[35]:


#Model Evaluation
# prediction on training data
training_data_prediction =regressor.predict(X_train)


# In[36]:


# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)


# In[37]:


# prediction on test data
test_data_prediction =regressor.predict(X_test)


# In[39]:


# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# In[ ]:




