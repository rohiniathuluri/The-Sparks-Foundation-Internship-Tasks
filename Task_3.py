#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics Intern Nov Batch 2021
# 
# # Author:Attuluri Rohini
# 
# The Sparks Foundation
# 
# (GRIPNOV21)
# 
# Task3: Exploratory Data Analysis - Retail
# 
# As a business manager, try to find out the weak areas where you can
# work to make more profit.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('SampleSuperstore.csv')


# In[3]:


data


# In[4]:


data.isna().any()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.corr()


# In[8]:


data.nunique()


# In[9]:


#correlation matrix

plt.figure(figsize=(10,5))
sns.heatmap(data.corr(),annot=True)
plt.show()


# In[10]:


data.corr()


# In[11]:


sns.pairplot(data)


# In[12]:


#creating new column cost
data['Cost'] = data['Sales']-data['Profit']
data['Cost'].head()


# In[13]:


data.head()


# In[14]:


#creating a new column PROFIT %


# In[15]:


data['Profit %'] = (data['Profit']/data['Cost'])*100


# In[16]:


data


# In[17]:


data.sort_values(['Profit %','Sub-Category'],ascending=False).groupby('Profit %').head(5)


# In[18]:


data['Sub-Category']


# In[19]:


data['Sub-Category'].value_counts()


# In[20]:


plt.figure(figsize=(16,10))
sns.set_style("whitegrid")
sns.countplot(x='Sub-Category',data=data,palette='Set3')


# In[21]:


data['Region'].value_counts()


# In[22]:


sns.countplot(x='Region',data=data)


# In[23]:


plt.figure(figsize=(10,10))
data['Sub-Category'].value_counts().plot.pie(autopct="%1.1f%%")


# In[ ]:





# In[24]:


data['Sales']


# In[25]:


data['State'].value_counts()


# 

# In[26]:


statewise=data.groupby(['State'])['Sales'].sum()
statewise.sort_values(ascending=False,inplace=True)
fig, ax = plt.subplots(figsize=(20,10))
plt.title('State vs Sales')
statewise.plot.bar()


# From the above graph we can clearly see that sales are high in California when compared to other states

# In[27]:


data['Profit']


# In[28]:


statewise=data.groupby(['State'])['Profit'].sum()
statewise.sort_values(ascending=False,inplace=True)
fig, ax = plt.subplots(figsize=(20,10))
plt.title("State vs Profit")
statewise.plot.bar()


#  From the above graph we can clearly identify and note that the profits in California is  higher when compared to other states and Newyork city is holding the second highest position after California.

# In[ ]:




