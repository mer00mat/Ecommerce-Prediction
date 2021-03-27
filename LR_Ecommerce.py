#!/usr/bin/env python
# coding: utf-8

# An Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure it out!

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv(r"C:\Users\mer00\Desktop\Ecommerce Customers.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[7]:


df.dropna()


# In[10]:


df.describe()


# In[11]:


df.columns


# In[9]:


sns.pairplot(df)


# In[18]:


sns.jointplot(x= 'Time on Website', y='Yearly Amount Spent', data=df)


# In[19]:


sns.jointplot(x= 'Time on App', y='Yearly Amount Spent', data=df, kind='hex')


# In[20]:


sns.lmplot(x= 'Length of Membership', y='Yearly Amount Spent', data=df)


# In[23]:


sns.heatmap(df.corr(), annot= True)


# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


df.columns


# In[36]:


X=df[['Avg. Session Length','Time on App','Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[38]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()


# In[39]:


lm.fit(X_train,y_train)


# In[40]:


print(lm.intercept_)


# In[46]:


coeff_df=pd.DataFrame(lm.coef_,X.columns,columns=["coeff"])
print(coeff_df)


# In[47]:


predictions=lm.predict(X_test)


# In[48]:


plt.scatter(predictions,y_test)


# In[51]:


sns.distplot(y_test-predictions,bins=50)


# In[50]:


from sklearn import metrics
from math import sqrt

print('MAE:', 
      metrics.mean_absolute_error(y_test, predictions), ' ',
      (1./len(y_test))*(sum(abs(y_test-predictions))))
print('MSE:', 
      metrics.mean_squared_error(y_test, predictions), ' ',
      (1./len(y_test))*(sum((y_test-predictions)**2)))
print('RMSE:', 
      np.sqrt(metrics.mean_squared_error(y_test, predictions)), ' ',
      sqrt((1./len(y_test))*(sum((y_test-predictions)**2))))


# In[52]:


coeff_df=pd.DataFrame(lm.coef_,X.columns,columns=["coeff"])
print(coeff_df)


# Time on App seems to generate more revenue when compared to Time on Website. Hence company must focus more on App improvement as well as Length of Membership plays the most important role

# In[ ]:




