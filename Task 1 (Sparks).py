#!/usr/bin/env python
# coding: utf-8

# # TASK 1 : Prediction Using Supervised ML
# 
# ## By : Shreyash Kerekar
# 
# ### The Sparks Foundation
# 
# ### Data Science and Business Analytics Intern
# 
# In this task we have to predict the percentage of an student based on the no. of study hours.
# 
# * This is simple linear Regression task which involves just two variables
# * Data set used     : http://bit.ly/w-data
# * Problem statement : Predicting the score if student studies for 9.25 hrs /day.

# ## Importing libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing dataset

# In[3]:


url = "http://bit.ly/w-data"
a = pd.read_csv(url)
print("Data imported Successfully! ")
print(a)


# In[4]:


a.shape


# # SCATTER PLOT

# In[5]:


plt.scatter(x=a.Hours,y=a.Scores)
plt.xlabel('Study Hours')
plt.ylabel('Student Marks')
plt.title('Student Marks Predictor')
plt.show()


# #### From the above graph, we can clearly see that there is postive linear relation between the number of hours studied and the percentage of score.

# In[6]:


#to check missing values
a.isnull().sum()


# # SPLITING DATA IN DEPENDENT AND INDEPENDENT VARIABLES

# The next step is to divide the data into "attributes"(inputs) and "labels"(outputs)

# In[7]:


X = a.iloc[:,:-1].values #independent variables
Y = a.iloc[:,1].values  #dependent variables


# In[8]:


print(X)


# # TRAIN-TEST SPLITTING

# In[9]:


# train test splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0) 


# In[10]:


# select a model and train it
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)


# In[11]:


# plotting the regression line
line=lr.coef_*X+lr.intercept_
# plotting for test data
plt.scatter(X,Y)
plt.plot(X, line);
plt.show


# # PREDICTIONS

# In[12]:


print(X_test) # testing data in hours
y_pred = lr.predict(X_test)


# In[13]:


#Comparing Actual VS Predict
df1 = pd.DataFrame({'Actual':Y_test,'Predicted' :y_pred})
print(df1)


# # MODEL EVALUATION

# In[14]:


# coef_ and intercept_ are coefficients and intercepts resp. for our model
print("Coefficients:\n",lr.coef_)
print("Intercept:\n",lr.intercept_)


# The Final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset.

# In[15]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,y_pred))


# In[16]:


# experience Variance Score : 1 is perfect prediction
from sklearn.metrics import r2_score
print('variance Score : %.2f' % r2_score(Y_test,y_pred))


# In[17]:


#plotting the graph
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test,y_pred,color='blue')
plt.title('Score vs Hours(Test Set)')
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.show()


# # SOLUTION

# In[18]:


hours = ([[9.25]])
own_pred = lr.predict(hours)
print("No. of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # CONCLUSION

# If a student studies for 9.25 hrs/day , according to model , predicted score is 93.69
