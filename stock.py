# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 20:54:43 2020

@author: Evanglin
"""
import sklearn
print('sklearn: %s' % sklearn.__version__)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Define time period to consider
start_date = "01-06-2018"
end_date   = "28-06-2018"
#import the dataset
data=pd.read_csv('trainset (1).csv')
#print(data)
#data.head()
# Set mask to select dates
mask = (data["Date"] > start_date) & (data["Date"] <= end_date)
# Select data between start and end date
data = data.loc[mask]
#get the number of columns and rows
print(data.head())
print(data.tail())
df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]
#split the data 70 for train and 30 for test
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)
#feature selection using random forest algorithm
feat_labels = ['Date','Open','High','Low','Close','Adj Close','Volume']
for feature in zip(feat_labels, clf.feature_importances_):
    print(feature)
sfm = SelectFromModel(clf, threshold=0.15)
sfm.fit(X_train, y_train)
#using support vector machine algorithm regression to train the dataset
#create the list x and y independent and dependent dataset
dates=[]
prices=[]
#Get the number of rows and columns in the data set 
data.shape
#Get all of the data except for the last row
data = data.head(len(data)-1)
print(data.shape)

df_dates = data.loc[:,'Date']
df_open = data.loc[:,'Open']
#Create the independent data set 'X' as dates
for date in df_dates:
  dates.append( [int(date.split('-')[0])] )
  
#Create the dependent data set 'y' as prices
for open_price in df_open:
  prices.append(float(open_price))

print(dates)
print (prices)
#Function to make predictions using 3 different support vector regression models with 3 different kernals & linear regression
def predict_prices(dates, prices, x):
  
  #Create the 3 Support Vector Regression models
  svr_lin = SVR(kernel='linear', C= 1e3,gamma=0.1)
  svr_poly= SVR(kernel='poly', C=1e3, degree=2,gamma=0.1)
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  
  #Train the SVR models 
  svr_lin.fit(dates,prices)
  svr_poly.fit(dates,prices)
  svr_rbf.fit(dates,prices)
  
  #Create the Linear Regression model
  lin_reg = LinearRegression()
  #Train the Linear Regression model
  lin_reg.fit(dates,prices)
  
  #Plot the models on a graph to see which has the best fit
  plt.scatter(dates, prices, color='black', label='Data')
  plt.plot(dates, svr_rbf.predict(dates), color='green', label='SVR RBF')
  plt.plot(dates, svr_poly.predict(dates), color='blue', label='SVR Poly')
  plt.plot(dates, svr_lin.predict(dates), color='red', label='SVR Linear')
  plt.plot(dates, lin_reg.predict(dates), color='orange', label='Linear Reg')
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.title('Regression')
  plt.legend()
  plt.show()
  return svr_rbf.predict(x)[0], svr_lin.predict(x)[0],svr_poly.predict(x)[0],lin_reg.predict(x)[0]
  
#Predict the price of GOOG on day 28
predicted_price = predict_prices(dates, prices, [[30]])
print(predicted_price)

  
  
  




