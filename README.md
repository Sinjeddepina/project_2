# project_2


import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Loaded ddata in2 a Pandas DataFrame
data = pd.read_csv('/UNZIP_FOR_NOTEBOOKS_FINAL/project_2/cc_general.csv')

# i define the dependent variable and independent variable(s) from the data sets patrick found on vredit history form kagel
X = data[['time', 'spending_patterns', 'credit_score']]
y = data['balance_amount']

# split the dta into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Split the data into training and testing sets (had trouble here but withspliting but turned out it was miss labebel 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Created a linear regression model for the training data
model = LinearRegression()
model.fit(X_train, y_train)

# used the model to make predictions on the test data
y_pred = model.predict(X_test)

# Calculated the "root mean square error (RMSE)" had to do somew research from what dave suggested to do in slack, and this was the conclussion
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)



In conclussion after predicting for balance amounts based on historical credit card data without being able to compare to our results unfortunitly i feel this model was pretty accurate To calculate the RSME, after first calculate the differences between the predicted and actual values. Then squareing each of these differences the mean of the squared differences, and then square root of the mean.

The RSME metric is a great statistics, engineering, and data science and perfect to assess the accuracy of models in our credit balkance predictor, It provides a numerical value that represents what we fill will be a good indicator for predicting credit balance.
