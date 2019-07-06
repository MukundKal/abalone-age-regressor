# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('abalone.csv',header = None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#encoding catogorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#encode the gender variable
encoder = LabelEncoder()
X[:,0] = encoder.fit_transform(X[:,0])
onehot = OneHotEncoder(categorical_features=[0])
X = onehot.fit_transform(X).toarray()

#remove one of the one-hot encodes feature columns - dummy variable trap
X=X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting regressor to the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=500, random_state= 0)
regressor.fit(X_train,y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Calculating accuracy.
from sklearn.metrics import  r2_score,mean_squared_error
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'accuracy: {score}')
print(f'mse: {mse}')




