import pandas as pd
import pickle
import preprocessing
import learn
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
import time

data = pd.read_csv('airline-test-samples.csv')
data = preprocessing.preprocess(data)

x_test = data[['airline', 'num_code', 'time_taken', 'type']]
y_test = data['price']

#Leaner
start = time.time()
pickled_model = pickle.load(open('len_model.sav', 'rb'))
y_pred = pickled_model.predict(x_test)

print("Time execution " ,time.time()- start)
print('MSE', metrics.mean_squared_error(y_pred, y_test))
print('True value: ' + str(np.asarray(y_test)))
print('Predicted value: ' + str(y_pred))
print('')

#Polynomial
start = time.time()
pickled_model = pickle.load(open('poly_model.sav', 'rb'))
x_test_poly = PolynomialFeatures(degree=7).fit_transform(x_test)
y_pred = pickled_model.predict(x_test_poly)

print("Time execution " ,time.time()- start)
print('MSE', metrics.mean_squared_error(y_pred, y_test))
print('True value: ' + str(np.asarray(y_test)))
print('Predicted value: ' + str(y_pred))
print('')

#GradientBoostingRegressor
start = time.time()
pickled_model = pickle.load(open('gbr.sav', 'rb'))
y_pred = pickled_model.predict(x_test)

print("Time execution " ,time.time()- start)
print('MSE', metrics.mean_squared_error(y_pred, y_test))
print('True value: ' + str(np.asarray(y_test)))
print('Predicted value: ' + str(y_pred))
print('')

#XGB Regressor
start = time.time()
pickled_model = pickle.load(open('xgb_r.sav', 'rb'))
y_pred = pickled_model.predict(x_test)

print("Time execution " ,time.time()- start)
print('MSE', metrics.mean_squared_error(y_pred, y_test))
print('True value: ' + str(np.asarray(y_test)))
print('Predicted value: ' + str(y_pred))
print('')
