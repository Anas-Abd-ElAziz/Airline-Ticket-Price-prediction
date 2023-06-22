import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import xgboost as xg
from sklearn.ensemble import GradientBoostingRegressor
import time


def correlateData(data):
  #correlation
  print(data.corr())
  top_feature = data.corr().index[abs(data.corr()['price'])>0.2]
  sns.heatmap(data[top_feature].corr(), annot = True)
  plt.show()

  return top_feature


def normalizeData(X):
  #normalization
  for coulm in X.columns:
    X[coulm] = (X[coulm] - X[coulm].min()) / (X[coulm].max() - X[coulm].min())
  
  return X


def train_poly_model(X, Y, d = 2):
  # polynomial model
  x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.20 ,shuffle=True) # train_test_split

  poly = PolynomialFeatures(degree=d) #declare poly transformer
  x_train = poly.fit_transform(x_train) #transforms features to higher degree
  x_test = poly.fit_transform(x_test) #transforms features to higher degree
  leaner = LinearRegression(normalize=True).fit(x_train, y_train) #declare leaner model & Normalize & fit
  y_train_predicted = leaner.predict(x_train) # predict train data-set
  y_test_predicted = leaner.predict(x_test) # predict test data-set
  scores = abs(cross_val_score(linear_model.LinearRegression(), x_train, y_train, scoring='neg_mean_squared_error', cv=5).mean())

  print('RMSE of train = ', metrics.mean_squared_error(y_train, y_train_predicted,squared = False))
  print('RMSE of test = ', metrics.mean_squared_error(y_test, y_test_predicted,squared = False))
  print('Average Mean price = ', Y.mean() )
  print('Cross val score = ', scores)
  print('true_value = ',np.asarray(y_test)[12])
  print('prediction_value = ',y_test_predicted[12])
  print('')

  return leaner

def train_linear_model(X, Y):
  #linear model
  X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(X, Y, test_size = 0.20,shuffle=False,random_state=10)
  sln= linear_model.LinearRegression()
  sln.fit(X_lin_train, y_lin_train)
  prediction=sln.predict(X_lin_test)
  true_value=np.asarray(y_lin_test)[0]
  predicted_value=prediction[0]
  print('Co-efficient of multiple linear regression',sln.coef_)
  print('Intercept of multiple linear regression model',sln.intercept_)
  print('Mean Square Error to multiple linear regression', metrics.mean_squared_error(y_lin_test, prediction))
  print('True value in the test set in millions is : ' + str(true_value))
  print('Predicted value in the test set in millions is : ' + str(predicted_value))
  print('')

  return sln

def train_XGB_Regressor_model(X, Y):
  #XGB_Regressor model
  start = time.time()
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=10)
  xgb_r = xg.XGBRegressor(objective='reg:linear', n_estimators=10)
  xgb_r.fit(x_train, y_train)
  pred = xgb_r.predict(x_test)
  print('XGB MSE', metrics.mean_squared_error(y_test, pred))
  print('true_value = ',np.asarray(y_test)[0])
  print('prediction_value = ',pred[0])
  print("Time execution " ,time.time()- start)
  print('')

  return xgb_r

def train_GradientBoostingRegressor_model(X, Y):
  #GradientBoostingRegressor model
  start = time.time()
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=10)
  gbr_params = {'n_estimators': 1000,'max_depth': 3, 'learning_rate': 0.01}
  gbr = GradientBoostingRegressor(**gbr_params).fit(x_train, y_train)
  pred = gbr.predict(x_test)
  print('GBR MSE', metrics.mean_squared_error(y_test, pred))
  print('true_value = ',np.asarray(y_test)[0])
  print('prediction_value = ',pred[0])
  print("Time execution " ,time.time()- start)
  print('')

  return gbr
