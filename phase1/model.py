import pandas as pd
import pickle
import preprocessing
import learn

data = pd.read_csv('airline-price-prediction.csv')

data = preprocessing.preprocess(data)

top_feature = learn.correlateData(data)

Y = data['price'] #Goal
X = data[top_feature]
X = X.drop(['price'], axis = 1)

X = learn.normalizeData(X)

sln = learn.train_linear_model(X, Y)
pickle.dump(sln, open('len_model.sav', 'wb'))

xgb_r = learn.train_XGB_Regressor_model(X,Y)
pickle.dump(xgb_r, open('xgb_r.sav', 'wb'))

gbr = learn.train_GradientBoostingRegressor_model(X,Y)
pickle.dump(gbr, open('gbr.sav', 'wb'))

leaner = learn.train_poly_model(X, Y, 7)
pickle.dump(leaner, open('poly_model.sav', 'wb'))








