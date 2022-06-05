import numpy as np
import pandas as pd
import classification_preprocessing as cp
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle

data = pd.read_csv("airline-price-classification.csv")
data = cp.preprocess(data)
Attributes = data.drop('TicketCategory', axis = 'columns')
Classes = data['TicketCategory']
X_train, X_test, y_train, y_test = train_test_split(Attributes, Classes, test_size=0.33, random_state=100)


#AdaBoost
AdaBoostModel = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=50).fit(X_train,y_train)
y_pred = AdaBoostModel.predict(X_test)
accuracy=np.mean(y_pred == y_test)*100
print ("The achieved accuracy using AdaBoost is " + str(accuracy))
pickle.dump(AdaBoostModel, open('AdaBoostModel.sav', 'wb'))



#DecisionTree
TreeModel = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=40, min_samples_leaf=15).fit(X_train,y_train)
y_pred = TreeModel.predict(X_test)
accuracy=np.mean(y_pred == y_test)*100
print ("The achieved accuracy using TreeModel is " + str(accuracy))
pickle.dump(TreeModel, open('TreeModel.sav', 'wb'))


#LogisticRegression
LogisticModel = LogisticRegression(solver='liblinear',max_iter=50).fit(X_train, y_train)
y_pred = LogisticModel.predict(X_test)
accuracy=np.mean(y_pred == y_test)*100
print ("The achieved accuracy using LogisticRegression is " + str(accuracy))
pickle.dump(LogisticModel, open('LogisticModel.sav', 'wb'))
#Low Correlation between features -> Low Accuracy
#No linear correlation between features and prediction