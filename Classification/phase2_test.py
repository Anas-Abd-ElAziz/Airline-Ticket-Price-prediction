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
import time

def learnData(fileName,modelName):
    df = pd.read_csv(f'{fileName}.csv')
    df = cp.preprocess(df)

    x_test = df.drop('TicketCategory', axis = 'columns')
    y_test = df['TicketCategory']

    pickled_model = pickle.load(open(f'{modelName}.sav', 'rb'))

    start = time.time()
    y_pred = pickled_model.predict(x_test)
    stop = time.time()
    testing_time = stop - start
    print("Testing time: {}".format(testing_time))

    accuracy=np.mean(y_pred == y_test)*100
    print (f"The achieved accuracy using {modelName} is " + str(accuracy))


learnData("airline-test-samples","AdaBoostModel")
learnData("airline-test-samples","TreeModel")
learnData("airline-test-samples","LogisticModel")
