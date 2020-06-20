from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

import pandas as pd

from sklearn import preprocessing

from sklearn import tree 
from sklearn.tree import export_graphviz

from IPython.display import Image

data = pd.read_csv('exported_data.csv')

X = data.iloc[:,:-1].values
y = data['label']
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

DecisionTreeClassifier_model = DecisionTreeClassifier()

DecisionTreeClassifier_model.fit(X_train, y_train)

DecisionTreeClassifier_prediction = DecisionTreeClassifier_model.predict(X_test)
scores = cross_validation.cross_val_score(DecisionTreeClassifier_model, X_test, y_test, cv=3)

print(scores)
print (scores.mean())
print("Decision Tree: ", accuracy_score(DecisionTreeClassifier_prediction, y_test))
print("Decision Tree Cross validation: ", cross_val_score(DecisionTreeClassifier_model, X, y, cv=10))


export_graphviz(dt, 
                out_file='DOT-files/tree.dot', 
                feature_names=cols)


Image(filename='tree.png')