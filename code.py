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

def convert(data):
    number = preprocessing.LabelEncoder()
    for i in range(5,39):
        colmn = str(i) + "SS"
        data[colmn] = number.fit_transform(data[colmn])
    data=data.fillna(-999)
    return data
    
def convert_test_vector(data):
    number = preprocessing.LabelEncoder()
    data[colmn] = number.fit_transform(data[colmn])
    return data

data = pd.read_csv('exported_data.csv')
# data.drop('1SS', axis=1, inplace=True)
# data.drop('2SS', axis=1, inplace=True)
# data.drop('3SS', axis=1, inplace=True)
# data.drop('4SS', axis=1, inplace=True)
# data=convert(data)
# print(data.head(5))
# data.to_csv (r'exported_data.csv', index = False, header=True)
# print(data.tail(30))


X = data.iloc[:,:-1].values
y = data['label']
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
# print(X_train, X_test, y_train, y_test)

# SVC_model = SVC()
# KNN_model = KNeighborsClassifier(n_neighbors=5)
GaussianNB_model = GaussianNB()
# DecisionTreeClassifier_model = DecisionTreeClassifier(random_state=0)
# LinearDiscriminantAnalysis_model = LinearDiscriminantAnalysis()
# RandomForestClassifier_model = RandomForestClassifier(n_estimators=100)


# SVC_model.fit(X_train, y_train)
# KNN_model.fit(X_train, y_train)
GaussianNB_model.fit(X_train, y_train)
# DecisionTreeClassifier_model.fit(X_train, y_train)
# LinearDiscriminantAnalysis_model.fit(X_train, y_train)
# LogisticRegression_model = LogisticRegression(random_state=0).fit(X, y)
# RandomForestClassifier_model.fit(X, y)

# SVC_prediction = SVC_model.predict(X_test)
# KNN_prediction = KNN_model.predict(X_test)
GaussianNB_prediction = GaussianNB_model.predict(X_test)
# DecisionTreeClassifier_prediction = DecisionTreeClassifier_model.predict(X_test)
# LinearDiscriminantAnalysis_prediction = LinearDiscriminantAnalysis_model.predict(X_test)
# LogisticRegression_prediction = LogisticRegression_model.predict(X_test)
# RandomForestClassifier_prediction = RandomForestClassifier_model.predict(X_test)


print("Naïve Bayes: ", accuracy_score(GaussianNB_prediction, y_test))
# print("SVM: ", accuracy_score(SVC_prediction, y_test))
# print("KNN: ", accuracy_score(KNN_prediction, y_test))
# print("Decision Tree: ", accuracy_score(DecisionTreeClassifier_prediction, y_test))
# print("Decision Tree Cross validation: ", cross_val_score(DecisionTreeClassifier_model, X, y, cv=10))
# print("Linear Discriminant Analysis: ", accuracy_score(LinearDiscriminantAnalysis_prediction, y_test))
# print("Random Forest: ", accuracy_score(RandomForestClassifier_prediction, y_test))

# print("Logistic: ", LogisticRegression_model.score(X_test, y_test))

# print(RandomForestClassifier_model.feature_importances_)

## Finding Feature importance
# x = RandomForestClassifier_model.feature_importances_
# dic = {}
# for i in range(5,39):
#     dic[i] = x[i-5]
# sorted_dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
# print(sorted_dic)

# print(confusion_matrix(GaussianNB_prediction, y_test))
print(GaussianNB_model.predict_proba([[-999,3,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999]]))
print(GaussianNB_model.predict([[np.nan,3,4,2,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999]]))