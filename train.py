import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np


df = pd.read_csv('cellular_churn_greece.csv')
X = df.drop('churned',axis = 1)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
file_name = 'churn_model.pkl'
pickle.dump(clf, open(file_name, "wb"))
X_test.to_csv('X_test.csv',index=False)
np.savetxt('preds.csv', y_pred)
