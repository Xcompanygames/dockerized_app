import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
test = pd.read_csv('X_test.csv').to_dict('records')
y_pred = list(np.loadtxt('preds.csv'))
y_pred_flask = []

for i in tqdm(test):
    r = requests.get('http://127.0.0.1:5000/predict_single', params=i).content
    y_pred_flask.append(float(r.decode()))

print(y_pred_flask)
print(y_pred)

print(y_pred_flask==y_pred)