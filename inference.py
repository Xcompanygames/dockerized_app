import pickle
import pandas as pd
import numpy as np
from flask import Flask, request

# model = pd.read_pickle(r'churn_model.pkl')
df = pd.read_csv('X_test.csv')
y_pred = np.loadtxt('preds.csv')
loaded_model = pickle.load(open('churn_model.pkl', 'rb'))

app = Flask(__name__)
@app.route("/predict_single")
def predict():

    # is_male,num_inters,late_on_payment,age,years_in_contract
    is_male = request.args.get("is_male")
    num_inters = request.args.get("num_inters")
    late_on_payment = request.args.get("late_on_payment")
    age = request.args.get("age")
    years_in_contract = request.args.get("years_in_contract")
    flask_pred = loaded_model.predict([[is_male,
                                       num_inters,
                                       late_on_payment,
                                       age,
                                       years_in_contract]])
    return f"{flask_pred[0]}"


if __name__ == '__main__':
    # model = pd.read_pickle(r'churn_model.pkl')
    # df = pd.read_csv('X_test.csv')
    # y_pred = np.loadtxt('preds.csv')
    #loaded_model = pickle.load(open('churn_model.pkl', 'rb'))

    app.run(host='0.0.0.0')
