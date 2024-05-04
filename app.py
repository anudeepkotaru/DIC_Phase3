from flask import Flask, jsonify, request, render_template
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import base64
from io import BytesIO
import os

app = Flask(__name__)

analytics = dict({0:0, 1:0, 2:0})

#Load the ML Random Forest Model
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
model = load_model()

#Plot the analytics graph
def plot_analytics():
    file_path = './static/plot.png'

    labels = ['Good', 'Poor', 'Standard']
    counts = [analytics[0], analytics[1], analytics[2]]
    plt.plot(labels, counts, marker='o')
    plt.xlabel('Prediction Outcome')
    plt.ylabel('Counts')
    plt.title('Prediction Analytics')
    plt.savefig(file_path,  dpi=300)
    plt.close()


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collect all form data
        input_features = [x for x in request.form.values()]
        feature_names = ['Customer_ID', 'Month', 'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 
                        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 
                        'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 
                        'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly', 
                        'Payment_Behaviour', 'Monthly_Balance']

        # Create DataFrame from the form, use None if the field is empty
        input_features = [request.form.get(field) or None for field in feature_names]

        # Create DataFrame with the correct data types
        data = pd.DataFrame([input_features], columns=feature_names)

        # Handle categorical values
        credit_mix = {'good':0, 'standard':1, 'other':2}
        data['Credit_Mix'] = credit_mix[data['Credit_Mix'][0]]

        # Handle types and conversion
        float_fields = ['Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
                        'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
        int_fields = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                    'Delay_from_due_date', 'Credit_History_Age_Months']

        for field in float_fields:
            data[field] = pd.to_numeric(data[field], errors='coerce')
        for field in int_fields:
            data[field] = pd.to_numeric(data[field], errors='coerce', downcast='integer')
        
        #Encoding results to labels
        y_labels = {0: "Good", 1:"Poor", 2:"Standard"}
        result = model.predict(data)

        analytics[result[0]] += 1
        plot_analytics()
        prediciton = y_labels[result[0]]
        print(result)

        #Return the results to the website
        return jsonify({'prediction': prediciton})
    
    else:
        return render_template('index_layout.html')


if __name__ == '__main__':
    app.run()
