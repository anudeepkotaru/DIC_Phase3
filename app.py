from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect all form data
    input_features = [x for x in request.form.values()]
    feature_names = ['Customer_ID', 'Month', 'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 
                     'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                     'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 
                     'Credit_Utilization_Ratio', 'Credit_History_Age_Months', 
                     'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly', 
                     'Payment_Behaviour', 'Monthly_Balance']

    # Create DataFrame
    # Collect data from the form, use None if the field is empty
    input_features = [request.form.get(field) or None for field in feature_names]

    # Create DataFrame with the correct data types
    data = pd.DataFrame([input_features], columns=feature_names)

    # Handle types and conversion (this may need to be adjusted based on the actual model requirements)
    float_fields = ['Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
                    'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    int_fields = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                  'Delay_from_due_date', 'Credit_History_Age_Months']

    for field in float_fields:
        data[field] = pd.to_numeric(data[field], errors='coerce')
    for field in int_fields:
        data[field] = pd.to_numeric(data[field], errors='coerce', downcast='integer')
    
    # Make prediction
    result = model.predict(data)
    y_labels = {0: "Good", 1:"Poor", 2:"Standard"}
    print(result)

    return render_template('result.html', prediction=y_labels[result[0]])


def load_model():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    train_data = pd.read_csv('clean_train_data.csv')
    # Encode the Customer_ID column
    label_encoder = LabelEncoder()
    train_data['Customer_ID'] = label_encoder.fit_transform(train_data['Customer_ID'])

    X = train_data.drop(columns=['Credit_Score'])
    y = train_data['Credit_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

    random_forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, min_samples_split=2)
    random_forest.fit(X_train, y_train)

    print('Model loaded successfully...!!')

    return random_forest



if __name__ == '__main__':
    model = load_model()
    app.run()
