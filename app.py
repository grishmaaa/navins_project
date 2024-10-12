from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('loan_approval_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route that renders the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        request.form['Gender'],
        request.form['Married'],
        request.form['Dependents'],
        request.form['Education'],
        request.form['Self_Employed'],
        request.form['ApplicantIncome'],
        request.form['CoapplicantIncome'],
        request.form['LoanAmount'],
        request.form['Loan_Amount_Term'],
        request.form['Credit_History'],
        request.form['Property_Area']
    ]

    # Convert form data to a NumPy array
    features = np.array(features).reshape(1, -1)

    # Make a prediction using the model
    prediction = model.predict(features)

    # Render the result
    if prediction == 1:
        return render_template('index.html', prediction_text='Loan Approved')
    else:
        return render_template('index.html', prediction_text='Loan Rejected')

if __name__ == '__main__':
    app.run(debug=True)
