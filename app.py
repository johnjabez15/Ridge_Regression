from flask import Flask, render_template, request
import os
import pickle
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Define the paths for the model and other directories
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "ridge_regression_model.pkl")

# Load the trained model and the scaler from the pickle file
try:
    with open(MODEL_PATH, "rb") as f:
        # We stored the model and scaler as a dictionary, so we load them back this way.
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
except FileNotFoundError:
    print(f"Error: The model file was not found at {MODEL_PATH}.")
    print("Please run ridge_model.py first to train and save the model.")
    exit()

@app.route('/')
def home():
    """Renders the home page with the sales prediction form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request from the form.
    
    It collects the user's input, scales it using the pre-trained scaler,
    and then uses the loaded Ridge Regression model to make a prediction.
    The predicted sales value is then displayed on a result page.
    """
    try:
        # Get the form data from the POST request
        tv_budget = float(request.form['tv_budget'])
        radio_budget = float(request.form['radio_budget'])
        newspaper_budget = float(request.form['newspaper_budget'])
        
        # Create a numpy array from the input, as the scaler expects a 2D array.
        # The order of features must match the order used during training:
        # ['TV_Ad_Budget', 'Radio_Ad_Budget', 'Newspaper_Ad_Budget']
        input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])

        # Scale the input data using the pre-trained scaler
        scaled_input_data = scaler.transform(input_data)
        
        # Make a prediction using the loaded model
        prediction = model.predict(scaled_input_data)[0]

        # Format the prediction to two decimal places
        formatted_prediction = f"${prediction:,.2f}k"

        # Render the result page with the prediction
        return render_template('result.html', prediction=formatted_prediction)

    except ValueError:
        # Handle cases where the input is not a valid number
        error_message = "Invalid input. Please enter valid numbers for all fields."
        return render_template('result.html', prediction=error_message)
    except Exception as e:
        # Handle any other unexpected errors
        error_message = f"An error occurred: {str(e)}"
        return render_template('result.html', prediction=error_message)

if __name__ == '__main__':
    # Run the app. It will be accessible at http://127.0.0.1:5000/
    app.run(debug=True)
