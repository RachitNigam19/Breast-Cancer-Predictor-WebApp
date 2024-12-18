# Importing all the dependencies
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize the Flask app
app = Flask(__name__)

# Loading the trained model through joblib
model = joblib.load('breast_cancer_model.pkl')

features = [
    "Enter the mean radius: ",
    "Enter the mean texture: ",
    "Enter the mean perimeter: ",
    "Enter the mean area: ",
    "Enter the mean smoothness: ",
    "Enter the mean compactness: ",
    "Enter the mean concavity: ",
    "Enter the mean concave points: ",
    "Enter the mean symmetry: ",
    "Enter the mean fractal dimension: ",
    "Enter the radius error: ",
    "Enter the texture error: ",
    "Enter the perimeter error: ",
    "Enter the area error: ",
    "Enter the smoothness error: ",
    "Enter the compactness error: ",
    "Enter the concavity error: ",
    "Enter the concave points error: ",
    "Enter the symmetry error: ",
    "Enter the fractal dimension error: ",
    "Enter the worst radius: ",
    "Enter the worst texture: ",
    "Enter the worst perimeter: ",
    "Enter the worst area: ",
    "Enter the worst smoothness: ",
    "Enter the worst compactness: ",
    "Enter the worst concavity: ",
    "Enter the worst concave points: ",
    "Enter the worst symmetry: ",
    "Enter the worst fractal dimension: ",
]

@app.route('/')
def home():
    return render_template('index.html', features=features, enumerate=enumerate)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        int_features = [float(x) for x in request.form.values()]  # Convert form values to floats
        final_features = np.array([int_features])  # Convert to NumPy array
        prediction = model.predict(final_features)  # Make prediction
        
        # Determine the output based on the prediction
        if prediction[0] == 0:
            output = 'The cancer is Malignant. Please consult a doctor for treatment.'
        else:
            output = 'The cancer is Benign. No immediate concern.'
        
        # Render the result on the same page
        return render_template('index.html', prediction_text=f'Prediction: {output}', features=features, enumerate=enumerate)
    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
