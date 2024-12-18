import numpy as np
import joblib

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

# Input prompts
features = [
    "Enter the mean radius: ", "Enter the mean texture: ", "Enter the mean perimeter: ",
    "Enter the mean area: ", "Enter the mean smoothness: ", "Enter the mean compactness: ",
    "Enter the mean concavity: ", "Enter the mean concave points: ", "Enter the mean symmetry: ",
    "Enter the mean fractal dimension: ", "Enter the radius error: ", "Enter the texture error: ",
    "Enter the perimeter error: ", "Enter the area error: ", "Enter the smoothness error: ",
    "Enter the compactness error: ", "Enter the concavity error: ", "Enter the concave points error: ",
    "Enter the symmetry error: ", "Enter the fractal dimension error: ", "Enter the worst radius: ",
    "Enter the worst texture: ", "Enter the worst perimeter: ", "Enter the worst area: ",
    "Enter the worst smoothness: ", "Enter the worst compactness: ", "Enter the worst concavity: ",
    "Enter the worst concave points: ", "Enter the worst symmetry: ", "Enter the worst fractal dimension: "
]

# Collect user input
print("\nPlease enter the following values:")
input_data = []

for feature in features:
    while True:
        try:
            value = float(input(feature))
            input_data.append(value)
            break
        except ValueError:
            print("Invalid input! Please enter a valid number.")

# Predict and display the result
input_data_array = np.array(input_data).reshape(1, -1)
prediction = model.predict(input_data_array)

if prediction[0] == 0:
    print("\nThe cancer is Malignant. Please consult a doctor for treatment.")
else:
    print("\nThe cancer is Benign. No immediate concern.")
