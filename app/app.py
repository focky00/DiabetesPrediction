from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open(r'C:\Users\ADMIN\Desktop\DataScienceLab\project\dsproject-2.2\diabetes_prediction_app\random_forest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Home route - renders the form


@app.route('/')
def home():
    return render_template('index.html')

# Prediction route - handles form submission and prediction


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    pregnancies = int(request.form.get('pregnancies', 0))
    glucose = int(request.form['glucose'])
    blood_pressure = int(request.form['blood_pressure'])
    skin_thickness = int(request.form.get('skin_thickness', 0))
    insulin = int(request.form.get('insulin', 0))
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(
        request.form.get('diabetes_pedigree_function', 0.0))
    age = int(request.form['age'])

    # Make prediction using the loaded model
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            bmi, diabetes_pedigree_function, age]])
    prediction = model.predict(input_data)

    print(prediction[0])
    # Prepare response
    if prediction[0] == 1:
        result = 'Diabetes'
    else:
        result = 'No Diabetes'

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
