# Diabetes Prediction EDA and Web Application

## Overview

This repository contains a project developed under Data Science and Visualization, where we explore and visualize a diabetes dataset, train various machine learning models, and deploy the best model using a Flask web application for predicting diabetes.

## Supervised by
[Prof. Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu), (Assistant Professor) Department of CSE, MIT Mysore
## Project Structure

```plaintext
Diabetes_Prediction/
│
├── dataset/
│   └── diabetes.csv                    # Dataset from Kaggle
│
├── notebooks/
│   └── diabetes_prediction.ipynb       # Jupyter Notebook for EDA and model training
│
├── app/
│   ├── templates/
│   │   ├── index.html                  # HTML form for user input
│   │   └── result.html                 # HTML page for displaying prediction result
│   ├── static/
│   │   └── style.css                   # CSS file for styling
│   ├── app.py                          # Flask application code
│   ├── random_forest.h5                # Trained Random Forest model in .h5 format
│   └── random_forest.pkl               # Trained Random Forest model
│
├── .gitignore                          # Git ignore file
├── README.md                           # Project description and instructions
└── requirements.txt                    # Python dependencies
```

## Features

- **Exploratory Data Analysis (EDA)**: Visualization and analysis of the diabetes dataset using `diabetes_prediction.ipynb`.
- **Model Training and Selection**: Training various machine learning models and selecting the best model based on performance metrics.
- **Web Application**: A Flask-based web application that allows users to input medical data and get predictions on the likelihood of diabetes.

## Data

The dataset used in this project is the diabetes dataset from Kaggle, which includes the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (target column where 1 indicates diabetes and 0 indicates no diabetes)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/suhaskm28/diabetes-prediction-app.git
    ```
2. On Windows use
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. Navigate to the project directory:
    ```bash
    cd Diabetes-Prediction
    ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Jupyter Notebook

1. Open the Jupyter Notebook:
    ```bash
    jupyter notebook notebooks/diabetes_prediction.ipynb
    ```
2. Run the notebook to perform EDA, train models, and select the best model.

### Flask Application

1. Ensure you have the trained Logistic Regression model saved as `app/logistic_regression_model.pkl`.
2. Run the Flask application:
    ```bash
    python app/app.py
    ```
3. Open your web browser and go to `http://127.0.0.1:5000` to use the application.

## Code Explanation

`app.py`
This is the main Flask application file.
```plaintext
from flask import Flask, render_template, request
import pickle
import numpy as np
import h5py

app = Flask(__name__)

# Load the trained model from .h5 file
model_h5_path = 'app/random_forest.h5'
with h5py.File(model_h5_path, 'r') as h5_file:
    model_byte_stream = h5_file['random_forest_model'][()]
    model = pickle.loads(model_byte_stream.tobytes())

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
    input_data = np.array(
        [[pregnancies, glucose, blood_pressure, bmi, diabetes_pedigree_function, age]])
    prediction = model.predict(input_data)

    # Prepare response
    if prediction[0] == 1:
        result = 'Diabetes'
    else:
        result = 'No Diabetes'

    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
```

## File Descriptions

- `dataset/diabetes.csv`: The dataset used for training and predictions.
- `notebooks/diabetes_prediction.ipynb`: Jupyter Notebook containing EDA, model training, and selection.
- `app/templates/index.html`: The home page with the input form for user data.
- `app/templates/result.html`: The results page displaying the prediction.
- `app/app.py`: The Flask application code.
- `app/logistic_regression_model.pkl`: The saved Logistic Regression model.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `README.md`: Project description and instructions.
- `requirements.txt`: Lists the Python dependencies for the project.

### Diabetes Prediction

#### Input Form
![Input Form](https://github.com/suhaskm28/Diabetes_Prediction/blob/main/images/Diabetes_form.png)

#### Prediction Result
![Prediction Result](https://github.com/suhaskm28/Diabetes_Prediction/blob/main/images/Diabetes_output.png)

### Diabetes Visualization

#### EDA - Pairplot
![Pairplot](https://github.com/suhaskm28/Diabetes_Prediction/blob/main/images/pair_plot.png)

#### EDA - Correlation Heatmap
![Correlation Heatmap](https://github.com/suhaskm28/Diabetes_Prediction/blob/main/images/correlation_matrix.png)

## Conclusion

This project demonstrates the application of data science techniques to predict diabetes using machine learning. By performing thorough exploratory data analysis (EDA), training multiple models, and deploying the best model using Flask, we provide a practical tool for diabetes prediction. 


## License
This project is licensed under the MIT License.



