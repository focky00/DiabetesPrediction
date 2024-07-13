# Diabetes Prediction EDA and Web Application

## Overview

This repository contains a project developed under Data Science and Visualization, where we explore and visualize a diabetes dataset, train various machine learning models, and deploy the best model using a Flask web application for predicting diabetes.

## Project Structure

```plaintext
diabetes-prediction-app/
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
│   ├── logistic_regression_model.pkl   # Trained Logistic Regression model
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
    git clone https://github.com/yourusername/diabetes-prediction-app.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Diabetes-Prediction
    ```
3. Install the required packages:
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

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License.
```
