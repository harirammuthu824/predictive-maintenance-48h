
# Predictive Maintenance System for Industrial Rotating Equipment

## Project Overview

This project develops a machine learning based predictive maintenance
system for industrial rotating equipment. The system analyzes machine
sensor parameters and predicts whether a machine failure may occur
within the next 48 hours.

## Objective

To predict industrial equipment failure 48 hours in advance using
machine sensor data and machine learning techniques.

## Sensor Parameters

- Vibration
- Temperature
- Pressure
- Motor Current
- RPM
- Acoustic Level

## Methodology

1. Generate and collect machine sensor data
2. Perform exploratory data analysis
3. Analyze sensor correlations
4. Create a 48-hour failure prediction target
5. Perform feature engineering
6. Handle missing values
7. Train a Random Forest classifier
8. Evaluate model performance
9. Analyze feature importance
10. Generate failure predictions

## Feature Engineering

The following features were created:

- 24-hour rolling mean
- 24-hour rolling standard deviation
- 6-hour sensor change

## Machine Learning Model

Random Forest Classifier

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Google Colab

## Project Output

The system provides:

- 48-hour failure prediction
- Failure probability
- Machine condition classification
- Important sensor features
- Model performance analysis

## Project Goal

The proposed system can support predictive maintenance by identifying
potential machine failures before breakdown and helping maintenance
teams plan inspections and servicing.

## Author

Biomedical Engineering Student- HARI RAM MUTHU S


