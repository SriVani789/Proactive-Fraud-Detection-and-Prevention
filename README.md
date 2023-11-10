# Proactive Fraud Detection and Prevention

## Overview
This project focuses on proactively detecting and preventing fraud in financial transactions using a machine learning approach. The goal is to identify fraudulent activities in financial data to enhance security measures and prevent potential risks.

## Contents
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Data
The financial transaction data is stored in a CSV file (`Fraud.csv`). It includes features such as amount, old balance, new balance, transaction type, etc., and a binary label indicating whether the transaction is fraudulent.

## Data Preprocessing
- Missing values are handled by dropping rows with missing data.
- Outliers in numerical columns (amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) are handled using the IQR method.
- A preprocessing pipeline is created to scale numerical features and one-hot encode categorical features.

## Feature Engineering
- Additional features are engineered, including 'hour_of_day' and 'day_of_week' based on the 'step' feature.

## Model Training
- The data is split into training and validation sets.
- A RandomForestClassifier is trained on the preprocessed data.

## Model Evaluation
- The trained model is evaluated using confusion matrix and classification report on the validation set.
- Missing values in target labels are checked and handled.

## Feature Importance
- Feature importance is determined using the RandomForestClassifier.
- The top features contributing to the model are displayed.

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Follow the instructions in the project code for data loading, preprocessing, model training, and evaluation.

## Dependencies
- pandas
- numpy
- scikit-learn
