# Proactive-Fraud-Detection-and-Prevention
Proactive Fraud Detection and Prevention in Financial Transactions: A Machine Learning Approach


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv('Fraud.csv')

# Handle missing values
data.dropna(inplace=True)

# Handle outliers
def handle_outliers(df, cols):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].apply(lambda x: max(lower_bound, min(upper_bound, x)))
    return df

numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
data = handle_outliers(data, numerical_cols)

# Feature engineering
data['hour_of_day'] = data['step'] % 24
data['day_of_week'] = (data['step'] // 24) % 7

# Split data into features (X) and target (y)
X = data.drop(['isFraud'], axis=1)
y = data['isFraud']

# Verify and convert categorical columns to string
categorical_cols = ['type']
X[categorical_cols] = X[categorical_cols].astype(str)

# Preprocessing pipeline for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing pipeline to training and validation sets
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# Impute missing values in X_train_processed
imputer = SimpleImputer(strategy='mean')
X_train_processed = imputer.fit_transform(X_train_processed)

# Check and handle missing values in target labels
y_train = y_train.fillna(0)

# Build and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_processed, y_train)

# Impute missing values in X_val_processed
X_val_processed = imputer.transform(X_val_processed)

# Evaluate the model
y_pred = model.predict(X_val_processed)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Feature importance
if 'cat' in preprocessor.named_transformers_:
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_feature_names)
else:
    feature_names = numerical_cols

feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)
