import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv('loan_prediction.csv')
df = df.drop('Loan_ID', axis=1)

# Fill missing values
df.fillna({
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'LoanAmount': df['LoanAmount'].median(),
    'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0],
    'Credit_History': df['Credit_History'].mode()[0]
}, inplace=True)

# Convert categorical columns into dummy variables
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])

# Convert target variable to binary
df['Loan_Status'] = df['Loan_Status'].map({'N': 0, 'Y': 1})

# Split features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Find best threshold (adjust to prioritize precision/recall balance)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_idx = np.argmax(f1_scores)  # Still optimizing F1, but let’s check alternatives
best_threshold = thresholds[best_idx]

# Evaluate model at different thresholds
print("Evaluating at Best F1 Threshold:")
y_pred = (y_pred_proba > best_threshold).astype(int)
print(f"Threshold: {best_threshold:.5f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test default threshold (0.5) for comparison
print("\nEvaluating at Default Threshold (0.5):")
y_pred_default = (y_pred_proba > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred_default))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_default))

# Save model and metadata
with open('loan_approval_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)
with open('best_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and metadata saved successfully!")