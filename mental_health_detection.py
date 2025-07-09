# Machine Learning Based Mental Health Detection
# Simple logistic regression model based on survey data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv('mental_health_survey.csv')

# Preprocess
data = data.dropna()
X = data[['age', 'gender', 'work_interfere', 'family_history']].copy()
X = pd.get_dummies(X, drop_first=True)
y = data['treatment'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
