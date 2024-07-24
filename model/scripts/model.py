import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier


adult_df = pd.read_csv("adult.csv", names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"], skipinitialspace=True)

# Split the dataset into features and target
X = adult_df.drop('income', axis=1)
y = adult_df['income']

# Encode categorical variables
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Calculate feature importances
rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(X_encoded, y)
importances = rf.feature_importances_

# Calculate cumulative importances
cumulative_importances = np.cumsum(importances)

# Find number of features for cumulative importance of 95%
num_features = np.where(cumulative_importances > 0.95)[0][0] + 1

# Extract the names of the most important features
feature_names = list(X.columns)
important_feature_names = [feature_names[i] for i in np.argsort(importances)[-num_features:]]

# Extract the most important features
important_features = X_encoded[:, np.argsort(importances)[-num_features:]]

# Handle the skewed nature of the data
positive_class = np.where(y == '>50K')[0]
negative_class = np.where(y == '<=50K')[0]

# Oversample the minority class
negative_class = np.random.choice(negative_class, size=len(positive_class), replace=True)

# Combine the positive and negative classes
X_train = np.concatenate([important_features[positive_class], important_features[negative_class]], axis=0)
y_train = np.concatenate([y[positive_class], y[negative_class]], axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model using the most important features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate the classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Print the accuracy score and the classification report
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
