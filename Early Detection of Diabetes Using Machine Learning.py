# Early Detection of Diabetes Using Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

def main():
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    data = pd.read_csv(url, names=columns)

    # Replace zeros with NaN for specific columns
    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_not_allowed:
        data[col] = data[col].replace(0, np.nan)

    # Fill missing values with median (fixed line)
    for col in zero_not_allowed:
        median = data[col].median()
        data[col] = data[col].fillna(median)

    # Visualize Glucose distribution by Outcome
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Outcome', y='Glucose', data=data)
    plt.title("Glucose Levels by Diabetes Outcome")
    plt.savefig('glucose_distribution.png')  # Save figure
    plt.close()

    # Prepare data
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, 'diabetes_model.pkl')

if __name__ == '__main__':
    main()




