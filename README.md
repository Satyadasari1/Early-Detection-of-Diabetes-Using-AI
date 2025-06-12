# Early Detection of Diabetes Using AI

## 🔬 Overview
This project leverages **Machine Learning**—specifically **Logistic Regression**—to predict whether a person is likely to be diabetic based on various health indicators. It supports early diagnosis using AI-powered models and is designed to be scalable via cloud platforms like **Microsoft Azure** and optionally deployable using **Streamlit**.

---

## 📊 Dataset
- **Source:** Pima Indians Diabetes Dataset  
- **Link:** [UCI Repository](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## ⚙️ Technologies & Tools Used
- **Programming Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib
- **Platform:** Microsoft Azure Machine Learning Studio
- **Optional Deployment:** Streamlit + Hugging Face Spaces

---

## 🧠 Machine Learning Model
- **Algorithm Used:** Logistic Regression
- **Data Split:** 80% training, 20% testing
- **Preprocessing:**
  - Replaced 0s in `Glucose`, `BloodPressure`, `BMI`, etc. with NaN
  - Filled missing values using the median
- **Evaluation:**
  - Accuracy: ~77%
  - Confusion Matrix
  - Classification Report
  - Visualization: Boxplot for glucose distribution by outcome

---

## 🗃️ Folder Structure


├── diabetes_prediction.py # Main model script
 ├── requirements.txt # List of Python dependencies
 ├── diabetes_model.pkl # Trained model file
 ├── glucose_distribution.png # Visualization chart
 ├── README.md # This documentation file

---

## 🚀 How to Run the Project Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Satyadasari1/early-diabetes-prediction.git
   cd early-diabetes-prediction

Install Dependencies

 pip install -r requirements.txt


Run the Script

 python diabetes_prediction.py


Model Output


Accuracy


Confusion matrix


Saved model file: diabetes_model.pkl


Saved visualization: glucose_distribution.png



🌐 Deployment Options
Microsoft Azure ML Studio: Full integration for training and deployment


Streamlit Web App: Optional web interface for real-time predictions


Hugging Face Spaces: Host Streamlit apps for free



📌 Future Enhancements
Add more sophisticated models (Random Forest, XGBoost)


Build a mobile/web application interface


Integrate real-time wearable data


Extend predictions to other chronic diseases



🧾 References
Pima Indians Diabetes Dataset


Scikit-learn Documentation


Microsoft Azure ML Studio


Hugging Face Spaces
