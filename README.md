# CardioPredict 🫀  
**Machine Learning based Cardiovascular Disease Prediction**

---

## 📌 Project Overview
CardioRisk Predictor is a machine learning project aimed at predicting the risk of cardiovascular disease using patient health data.  
The system uses multiple algorithms like:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**  

The dataset contains features such as **age, blood pressure, cholesterol, glucose levels, and lifestyle habits**.  
Early detection of heart disease can save lives, and this project demonstrates how ML can assist in making accurate predictions.

---

## ✅ Features
✔ Data Preprocessing (remove outliers, scale features, handle categorical variables)  
✔ Exploratory Data Analysis (EDA) with visualizations  
✔ Model Training and Accuracy Comparison  
✔ Confusion Matrix and Classification Report for all models  
✔ Feature Importance Analysis using Random Forest  

---

## 📂 Dataset
The dataset used: **cardio_train.csv** (Kaggle Cardio Dataset).  
It contains:
- **Features:** age (in years), gender, height, weight, blood pressure, cholesterol, glucose, smoking/alcohol activity indicators.
- **Target:** `cardio` (0 = No disease, 1 = Disease)

---

## 🛠️ Tech Stack
- **Python 3.x**
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn

---

## 📊 Workflow
1. **Data Preprocessing**
   - Convert `age` from days to years
   - Remove invalid blood pressure values
   - Drop unnecessary columns
2. **EDA**
   - Histograms, Correlation Matrix, Count Plots
3. **Model Training**
   - Train 5 ML models (LR, DT, RF, SVM, KNN)
4. **Evaluation**
   - Accuracy, Confusion Matrix, Classification Report
   - Feature Importance from Random Forest

---

## 📷 Example Visualizations
- Correlation Heatmap  
- Target Distribution  
- Feature Importance  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/KaRTiK0821/CardioPredict.git
