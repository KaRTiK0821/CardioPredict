# =========================================
# Cardiovascular Disease Prediction Project
# =========================================
# This script predicts heart disease using different ML algorithms
# Dataset: cardio_train.csv
# Author: [Your Name]
# =========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# To ignore warnings for clean output
import warnings
warnings.filterwarnings("ignore")

# =========================================
# Step 2: Load Dataset
# =========================================
data = pd.read_csv("cardio_train.csv", sep=';')  # dataset uses semicolon as separator
print("Dataset Shape:", data.shape)
print("First 5 rows:\n", data.head())

# =========================================
# Step 3: Preprocessing
# =========================================

# Drop 'id' column since it's not useful
data.drop('id', axis=1, inplace=True)

# Convert age from days to years for better understanding
data['age'] = (data['age'] / 365).round().astype(int)

# Check for null values (just in case)
print("Missing values:\n", data.isnull().sum())

# Quick look at data types
print(data.dtypes)

# Handling outliers (blood pressure values)
# Some systolic (ap_hi) or diastolic (ap_lo) values are unrealistic (like negative or very high)
data = data[(data['ap_hi'] >= 80) & (data['ap_hi'] <= 200)]
data = data[(data['ap_lo'] >= 50) & (data['ap_lo'] <= 150)]

print("Shape after removing outliers:", data.shape)

# =========================================
# Step 4: Exploratory Data Analysis (EDA)
# =========================================

# Distribution of target variable
plt.figure(figsize=(5, 4))
sns.countplot(x='cardio', data=data, palette='coolwarm')
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.show()

# Age distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

# =========================================
# Step 5: Prepare Data for Modeling
# =========================================

X = data.drop('cardio', axis=1)
y = data['cardio']

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for models like SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================
# Step 6: Train Multiple Models
# =========================================

# Dictionary to store model accuracy
model_accuracy = {}

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
model_accuracy['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
model_accuracy['Decision Tree'] = accuracy_score(y_test, y_pred_dt)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
model_accuracy['Random Forest'] = accuracy_score(y_test, y_pred_rf)

# SVM
svm_clf = SVC()
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)
model_accuracy['SVM'] = accuracy_score(y_test, y_pred_svm)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
model_accuracy['KNN'] = accuracy_score(y_test, y_pred_knn)

# =========================================
# Step 7: Print Accuracy of All Models
# =========================================
print("\nModel Accuracy Comparison:")
for model, acc in model_accuracy.items():
    print(f"{model}: {acc:.4f}")

# =========================================
# Step 8: Detailed Metrics for Each Model
# =========================================
def show_metrics(name, y_true, y_pred):
    print(f"\n{name} Model Metrics")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

show_metrics("Logistic Regression", y_test, y_pred_lr)
show_metrics("Decision Tree", y_test, y_pred_dt)
show_metrics("Random Forest", y_test, y_pred_rf)
show_metrics("SVM", y_test, y_pred_svm)
show_metrics("KNN", y_test, y_pred_knn)

# =========================================
# Step 9: Feature Importance (Random Forest)
# =========================================
importances = rf.feature_importances_
features = X.columns
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.show()
