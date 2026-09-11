# Comparative Study of Machine Learning Algorithms for Predicting Electromagnetic Field Levels in Railway Transport

This repository contains the core codebase and machine learning pipeline developed for my Bachelor’s thesis.

## 🚀 Project Overview
The project focuses on predicting electromagnetic field (EMF) levels in railway environments to ensure strict compliance with human safety exposure limits (ICNIRP guidelines). It implements an end-to-end data science pipeline—from raw data processing to predictive modeling and model interpretability.

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & Manipulation:** Pandas, NumPy (IQR outlier removal, data scaling)
* **Machine Learning Architectures:** Scikit-Learn (Random Forest, Linear Regression, KNN, Gradient Boosting), Keras/TensorFlow (ANN, 1D CNN)
* **Model Explainability:** SHAP

## 📊 Key Pipeline Steps
1. **Data Cleaning:** Rigorous handling of raw physical measurements using the Interquartile Range (IQR) method to eliminate anomalies.
2. **Model Training & Evaluation:** Comparative training across multiple traditional ML algorithms and deep learning architectures designed to prevent underestimations and prioritize safety.
3. **Interpretability:** Applying SHAP analysis to extract feature importance.

## 📈 Results & Visualizations
![Model Results](images)
