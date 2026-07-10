<div align="center">

# 🛠️ Predictive Maintenance of Industrial Machinery

### ML model that predicts industrial machine failure type from real-time sensor data — enabling proactive maintenance on IBM Cloud.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![IBM Cloud](https://img.shields.io/badge/IBM%20Cloud-Watson%20Studio%20%2B%20WML-052FAD?logo=ibm&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployed%20%26%20Tested-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-~98%25-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## 🔍 Overview

Unplanned failures in industrial machinery lead to costly downtime and reactive maintenance. This project builds a **multi-class classification model** that identifies *which type* of failure is likely — Tool Wear, Heat Dissipation, Power Failure, Overstrain, or Random Failure — from live operational sensor readings, so maintenance teams can intervene before a breakdown happens.

> Mechanical Engineering · Machine Learning Project · Problem Statement No. 39
> Built and deployed end-to-end on **IBM Cloud Lite**.

---

## 🎯 Problem Statement

<details>
<summary><b>Click to expand full problem statement</b></summary>

<br>

Develop a predictive maintenance model for a fleet of industrial machines to anticipate failures before they occur, by analyzing sensor data to identify patterns that precede a failure. The model should classify the type of failure based on real-time operational data.

**Technology requirement:** IBM Cloud Lite services (Watson Studio + Watson Machine Learning).

</details>

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [Machine Predictive Maintenance Classification (Kaggle)](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) |
| **Size** | 10,000 records × 10 columns |
| **Features** | Air/Process temperature, Rotational speed, Torque, Tool wear, Machine quality Type (L/M/H) |
| **Target** | `Failure Type` — 6 classes |
| **Class balance** | ⚠️ Highly imbalanced — 96.5% "No Failure" |

<details>
<summary><b>📈 Failure type distribution</b></summary>

<br>

| Failure Type | Count | % |
|---|---:|---:|
| No Failure | 9,652 | 96.52% |
| Heat Dissipation Failure | 112 | 1.12% |
| Power Failure | 95 | 0.95% |
| Overstrain Failure | 78 | 0.78% |
| Tool Wear Failure | 45 | 0.45% |
| Random Failures | 18 | 0.18% |

</details>

---

## ⚙️ Project Workflow
1. **Exploratory Data Analysis** — distribution plots, correlation heatmap, boxplots across failure types
2. **Feature Engineering** — encoded `Type`; derived `Power [W] = Torque × Rotational Speed`; derived `Temp Diff [K]`
3. **Preprocessing** — stratified train/test split, `StandardScaler`
4. **Class Imbalance Handling** — SMOTE oversampling (training set only)
5. **Model Training** — Random Forest Classifier
6. **Hyperparameter Tuning** — GridSearchCV
7. **Evaluation** — accuracy, F1, classification report, confusion matrix, feature importance
8. **Deployment** — published & deployed via IBM Watson Machine Learning as a live REST scoring endpoint

---

## 🧰 Tech Stack

<div align="center">

| Category | Tools |
|---|---|
| Language | ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) |
| Data | ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) |
| Visualization | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C) ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB) |
| ML | ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikitlearn&logoColor=white) |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Cloud | ![IBM Cloud](https://img.shields.io/badge/-IBM%20Cloud-052FAD?logo=ibm&logoColor=white) Watson Studio · Watson Machine Learning · Cloud Object Storage |

</div>

---

## 📈 Results

<div align="center">

| Metric | Score |
|:---:|:---:|
| **Test Accuracy** | 🎯 ~98% |
| **Weighted F1-Score** | ⭐ ~0.98 |
| **Best Algorithm** | Random Forest + SMOTE |

</div>

The model performs strongly on majority classes (No Failure, Power Failure, Heat Dissipation) and reasonably on Overstrain Failure. Performance on **Random Failures** and **Tool Wear Failure** is limited by very low sample counts in the raw dataset — see [Limitations](#-limitations--future-work).

**Top predictive features:** Torque, Rotational Speed, and the engineered Power feature.

---

## 🚀 Live Deployment

This model is deployed on **IBM Watson Machine Learning** as a live, callable REST scoring endpoint — tested with 6 real sensor readings, correctly returning all 6 expected failure categories (No Failure, Power Failure, Heat Dissipation Failure, Overstrain Failure, Tool Wear Failure).

<details>
<summary><b>🔗 Deployment details</b></summary>

<br>

- **Status:** `Deployed · Online` ✅
- **Endpoint type:** IBM Watson Machine Learning public scoring endpoint
- **Tested with:** 6 sample rows spanning all failure categories
- **Result:** 6/6 correct predictions

</details>

---



<details>
<summary><b>📦 requirements.txt</b></summary>

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
jupyter
```

</details>

---



## ☁️ IBM Cloud Deployment Guide

1. **Watson Studio** — hosts the project and notebook
2. **Watson Machine Learning (Lite)** — stores and deploys the trained model as an online REST scoring endpoint
3. **Cloud Object Storage (Lite)** — backs the Watson Studio project's asset storage








