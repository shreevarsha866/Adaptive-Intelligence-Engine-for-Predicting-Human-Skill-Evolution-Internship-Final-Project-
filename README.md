# 🧠 Adaptive Intelligence Engine for Predicting Human Skill Evolution in Digital Platforms

> **Internship Final Project — VCodez Data Science Internship (Jan 2026)**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)](https://powerbi.microsoft.com)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ig-soHvsEXjZteusuzT2x-QKEcu2UU1w?usp=sharing)

---

## 📌 Project Overview

This project builds an **Adaptive Intelligence Engine** that predicts student skill evolution on digital learning platforms using academic performance and behavioral data. The system applies a complete data science pipeline — from raw data cleaning to model deployment — to determine how well a student will develop their skills based on study habits, learning style, motivation, and prior academic history.

The final model (**Random Forest Regression**) achieved an **R² of 0.9133**, explaining over 91% of the variance in student skill outcomes — outperforming the Ridge Regression baseline significantly.

---

## 🎯 Business Problem

Educational platforms need to identify **which students are at risk of poor skill development early** — so they can intervene with personalized support. This engine provides a data-driven solution to predict final skill scores, enabling:

- 📚 Early identification of at-risk learners
- 🎯 Personalized learning pathway recommendations
- 📊 Data-backed decision making for educators and platform administrators

---

## 📊 Key Results

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression (Baseline) | 0.8628 | 8.52 | 11.34 |
| Ridge Regression (Tuned) | 0.8628 | 8.52 | 11.34 |
| **Random Forest (Final)** | **0.9133** | **7.20** | **9.02** |

✅ **Random Forest improved R² by ~5%** and reduced prediction error compared to the tuned baseline — confirming its ability to capture non-linear learning patterns.

---

## 🗂️ Project Pipeline

```
Raw Data (.xlsx)
     │
     ▼
1. Data Inspection        → shape, dtypes, nulls, duplicates
     │
     ▼
2. Data Cleaning          → fix special values, fill nulls (mean/mode)
     │
     ▼
3. Exploratory Analysis   → distributions, correlations, EDA visualizations
     │
     ▼
4. Feature Engineering    → one-hot encoding, StandardScaler, train-test split
     │
     ▼
5. Model Training         → Ridge Regression vs Random Forest Regression
     │
     ▼
6. Model Evaluation       → R², MAE, RMSE comparison
     │
     ▼
7. Feature Importance     → top 10 predictors identified
     │
     ▼
8. Model Saving           → Joblib (.pkl)
     │
     ▼
9. Power BI Dashboard     → business insights visualization
```

---

## 📂 Dataset Features

The dataset contains student academic and behavioral attributes including:

| Category | Features |
|---|---|
| **Academic** | G1 (mid-term 1), G2 (mid-term 2), G3 (final score — target), failures, studytime |
| **Behavioral** | absences, screen_time_hours, learning_style, motivation_level |
| **Demographic** | address (Urban/Rural), parent_education, famsup, schoolsup |

> **Target Variable:** `G3` — Final score (quantitative proxy for skill evolution)

---

## 🔍 EDA Highlights

- 📈 **G1 and G2** (prior academic scores) are the **strongest predictors** of final skill level
- ❌ **Failure history** significantly reduces predicted skill evolution
- 📱 Higher **screen time** shows a moderate negative correlation with final scores
- 🧠 **Learning style and motivation** contribute moderately to outcomes
- 🏠 Demographic features (address, parent education) have minimal predictive impact

---

## ⚙️ Feature Engineering

```python
# Target variable
X = df.drop(columns=['G3'])
y = df['G3']

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

---

## 🤖 Model Training

### Ridge Regression — Baseline
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_ridge = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')
grid_ridge.fit(X_train, y_train)
```
> R² = 0.8628 | MAE = 8.52 | RMSE = 11.34

### Random Forest Regression — Final Model
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
```
> **R² = 0.9133 | MAE = 7.20 | RMSE = 9.02** ✅

---

## 🏆 Top 10 Feature Importances

> Prior academic scores and failure history dominate skill evolution prediction

1. 🥇 G2 — Mid-term score 2
2. 🥈 G1 — Mid-term score 1
3. 🥉 failures — Number of past failures
4. absences — Number of absences
5. studytime — Weekly study time
6. screen_time_hours — Daily screen usage
7. motivation_level
8. learning_style
9. parent_education
10. address (Urban/Rural)

---

## 💾 Model Saving

```python
import joblib

joblib.dump(rf, "random_forest_skill_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

---

## 📊 Power BI Dashboard

An interactive **Power BI dashboard** was built to visualize:
- 📈 Skill evolution trends across learning styles
- 🎯 Score distribution by motivation level and study time
- ❌ Failure rate analysis by behavioral attributes
- 🗺️ Demographic impact on skill outcomes

> Dashboard file: `final_project_power_bi01.pbix`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core programming |
| **Pandas & NumPy** | Data manipulation |
| **Matplotlib & Seaborn** | EDA visualizations |
| **Scikit-learn** | ML modeling & evaluation |
| **Joblib** | Model persistence |
| **Power BI** | Business dashboard |
| **Google Colab** | Development environment |

---

## 🚀 How to Run

**Option 1 — Google Colab (Recommended)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ig-soHvsEXjZteusuzT2x-QKEcu2UU1w?usp=sharing)

**Option 2 — Run Locally**

```bash
# Clone the repo
git clone https://github.com/shreevarsha866/Adaptive-Intelligence-Engine-for-Predicting-Human-Skill-Evolution-Internship-Final-Project-

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl

# Run the notebook
jupyter notebook FINAL_PROJECT_DS.ipynb
```

---

## 📁 Repository Structure

```
📦 Adaptive-Intelligence-Engine
 ┣ 📓 FINAL_PROJECT_DS.ipynb       ← Main notebook
 ┣ 📊 final_project_power_bi01.pbix ← Power BI dashboard
 ┣ 📄 final_projectDS_powerbi.csv   ← Cleaned dataset export
 ┣ 🤖 random_forest_skill_model.pkl ← Saved RF model
 ┣ ⚙️  scaler.pkl                   ← Saved StandardScaler
 ┗ 📖 README.md
```

---

## 💡 Conclusion

- Machine learning can effectively predict student skill evolution using academic and behavioral data
- **Random Forest significantly outperformed Ridge Regression** by capturing non-linear learning patterns
- **Prior academic performance and failure history** are the strongest determinants of skill outcomes
- The Power BI dashboard translates model insights into actionable business intelligence

---

## 🔮 Future Work

- Incorporate real-time interaction data from live digital platforms
- Explore deep learning models (LSTM, Neural Networks) for sequential learning data
- Build personalized recommendation system based on predicted skill gaps
- Deploy as a web application using FastAPI + Streamlit

---

## 👩‍💻 Author

**Shreevarsha S**
Data Science Professional | ML & NLP Enthusiast

[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-7b6ef6?logo=globe)](https://shreevarsha866.github.io/Shreevarsha_Portfolio)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/s-shreevarsha-503887218/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/shreevarsha866)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:varshashree866@gmail.com)

---

*⭐ If you found this project helpful, please give it a star!*
