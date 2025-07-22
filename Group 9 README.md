# Group 9 Capstone Project: Predicting Income Level Using Census Data

## Project Overview

This project is the final capstone for CSIS 503 - Data Science & Analytics at Osiri University. Group 9 applied the full data science lifecycle from data acquisition through modeling and storytelling using the UCI Adult Census dataset. Our objective: **predict whether an individual earns more than $50K/year based on demographic and employment factors**.

---

## Problem Statement

**Can we accurately predict if a person earns >$50K/year using features such as age, education, occupation, and hours worked?**

This study explores how socioeconomic and demographic attributes influence income level. The goal is to uncover insights relevant to policy-making, workforce equity, and education outcomes.

---

## Dataset Summary

- **Source:** UCI Machine Learning Repository  
- **Name:** Adult Income Dataset  
- **Size:** 48,842 rows  
- **Original Features:** 15  
- **Post-Processing Features:** 99 (after one-hot encoding and binning)

🔗 [View Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

---

## Methodology

### Data Preparation
- Removed rows with missing or ambiguous values
- Feature engineering:
  - Binned age into groups
  - One-hot encoded categorical features
- Final dataset included 99 processed features

### Exploratory Data Analysis (EDA)
- Identified key drivers of income such as education level, capital gains, hours worked, and marital status
- Used visualizations (heatmaps, bar plots, histograms) to explore feature distributions and correlations

### Modeling
- Implemented:
  - **Logistic Regression**
  - **Random Forest Classifier**
- Evaluation Metrics:
  - Accuracy
  - F1 Score
  - AUC (Area Under ROC Curve)
  - Confusion Matrix

---

## Results Summary

| Model                | Accuracy | AUC    | F1 Score |
|---------------------|----------|--------|----------|
| Logistic Regression | 84%      | 0.88   | 0.68     |
| Random Forest       | 87%      | 0.91   | 0.72     |

**Random Forest** performed best across all metrics.

---

## Key Insights

- **Education level** and **capital gain** are strong predictors of higher income
- **Gender disparity**: Males are more likely to earn >$50K
- **Marital status** and **weekly working hours** also play significant roles

---

## Future Enhancements

- Tune models using `GridSearchCV`
- Test ensemble models like `XGBoost` and `LightGBM`
- Address class imbalance using `SMOTE`
- Add interpretability tools like `SHAP`
- Wrap workflow in a full `scikit-learn` pipeline

---

## Team Roles (Group 9)

| Member               | Role Description                                                  |
|----------------------|--------------------------------------------------------------------|
| ThankGod Israel      | **Team Lead** – Data cleaning, EDA, final report writing           |
| Member 1             | **Modeling Lead** – Developed and evaluated ML models             |
| Member 2             | **Visualization Lead** – Built plots and explained feature importances |
| Member 3             | **QA Reviewer / GitHub Manager** – Cleaned notebook, structured repo |
| Member 4             | **Presentation Specialist** – Designed and formatted slides        |

---

## Repository Structure

```
Capstone-Income-Prediction
capstone.ipynb               # Final notebook with code and results
adult_combined.csv           # Cleaned dataset
top_10_feature_importance.png
Capstone_Report.md           # Written project summary
Capstone_Slides.pdf          # Final presentation
README.md                    # Project overview (this file)
```

---

## Instructors

- **Dr. Noble Anumbe** [noble@osiriuniversity.org]
- **TA:** Sebastian Boscan Villalobos

---

## Key Dates

- Dataset Approval: **July 26, 2025**  
- Final Presentation: **August 16, 2025**  
- Submission Deadline: **August 31, 2025**

---

> Data will talk to you if you're willing to listen. Jim Bergeson
