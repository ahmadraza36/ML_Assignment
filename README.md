# 🎬 Movie Box Office Revenue Prediction

## 1. Project Title
**Predicting Film Box Office Revenue Using Supervised Machine Learning & Data Pipeline Engineering**

---

## 2. Problem Statement
The film industry involves high-stakes financial investments, where accurately predicting box office success before release is critical for studios, streaming platforms, and investors. The goal of this project is to build an end-to-end Machine Learning pipeline to clean, process, and analyze metadata from over 45,000 films, ultimately predicting global box office revenue (`log_revenue`). We address extreme skewness, missing structural attributes, and non-linear feature interactions while implementing core pipeline components from scratch.

---

## 3. Group Members

| Student Name | Roll Number | Contribution |
| :--- | :--- | :--- |
| **Md. Ahmad Raza** | *[Insert Roll No.]* | Data Cleaning, Pipeline Architecture, From-Scratch Modules & ML Models |
| **Student 2** | *[Insert Roll No.]* | *[Insert Contribution]* |
| **Student 3** | *[Insert Roll No.]* | *[Insert Contribution]* |
| **Student 4** | *[Insert Roll No.]* | *[Insert Contribution]* |

---

## 4. Dataset Description
The project utilizes the **TMDB 5000 / Movies Metadata Dataset**, containing comprehensive metadata for over 45,000 movies released on or before July 2017. 

* **Total Raw Records:** 45,466 films
* **Target Variable:** `revenue` (Continuous financial metric transformed via $\ln(\text{Revenue} + 1)$)
* **Key Attributes:** `budget`, `popularity`, `vote_count`, `vote_average`, `runtime`, `genres`, `original_language`, `status`

---

## 5. Dataset Source
* **Source Platform:** Kaggle Datasets / The Movie Database (TMDB)
* **Data URL:** [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/sibamsamanta07/movies-dataset-45k-films-with-budget-and-revenue?resource=download)

---

## 6. Preprocessing Techniques Implemented
* **Duplicate Removal:** Identified and removed duplicate film IDs (`subset=['id']`) across the raw metadata.
* **Missing Value Imputation:** Applied numerical median imputation for missing `runtime` entries ($95.0\text{ minutes}$) to preserve distribution symmetry.
* **Skewness Correction:** Applied log transformation ($\ln(X + 1)$) to right-skewed revenue data, reducing target skewness from **$+5.1446$** to **$-1.6966$**.
* **Categorical Encoding:** 
  * **Label Encoding:** Encoded target ordinal stage statuses (`status`) from scratch.
  * **One-Hot Encoding:** Built binary indicator matrices for primary nominal genre tags (`primary_genre`).
* **Feature Scaling:** Evaluated standard $Z$-score scaling vs. Min-Max normalization for continuous predictors.

---

## 7. Feature-Selection Techniques Implemented
* **Pearson Correlation Matrix:** Built custom linear correlation evaluation across numerical features against target `revenue` and `log_revenue`.
* **Multicollinearity Elimination:** Removed redundant raw values (`budget`, `revenue`) in favor of stabilized log-scale features.
* **Low-Relevance / High-Cardinality Exclusion:** Dropped non-predictive metadata strings, raw text descriptions (`overview`, `tagline`), and unique URLs (`poster_path`, `homepage`).

---

## 8. From-Scratch Implementations
The following algorithms and data engineering functions were built without relying on external pre-built wrapper libraries (such as `scikit-learn` transformers):

1. **Label Encoder:** Native Python dictionary lookup and index-mapping logic for ordinal categories.
2. **One-Hot Encoder:** Custom binary column generation utilizing list comprehensions and dictionary mappings.
3. **Outlier Boundary Detection:**
   * **IQR Bounds:** $Q_1$, $Q_3$, and Outlier limits ($Q_1 - 1.5\text{IQR}$, $Q_3 + 1.5\text{IQR}$).
   * **Z-Score Calculation:** Mean ($\mu$), Standard Deviation ($\sigma$), and outlier detection where $|Z| > 3$.
4. **Skewness Formula Engine:** Sample skewness calculation via:
   $$\text{Skewness} = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left( \frac{X_i - \bar{X}}{s} \right)^3$$
5. **Pearson Correlation Coefficient Engine:** Custom sample covariance and standard deviation cross-product logic:
   $$r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2} \sqrt{\sum (Y_i - \bar{Y})^2}}$$

---

## 9. Results

| Model Name | MAE | RMSE | $R^2$ Score (Variance Explained) |
| :--- | :--- | :--- | :--- |
| **Baseline Model (Mean Predictor)** | $1.7463$ | $2.5733$ | $-0.0009$ ($0.0\%$) |
| **Linear Regression Benchmark** | $1.1455$ | $1.7396$ | $0.5426$ ($54.26\%$) |
| **Random Forest Regressor** | **$1.0508$** | **$1.6227$** | **$0.6020$ ($60.20\%$)** |

---

## 10. Selected Features
The final vector of predictors selected for model training:

1. `log_budget`: Log-transformed financial production budget ($\ln(\text{Budget} + 1)$).
2. `vote_count`: Total number of user ratings logged.
3. `popularity`: TMDB continuous popularity engagement score.
4. `vote_average`: Mean score awarded by viewers ($0\text{ to }10$).
5. `runtime`: Duration of the film in minutes.
6. `primary_genre`: One-Hot encoded primary genre indicators.

---

## 11. Key Findings
* **Strongest Linear Predictors:** User engagement metric **`vote_count`** ($r = 0.77$) and production budget **`log_budget`** ($r = 0.70$) exhibited the highest predictive correlation with revenue.
* **Non-Parametric Outlier Robustness:** The IQR method identified $5,416$ duration boundary outliers due to tight variance around $90\text{–}100\text{ minute}$ films, whereas the $Z$-Score method ($|Z| > 3$) flagged $303$ severe numerical anomalies.
* **Non-Linear Model Advantage:** The **Random Forest Regressor** outperformed standard linear regression, expanding explained variance ($R^2$) to **$60.20\%$** while minimizing mean prediction error ($1.0508$ log-units).

---

## 12. Instructions to Run the Code

### **Prerequisites**
Ensure Python 3.8+ is installed on your environment.

### **1. Clone Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/movie-revenue-prediction.git](https://github.com/YOUR_USERNAME/movie-revenue-prediction.git)
cd movie-revenue-prediction
