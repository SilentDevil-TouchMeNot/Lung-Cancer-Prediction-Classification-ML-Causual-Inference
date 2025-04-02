# 🏥 Lung Cancer Prediction & Causal Inference 📊

## 📌 Project Overview
This project focuses on predicting lung cancer cases and examining relationships among various risk factors using statistical tests and pre‐processing techniques. The work is implemented in Python and involves detailed exploratory data analysis (EDA), data cleaning, scaling, multicollinearity diagnostics, testing the linearity assumption of logistic regression (using the Box-Tidwell test), model building, applying different classification models, testing their performances, evaluating the models and hyperparameter tuning and lastly doing a Causual Inference.

---

## 📂 Dataset
- **Source:** A CSV file named “Lung Cancer Dataset.csv” located in the `/kaggle/input/lung-disease-data/` directory.
- **Size:** 5000 samples with 18 columns.
- **Features Include:**
  - Demographic (e.g., AGE, GENDER)
  - Lifestyle (e.g., SMOKING, ALCOHOL_CONSUMPTION)
  - Clinical/Physical symptoms (e.g., FINGER_DISCOLORATION, BREATHING_ISSUE, THROAT_DISCOMFORT)
  - Environmental factors (e.g., EXPOSURE_TO_POLLUTION)
  - Family history (e.g., FAMILY_HISTORY, SMOKING_FAMILY_HISTORY)
  - Others: MENTAL_STRESS, LONG_TERM_ILLNESS, ENERGY_LEVEL, IMMUNE_WEAKNESS, CHEST_TIGHTNESS, STRESS_IMMUNE
- **Target Variable:** `PULMONARY_DISEASE` (originally as “YES”/“NO”, then mapped to binary: 1 for YES, 0 for NO).

---

## 🔄 Data Preprocessing
1. **Data Import & Initial Checks:**
   - Imported necessary libraries (Pandas, NumPy, Seaborn, Matplotlib, SciPy, Statsmodels,etc).
   - Read the CSV file and displayed both the first few and last few rows to inspect data consistency.
2. **Data Cleaning:**
   - Verified there were **no missing values** (using `data.isnull().sum()`) and confirmed that all 5000 rows were complete.
   - Checked for duplicate rows and found none; the data was then stored in a cleaned DataFrame.
3. **Data Type Consistency:**
   - Converted the target column `PULMONARY_DISEASE` from categorical (YES/NO) to numeric binary values.
4. **Descriptive Statistics & Visualizations:**
   - Generated summary statistics (mean, std, min, max) for all features.
   - Produced histograms for each numerical feature, scatter matrix plots, Q–Q plots, and KDE plots to assess distributions and detect skewness/kurtosis.
5. **Data Splitting & Scaling:**
   - **Train-Test Split:** Divided the data into 70% training and 30% testing sets with stratification on the target variable.
   - **Scaling (Case 1 was used):**
     - Applied **StandardScaler** on features assumed to be Gaussian (e.g., ENERGY_LEVEL and OXYGEN_SATURATION were initially candidates).
     - Applied **MinMaxScaler** on non-Gaussian features (e.g., AGE was scaled separately).

---

## 🚀 Steps Followed
1. **Exploratory Data Analysis (EDA):**
   - Inspected data distributions and correlations among features.
   - Visualized the correlation matrix; for instance, found a notable correlation (≈0.77) between `FAMILY_HISTORY` and `SMOKING_FAMILY_HISTORY`.
2. **Feature Selection & Multicollinearity Check:**
   - Calculated the Variance Inflation Factor (VIF) for predictors.
   - **VIF Elimination Process:**
     - Initial VIF values revealed extremely high values for `OXYGEN_SATURATION` (≈80.39) and `ENERGY_LEVEL` (≈47.85).
     - Iteratively dropped features with VIF values above 10, first removing `OXYGEN_SATURATION` and then `ENERGY_LEVEL`, until all remaining features had VIF below 10.
3. **Testing Model Assumptions:**
   - Performed a Box-Tidwell test on the `AGE` feature to assess the linearity assumption for logistic regression.
   - Multiple datasets were constructed by dropping different combinations of features (e.g., dropping `FAMILY_HISTORY` and/or `SMOKING_FAMILY_HISTORY`) and the test was run on each.
   - The Box-Tidwell test output showed that the interaction term for `AGE` was not statistically significant (p-value > 0.05), indicating no violation of the linearity assumption.
   - Different models were fit, trained and tested. Compared their performances based on Precision,Recall,Accuracy,F1,AUC-ROC Curve and other parameters.
   - Performed a Causual Inference as well.

## 🛠️ Models Tested

The following models were evaluated during the project:

- **Decision Tree:**  
  - *Implementation:* `DecisionTreeClassifier()`

- **Random Forest:**  
  - *Implementation:* `RandomForestClassifier()`

- **XGBoost:**  
  - *Implementation:* `XGBClassifier()`

- **LightGBM (LGB):**  
  - *Implementation:* `LGBMClassifier()`

- **CatBoost (CAT):**  
  - *Implementation:* `CatBoostClassifier()`

- **Support Vector Machine (SVM):**  
  - *Implementation:* `SVC(probability=True)`

- **K-Nearest Neighbors (KNN):**  
  - *Implementation:* `KNeighborsClassifier()`

- **Naive Bayes:**  
  - *Implementation:* `GaussianNB()`

- **Neural Network Architectures:**  
  - **Sequential Dense:**  
    - *Implementation:* `get_seq_dense()`
  - **Sequential Dropout:**  
    - *Implementation:* `get_seq_dropout()`

- **Logistic Regression Variants:**  
  - **Logistic Regression with VIF-based Feature Selection**
  - **Logistic Regression without VIF**
  - **Regularized Logistic Regression Models:**  
    - *Elastic Net*  
    - *Lasso*  
    - *Ridge*

    # For different Scenerios these moels were tested and evaluated.

---


## 🎯 Results & Findings
- **Data Distribution:**
  - The training set showed a class distribution of approximately 59% (No disease) and 41% (Disease), which was well preserved in the test set.
  - Model training and evaluation results were performed which can be viewed from the pdf.

---

## 🔍 Causal Inference

In this project, we applied causal inference techniques to better understand the relationship between key risk factors (e.g., SMOKING) and lung cancer (PULMONARY_DISEASE). The process involved the following steps:

1. **Causal Structure Learning:**
   - We filtered the dataset to include relevant features.
   - Using **pgmpy**'s **HillClimbSearch** with the BDeu scoring method, we estimated the best-fitting Bayesian network structure (limited to 20 iterations).
   - The learned network was converted into a directed graph and visualized using NetworkX. We confirmed that the graph is a valid Directed Acyclic Graph (DAG) with no cycles.

2. **Bayesian Network Construction & Inference:**
   - The identified DAG was transformed into a **Discrete Bayesian Network**.
   - Using Maximum Likelihood Estimation (MLE), we attached Conditional Probability Distributions (CPDs) to the network.
   - With **Variable Elimination**, we performed inference queries. For example, querying for P(PULMONARY_DISEASE) given SMOKING = 1 yielded:
     - P(PULMONARY_DISEASE = 0) ≈ 0.4313  
     - P(PULMONARY_DISEASE = 1) ≈ 0.5687

3. **Causal Effect Estimation Using DoWhy:**
   - The **DoWhy** package was utilized to define the causal model with:
     - **Treatment:** SMOKING  
     - **Outcome:** PULMONARY_DISEASE  
     - **Confounders:** AGE, EXPOSURE_TO_POLLUTION, IMMUNE_WEAKNESS, BREATHING_ISSUE, and FAMILY_HISTORY
   - The model identified the backdoor estimand, and we estimated the average treatment effect (ATE) via propensity score matching. The estimated causal effect was **0.4816**, suggesting that SMOKING increases the probability of lung cancer by approximately 48.16 percentage points when controlling for the specified confounders.

4. **Robustness & Sensitivity Analysis:**
   - **Placebo Treatment Refuter:** When replacing SMOKING with a placebo treatment, the estimated effect dropped to **-0.00151** (p-value: 0.98), indicating that the original effect is robust.
   - **Random Common Cause Refuter:** Adding a random common cause did not alter the effect, which remained at **0.4816** (p-value: 1.0), further confirming the stability of the causal estimate.

5. **Conclusion:**
   - The causal inference analysis provides strong evidence that SMOKING is a significant risk factor for lung cancer. The estimated average treatment effect of **0.4816** is robust to various refutation tests, indicating that the effect is not due to random chance or confounding factors.
   - These findings support the hypothesis that smoking has a direct causal influence on lung cancer risk. Additionally, the rigorous structure learning and robustness checks reinforce the validity of the causal model.
   - Overall, this analysis lays the groundwork for further exploration using advanced causal inference techniques, which could help in formulating targeted interventions and healthcare policies aimed at reducing lung cancer incidence.

---

## 💡 Future Directions
- **Deployment & Interpretation:**
  - Develop interactive dashboards for model interpretability (using tools such as SHAP) and consider deploying the model as a web application for clinical decision support.
- **Data Expansion:**
  - Incorporate additional features or larger datasets to improve model generalization and predictive accuracy.

---

## 🛠️ Technologies & Libraries Used
- **Python:** Core programming language.
- **Pandas & NumPy:** Data manipulation and analysis.
- **Matplotlib & Seaborn:** Data visualization.
- **Scikit-learn:** Data splitting, scaling, and VIF computation.
- **Statsmodels:** Statistical tests and logistic regression (for the Box-Tidwell test).
- **SciPy:** Additional statistical functions.
- Etc.

---
## This Repository will be updated with a report file later and some further additions will be improvised.
## 🤝 Contribution
Feel free to **fork** this project, raise issues, or contribute by making a pull request. For any questions or suggestions, please contact me at [rahuldebnath1438@gmail.com] 
