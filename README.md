## credit-risk-classification
**Module 20 Challenge - Supervised Machine Learning**

In this scenario, I employed Python and supervised machine learning techniques within a Jupyter Notebook to develop a logistic regression model. This model was trained to assess the creditworthiness of borrowers.


**Repository Folders and Contents:**
- Credit_Risk:
  - Resources:
    - lending_data.csv
  - credit_risk_classification.ipynb


## Table of Contents

- [Overview of the Analysis](#overview-of-the-analysis)
- [Results](#results)
- [Summary](#summary)
- [Getting Started](#getting-started)
- [Installing](#installing)
- [Contributing](#contributing)


## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

**Resource File I Used:**
  - lending_data.csv

**My Jupyter Notebook Python Cleaning Script:**
  - credit_risk_classification.ipynb

**Tools/Libraries I Imported:**
  - import numpy as np # For numerical operations and calculations
  - import pandas as pd # To read and manipulate the lending data as a dataframe
  - from pathlib import Path # To specify the the file path for reading the csv file
  - from sklearn.model_selection import train_test_split # To split the dataset into training and testing data
  - from sklearn.linear_model import LogisticRegression # To create and train a logistic regression model
  - from sklearn.metrics import confusion_matrix, classification_report # To evaluate the model's performance
  - import seaborn as sns # To create pairplots and heatmaps to visualize data relationships and correlations
  - import matplotlib.pyplot as plt # To create and display visualizations, including heatmaps and confusion matrices
  - from scipy import stats # To calculate the Pearson correlation coefficient
  - from statsmodels.stats.outliers_influence import variance_inflation_factor # To test for multicolinearity in independant variables

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

**Scatterplot Matrix:**

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/0289d6f7-abc5-4a1c-9bdb-5f4a6d72d916)

**Correlation Matrix Heatmap:**

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/a6b4bc24-0481-4536-98dd-b9a100a58185)

**Variation Inflation Factor Values:**

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/0de49276-1984-4198-8357-e49246b1f907)

**Confusion Matrix:**

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/a89e9d09-d928-42df-bac7-2e7c0141968a)

**Classification Report:**

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/20539208-e730-478c-82cf-16d30a4fb3c1)




## Summary

Overall, the logistic regression model appears to perform well in predicting both healthy loans and high-risk loans. The precision for high-risk loans is 0.85, indicating that the model makes fewer false positive predictions for high-risk loans. The recall for high-risk loans is 0.91, suggesting that the model effectively captures a significant proportion of high-risk loans. The accuracy of the model is 0.99, showing that the overall model has high accuracy for predicting both healthy and high-risk loans.

However, when looking at the correlation coefficient and the VIF scores, there is evidence of multicollinearity among at least two or more of the independent variables. As mentioned earlier, this leads to challenges in calculating the coefficients for each independent variable and accurately assessing their impact on the dependent variable, along with the potential for overfitting. Some suggestions to potentially improve the model include:

  - Removing one or more of the highly correlated variables.
  - Combining correlated variables to create composite variables.
  - Acquiring more data to better distinguish the relationships between the variables.

## Getting Started

**Programs/software I used:**
  - Jupyter Notebook: python programming tool, was used for data manipulation and consolidation.

**To activate dev environment and open Jupyter Notebook:**
  - Open Anaconda Prompt
  - Activate dev environment, type 'conda activate dev'
  - Navigate to the folder where repository is saved on local drive
  - Open Jupyter Notebook, type 'Jupyter Notebook'

## Installing

**Install scikit-learn library**
  - https://scikit-learn.org/stable/install.html
  
## Contributing
  - How to create pairplots plots: https://seaborn.pydata.org/generated/seaborn.pairplot.html
  - How to create correlation matrix heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html
  - How to calculate variation inflation factor: https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
