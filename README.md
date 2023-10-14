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

**Purpose of the analysis:** My goal in this analysis was to create a machine learning model to predict the creditworthiness of borrowers. I aimed to assess whether a borrower's loan should be categorized as "Healthy Loan" or "High-Risk Loan" based on the available financial information.
  
**Data Used:** I worked with a dataset contained in the "lending_data.csv" file, located in the "Resources" folder. This dataset contained various financial and credit-related information about borrowers, such as loan size, interest rate, debt to income ratio, number of accounts, derogatory marks, total debt and loan status. The target variable I wanted to predict was "loan_status," which is binary and indicates whether a loan is healthy (0) or high-risk (1).
  
**Machine Learning Stages:**

<ins>1. Data Loading:<\ins>
   
    - I started by importing the necessary libraries and loading the lending data from the CSV file into a Pandas DataFrame.

2. Exploratory Data Analysis (EDA):

    - I began by examining the first and last rows of the dataset to get an initial sense of its structure.
    - I created a scatterplot matrix (pairplot) and a correlation matrix heatmap to visualize data distributions and identify relationships between variables.
    - To assess multicollinearity, I calculated Variance Inflation Factors (VIF).

3. Data Preprocessing:

    - I divided the dataset into features (independent variables) and labels (the dependent variable).
    - I calculated VIF scores to detect multicollinearity among independent variables.

4. Model Building:

    - I split the data into training and testing sets using train_test_split.
    - I instantiated a logistic regression model and trained it using the training data.
  
5. Model Evaluation:

    - I assessed the model's performance using a confusion matrix, classification report, and various metrics, including accuracy, precision, recall, and F1-score.
    - The confusion matrix provided counts of true positives, true negatives, false positives, and false negatives.
    - The classification report displayed precision, recall, F1-score, and support for each class (healthy and high-risk loans).


**Methods Used:**

  - I used a logistic regression model for binary classification of loans.
  - To identify multicollinearity, I calculated Variance Inflation Factor (VIF) scores.
  - I split the dataset into training and testing sets.
  - Model evaluation was performed using a confusion matrix and a classification report.

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

**1. Scatterplot Matrix:**
The charts below visually indicate a high correlation between most of the independent variables in the lending_df dataframe.

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/0289d6f7-abc5-4a1c-9bdb-5f4a6d72d916)


**2. Correlation Matrix Heatmap:**
The matrix below once again reveals a very high correlation between the independent variables, as indicated by the predominantly high Pearson correlation coefficient values, most of which are over 0.80. This suggests the presence of multicollinearity within the lending_df dataframe.

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/a6b4bc24-0481-4536-98dd-b9a100a58185)


**3. Variation Inflation Factor Values:**
The extremely high VIF values (any score exceeding 5) indicate the presence of multicollinearity within the lending_df dataframe. This implies that accurately determining the coefficients for each independent variable and their true impact on the dependent variable will be challenging. Additionally, there is potential for overfitting, meaning the model may capture noise in the data due to highly correlated variables, rather than the genuine underlying relationships.

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/0de49276-1984-4198-8357-e49246b1f907)

**4. Confusion Matrix:**
The confusion matrix shows that the model correctly predicted the vast majority of healthy loans in the dataset (18663). The model also accurately predicted 563 high-risk loans. However, there were 102 false positives, meaning the model incorrectly predicted high-risk loans that were actually healthy loans. Additionally, there were 56 false negatives, indicating cases where the model incorrectly predicted healthy loans that were actually high-risk loans.

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/a89e9d09-d928-42df-bac7-2e7c0141968a)

![image](https://github.com/KTamas03/credit-risk-classification/assets/132874272/bf057b41-ef1c-49fa-b11e-a846efc05ba4)

**5. Classification Report:**
In this classification report, the accuracy score of 0.99 signifies that the model's predictions are correct 99% of the time. Accuracy measures the overall correctness of the model's predictions, encompassing both true positives and true negatives.

Furthermore, the precision score for high-risk loans stands at 0.85, indicating that 85% of the loans predicted as high-risk are indeed high-risk loans. This metric reflects the model's proficiency in avoiding false positive errors. Precision assesses how accurately the model identifies positive class predictions (e.g., high-risk loans).

In addition, the recall score for high-risk loans is 0.91, signifying that the model correctly identifies 91% of the actual high-risk loans. This metric, also known as sensitivity, gauges the model's capability to avoid false negatives. It quantifies how many of the true positive cases (high-risk loans) the model successfully predicted.

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
