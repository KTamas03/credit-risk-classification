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

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarise the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

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
