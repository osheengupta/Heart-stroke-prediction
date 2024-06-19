# Heart stroke prediction using Logistic Regression and Random Forrest

This project's main goal is to create a predictive model that reliably identifies people who are at risk of stroke based on their health characteristics. This is a classification problem that predicts whether a person has a stroke or not with a good amount of accuracy. This project was done as part of Data Mining class at the California State University East Bay.

## Table of Contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Technologies and Tools](#technologies-and-tools)
* [Process](#process)
* [Code Examples](#code-examples)
* [Features](#features)
* [Status](#status)
* [Contact](#contact)

## General Info
Our aim is to develop a comprehensive predictive model for recognizing stroke events. It will provide insights and give a predictive model to help identify individuals at risk. This model will give healthcare professionals practical insights to enable early intervention and prevention. Our research aims to improve understanding of the factors that contribute to stroke risk, thus improving patient outcomes and healthcare delivery of a stroke. 

## Features
| Variables | Description |
| --- | --- |
| ID | Patients ID|
| GENDER | Gender of the patients (‘Male’, ‘Female’)|
| AGE | Age of the Patients |
| HYPERTENSION | Patients with Hypertension (“1”-if they had hypertension,”0”-if otherwise) |
| HEART_DISEASE | Patients with Heart disease (“1”-if they had any heart disease,”0”-if otherwise) |
| EVER_MARRIED | Marital Status (“Yes”-if they were married, “No”-if otherwise) |
| WORK_TYPE | Work type of the patients (“Children”,” Self-employed”, “Private”,” Never worked”,” Govt job”) |
| RESIDENCE_TYPE | Area where the Patients lived (“Rural”,” Urban”) |
| AVG_GLUCOSE_LEVEL | Average level of glucose of each patient |
| BMI | Basal Metabolic Index of each patient |
| SMOKING_STATUS | (“Formerly smoked”, “Never smoked”,” Smokes”,” Unknown”) |
| STROKE | Stroke rate (“1”-if they had a stroke,”0”-had no stroke) |


## Screenshots

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


## Process

A step by step guide that will tell you how to get the development environment up and running.

1.	Obtain Data for Analysis - The dataset can be found here.
2.	Explore, Clean and Preprocess Data; Reduce Data Dimension - The data set has 5110 rows and 12 columns. The column names were changed to uppercase. The column "ID" was dropped as it was irrelevant. The object type variables were converted to dummy      variables and one of their classes was dropped to prevent multicollinearity.
    There were some missing data points in BMI which was imputed by mean BMI value.
3. Determine the data mining task - Logistic Regression model and Decision trees have been used on the partitioned data where 60% was training data and 40% was test data.
4. Evaluation of models - Confusion matrix have been drwn and compared for both the models.
   

## Code Examples

A few examples of useful commands/code snippets.

### Logistic Regression Model
```
log_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
log_reg.fit(train_X, train_y)

# Show intercept and coefficients of the multiple predictors' logistic model.
print('Parameters of Logistic Regresion Model with Multiple Predictors')
print('Intercept:', np.round(log_reg.intercept_[0], decimals=3))
print('Coefficients for Predictors')
print(pd.DataFrame({'Coeff:': np.round(log_reg.coef_[0], decimals=3)}, 
                    index=X.columns).transpose())
```
### Decision tree 
```
param_grid = {
    'max_depth': list(range(2, 20)),  
    'min_impurity_decrease': [0, 0.0005, 0.001], 
    'min_samples_split': list(range(10, 30)),
}

# Apply GridSearchCV() fucntion for various combinations of
# DecisionTreeClassifier() improved parameters. 
gridSearch = GridSearchCV(DecisionTreeClassifier(), 
                param_grid, cv=5, n_jobs=1)
gridSearch.fit(train_X, train_y)

# Display best improved paramenters of classification tree. 
print()
print(f'Improved score:{gridSearch.best_score_:.4f}')
print('Improved parameters: ', gridSearch.best_params_)

# Create classification tree based on the improved parameters.
bestClassTree = gridSearch.best_estimator_

# Display classification tree based on improved parameters.
print('Best Classification Tree with Grid Search')
plotDecisionTree(bestClassTree, feature_names=train_X.columns)
```
### Random Forrest
```
# Apply RandomForestClassifier() function to develop a combined
# (ensemple) classification tree using Random Forest algorithm.
rf = RandomForestClassifier(n_estimators=500, random_state=1)
rf.fit(train_X, train_y)

# Display number of nodes in Random Forest trees.
n_nodes = rf.estimators_[0].tree_.node_count
print('Number of Nodes in Tree in Random Forest:', n_nodes)
```

## Status

Project is: _finished_ and was done as a part of Data Mining course at California State University East Bay.

## Contact
