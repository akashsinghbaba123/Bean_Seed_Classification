#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Step 1: Load the Dataset
data = pd.read_csv("C:/Users/AKASH/Downloads/beans-230421-141141.csv")


# In[3]:


#shape of our data set
data.shape


# In[4]:


#data types of variables
data.dtypes


# In[5]:


#name of columns
data.columns


# In[6]:


#storing the independent variables
variables=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
       'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
       'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
       'ShapeFactor3', 'ShapeFactor4']


# In[7]:


#checking the null values in variables
data.isna().sum()


# In[8]:


# Identify the float continuous variables
float_continuous_variables = ['MinorAxisLength', 'ConvexArea', 'Compactness', 'ShapeFactor1', 'ShapeFactor4']

# Fill missing values with the median
for variable in float_continuous_variables:
    median_value = data[variable].median()
    data[variable].fillna(median_value, inplace=True)

# Verify the missing values are filled
missing_values = data[float_continuous_variables].isnull().sum()
print(missing_values)


# In[9]:


# Step 2: Exploratory Data Analysis (EDA)
# Visualize the distribution of variables
for variable in variables:
    
    plt.figure()
    sns.boxplot(data=data, y=variable)
    plt.title(f"Box Plot of {variable}")
    plt.show()
    
#We can see outliers in the variables 


# In[10]:


#EDA 
# Calculate correlation between features

correlation = data[variables].corr()
correlation


# In[11]:


# values count for Categorical variable and it classess
data['Class'].value_counts()


# In[28]:


#Label Encoding for the Class variable with numeric values 
data['Class']=data['Class'].map({'DERMASON':1,'SIRA':2,'SEKER':3,'HOROZ':4,'CALI':5,'BARBUNYA':6,'BOMBAY':7})


# In[29]:


# Calculate correlation between features and Target variable
correlation1 = data[variables].corrwith(data['Class'])
correlation1


# In[30]:


# Summarize descriptive statistics
data.describe()


# In[23]:


#depndent and independent variable
X = data.drop('Class', axis=1)
y = data['Class']

# Perform data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X=pd.DataFrame(X,columns=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
       'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
       'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
       'ShapeFactor3', 'ShapeFactor4'])
X.head()


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


# Step 5: Models

#Importing the necessary algorithm as told in question
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier

# Linear model with regularization (Logistic Regression)
logreg = LogisticRegression(penalty='l2', max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted')
print('F1 Score (Logistic Regression):', f1_logreg)


# Logisticregression is quite good f1_score 

# In[26]:


# Bagging model (Random Forest)
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
print('F1 Score (Random Forest):', f1_rf)

# Boosting model (AdaBoost)
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
f1_adaboost = f1_score(y_test, y_pred_adaboost, average='weighted')
print('F1 Score (AdaBoost):', f1_adaboost)



# Step 6: Stacking of Models

# Initialize base models
base_models = [
    LogisticRegression(penalty='l2', max_iter=1000),
    RandomForestClassifier(),
    AdaBoostClassifier()
]

# Initialize meta model (Logistic Regression)
meta_model = LogisticRegression(penalty='l2', max_iter=1000)

# Initialize stacking classifier
stacking = StackingClassifier(
    classifiers=base_models,
    meta_classifier=meta_model,
    use_probas=True,
    average_probas=False
)

# Fit the stacking classifier
stacking.fit(X_train, y_train)

# Predict using the stacking classifier
y_pred_stacking = stacking.predict(X_test)
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')
print('F1 Score (Stacking):', f1_stacking)


# We can see in the above f1 score that our Randomforest  and Stacking is working fine for this dataset

# In[27]:


# Step 7: Hyperparameter Tuning
# Grid search for Random Forest hyperparameters
param_grid_rf = {
    'n_estimators': [100, 200],   
    'max_depth': [5, 10]    #If our model not perform well on this hyperparameter then we can all add more hyperparameter
    
}


grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=3)
grid_search_rf.fit(X_train, y_train)

#  the best Random Forest model
best_rf = grid_search_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
f1_best_rf = f1_score(y_test, y_pred_best_rf, average='weighted')
print('Best F1 Score (Random Forest):', f1_best_rf)

# Grid search for AdaBoost hyperparameters
param_grid_adaboost = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

grid_search_adaboost = GridSearchCV(estimator=adaboost, param_grid=param_grid_adaboost, cv=3)
grid_search_adaboost.fit(X_train, y_train) 

# Get the best AdaBoost model
best_adaboost = grid_search_adaboost.best_estimator_
y_pred_best_adaboost = best_adaboost.predict(X_test)
f1_best_adaboost = f1_score(y_test, y_pred_best_adaboost, average='weighted')
print('Best F1 Score (AdaBoost):', f1_best_adaboost)


# Now Clearly we can see that out randomforest is working well for this bean classification problem so we need to use randomforest here.

# In[29]:


# Step 8: Analyze Feature Importance
# Random Forest Feature Importance
importances = best_rf.feature_importances_
feature_importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False)
print('Random Forest Feature Importance:')
print(feature_importance_rf)  


# In[30]:


# AdaBoost Feature Importance
importances = best_adaboost.feature_importances_
feature_importance_adaboost = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_adaboost = feature_importance_adaboost.sort_values(by='Importance', ascending=False)
print('AdaBoost Feature Importance:')
print(feature_importance_adaboost)  


# # According to importance of feature we can train our model while using the top features 

# In[35]:


X = data[['Area', 'Perimeter', 'MajorAxisLength',       
       'AspectRation','EquivDiameter','ShapeFactor1','ShapeFactor3']]
y = data['Class']

# Perform data preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X=pd.DataFrame(X,columns=['Area', 'Perimeter', 'MajorAxisLength',       
       'AspectRation','EquivDiameter','ShapeFactor1','ShapeFactor3'])
X.head()


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


#  Models

#Importing the necessary algorithm as told in question
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier

# Linear model with regularization (Logistic Regression)
logreg = LogisticRegression(penalty='l2', max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
f1_logreg = f1_score(y_test, y_pred_logreg, average='weighted')
print('F1 Score (Logistic Regression):', f1_logreg)


# In[38]:


# Bagging model (Random Forest)
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
print('F1 Score (Random Forest):', f1_rf)

# Boosting model (AdaBoost)
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)
y_pred_adaboost = adaboost.predict(X_test)
f1_adaboost = f1_score(y_test, y_pred_adaboost, average='weighted')
print('F1 Score (AdaBoost):', f1_adaboost)



#  Stacking of Models

# Initialize base models
base_models = [
    LogisticRegression(penalty='l2', max_iter=1000),
    RandomForestClassifier(),
    AdaBoostClassifier()
]

# Initialize meta model (Logistic Regression)
meta_model = LogisticRegression(penalty='l2', max_iter=1000)

# Initialize stacking classifier
stacking = StackingClassifier(
    classifiers=base_models,
    meta_classifier=meta_model,
    use_probas=True,
    average_probas=False
)

# Fit the stacking classifier
stacking.fit(X_train, y_train)

# Predict using the stacking classifier
y_pred_stacking = stacking.predict(X_test)
f1_stacking = f1_score(y_test, y_pred_stacking, average='weighted')
print('F1 Score (Stacking):', f1_stacking)


# # Wonderful For the above score we can say that by using Top 7 features insted of 16 features we can get approx same score of our model 
# 
# Reduced model complexity With fewer features, the model becomes simpler and easier to interpret. It can also lead to faster training and inference times.

# In[ ]:




