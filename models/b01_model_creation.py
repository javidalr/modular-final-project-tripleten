# Librerias ============================================================================================
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat

# Variable de aleatoriedad
seed = 200

# Funciones ============================================================================================
from functions.f00_functions import calculate_metrics
from functions.f00_functions import hyperparam_tuning_calculate_metrics

# Extraccion de datos ============================================================================================
features_train = pd.read_feather('files/datasets/intermediate/a05_features_train.feather')
features_test = pd.read_feather('files/datasets/intermediate/a05_features_test.feather')
target_train = pd.read_feather('files/datasets/intermediate/a03_target_train.feather')
target_test = pd.read_feather('files/datasets/intermediate/a03_target_test.feather')

# Models ============================================================================================

# Logistic Regression ============================================================================================
logistic_regression_model = LogisticRegression().fit(features_train, target_train)
logistic_regression_score = calculate_metrics(logistic_regression_model, features_train, target_train, features_test, target_test, 'Logistic Regression')
logistic_regression_hyper_tuning_model = LogisticRegression()
param_distributions = {
                        'C': [0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear', 'saga'],
                        'max_iter': [100, 200, 300], 
                        'random_state': [seed]
}
logistic_regression_hyper_tuning_score = hyperparam_tuning_calculate_metrics(logistic_regression_hyper_tuning_model, param_distributions, features_train, target_train, features_test, target_test, 'Logistic Regression')

# Decision Tree Classifier ============================================================================================

dtc = DecisionTreeClassifier().fit(features_train, target_train)
dtc_score = calculate_metrics(dtc, features_train, target_train, features_test, target_test, 'Decision Tree Classifier')

dtc_hyper_tuning = DecisionTreeClassifier()
param_distributions = {
                        'criterion': ['gini', 'entropy'], 
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2', None],
                        'random_state': [seed]
}

dtc_hyper_tuning_score = hyperparam_tuning_calculate_metrics(dtc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'Decision Tree Classifier')

# Random Forest Classifier ============================================================================================

rfc = RandomForestClassifier().fit(features_train, target_train)
rfc_score = calculate_metrics(rfc, features_train, target_train, features_test, target_test, 'Random Forest Classifier')

rfc_hyper_tuning = RandomForestClassifier()
param_distributions = {
                        'n_estimators': [100, 200, 300, 400, 500],
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None, 10, 20, 30, 40, 50],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'max_features': ['sqrt', 'log2'],
                        'bootstrap': [True, False],
                        'random_state': [seed]
}

rfc_hyper_tuning_score = hyperparam_tuning_calculate_metrics(rfc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'Random Forest Classifier')

# XGBoost Classifier ============================================================================================

xgbc = XGBClassifier().fit(features_train, target_train)
xgbc_score = calculate_metrics(xgbc, features_train, target_train, features_test, target_test, 'XGBoost Classifier')

xgbc_hyper_tuning = XGBClassifier()
param_distributions = {
                        'max_depth'    : sp_randInt(5, 50),
                        'n_estimators' : sp_randInt(50, 800),    
                        'learning_rate': sp_randFloat(),    
                        'subsample'    : sp_randFloat(),
                        'random_state' : [seed]
}

xgbc_hyper_tuning_score = hyperparam_tuning_calculate_metrics(xgbc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'XGBoost Classifier')

# LightGBM Classifier ============================================================================================

lgbmc = LGBMClassifier().fit(features_train, target_train)
lgbmc_score = calculate_metrics(lgbmc, features_train, target_train, features_test, target_test, 'LightGBM Classifier')

lgbmc_hyper_tuning = LGBMClassifier()
param_distributions = {
                        'max_depth'    : sp_randInt(5, 50),    
                        'n_estimators' : sp_randInt(50, 800),    
                        'learning_rate': sp_randFloat(),    
                        'subsample'    : sp_randFloat(),
                        'random_state' : [seed]
}

lgbmc_hyper_tuning_score = hyperparam_tuning_calculate_metrics(lgbmc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'LightGBM Classifier')

# CatBoost Classifier ============================================================================================

cbc = CatBoostClassifier(verbose=500).fit(features_train, target_train)
cbc_score = calculate_metrics(cbc, features_train, target_train, features_test, target_test, 'CatBoost Classifier')

cbc_hyper_tuning = CatBoostClassifier(verbose=500)

param_distributions = {
                        'max_depth'    : sp_randInt(5, 15),    
                        'n_estimators' : sp_randInt(50, 500),    
                        'learning_rate': sp_randFloat(),    
                        'subsample'    : sp_randFloat(),
                        'random_state' : [seed]
}

cbc_hyper_tuning_score = hyperparam_tuning_calculate_metrics(cbc_hyper_tuning, param_distributions, features_train, target_train, features_test, target_test, 'CatBoost Classifier')