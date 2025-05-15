# Librerias ============================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Importacion de datos ============================================================================================

df_contract, df_internet, df_personal, df_phone = pd.read_csv('files/datasets/input/contract.csv')\
                                                , pd.read_csv('files/datasets/input/internet.csv')\
                                                , pd.read_csv('files/datasets/input/personal.csv')\
                                                , pd.read_csv('files/datasets/input/phone.csv')

# Unificar datos ============================================================================================

df_telecom = df_contract.merge(df_personal, on='customerID', how='outer')\
                        .merge(df_internet, on='customerID', how='outer')\
                        .merge(df_phone, on='customerID', how='outer')