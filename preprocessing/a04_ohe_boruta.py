# Librerias ============================================================================================
import pandas as pd

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler

# Variable de aleatoriedad
seed = 200

# Extraccion de datos ============================================================================================
features_train = pd.read_feather('files/datasets/intermediate/a03_features_train.feather')
features_test = pd.read_feather('files/datasets/intermediate/a03_features_test.feather')
target_train = pd.read_feather('files/datasets/intermediate/a03_target_train.feather')

# Codificacion Variables Categoricas ============================================================================================
features_train = pd.get_dummies(features_train, drop_first=True).astype(int)
features_test = pd.get_dummies(features_test, drop_first=True).astype(int)

# Boruta ============================================================================================
boruta_rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1, class_weight='balanced')
boruta_selector = BorutaPy(estimator=boruta_rf, n_estimators='auto', verbose=2, random_state=seed)
boruta_selector.fit(features_train.values, target_train.values)
important_features = features_train.columns[boruta_selector.support_].tolist()

# Seleccion de caracteristicas importantes
features_train.columns = features_train.columns.str.replace(' ', '_')
features_test.columns = features_test.columns.str.replace(' ', '_')
eda_features = ['paperlessbilling','seniorcitizen', 'partner', 'dependents', 'onlinesecurity','techsupport','type_One_year','type_Two_year', 'paymentmethod_Credit_card_(automatic)', 'paymentmethod_Electronic_check', 'paymentmethod_Mailed_check', 'internetservice_Fiber_optic', 'internetservice_No_internet', 'charge_category_mid_low', 'charge_category_middle', 'charge_category_mid_high', 'charge_category_high', 'total_charge_category_mid_low', 'total_charge_category_middle', 'total_charge_category_mid_high', 'total_charge_category_high']
final_features = list(set(important_features + eda_features))
features_train = features_train[final_features]
features_test = features_test[final_features]

# Guardado de datasets ============================================================================================
features_train.to_feather('files/datasets/intermediate/a04_features_train.feather')
features_test.to_feather('files/datasets/intermediate/a04_features_test.feather')
