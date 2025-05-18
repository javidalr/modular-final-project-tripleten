# Librerias ============================================================================================
import pandas as pd

from preprocessing.a04_ohe_boruta import eda_features
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler

# Variable de aleatoriedad
seed = 200

# Extraccion de datos ============================================================================================
features_train = pd.read_feather('files/datasets/intermediate/a04_features_train.feather')
features_test = pd.read_feather('files/datasets/intermediate/a04_features_test.feather')
target_train = pd.read_feather('files/datasets/intermediate/a03_target_train.feather')

# Escalado de variables ============================================================================================
cols = ['monthlycharges', 'totalcharges', 'year', 'month']
scaler = StandardScaler().fit(features_train[cols])
features_train[cols] = scaler.transform(features_train[cols])

# Balance de clases ============================================================================================
# Aplicando SMOTENC
cat_features = [features_train.columns.get_loc(col) for col in eda_features]
smote_nc = SMOTENC(categorical_features=cat_features, random_state=seed)
features_train, target_train = smote_nc.fit_resample(features_train, target_train)

# TODO #! ENCAJAR ESTE CODIGO EN EDA
# target_class_frequency = target_train.value_counts(normalize=True)*100
# print(target_class_frequency)
# target_class_frequency.plot(kind='bar', title='Balance de clase: Variable objetivo');

# Guardado de datasets ============================================================================================
features_train.to_feather('files/datasets/intermediate/a05_features_train.feather')
features_test.to_feather('files/datasets/intermediate/a05_features_test.feather')

