# Librerias ============================================================================================
import pandas as pd

from sklearn.model_selection import train_test_split

# Variable de aleatoriedad
seed = 200

# Extraccion de datos ============================================================================================
df_telecom = pd.read_feather('files/datasets/intermediate/a02_datos_preprocesados.feather')

# Train Test Split ============================================================================================
features = df_telecom.drop('churn', axis=1)
target = df_telecom['churn']

features_train, features_test, target_train, target_test = train_test_split(features, 
                                                                            target, 
                                                                            test_size = 0.2, 
                                                                            random_state = seed)

# Guardado de datasets ============================================================================================
features_train.to_feather('files/datasets/intermediate/a03_features_train.feather')
features_test.to_feather('files/datasets/intermediate/a03_features_test.feather')
target_train.to_frame(name='churn').reset_index(drop=True).to_feather('files/datasets/intermediate/a03_target_train.feather')
target_test.to_frame(name='churn').reset_index(drop=True).to_feather('files/datasets/intermediate/a03_target_test.feather')