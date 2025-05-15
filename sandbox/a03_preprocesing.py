# Librerias ============================================================================================
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler

# Semilla de aleatoriedad
seed = 200

# Train Test Split ============================================================================================

features = df_telecom.drop('churn', axis=1)
target = df_telecom['churn']

features_train, features_test, target_train, target_test = train_test_split(features, 
                                                                            target, 
                                                                            test_size = 0.2, 
                                                                            random_state = seed)


# TODO Revisar los output de cada ejecucion para que cada uno entregue uno distinto

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
eda_features = ['paperless_billing','senior_citizen', 'partner', 'dependents', 'online_security','tech_support','type_One_year','type_Two_year', 'payment_method_Credit_card_(automatic)', 'payment_method_Electronic_check', 'payment_method_Mailed_check', 'internet_service_Fiber_optic', 
                'internet_service_No_internet', 'charge_category_mid_low', 'charge_category_middle', 'charge_category_mid_high', 'charge_category_high', 'total_charge_category_mid_low', 'total_charge_category_middle', 'total_charge_category_mid_high', 'total_charge_category_high']
final_features = list(set(important_features+eda_features))
features_train = features_train[final_features]
features_test = features_test[final_features]

# Escalado de variables ============================================================================================

cols = ['monthly_charges', 'total_charges', 'year', 'month']
scaler = StandardScaler().fit(features_train[cols])
features_train[cols] = scaler.transform(features_train[cols])

# Balance de clases ============================================================================================

# Aplicando SMOTENC
cat_features = [features_train.columns.get_loc(col) for col in eda_features]
smote_nc = SMOTENC(categorical_features=cat_features, random_state=seed)
features_train, target_train = smote_nc.fit_resample(features_train, target_train)

target_class_frequency = target_train.value_counts(normalize=True)*100
print(target_class_frequency)
target_class_frequency.plot(kind='bar', title='Balance de clase: Variable objetivo');
