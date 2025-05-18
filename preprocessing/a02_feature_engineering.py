# Librerias ============================================================================================
import pandas as pd

# Ingenieria de Caracteristicas ============================================================================================
# Extraccion de datos ============================================================================================
df_telecom = pd.read_feather('files/datasets/intermediate/a01_datos_preprocesados.feather')

# Cambio formato columnas
df_telecom['begindate'] = df_telecom['begindate'].apply(pd.to_datetime).dt.to_period('M')
df_telecom['totalcharges'] = pd.to_numeric(df_telecom['totalcharges'], errors = 'coerce')

# Aplicacion de OHE
columns_convert = ['partner', 
                'dependents', 
                'onlinesecurity', 
                'onlinebackup',                    
                'deviceprotection',                 
                'techsupport',    
                'streamingtv', 
                'streamingmovies',                   
                'multiplelines',
                'paperlessbilling'
]

df_telecom[columns_convert] = df_telecom[columns_convert].replace({'Yes': 1, 'No': 0})
df_telecom[columns_convert] = df_telecom[columns_convert].fillna(0).astype(int)

# Creando las columnas 'month' y 'year'
df_telecom['begindate'] = df_telecom['begindate'].astype(str)
df_telecom[['year', 'month']] = df_telecom['begindate'].str.split('-', expand=True)
df_telecom['year'] = df_telecom['year'].astype(int)
df_telecom['month'] = df_telecom['month'].astype(int)
df_telecom = df_telecom.drop('begindate', axis=1)

# Creacion de nueva columna 'churn'
df_telecom['churn'] = df_telecom['enddate'].apply(lambda x : 0 if x == 'No' else 1)

# Eliminar columnas innecesarias
df_telecom = df_telecom.drop('customerid', axis=1)

# Crear columna 'charge_category', con base en 'monthly_charges'
df_telecom['charge_category'] = pd.cut(df_telecom['monthlycharges'], bins=[0, 30, 50, 70, 90, 120], labels=['low', 'mid_low', 'middle', 'mid_high', 'high'], right=False)

# Crear columna 'total_charge_category', con base en 'total_charges'
df_telecom['total_charge_category'] = pd.cut(df_telecom['totalcharges'], bins=[0, 2000, 4000, 6000, 8000, 10000], labels=['low', 'mid_low', 'middle', 'mid_high', 'high'], right=False)

# Valores ausentes (Nulos) ============================================================================================
# Eliminacion de columna 'end_date'
df_telecom.drop('enddate', axis=1, inplace=True)

# Realizar reemplazo valores NaN por 'No internet'
df_telecom['internetservice'] = df_telecom['internetservice'].fillna('No internet')

# Eliminar registros nulos de df_telecom
df_telecom.dropna(subset=['totalcharges'], inplace=True)

# Valores duplicados ============================================================================================
#! No existen registros duplicados

# Guardar dataframe ============================================================================================
df_telecom.to_feather('files/datasets/intermediate/a02_datos_preprocesados.feather')