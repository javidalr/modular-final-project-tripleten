# Librerias ============================================================================================
import pandas as pd
import re


# Funciones ============================================================================================

# 1.Modifica formato nombre de columnas (snake_case)

def camel_to_snake(string):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    string = re.sub('(.)([0-9]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()

# Ingenieria de Caracteristicas ============================================================================================

# Renombrar columnas
new_cols = []
for string in df_telecom.columns:
    new_cols.append(camel_to_snake(string))
df_telecom.columns = new_cols
df_telecom.columns

# Cambio formato columna 'begin_date'
df_telecom['begin_date'] = df_telecom['begin_date'].apply(pd.to_datetime).dt.to_period('M')

# Creando las columnas 'month' y 'year'
df_telecom['begin_date'] = df_telecom['begin_date'].astype(str)
df_telecom[['year', 'month']] = df_telecom['begin_date'].str.split('-', expand=True)
df_telecom['year'] = df_telecom['year'].astype(int)
df_telecom['month'] = df_telecom['month'].astype(int)
df_telecom = df_telecom.drop('begin_date', axis=1)

# Creacion de nueva columna 'churn'
df_telecom['churn'] = df_telecom['end_date'].apply(lambda x : 0 if x == 'No' else 1)

# Eliminar columnas innecesarias
df_telecom = df_telecom.drop('customer_id', axis=1)

# Crear columna 'charge_category', con base en 'monthly_charges'
df_telecom['charge_category'] = pd.cut(df_telecom['monthly_charges'], bins=[0, 30, 50, 70, 90, 120], labels=['low', 'mid_low', 'middle', 'mid_high', 'high'], right=False)

# Crear columna 'total_charge_category', con base en 'total_charges'
df_telecom['total_charge_category'] = pd.cut(df_telecom['total_charges'], bins=[0, 2000, 4000, 6000, 8000, 10000], labels=['low', 'mid_low', 'middle', 'mid_high', 'high'], right=False)


# Nulos ============================================================================================

# Eliminacion de columna 'end_date'
df_telecom.drop('end_date', axis=1, inplace=True)

# Realizar reemplazo valores NaN por 'No internet'
df_telecom['internet_service'] = df_telecom['internet_service'].fillna('No internet')

# Eliminar registros nulos de df_telecom
df_telecom.dropna(subset=['total_charges'], inplace=True)


# Duplicados ============================================================================================

# No existen observaciones duplicadas