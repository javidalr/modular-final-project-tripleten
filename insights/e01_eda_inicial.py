# Librerias ============================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funciones ============================================================================================

# 1. Contexto general del conjunto de datos

def info_data(data):
    print('1. Tabla de datos')
    display(data.sample(10))
    print('='*100)
    print()

    print('2. Registros y variables')
    print()
    print('Registros: ', data.shape[0])
    print()
    print('Variables: ', data.shape[1])
    print('='*100)
    print()

    print('3. Resumen general del dataset')
    print(data.info())
    print('='*100)
    print()

    print('4. Resumen estadistico del dataset')
    print()
    print(data.describe())
    print('='*100)
    print()

    print('5. Valores duplicados por variable')
    print()
    print(data.duplicated().sum())
    print('='*100)
    print()

    print('6. Valores ausentes por variable')
    print()
    print(data.isna().sum())
    print('='*100)
    print()


# EDA Inicial ============================================================================================

# Informacion preliminar
info_data(df_telecom)

'''
**Hallazgos**

- El conjunto de datos final contiene 7043 observaciones y 20 variables.

- Las tablas de `ìnternet` y `phone` tenian valores faltantes, los cuales se traspasaron al dataset final.

- Estandarizar el formato de nombres de las columnas para mantener la coherencia de la tabla.

- No existen valores duplicados en el conjunto de datos.

- Gran parte de las variables del set de datos son tipo object, incluyendo `BeginDate` y `EndDate` que deben ser tipo datetime, y `TotalCharges`, que debe ser tipo float.

- Solo existen 2 columnas numéricas, `SeniorCitizen` y `MonthlyCharges`.

-  `MonthlyCharges` tiene una media de 64.76 dólares por mes, y una mediana de 70.35 dólares, lo que puede implicar presencia de valores atípicos(outliers).

-  `SeniorCitizen` solo contiene valores 1 y 0, para indicear si el clientes es adulto mayor o no.

-  Los valores ausentes estan presentes en 8 columnas, y corresponden a casi un 22% del total de registros, por ello, es necesario indagar y aplicar métodos y/o técnicas apropiadas para su tratamiento, con el fin de conservar la integridad de los datos.

-  Las columnas se componen de datos de 'yes/no', deben ser representadas de manera uniforma como la columna `SeniorCitizen`.

-  `TotalCharges` tiene tipo de dato erroneo, lo que impide ver sus valores en el análisis descriptivo.

-  `EndDate` sera la colummna objetivo del modelo de machine learning del proyecto, la cual indica el estado del cliente, si sigue usando los servicios o los ha dado de baja.

-  Al ser un modelo de clasificación, es necesario que la columna objetivo este representada por datos de valores binarios, por lo que es necesario transformar sus datos a valores de 0 y 1. 
'''

# Matriz de correlaciones
df_telecom.select_dtypes(include='number').corr()['churn'].sort_values(ascending=False)