# Librerias ============================================================================================

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Importacion de datos ============================================================================================

# Carga desde base de datos PostgreSQL ============================================================================================

load_dotenv()

db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")

try:
    db_url = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(db_url)

    tablas = ['contract', 'personal', 'internet', 'phone']
    data = {} 

    for tabla in tablas:
        query = f'select * from {tabla};'
        data[tabla] = pd.read_sql_query(query, engine)

    df_contract = data['contract']
    df_personal = data['personal']
    df_internet = data['internet']
    df_phone = data['phone']

except Exception as e:
    print("❌ Error al conectar o consultar la base de datos:", e)

# Unificar datos ============================================================================================

df_telecom = df_contract.merge(df_personal, on='customerid', how='outer')\
                        .merge(df_internet, on='customerid', how='outer')\
                        .merge(df_phone, on='customerid', how='outer')

# Guardar dataframe ============================================================================================

df_telecom.to_feather('files/datasets/intermediate/a01_datos_preprocesados.feather')