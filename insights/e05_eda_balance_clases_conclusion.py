# Librerias ============================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EDA Balance de clases ============================================================================================

class_frequency = df_telecom['churn'].value_counts(normalize=True)*100
print(class_frequency)
class_frequency.plot(kind='bar', title='Balance de clase: Variable objetivo');   
'''
El desbalance de los datos permite determinar que aproximadamente el 27% de los clientes se fugan de la compañia, y el 73% restante son clientes que estan fidelizados, esta materia se tratará más adelante en el proyecto.
'''

# EDA Conclusion ============================================================================================

'''
Se puede deducir,  que son varias las razones que influencian la cancelación del servicio, por ello, se aplicaran las siguientes determinaciones:

- Enfocar el proyecto, en las variables con alta correlación en el riesgo de abandono, se descartan variables de baja correlación.

- Varias características presentan la tasa de cancelación más alta(superior al 40%),como los clientes adultos mayores(41,68%), la suscripción mensual(42,71%), el método de pago por cheque electrónico(45,29%) y el servicio de internet por fibra óptica(41,89%), las cuales requieren un interes especial para mitigar el aumento de la tasa de deserción de clientes.

- Los clientes con pareja/esposa, hijos o alguien dependiente, muestran menor deserción que los clientes solteros, sini hijos o sin alguien dependiente.

- Los clientes que no utilizan servicios de seguridad o soporte técnico presentan un mayor riesgo de cancelación.

- No se observa una diferencia significativa en la tasa de cancelación entre clientes hombres y muejeres.

Mediante estos hallazgos la Interconnect puede diseñar estreategias especificas y desarrollar programas promocionales directos para reterner clientes, enfocandose en las categórias de alto riesgo.

## **4. Entrenamiento y prueba del modelo**

En esta sección, se procede a desarrollar varios modelos de machine learning, con el fin de realizar las predicciones de los datos. 

El objetivo, es lograr la puntuación más alta posible en el conjunto de prueba, utilizando principalmente la métrica AUC-ROC.

Los criterios de evaluación para el modelado son los siguientes:

- AUC-ROC < 0.75 — 0 SP

- 0.75 ≤ AUC-ROC < 0.81 — 4 SP

- 0.81 ≤ AUC-ROC < 0.85 — 4.5 SP

- 0.85 ≤ AUC-ROC < 0.87 — 5 SP

- 0.87 ≤ AUC-ROC < 0.88 — 5.5 SP

- AUC-ROC ≥ 0.88 — 6 SP

Para ello, se desarrollarán las siguientes etapas:

- Preparar los datos para el modelado, lo que incluye la partición de los datos, la codificación OHE para las variables categóricas, aplicar Boruta(selección de características) y el escalado en los datos de entrenamiento y prueba.

- Entrenar y evaluar distintos modelos de machine learning de clasificación, como Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost.

- Realizar ajuste de hiperparámetros para obtener el mejor desempeño de los modelos.

Con este procedimiento, se espera desarrollar el mejor modelo de machine learning, robusto y preciso, que pueda predecir de manera efectiva la deserción de clientes, permitiendo a Interconnect tomar medidas proactivas respecto a la retención de clientes. 
'''