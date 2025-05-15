# Librerias ============================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Funciones ============================================================================================

# 1. Analiza distribución y elabora gráfico circular 

def analyze_column(column):
    value_counts = data[column].value_counts()
    percentages = (value_counts / len(data)) * 100
    df = pd.DataFrame({'Cantidad de valores': value_counts, 'Porcentajes': percentages.map('{:.2f}%'.format)})
    display(df)
    plt.pie(value_counts, labels=value_counts.index, autopct='%.1f%%')
    plt.title(f'Distribución de {column}') 
    plt.show()


# 2. Calcula la taza de abandono y la visualiza

def analyze_churn_rate(column):

    churn_rate = data.groupby(column, observed=True)['churn'].value_counts().unstack(fill_value=0)
    churn_rate['churn_rate'] = (churn_rate[1] / churn_rate.sum(axis=1) * 100).round(2).astype(str) + '%'
    churn_rate.plot(kind='bar', stacked=False)
    plt.title('Tasa de abandono por ' + column)
    plt.xlabel(column)
    plt.ylabel('Cantidad')
    plt.legend(title='Churn', loc='upper right')
    plt.gca().set_xticklabels(churn_rate.index, rotation=45, ha='right')
    plt.show()
    return churn_rate

# EDA Variables Numericas============================================================================================

# Correlaciones Positivas ============================================================================================

# monthly_charges
plt.figure(figsize=(15, 10), dpi=80)
plt.subplot(2, 1, 1)
plt.hist(data['monthly_charges'], bins=20, alpha=0.5, label='Distribución de la facturación mensual')
plt.xlabel('Facturación mensual')
plt.ylabel('Frecuencia')
plt.legend(loc='upper right')
plt.subplot(2, 1, 2)
plt.boxplot(data['monthly_charges'], vert=False)
plt.xlabel('Facturación Mensual')
plt.ylabel('Valor')
plt.title('Facturación mensual Boxplot')
plt.tight_layout()
plt.show()
analyze_column('charge_category')
monthly_charge_churn_rate = analyze_churn_rate('charge_category')
print(monthly_charge_churn_rate)

'''
**Hallazgos**

- La distribución se concentra en los valores bajos, principalmente alrededor de los 20 dólares.

- La distribución mejora para montos de facturación superiores a los 70 dólares.

**Hallazgos:**

- La distribución de los datos esta concentrada en las categórias `mid-high`, `high` y `low`, mientras las categorias `middle` y `mid-low` tienen relativamente una baja representación.

**Hallazgos**

- Mediante el análisis de correlación, se deduce que un valor alto en `monthly_charges`, mayor probabilidad de cancelación.

- El análisis mas detallado confirma esta hipótesis inicial, ya que las tasas de abandono son más altas, en los segmentos 'mid-high'(70-90 dólares) y 'high'(90-120 dólares) de facturación mensual, en comparacion con grupos que tiene una facturación más baja.

**Recomendaciones**

- Interconnect debe prestar mayor interes en los clientes con facturaciones altas, ya que son más propensos a dar de baja el servicio.

- Incentivar los programas promocionales para que el objetivo de estos sean los segmentos 'mid-high' y 'high'.
'''

# papers_billing
analyze_column('paperless_billing')
paperless_churn_rate = analyze_churn_rate('paperless_billing')
paperless_churn_rate

'''
Segun los datos, la mayoria de los clientes utiliza el método de facturación electrónica (sin papeles).

**Hallasgoz**

- Los clientes que utilizan el método nde facturación sin papeles(valor 1) tienen evidentemente casi eldoble de probabilidad de anular los servicios, en comparación con los clientes que utilizan la facturación por papel(valor 0)

**Insights**

- Las facturas sin papel pueden ser más faciles de cancelar, ya que se procesan en línea, a través de la página web. 

- Las facturas físicas(recibos impresos), pueden crear una barrera para los clientes al tratar de abandonar el servicio, ya que es posible que requieran un proceso más burocratico que la cancelación en línea.

**Recomendaciones**

- Fomentar que los clientes cambien su método de facturación electrónica(sin papel) hacia la facturación física(con papel), puede que no sea prudente en la era digital, pero se puede utilizar como un disuasivo para aquellos que consideren cancelar el servicio.

- Incentivar los programas promocionales para que el objetivo de estos sean los clientes de facturación electrónica, e incentivar su permanencia.

##### **3.1.1.3 Columna `senior_citizen`**

Igual que ambos procesos anteriores, la correlación obtenida indica que los clientes de avanzada edad o con valor 1 en la columna `senior_citizen`, tienen mayor probabilidad de cambiar de compañia. Por ello, se ha de investigar, para confirmar esta hipótesis.
'''


# senior_citizen
analyze_column('senior_citizen')
senior_citizen_churn_rate = analyze_churn_rate('senior_citizen')
senior_citizen_churn_rate
'''
Solo el 16.24% de los clientes son clasificados como adultos mayores. Ahora, se examinará la tasa de abandono de estos clientes

**Hallazgos**

- En términos absolutos, hay más clientes no mayores (valor 0) que han abandonado el servicio, con un total de 1393 clientes.

- Sin embargo, la tasa de cancelación entre los clientes mayores (valor 1) es significativamente más alta, alcanzando el 41.68%.

- El número de clientes mayores que han dejado el servicio es en efecto "solo" 476, pero este número es cercano al de los clientes mayores que se han quedado, que son 666, por lo que la proporción de cancelación indica aproximadamente que de 2 clientes mayores, 1 abandona el servicio.

- Esto indica que los clientes mayores tienen un mayor riesgo de cancelar el servicio.

**Recomendaciones**

- A pesar de que el número de clientes mayores es relativamente bajo, es fundamental poner atención a la proporción de cancelación y no pasarla por alto.

- Incentivar los programas promocionales para que el objetivo de estos sean los clientes adultos mayores, que se adecuen y respondan a su necesidad, con el fin de conseguir su permanencia. 

#### **3.1.2 Correlación Negativa**

En esta sección se procederá a examinar las columnas que tienen una correlación negativa significativa con la variable objetivo('churn') como son: `total_charges`, `online_security`, `tech_support`, `partner` y `dependents`
'''