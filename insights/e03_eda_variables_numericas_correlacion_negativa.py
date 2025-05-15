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

# EDA Variables Numericas ============================================================================================

# Correlaciones Negativas ============================================================================================

# total_charges
plt.figure(figsize=(15, 10), dpi=80)
plt.subplot(2, 1, 1)
plt.hist(data['total_charges'], bins=20, alpha=0.5, label='Distribución de Facturación Total')
plt.xlabel('Factura Total')
plt.ylabel('Frecuencia')
plt.legend(loc='upper right')
plt.subplot(2, 1, 2)
plt.boxplot(data['total_charges'], vert=False)
plt.xlabel('Factura Total')
plt.ylabel('Valor')
plt.title('Factura Total Boxplot')
plt.tight_layout()
plt.show()
analyze_column('total_charge_category')
total_charge_churn_rate = analyze_churn_rate('total_charge_category')
total_charge_churn_rate
'''
**Hallazgos**

- La distribución de la columna esta sesgada a la derecha, lo que advierte que la mayoria de los clientes tiene una facturación total baja.

- Gran parte de los clientes tiene un valor total de facturación inferior a los 1000 dólares.

De forma similar, que el estudio anterior, se procede a crear una nueva columna con la agrupación de los datos basada en la columna `total_charges`.

El 59.4% de los clientes, estan en el segemento 'low', lo que indica una facturación total baja. Esto deduce, que casi el 60% de los clientes tiene mayor probabilidad de influir de forma negativa en la tasa de deserción de clientes.

**Hallazgos**

- Al igual que el análisis preliminar, existe la correlación negativa estra `total_charges` y `churn`, que indica un mayor potencial de cancelación asociado a los valores bajos en la columna `total_charges`.

- La tasa de pérdida de clientes, para clientes con valores de facturación total bajos o muy bajos( 0 - 2000 dólares) es más alto, en comparación con valores totales de facturación más altos.

**Insights**

- Los valores más altos, señalan una suscripción de myor duración, lo que refleja la fidelización del cliente.

- Los clientes con valores de facturación total más altos, tienen probabilidades mas bajas de eliminar el servicio.

**Recomendaciones**

- Incentivar los programas promocionales para lograr fidelizar a los clientes que estan en el segmento de facturación total baja, con el objetivo de minimizar la tasa de cancelación que tiene la compañia. 
'''

# online_security
analyze_column('online_security')
online_security_churn_rate = analyze_churn_rate('online_security')
online_security_churn_rate
'''
La mayoria de los clientes no usa el servicio de seguridad en línea

**Hallazgos**

- Los clientes que no utilizan el servicio de seguridad en línea estan significativamentes mas proclives a finalizar los servicios con la compañia

- La cantidad de clientes que anularon sus servicios y no usanron el servicio de seguridad en línea (1574), es aproximadamente, los clientes que usaron el servicio y eligieron seguir con sus servicios(1720).

- Si el cliente, usa el servicio de seguridad en línea o lo utilizó en algún periodo, es muy dificil que terminen su contrato.

**Insights**

- El servicio de seguridad en línea, tiene un desempeño muy relevante para el cliente, lo impacta positivamente en la tasa de cancelación.

- Los clientes que adquieren este servicio tienden a tener mayor confianza y seguridad en la compañia, y en los servicios ofrecidos por esta, lo que reduce la posibilidad de término de contrato y de la misma manera, aumenta la permanencia de los clientes. 

- Este servicio permite el aumento de tasa de retención de clientes, ya que los clientes que usan el servicio asi lo demuestran. 

**Recomendaciones**

- Resaltar y promover el servicio de seguridad en línea, con el fin de atraer más clientes, aumentar la tasa de retención y disminuir la tasa de cancelación.
'''

# tech_support
analyze_column('tech_support')
tech_support_churn_rate = analyze_churn_rate('tech_support')
tech_support_churn_rate
'''
Gran parte de los clientes asociados a la empresa, no usan los servicios de soporte técnico.

**Hallazgos**

- Los clientes que no usan el servicio de soporte técnico, tienen más probabilidad de cancelar su servicio, en comparacion a los que si lo han utilizado.

- La cantidad de clientes que cancelaron sus servicios es aproximadamente equivalente con la cantidad de clientes que usan el servicio de soporte técnico y permanecen en la compañia.

**Insights**

- El servicio de soporte técnico puede desempeñar un papel fundamental en la retención de los clientes.

- Los clientes al tener acceso a la asistencia técnica, tienen mas probabilidad de sentirse respaldados y satisfechos con los servicios contratados.

**Recomendaciones**

- Reforzar y promover el servicio de soporte técnico, para incentivar a los clientes el uso del servicio.

- Con una asistencia técnica confiable, Interconnect puede cambiar la satisfacción del cliente y reducir las cancelaciones.
'''

# dependents
analyze_column('dependents')
dependents_churn_rate = analyze_churn_rate('dependents')
dependents_churn_rate
'''
La mayoria de los suscriptores no tienen hijos ni personas a su cargo.

**Halllazgos**

- Los consumidores sin personas a su cargo tienden a tener una tasa de cancelación más alta, lo que sugiere que son más propensos a eliminar el servicio, en comparación con los que si tienen personas a su cargo.

**Insights**

- Los usuarios con personas a cargo puede que tengan más compromiso con el servicio debido a sus necesidades y preferencias familiares, lo que reduce la cancelación de la suscripción.

**Recomendaciones**

- Diseño de promociones y ofertas especiales orientadas a los consumidores con personas a su cargo, para reforzar la preferencia y la mantencion del servicio.

- Indagar con los usuarios que no tienen personas a su cargo, mediante encuestas o retroalimentación de ellos, para asi comprender de mejor manera las razones de su abandono, con el fin de implementar entrategias que aborden sus necesidades y mitiguen el aumento de la tasa de cancelación.
'''

# partner
analyze_column('partner')
partner_churn_rate = analyze_churn_rate('partner')
partner_churn_rate
'''
El número de suscriptores con y sin pareja, esta relativamente balanceado.

**Hallazgos**

- la tasa de cancelación de los usuarios que no tienene pareja es significativamente más alta, llegando un 32.98% en comparación de los que si tienen pareja, cuya tasa de abandono es de 19.72%.

- La cantidad de usuarios que permanece en la compañia y no tiene pareja , es menor que aquellos que si tienen pareja.

**Insights**

- Tener pareja genera una reducción de la probabilidad de los usuarios que cancelan su suscripción.

- Los clientes sin pareja, podrian tener experiencias o necesidades diferentes que los lleva a tener una mayor tasa de abandono.

**Recomendaciones**

- Diseñar promociones para y ofertas dirigidas al segmento de usuarios que no tienen pareja, con el fin de reducir la tasad e cancelación asociado a este segmento.
'''