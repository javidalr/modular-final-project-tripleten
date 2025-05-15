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


# EDA Variables Categoricas ============================================================================================

# type
analyze_column('type')
type_churn_rate = analyze_churn_rate('type')
type_churn_rate
'''
La mayoria de los suscriptores, alrededor del 55.1% prefiere pagar sus servicios de forma mensual.

**Hallazgos**

- Los clientes de suscripción mensual son los mas vulnerables para abandonar el servicio, ya que su cancelación es más alta que otros tipos de suscripción. Esto indica, que este tipo de clientes pueden ser mas sensibles a cambios en los precios o en la calidad del servicio, esto los mantiene explorando otras opciones y/o cambio de proveedor.

- Paralelamente, los clientes que optaron por las formas de pago mas extensas(anual o bienal), son clientes más fidelizados y tienen menor potencial de abandono del servicio, lo que deduce como mayor lealtad. Esta lealtad se puede atribuir a diversos factores como inventivos de precio, confianza por la calidad de servicio o un compromiso más fuerte con el uso de servicio a largo plazo.

**Insights**

- El tipo de suscripción desempeña un rol fundamental en la tasa de retención de clientes.

- Ofrecer mayores beneficios e incentivos, para generear compromisos de suscripciones mas largos, puede ser una eficaz estrategia para aumentar la lealtad de los clientes y reducir la tasa de perdida de clientes.

- Monitorear y abordar las necesidades y preocupaciones de los clientes con planes mensuales, lo que ayude a mejorar su satisfacción de los consumidores y dismunir la probabilidad de deserción.

**Recomendaciones**

- Aumentar las promociones e incentivos para las suscripciones más extensas, con el fin de animar a los suscriptores a usar el servicio de forma prologada y asi mejorar la contención.

- Considerar ofrecer tarifas con descuento o características adicionales al servicio básico, para los clientes de contratos anuales o bienales. Abordar estos casos de forma proactiva puede ayudar aumentar la satisfacción del clientes y aminorar la tasa de de cancelación en el segmento.
'''

# payment_method
analyze_column('payment_method')
payment_method_churn_rate = analyze_churn_rate('payment_method')
payment_method_churn_rate
'''
La distribución de los consumidores basada en los métodos de pago es relativamente semejante, pero el método de 'cheque electrónico' tiene una participación mayor al resto.

**Hallazgos**

- La tasa de abandono para el método de pago de 'cheque electrónico' es significativamente superior, que la de otros medios de pago.

- La 'tarjeta de credito' es el que presenta menor riesgo de deserción.

**Insights**

- La alta tasa de cancelación de los clientes con 'cheque electrónico', puede señalar que estos consideran este medio de pago menos conveniente, comparado con otras opciones. Esto puede ocasionarse por factores como error en las transacciones, demoras en el procesamiento o temas relacionados a la seguridad.

- Por otra parte, la 'tarjeta de crédito' sugiere que los consumidores con este medio pago pueden tener una situacion financiera más estable y se pueden legar a estar mas comprometidos con el uso del servicio. Los pagos con 'tarjeta de crédito' ofrecen facturación automática lo que puede generar una mayor impresión de seguridad, lo que fomenta la retención de clientes.

**Recomendaciones**

- Realizar esfuerzos para incentivar a los suscriptores a modificar su medio de pago, de 'cheque electrónico' por uno más confiable y conveniente. Esto puede incorporar descuentos o promociones para quienes paguen con 'tarjeta de crédito' u otros métodos de baja deserción.

- Implementar incentivos promocionales , con el fin de fomentar el uso de las tarjetas de credito u otro medio de pago con menor riesgo de cancelación, para poder mejorar los niveles de retención. Por último, mejorar la experiencia general de pago para los usuarios de cheques electrónicos, abordando problemas y ofreciendo soporte, esto puede ayudar a reducir la tasa de abandono de este segmento.
'''

# internet_service
analyze_column('internet_service')
internet_service_churn_rate = analyze_churn_rate('internet_service')
internet_service_churn_rate
'''
La gran mayoria de los suscriptores que tienen servicio de internet, usan fibra óptica, mientras que un alto número de clientes no tienen contratado ningún servicio de internet. 

**Hallazgos**

- La mayoría de los clientes(44%) utiliza el servicio de internet por fibra óptica, mientras que una cantidad significativa de se clientes no usa ningún tipo de servicio de internet.

- Los suscriptores que usan fibra óptica tienen mayor potencial de abandonar el servicio, en comparación con aquellos que utilizan DSL o que no usan servicios de internet.

**Insights**

- La alta tasa de cancelación entre lso usuarios de internet, especialmente los de fibra óptica, puede atribuirse a factores como la calidad del servicio, velocidad de internet, los precios o la competencia de otros proveedores del servicio de internet.

- Los clientes que no usan servicios de internet, pueden haber optado otros medios de comuniacación o simplemente no tener la necesidad de conectividad, lo que los hace menos propensos a cancelar.

**Recomendaciones**

- Realizar análisis exhaustivo de las razones detras de la alta tasa de cancelación entre los usuarios de fibra óptica. Esto puede incluir la recolección de comentarios de los clientes, la identificación de puntos problematicos y la implementación de mejoras necesarias para aumentar la satisfacción.

- Ofrecer promociones atractivas o incentivos para retener a los clientes actuales de internet por fibra óptica y animarlos a retener el servicio.

- Considerar la diversificación del servicio de internet, el ofrecimiento de DSL u otra altenativa de banda ancha, para adarptarse a las necesidades y/o preferencias de los clientes.

- Los clientes que no utilizan servicios de internet, evaluar oportunidades de ofrecer paquetes básicos de internet u ofertas combinadas que mejoren la experiencia general del usuario.
'''

# gender
analyze_column('gender')
gender_service_churn_rate = analyze_churn_rate('gender')
gender_service_churn_rate
'''
La proporción entre clientes hombres y mujeres es relativamente balanceada.

**Hallazgos**

- La tasa de cancelación para clientes hombre o mujeres es similar, lo quew sugiere que el sexo de los suscriptores no influyw significativamente en la probabilidad  de que los clientes abandonen el servicio.

- No se observan diferencias notables en el comportamiento de diserción, entre clientes masculinos y femeninos.

**Insights**

- El análisis sugiere revisar otros factores mas alla del género, que desempeñen un papel más fundamental en la tasa de abandono de los clientes.  Es primordial enfocarse en estos otros factores para entender las razones detras de la pérdida de clientes.

**Recomendaciones**

- En el diseño de programas promocionales o estrategias de reducción de cancelación, se recomienda preferir otros factores más relevantes como el tipo de suscripción, el método de pago, la calidad del servicio de internet y la atención al cliente, ya que su impacto parece ser más significativo en la retención.

- Independiente, que el género no sea una variable relevante para el proyecto, debido a su nula influencia en la tasa de abandono, es necesario seguir realizando investigaciones de mercado continuas y análisis de retroalimentación del cliente para reconocer posibles cambios en las preferencias o en los patrones de comportamientos de estos segmentos.
'''