"""01 · Análisis exploratorio de datos (EDA).

Entrada:  data/raw/maternal_health_risk.csv
Salidas:  figuras en results/figures/ y tablas en results/tables/

    - Resumen de estadísticas.
    - Gráficos de análisis exploratorio.
    - Histogramas y gráficos de densidad para determinar la distribución.
    - Matriz de correlaciones gráfica de todas las variables.

Uso: python src/01_eda.py
"""
#%%
from config import DATA_RAW, FIGURES, TABLES, PREDICTORS, TARGET, LEVELS, L_NOMBRES, ORDEN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv(DATA_RAW)
df.info()
#   Hay un total de 7 columnas, incluyendo el target, 2 de ellas son tipo float (Blood Sugar y Body Temperature)
#   4 son de valores enteros (Age, Systolic BP, Diastolic BP y Heart Rate) y 1 es categórica (el target-nivel de riesgo).
#   Desde ya se observa que no hay datos nulos.

#%%
descriptivas = df.describe().round(2)
descriptivas.to_csv(TABLES / "eda_descriptivas_con_duplicados.csv")
descriptivas_agrupadas = df.groupby(TARGET)[PREDICTORS].median().round(2)
descriptivas_agrupadas.to_csv(TABLES / "eda_medianas_por_clase_con_duplicados.csv")
print(descriptivas)
print(descriptivas_agrupadas)

#   A simple vista, la variable edad tiene valores atípicos, como edades máximas de 70 años lo que podría tratarse de errores de tipeo.
#   Respecto a esta misma variable, se observa la diferencia cuando se hace la agrupación por niveles. Se plantea la hipótesis de que será una variable influyente.
#   En cuanto a presión en la sangre (BP), el dataset general tiene unos mínimo sy unos máximos muy afuera de los umbrales que se consideran médicamente estables (90/60 mmHg a 120/80 mmHg).
#   Se encuentran casos tanto de hipotensión como de hipertensión, sin embargo, en promedio y en cuartile se observa también una gran mayoría de valores normales.
#   Para el azúcar en sangre se encuentra una limitación ya que no se especifíca si fue una medición tomada o no en ayunas. Sin embargo, tiene potencial de ser una variable
#   útil para la clasificación de alto riesgo, aunque no tanto para separar reigo medio de bajo.
#   La temperatura corporal fue medida en F, donde el rango acepable está entre 97 y 100. Debajo de 97 no hay ningun dato pero, sí hay extremos muy alto.
#   Finalmente en frecuencia cardiaca, todo el datrase se encuentra en valores normales(60-100).

#   En terminos generales, al agrupar las diferencias de medianas más claras se observan en la clase de alto riesgo.



#%% 
print(df.isnull().sum())
print(df.duplicated().sum())
HeartRate_menor_7 = int((df.HeartRate < 7).sum())
Age_menor_13 = int((df.Age < 13).sum())
Age_mayor_60 = int((df.Age > 50).sum())
print("Valores poco probables")
print(f"Frec. cardiaca {HeartRate_menor_7}, Más de 60 años {Age_mayor_60}, Menosde 13 años {Age_menor_13}")
reg_unicos = int(len(df) - df.duplicated().sum())
print(f"Solo hay {reg_unicos} registros sin duplicados")

#   En la evaluación de duplicados, se observa que más de la mitad del dataset corresponde a resgistros duplicados, hay perfiles donde todas las filas son exactamente iguales.
#   por lo tanto, deciden eliminarse más adelante con el fin de no estrechar la likelihood ni obtener posteriors optimistas.
#   También se analizaron variables que tenian valores poco probables como edades fuera del rango (13 a 60)
#   incluso imposibles como 7 lpm. 
#   Finalmente, se corroboró que solo 452 registros son únicos en la 7 variables, inlcuyendo el target, es con estos que se trabajará posteriormente.

clases = pd.DataFrame({
    "con_duplicados": df[TARGET].value_counts(),
    "sin_duplicados": df.drop_duplicates()[TARGET].value_counts(),
}).loc[list(LEVELS)]
clases.to_csv(TABLES / "eda_clases_duplicados.csv")
print(clases)

#   Se observa la diferencia el la distribución de las clases dependiendo si existen duplicados o no, en ambos casos, riesgo bajo sigue siendo el más presente.
#%%
df_u = df.drop_duplicates()

conteo = {
    'Con duplicados (n = %d)' % len(df):   df.RiskLevel.value_counts().reindex(ORDEN),
    'Sin duplicados (n = %d)' % len(df_u): df_u.RiskLevel.value_counts().reindex(ORDEN),
}

x = np.arange(len(ORDEN))
ancho = 0.38
colores = ['#9DB4E8', '#E8A48C']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, tipo in zip(axes, ['n', '%']):
    for i, (nombre, c) in enumerate(conteo.items()):
        valores = c.values if tipo == 'n' else c.values / c.sum() * 100
        barras = ax.bar(x + (i - 0.5) * ancho, valores, ancho,
                        label=nombre, color=colores[i])
        etiquetas = [f'{v:.0f}' if tipo == 'n' else f'{v:.1f}%' for v in valores]
        ax.bar_label(barras, labels=etiquetas, padding=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(['Bajo', 'Medio', 'Alto'])
    ax.set_xlabel('Nivel de riesgo')
    ax.set_ylabel('Número de registros' if tipo == 'n' else 'Porcentaje de registros')
    ax.set_title('Frecuencia absoluta' if tipo == 'n' else 'Frecuencia relativa')
    ax.spines[['top', 'right']].set_visible(False)
    ax.margins(y=0.15)

axes[0].legend(frameon=False)
fig.suptitle('Distribución de RiskLevel con y sin registros duplicados', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES / 'eda_clases_duplicados.png', dpi=300, bbox_inches='tight')
plt.show()




#%%
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, var in zip(axes.ravel(), PREDICTORS):
    sns.histplot(df[var], kde=True, ax=ax, color="#2F79F0")
    ax.set_title(var)
fig.suptitle("Distribución de las variables fisiológicas (N = %d)" % len(df))
fig.tight_layout()
fig.savefig(FIGURES / "eda_histogramas_con_duplicados.png", dpi=150)

#%%

df_u = df_u[df_u.HeartRate > 40]        # quita la fila con 7 lpm

colores = {'low risk': '#66C2A5', 'mid risk': '#FC8D62', 'high risk': '#8DA0CB'}

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))

for ax, var in zip(axes.flat, PREDICTORS):
    sns.boxplot(x='RiskLevel', y=var, data=df_u, order=ORDEN,
                palette=colores, ax=ax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(L_NOMBRES)
    ax.set_xlabel('Nivel de riesgo')

fig.suptitle('Variables fisiológicas según nivel de riesgo (registros únicos)')
plt.tight_layout()
fig.savefig(FIGURES / "eda_boxplots_por_riesgo_sin_duplicados.png", dpi=150)
plt.show()

#   Usando el dataset de 452 registros únicos, dSE observa como en caso todas las variables se solapan las clases.
#   De  nuevo, se coorrobora visualmente que la clase más fácil de separar es la de riesgo alto, especialmente haciendo
#   Uso de variables como la edad y  el azúcar en sangre, esto último corresponde a hipótesis del grupo sin ser validadad aún por un modelo.
#   Lo que se sospechaba de presencia de atipicos en la variable de edad es correcto, edades más arriba de 55 aproximadamee son bastante inusuales
#   y más sin fecundación asistida, elemento que igualmente no se menciona a detalle en la fuente.
#   Los otros atípicos observados se encuentran en la variable de BS, la referente a azúcar, de por sí estos valores son bastante altos de acuerdo a la literatura.
#%%
df["y"] = df[TARGET].map(LEVELS)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, metodo in zip(axes, ["pearson", "spearman"]):
    c = df[PREDICTORS + ["y"]].corr(method=metodo)
    sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Correlación de {metodo.capitalize()} (y = riesgo ordinal)")
fig.tight_layout()
fig.savefig(FIGURES / "eda_correlaciones_con_duplicados.png", dpi=150)

#   La variable con menos correlaciones en el dataset es la de temperatura corporal, esto se debe a que tienenmuy pocos valores únicos y de hecho su distribución es muy estrecha.
#   Por otro lado, como es de esperarse por su naturaleza médica, lo relacionado a la presión arterial tiene bastante correlación.


#   Se decide eliminar lo completamente duplicado, teniendo en cuenta las limitaciones y la importancia de no sobreestimar una posterior, asignando una likelihood que tiene muchas observaciones individuales similares.
#   Además resuleve problemas posteriores de fuga de información ya que un registro repetido posría aparecer en entrenamiento y en validación generando unas poseriors optimistas.
#   Así mismo, se plantea una prueba de sensibilidad con el dataset completo y comparar el 'peso de cada variable en cada caso y el desempeño de los dos escenarios. 
# 
#   No se enfrentan problemas de datos nulos y solo hay un evidente error de tipeo, el 7 latidos por minuto. Los otros valores que se consideran bastante atípicos pueden ser poco probables pero no imposibles.    


