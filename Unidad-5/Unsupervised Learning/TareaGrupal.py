# Groupp work 

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ydata_profiling import ProfileReport


# =========================================================
# 1. CONFIGURACIÓN DE RUTAS
# =========================================================

carpeta_actual = Path(__file__).parent

ruta_csv = carpeta_actual / "inec_iaee_agua_2014.csv"

ruta_reporte = carpeta_actual / "reporte_aguas.html"

carpeta_imagenes = carpeta_actual / "imagenes"

carpeta_imagenes.mkdir(exist_ok=True)


# =========================================================
# 2. CARGA DE DATOS
# =========================================================

df_aguas = pd.read_csv(ruta_csv)

print("Número de filas y columnas:")
print(df_aguas.shape)

print("\nInformación general:")
df_aguas.info()


# =========================================================
# 3. DESCRIPCIÓN DE VARIABLES
# =========================================================

# AQUÍ PONER QUÉ SIGNIFICA CADA VARIABLE
#
# Ejemplo:
#
# id_empresa:
# Identificador de la empresa.
#
# agu_con_anu_m3:
# Consumo anual de agua en metros cúbicos.
#
# agu_con_anu_usd:
# Gasto anual asociado al consumo de agua en dólares.
#
# Completar utilizando el diccionario oficial del INEC.


# =========================================================
# 4. PERFILAMIENTO GENERAL
# =========================================================

perfil_datos = ProfileReport(
    df_aguas,
    title="Reporte de Perfilado de Datos de Agua en el Ecuador",
    explorative=True
)

perfil_datos.to_file(ruta_reporte)

print("\nReporte HTML generado correctamente.")
print("Ruta del reporte:")
print(ruta_reporte)


# =========================================================
# 5. SELECCIÓN DE VARIABLES PARA K-MEANS
# =========================================================

variables = [
    'agu_con_anu_m3',
    'agu_con_anu_usd'
]

df_cluster = df_aguas[variables].copy()


# =========================================================
# 6. CONVERSIÓN A VARIABLES NUMÉRICAS
# =========================================================

for columna in variables:

    df_cluster[columna] = pd.to_numeric(
        df_cluster[columna],
        errors='coerce'
    )


print("\nInformación del nuevo DataFrame:")
df_cluster.info()

print("\nEstadísticas descriptivas:")
print(df_cluster.describe())

print("\nValores faltantes:")
print(df_cluster.isnull().sum())


# =========================================================
# 7. ELIMINACIÓN DE FILAS SIN INFORMACIÓN
# =========================================================

df_cluster = df_cluster.dropna()

print("\nNúmero de observaciones disponibles para clustering:")
print(len(df_cluster))

print("\nEstadísticas después de eliminar faltantes:")
print(df_cluster.describe())


# =========================================================
# 8. VISUALIZACIÓN DE OUTLIERS ANTES DE ELIMINARLOS
# =========================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_cluster
)

plt.title(
    "Boxplot antes de eliminar valores atípicos"
)

plt.ylabel("Valor")

plt.tight_layout()

plt.savefig(
    carpeta_imagenes / "01_boxplot_antes_outliers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nImagen guardada: 01_boxplot_antes_outliers.png"
)


# =========================================================
# 9. IDENTIFICACIÓN Y ELIMINACIÓN DE OUTLIERS EXTREMOS
# =========================================================

print(
    "\nObservaciones antes de eliminar outliers:"
)

print(
    len(df_cluster)
)


# Identificamos las empresas cuyo consumo anual
# es igual o superior a 1,000,000 m³

outliers_extremos = df_cluster[
    df_cluster['agu_con_anu_m3'] >= 1_000_000
]


print(
    "\nValores atípicos extremos identificados:"
)

print(
    outliers_extremos
)


# Eliminamos esos valores extremos

df_cluster = df_cluster[
    df_cluster['agu_con_anu_m3'] < 1_000_000
].copy()


print(
    "\nObservaciones después de eliminar outliers:"
)

print(
    len(df_cluster)
)


# =========================================================
# 10. BOXPLOT DESPUÉS DE ELIMINAR OUTLIERS
# =========================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df_cluster
)

plt.title(
    "Boxplot después de eliminar valores atípicos"
)

plt.ylabel("Valor")

plt.tight_layout()

plt.savefig(
    carpeta_imagenes / "02_boxplot_despues_outliers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Imagen guardada: 02_boxplot_despues_outliers.png"
)


# =========================================================
# 11. DISTRIBUCIÓN DE LAS VARIABLES SELECCIONADAS
# =========================================================

df_cluster.hist(
    figsize=(10, 4),
    bins=30
)

plt.tight_layout()

plt.savefig(
    carpeta_imagenes / "03_distribucion_variables.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Imagen guardada: 03_distribucion_variables.png"
)


# =========================================================
# 12. GRÁFICA DEL NUEVO DATASET
# =========================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_cluster,
    x='agu_con_anu_m3',
    y='agu_con_anu_usd'
)

plt.title(
    "Consumo anual de agua vs gasto anual"
)

plt.xlabel(
    "Consumo anual de agua (m³)"
)

plt.ylabel(
    "Gasto anual de agua (USD)"
)

plt.tight_layout()

plt.savefig(
    carpeta_imagenes / "04_consumo_vs_gasto.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Imagen guardada: 04_consumo_vs_gasto.png"
)


# =========================================================
# 13. ESTANDARIZACIÓN
# =========================================================

scaler = StandardScaler()

X = scaler.fit_transform(
    df_cluster
)

print(
    "\nPrimeras cinco observaciones estandarizadas:"
)

print(
    X[:5]
)


# =========================================================
# 14. IMPLEMENTACIÓN INICIAL DE K-MEANS CON k = 2
# =========================================================

modelo_kmeans = KMeans(
    n_clusters=2,
    random_state=75,
    n_init=10
)

modelo_kmeans.fit(
    X
)

df_cluster_k2 = df_cluster.copy()

df_cluster_k2['Cluster'] = (
    modelo_kmeans.labels_
)

print(
    "\nPrimeras observaciones con k = 2:"
)

print(
    df_cluster_k2.head()
)


# =========================================================
# 15. VISUALIZACIÓN PARA k = 2
# =========================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_cluster_k2,
    x='agu_con_anu_m3',
    y='agu_con_anu_usd',
    hue='Cluster',
    palette='viridis'
)

plt.title(
    "Clusters con k = 2"
)

plt.xlabel(
    "Consumo anual de agua (m³)"
)

plt.ylabel(
    "Gasto anual en agua (USD)"
)

plt.tight_layout()

plt.savefig(
    carpeta_imagenes / "05_clusters_k2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Imagen guardada: 05_clusters_k2.png"
)


# =========================================================
# 16. INERCIA PARA k = 2
# =========================================================

inercia_k2 = (
    modelo_kmeans.inertia_
)

print(
    "\nInercia para k = 2:"
)

print(
    inercia_k2
)


# =========================================================
# 17. MÉTODO DEL CODO
# =========================================================

inercias = []

valores_k = range(
    1,
    11
)

for k in valores_k:

    modelo = KMeans(
        n_clusters=k,
        random_state=75,
        n_init=10
    )

    modelo.fit(
        X
    )

    inercias.append(
        modelo.inertia_
    )


print(
    "\nValores de inercia:"
)

for k, inercia in zip(
    valores_k,
    inercias
):

    print(
        f"k = {k}: {inercia}"
    )


# =========================================================
# 18. VISUALIZACIÓN DEL MÉTODO DEL CODO
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    valores_k, inercias, marker='o'
)

plt.xlabel(
    "Número de clusters (k)"
)

plt.ylabel(
    "Inercia"
)

plt.title(
    "Método del codo"
)

plt.xticks(
    valores_k
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    carpeta_imagenes / "06_metodo_del_codo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Imagen guardada: 06_metodo_del_codo.png"
)


# =========================================================
# 19. SELECCIÓN DEL k ÓPTIMO
# =========================================================

# IMPORTANTE:
# Revisar el gráfico del método del codo
# y cambiar este número si es necesario.

k_optimo = 3


print(
    "\nNúmero de clusters seleccionado:"
)

print(
    k_optimo
)


# =========================================================
# 20. K-MEANS CON EL k ÓPTIMO
# =========================================================

modelo_kmeans_final = KMeans(
    n_clusters=k_optimo,
    random_state=75,
    n_init=10
)

modelo_kmeans_final.fit(
    X
)


df_cluster_final = (
    df_cluster.copy()
)

df_cluster_final['Cluster'] = (
    modelo_kmeans_final.labels_
)


# =========================================================
# 21. VISUALIZACIÓN DE LOS CLUSTERS FINALES
# =========================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_cluster_final,
    x='agu_con_anu_m3',
    y='agu_con_anu_usd',
    hue='Cluster',
    palette='viridis'
)

plt.title(
    f"Clusters finales con k = {k_optimo}"
)

plt.xlabel(
    "Consumo anual de agua (m³)"
)

plt.ylabel(
    "Gasto anual en agua (USD)"
)

plt.tight_layout()

plt.savefig(
    carpeta_imagenes / "07_clusters_finales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Imagen guardada: 07_clusters_finales.png"
)


# =========================================================
# 22. INTERPRETACIÓN DE LOS CLUSTERS
# =========================================================

resumen_clusters = (
    df_cluster_final
    .groupby('Cluster')
    .agg(
        {
            'agu_con_anu_m3': [
                'mean', 'median', 'min', 'max'
            ],

            'agu_con_anu_usd': [
                'mean', 'median', 'min', 'max'
            ]
        }
    )
)


print(
    "\nResumen de los clusters:"
)

print(
    resumen_clusters
)


print(
    "\nNúmero de empresas por cluster:"
)

print(
    df_cluster_final[
        'Cluster'
    ]
    .value_counts()
    .sort_index()
)


# =========================================================
# 23. CENTROIDES
# =========================================================

centroides_estandarizados = (
    modelo_kmeans_final.cluster_centers_
)


centroides_originales = (
    scaler.inverse_transform(
        centroides_estandarizados
    )
)


df_centroides = pd.DataFrame(
    centroides_originales,
    columns=variables
)


df_centroides['Cluster'] = range(
    k_optimo
)


print(
    "\nCentroides de cada cluster:"
)

print(
    df_centroides
)


# =========================================================
# 24. VISUALIZACIÓN DE CLUSTERS CON CENTROIDES
# =========================================================

plt.figure(figsize=(8, 6))


sns.scatterplot(
    data=df_cluster_final,
    x='agu_con_anu_m3',
    y='agu_con_anu_usd',
    hue='Cluster',
    palette='viridis',
    alpha=0.7
)


plt.scatter(
    df_centroides[
        'agu_con_anu_m3'
    ],

    df_centroides[
        'agu_con_anu_usd'
    ],

    marker='X',
    s=200,
    edgecolors='black',
    label='Centroides'
)


plt.title(
    f"Clusters y centroides con k = {k_optimo}"
)


plt.xlabel(
    "Consumo anual de agua (m³)"
)


plt.ylabel(
    "Gasto anual en agua (USD)"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    carpeta_imagenes / "08_clusters_centroides.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Imagen guardada: 08_clusters_centroides.png"
)


# =========================================================
# 25. CONCLUSIONES / HALLAZGOS
# =========================================================

print(
    "\nProceso terminado correctamente."
)

print(
    "\nLas imágenes fueron guardadas en:"
)

print(
    carpeta_imagenes
)

print(
    "\nEl reporte HTML fue guardado en:"
)

print(
    ruta_reporte
)