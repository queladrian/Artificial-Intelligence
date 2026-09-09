# ============================================================
# CLASIFICACIÓN DEL TAMAÑO DE EMPRESAS CON k-NN
# Dataset: Consumo de agua de empresas del Ecuador
# Fuente: INEC
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

df = pd.read_csv(
    r"C:\Users\Adrian\Downloads\SEPTIMO SEMESTRE\INTELIGENCIA  ARTIFICIAL\Assingments\inec_iaee_agua_2014.csv"
)

variables = [
    "agu_con_anu_m3",
    "agu_con_anu_usd"
]

objetivo = "TAMANIO_EMPRESA"


# ============================================================
# 2. LIMPIEZA
# ============================================================

df_modelo = df[variables + [objetivo]].copy()

df_modelo[variables] = df_modelo[variables].apply(
    pd.to_numeric,
    errors="coerce"
)

df_modelo = df_modelo.dropna()

df_modelo = df_modelo[
    df_modelo["agu_con_anu_m3"] < 1_000_000
].copy()


print("======================================")
print("INFORMACIÓN DE LA BASE DE DATOS")
print("======================================")

print("Fuente: INEC")
print("Datos originales:", df.shape)
print("Datos utilizados:", df_modelo.shape)

print("\nCantidad por tamaño de empresa:")
print(df_modelo[objetivo].value_counts().sort_index())


# ============================================================
# 3. GRÁFICA DE DISPERSIÓN
# ============================================================

plt.figure(figsize=(8, 6))

for clase in sorted(df_modelo[objetivo].unique()):

    datos_clase = df_modelo[
        df_modelo[objetivo] == clase
    ]

    plt.scatter(
        datos_clase["agu_con_anu_m3"],
        datos_clase["agu_con_anu_usd"],
        label=str(clase),
        alpha=0.7
    )

plt.xlabel("Consumo anual de agua (m³)")
plt.ylabel("Costo anual del agua (USD)")

plt.title(
    "Consumo de agua según tamaño de empresa"
)

plt.legend(title="Tamaño de empresa")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 4. VARIABLES DEL MODELO
# ============================================================

X = df_modelo[variables]
y = df_modelo[objetivo]


# ============================================================
# 5. ENTRENAMIENTO Y PRUEBA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=75,
    stratify=y
)

print("\n======================================")
print("DIVISIÓN DE DATOS")
print("======================================")

print("Datos totales:", len(X))
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

print(
    f"Entrenamiento: "
    f"{len(X_train) / len(X) * 100:.2f}%"
)

print(
    f"Prueba: "
    f"{len(X_test) / len(X) * 100:.2f}%"
)


# ============================================================
# 6. ESTANDARIZACIÓN
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 7. BÚSQUEDA DEL MEJOR k
# ============================================================

valores_k = range(1, 21)

accuracies = []

for k in valores_k:

    modelo = KNeighborsClassifier(
        n_neighbors=k
    )

    modelo.fit(
        X_train_scaled,
        y_train
    )

    predicciones = modelo.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predicciones
    )

    accuracies.append(accuracy)


# Mejor k
mejor_accuracy = max(accuracies)

indice = accuracies.index(
    mejor_accuracy
)

k_optimo = list(valores_k)[indice]


print("\n======================================")
print("SELECCIÓN DE k")
print("======================================")

for k, accuracy in zip(
    valores_k,
    accuracies
):
    print(
        f"k = {k:2d} -> "
        f"Accuracy = {accuracy:.4f}"
    )


print(
    f"\nk óptimo = {k_optimo}"
)

print(
    f"Mejor accuracy = "
    f"{mejor_accuracy:.4f}"
)


# ============================================================
# 8. GRÁFICA: ACCURACY VS k
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    valores_k,
    accuracies,
    marker="o"
)

plt.scatter(
    k_optimo,
    mejor_accuracy,
    s=150,
    label=f"k óptimo = {k_optimo}"
)

plt.xlabel("Número de vecinos (k)")
plt.ylabel("Accuracy")

plt.title(
    "Selección del número óptimo de vecinos"
)

plt.xticks(list(valores_k))

plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. MODELO FINAL
# ============================================================

modelo_final = KNeighborsClassifier(
    n_neighbors=k_optimo
)

modelo_final.fit(
    X_train_scaled,
    y_train
)

y_pred = modelo_final.predict(
    X_test_scaled
)


# ============================================================
# 10. MÉTRICAS
# ============================================================

accuracy_final = accuracy_score(
    y_test,
    y_pred
)

print("\n======================================")
print("RESULTADOS DEL MODELO")
print("======================================")

print(
    f"Accuracy: {accuracy_final:.4f}"
)

print("\nReporte de clasificación:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 11. MATRIZ DE CONFUSIÓN
# ============================================================

clases = sorted(
    y.unique()
)

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=clases
)


plt.figure(figsize=(6, 5))

plt.imshow(
    matriz,
    cmap="Blues"
)

plt.title(
    "Matriz de confusión"
)

plt.xlabel(
    "Clase predicha"
)

plt.ylabel(
    "Clase real"
)

plt.xticks(
    range(len(clases)),
    clases
)

plt.yticks(
    range(len(clases)),
    clases
)


for i in range(len(clases)):

    for j in range(len(clases)):

        plt.text(
            j,
            i,
            matriz[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()
plt.show()


# ============================================================
# 12. PRECISION, RECALL Y F1-SCORE
# ============================================================

reporte = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

metricas = pd.DataFrame(
    reporte
).T


metricas_clases = metricas.loc[
    [str(clase) for clase in clases],
    [
        "precision",
        "recall",
        "f1-score"
    ]
]


metricas_clases.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Métricas de clasificación por clase"
)

plt.xlabel(
    "Tamaño de empresa"
)

plt.ylabel(
    "Valor"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=0
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()