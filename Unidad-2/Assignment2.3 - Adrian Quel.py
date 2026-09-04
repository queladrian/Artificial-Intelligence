
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_ecommerce = pd.read_csv('ArtificialIntelligence/Datasets/ecommerce_transactions.csv')

print(df_ecommerce.head())

df_ecommerce.info()

df_ecommerce.isnull().sum()

df_ecommerce = df_ecommerce.dropna()

sns.boxplot(y=df_ecommerce['Quantity'])

plt.show()

Q1 = df_ecommerce['Quantity'].quantile(0.25)
Q3 = df_ecommerce['Quantity'].quantile(0.75)

IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

liminf = Q1 - 1.5 * IQR
limsup = Q3 + 1.5 * IQR

outliers = (df_ecommerce['Quantity'] < liminf) | (df_ecommerce['Quantity'] > limsup)

print("Number of outliers:", outliers.sum())

df_ecommerce = df_ecommerce[~outliers]

sns.boxplot(y=df_ecommerce['Quantity'])

plt.show()

df_ecommerce = df_ecommerce[df_ecommerce['Quantity'] > 0]

sns.boxplot(y=df_ecommerce['Quantity'])

plt.show()

df_ecommerce['Revenue'] = df_ecommerce['Price'] * df_ecommerce['Quantity']

print(df_ecommerce.head())

group_products = df_ecommerce.groupby('ProductName')

sum_product = group_products['Quantity'].sum()

product_most_sold = sum_product.sort_values(ascending=False)

print(product_most_sold.head(10))

plt.figure(figsize=(12, 6))

sns.barplot(
    x=product_most_sold.head(10),
    y=product_most_sold.head(10).index,
    palette='viridis'
)

plt.title('Top 10 Most Sold Products')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product Name')

plt.show()

group_products = df_ecommerce.groupby('ProductName')

sum_product_incomes = group_products['Revenue'].sum()

products_incomes = sum_product_incomes.sort_values(ascending=False)

print(products_incomes.head(10))

plt.figure(figsize=(12, 6))

sns.barplot(
    x=products_incomes.head(10),
    y=products_incomes.head(10).index,
    palette='viridis'
)

plt.title('Top 10 Productos con más Ingresos')
plt.xlabel('Ingresos')
plt.ylabel('Producto')

plt.show()

df_ecommerce['Date'] = pd.to_datetime(
    df_ecommerce['Date'],
    format='%m/%d/%Y'
)

df_ecommerce['Month'] = df_ecommerce['Date'].dt.month

ventas_por_mes = df_ecommerce.groupby('Month')['Revenue'].sum()

plt.figure(figsize=(12, 6))

sns.barplot(
    x=ventas_por_mes.index,
    y=ventas_por_mes.values,
    palette='viridis'
)

plt.title('Ganancias por Mes')
plt.xlabel('Mes')
plt.ylabel('Ganancias')

plt.show()

df_ecommerce['DayOfWeek'] = df_ecommerce['Date'].dt.day_name()

ventas_por_dia_semana = df_ecommerce['DayOfWeek'].value_counts()

plt.figure(figsize=(12, 6))

sns.barplot(
    x=ventas_por_dia_semana.index,
    y=ventas_por_dia_semana.values,
    palette='viridis'
)

plt.title('Ventas por Día de la Semana')
plt.xlabel('Día de la Semana')
plt.ylabel('Número de transacciones')

plt.show()


