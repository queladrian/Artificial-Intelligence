import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport

####################### EDA on the California Housing Prices dataset with Profile Report ###########################################

#First, we uppload the dataset that corresponds to the California Housing Prices DATA.

df_california= pd.read_csv('ArtificialIntelligence/Datasets/california_housing_prices.csv')

print(df_california.head(15))

print(df_california.describe())

df_california.info()

df_california.hist(figsize=(20, 15))
plt.show()

# To plot the dispersion graphs, we use the scatterplot function from the seaborn library. We will plot the longitude and latitude of the houses in California to visualize their distribution.

sns.scatterplot(x='longitude', y='latitude', data=df_california)
plt.show()

sns.scatterplot(x='longitude', y='latitude', data=df_california, alpha=0.1, hue='median_house_value')
plt.show()

sns.scatterplot(x='longitude', y='latitude', data=df_california, alpha=0.4, hue='ocean_proximity')
plt.show()

# To visualize the profile report, we use the ydata-profiling library. We will generate a profile report of the dataset to analyze its characteristics and identify potential issues.

profile = ProfileReport(df_california, title='Reporte Perfilamiento de Datos', explorative=True)


profile.to_file("profile_report.html")



