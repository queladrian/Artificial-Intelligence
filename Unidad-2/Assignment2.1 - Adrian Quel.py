import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns   

################### Hipotesis testing and EDA on the Titanic dataset. #################################################################

#First, we uppload the dataset that corresponds to the Titanic DATA.

df_titanic= pd.read_csv('ArtificialIntelligence/Datasets/titanic.csv')

print(df_titanic.info())

# We use .info() to know the name of columns and the type of data of each column. 
#To prove our two hypothesis, we remove the columns that are not relevant to our analysis with .drop().

df_titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#   Next, we need to have information about null values in our dataset.

df_titanic.dropna(inplace=True)
df_titanic.isnull().sum()

survived = df_titanic[df_titanic.Survived == 1] # We need to identify the passengers that survived and those that did not survive. We create a new dataframe called

not_survived = df_titanic[df_titanic.Survived == 0] # We create a new dataframe to identify the passengers that did not survive.

# We want to visualize the data to verify our first hypothesis: Las mujeres tuvieron más probabilidades de sobrevivir

sns.countplot(x='Survived', data=df_titanic, hue='Sex')
plt.title('Sobrevivientes del Titanic')
plt.xlabel('Sobrevivió')
plt.ylabel('Cantidad')
plt.xticks([0,1], ['No', 'Sí'])
plt.show()

# Next, we want to verify our second hypothesis: Los pasajeros de primera clase tuvieron más probabilidades de sobrevivir. 

sns.countplot(x='Survived', data=df_titanic, hue='Pclass')
plt.title('Sobrevivientes del Titanic por Clase')   
plt.xlabel('Sobrevivió')
plt.ylabel('Cantidad')
plt.xticks([0,1], ['No', 'Sí'])
plt.show()

