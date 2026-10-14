########## LIBRARIES ###############

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

########## DATASETS ###############

X_train = pd.read_csv(r"C:\Users\Adrian\Downloads\SEPTIMO SEMESTRE\INTELIGENCIA  ARTIFICIAL\Assingments\Midterm\X_train_Kaggle.csv")
y_train = pd.read_csv(r"C:\Users\Adrian\Downloads\SEPTIMO SEMESTRE\INTELIGENCIA  ARTIFICIAL\Assingments\Midterm\y_train_Kaggle.csv")
X_test = pd.read_csv(r"C:\Users\Adrian\Downloads\SEPTIMO SEMESTRE\INTELIGENCIA  ARTIFICIAL\Assingments\Midterm\X_test_Kaggle.csv")

# We save this column for the final submission file
X_test_id = X_test['ID'].copy()

########## DATA PREPARATION ###############

# We delete unnecessary columns that won't be used for training the model
columnas_a_eliminar = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month']
X_train = X_train.drop(columns=columnas_a_eliminar) 
X_test = X_test.drop(columns=columnas_a_eliminar)

# We correct the data types of the columns to ensure they are consistent and suitable for modeling
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype(str).str.strip().str.lower()
        X_test[col] = X_test[col].astype(str).str.strip().str.lower()

# 3. We classify the columns into text and numerical types for appropriate handling
columnas_texto = X_train.select_dtypes(include=['object']).columns
columnas_numericas = X_train.select_dtypes(exclude=['object']).columns

# We fill missing values in numerical columns with the mean of the training set 
for columna in columnas_numericas:
    mean_value = X_train[columna].mean()
    X_train[columna] = X_train[columna].fillna(mean_value)
    X_test[columna] = X_test[columna].fillna(mean_value)

# We fill missing values in text columns with 'unknown'
for columna in columnas_texto:
    X_train[columna] = X_train[columna].fillna('unknown')
    X_test[columna] = X_test[columna].fillna('unknown')

# We use LabelEncoder to convert text columns into numerical values for model training
for columna in columnas_texto:
    le = LabelEncoder()
    
    # We fit the encoder with all unique values from both training and test sets, plus 'unknown' to ensure consistency
    valores_unicos = list(X_train[columna].unique()) + list(X_test[columna].unique()) + ['unknown']
    le.fit(valores_unicos)
    
    # We transform the text columns in both training and test sets into numerical values
    X_train[columna] = le.transform(X_train[columna])
    X_test[columna] = le.transform(X_test[columna])

# We align the training and test datasets to ensure they have the same columns, filling any missing columns in the test set with zeros
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# We encode the target variable 'class' into numerical values using LabelEncoder
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train.values.ravel())

########## MODEL TRAINING ###############

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), # Neural Architecture: 2 hidden layers with 100 and 50 neurons respectively
                           max_iter=500, 
                           random_state=42))
])

modelo = pipeline.fit(X_train, y_train_encoded)  # We train using the encoded labels


y_pred_encoded = modelo.predict(X_test)

############### SUBMISSION ###############


submission = pd.DataFrame({
    'class': y_pred_encoded
})

submission.to_csv('submission_midterm.csv', index=False)




