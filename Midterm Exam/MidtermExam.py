
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

########## DATASETS ##########

X_train = pd.read_csv(r"C:\Users\Adrian\Downloads\SEPTIMO SEMESTRE\INTELIGENCIA  ARTIFICIAL\Assingments\Midterm\X_train_Kaggle.csv")
y_train = pd.read_csv(r"C:\Users\Adrian\Downloads\SEPTIMO SEMESTRE\INTELIGENCIA  ARTIFICIAL\Assingments\Midterm\y_train_Kaggle.csv")
X_test = pd.read_csv(r"C:\Users\Adrian\Downloads\SEPTIMO SEMESTRE\INTELIGENCIA  ARTIFICIAL\Assingments\Midterm\X_test_Kaggle.csv")

########## DATA PREPARATION ##########

# We delete the unnecessary columns 
columnas = ["ID", "Customer_ID", "Name", "SSN", "Month"]

X_train = X_train.drop(columns=columnas)
X_test = X_test.drop(columns=columnas)

# We have to convert the cathegorical columns into numerical columns. Also we have to fill tha NaN values. 

for col in X_train.columns:
    if X_train[col].dtype == "object":

        X_train[col] = X_train[col].fillna("unknown").astype(str)
        X_test[col] = X_test[col].fillna("unknown").astype(str)

        valores = pd.concat([X_train[col], X_test[col]]).unique() #We use the "concat" function to combine the possible results and ensure they are unique.
        equivalencias = {v: i for i, v in enumerate(valores)}

        X_train[col] = X_train[col].map(equivalencias)
        X_test[col] = X_test[col].map(equivalencias)

    else:
        X_train[col] = X_train[col].fillna(X_train[col].mean())
        X_test[col] = X_test[col].fillna(X_train[col].mean())

########## MODEL TRAINING ##########

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)

modelo.fit(X_train, y_train["Credit_Score"])

########## PREDICTIONS ##########

predicciones = modelo.predict(X_test)

########## SUBMISSION ##########

submission = pd.DataFrame({
    "Credit_Score": predicciones
})

submission.to_csv(r"C:\Users\Adrian\Downloads\SEPTIMO SEMESTRE\INTELIGENCIA  ARTIFICIAL\Assingments\Midterm\submission_midterm.csv", index=False)

print(submission.head())





