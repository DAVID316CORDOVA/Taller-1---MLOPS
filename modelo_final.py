## SCRIPT DE ENTRENAMIENTO DE MODELOS PARA PINGÜINOS
# Este script carga el dataset de pingüinos de Palmer, limpia los datos,
# convierte variables categóricas a numéricas, divide en entrenamiento y prueba,
# entrena varios modelos, guarda los modelos y escaladores, y calcula el ROC AUC en test.

## IMPORTANDO LAS LIBRERIAS
import os  # Interactuar con el sistema de archivos.
import pandas as pd  # Manipulación y análisis de datos.
from palmerpenguins import load_penguins  # Cargar dataset de pingüinos.
pd.options.display.float_format="{:,.2f}".format  # Formato de impresión para floats.
from sklearn.model_selection import train_test_split  # Separar train y test.
import pickle  # Guardar modelos en formato binario.
import joblib  # Guardar objetos de Python eficientemente.
import numpy as np  # Operaciones numéricas.
from sklearn.preprocessing import StandardScaler  # Escalado de variables numéricas.
from sklearn.linear_model import LogisticRegression  # Modelo regresión logística.
from sklearn.neighbors import KNeighborsClassifier  # Modelo KNN.
from sklearn.tree import DecisionTreeClassifier  # Árbol de decisión.
from sklearn.metrics import roc_auc_score  # Calcular ROC AUC.

###########################################################
# Cargar los datos como DataFrame de pandas
df = load_penguins()  # Carga dataset en un DataFrame.

print("Muestra del modelo")
print(df.head())  # Primeras filas.

print("*"*100)
print("Averiguando si hay nulos")
print((df.isnull().sum()/df.shape[0])*100)  # Porcentaje de nulos por columna.

print("*"*100)
print("Viendo todos los nulos")
print(df[df.isna().any(axis=1)].loc[:, df.isna().any(axis=0)])  # Filas con nulos.

print("*"*100)
print("Eliminando los nulos")
df = df.dropna()  # Elimina filas con valores faltantes.

# Definir variables categóricas para pasar a dummies
categorical = ["sex", "island"]

for line in categorical:
    print("La columna " + line + " contiene", str(len(df[line].unique())), "valores únicos")

# Convertir variables categóricas a dummies
for column in categorical:
    nuevas_features = pd.get_dummies(df[column], prefix=column, drop_first=False).astype(int)
    df = pd.merge(df, nuevas_features, left_index=True, right_index=True)
    df = df.drop(columns=column)

print("*"*100)
print("\nPrimeras filas con las nuevas variables dummy:")
print(df.head())

print("*"*100)
print("Convirtiendo a valores numéricos los valores de la columna objetivo")
df['species'] = df['species'].apply(lambda x: 
    1 if x == 'Adelie' else 
    2 if x == 'Chinstrap' else 
    3 if x == 'Gentoo' else 
    None)
print(df.head())

# Variables predictoras y objetivo
X = df.drop(columns=["species"])
y = df[["species"]]

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar datos
scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)  # Importante: usar transform en test

# Ajustar formato de y_train
y_train_ = y_train.values.reshape(-1, 1).ravel()

# Definir modelos
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5)
}

# Entrenar, guardar y calcular ROC AUC
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train_)
    
    # Predicciones de probabilidad en test
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calcular ROC AUC multiclase
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    # Guardar modelo
    model_path = f"models/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Guardar escalador
    scaler_path = f"models/scaler_{model_name}.pkl"
    joblib.dump(scaler_x, scaler_path)
    
    print(f"Modelo '{model_name}' y scaler guardados en carpeta models/ | ROC AUC: {roc_auc:.4f}")

print("Entrenamiento y guardado completado.")
