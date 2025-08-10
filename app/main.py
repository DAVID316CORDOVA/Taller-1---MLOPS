# Importar FastAPI y utilidades para manejo de errores y parámetros
from fastapi import FastAPI, HTTPException, Query
# Importar BaseModel y Field de Pydantic para validar y documentar la entrada
from pydantic import BaseModel, Field
# Importar numpy para construir arrays numéricos que aceptan los modelos
import numpy as np
# Importar joblib para cargar objetos serializados (por ejemplo scalers)
import joblib
# Importar pickle para cargar modelos serializados
import pickle
# Importar logging para registrar información y errores en tiempo de ejecución
import logging
# Importar os por si se necesita (p. ej. rutas), se deja disponible para uso futuro
import os

# Configuración básica del logging
logging.basicConfig(level=logging.INFO)  # Se establece un nivel INFO por defecto para registrar mensajes útiles.
logger = logging.getLogger(__name__)     # Se obtiene un logger con el nombre del módulo para registros coherentes.

# Instanciar la aplicación FastAPI con metadatos
app = FastAPI(
    title="Penguin Species Prediction API",   # Se define un título que aparecerá en la documentación automática.
    description="API para predecir la especie de pingüino con distintos modelos",  # Descripción para la docs.
    version="2.0.0"  # Versionado de la API para control de cambios.
)

# Variables globales compartidas por la app
models = {}   # Se inicializa un diccionario vacío que almacenará modelos y scalers cargados al inicio.
# Mapeo entre id (numérico) y nombre de especie para devolver resultados legibles.
species_mapping = {1: "Adelie", 2: "Chinstrap", 3: "Gentoo"}

# ==== Cargar modelos y scalers al arrancar la app ====
@app.on_event("startup")
async def load_models():
    # Declarar que se usará la variable global 'models' para poblarla con los objetos cargados.
    global models
    try:
        # Diccionario que mapea el nombre lógico del modelo a las rutas (modelo, scaler).
        # Se usa este mapeo para permitir añadir/quitar modelos fácilmente.
        model_files = {
            "logistic_regression": ("models/logistic_regression.pkl", "models/scaler_logistic_regression.pkl"),
            "decision_tree": ("models/decision_tree.pkl", "models/scaler_decision_tree.pkl"),
            "knn": ("models/knn.pkl", "models/scaler_knn.pkl")
        }
        # Iterar sobre cada entrada para cargar el modelo y su scaler correspondiente.
        for name, (model_path, scaler_path) in model_files.items():
            # Abrir el archivo del modelo serializado con pickle y cargarlo en memoria.
            # Se usa pickle porque previamente los modelos fueron guardados con pickle.
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            # Cargar el scaler con joblib; joblib es eficiente para objetos numpy/pickles grandes (scalers).
            scaler = joblib.load(scaler_path)
            # Guardar en el diccionario 'models' una estructura con el modelo y el scaler
            # para facilitar su uso posterior (models[name]["model"], models[name]["scaler"]).
            models[name] = {"model": model, "scaler": scaler}
        
        # Registrar en INFO qué modelos se han cargado correctamente.
        logger.info(f"Modelos cargados: {list(models.keys())}")

    except Exception as e:
        # Registrar el error en caso de fallo durante la carga y relanzarlo para evitar arrancar en estado inconsistente.
        logger.error(f"Error cargando modelos: {str(e)}")
        raise e


# ==== Esquema de entrada usando Pydantic ====
class PenguinFeatures(BaseModel):
    # Cada campo representa una característica que espera el modelo en el orden que fue entrenado.
    # Se usan ejemplos (Field(..., example=...)) para que la documentación automática muestre ejemplos útiles.
    bill_length_mm: float = Field(..., example=39.1)         # Largo del pico (mm) — requerido.
    bill_depth_mm: float = Field(..., example=18.7)          # Profundidad del pico (mm) — requerido.
    flipper_length_mm: float = Field(..., example=181.0)     # Largo de las aletas (mm) — requerido.
    body_mass_g: float = Field(..., example=3750.0)          # Masa corporal (g) — requerido.
    year: int = Field(..., example=2007)                     # Año de observación — requerido (se usó en entrenamiento).
    # Variables dummy para sex y island — se esperan 0/1 tal como se generaron en el preprocesado.
    sex_Female: int = Field(..., example=0)                  # Dummy sex=Female (0/1).
    sex_Male: int = Field(..., example=1)                    # Dummy sex=Male (0/1).
    island_Biscoe: int = Field(..., example=0)               # Dummy isla Biscoe (0/1).
    island_Dream: int = Field(..., example=0)                # Dummy isla Dream (0/1).
    island_Torgersen: int = Field(..., example=1)            # Dummy isla Torgersen (0/1).


# ==== Endpoint de predicción ====
@app.post("/predict")
def predict(
    features: PenguinFeatures,  # El cuerpo de la petición será validado por Pydantic contra PenguinFeatures.
    model_name: str = Query(..., enum=["logistic_regression", "decision_tree", "knn"])  
    # Se obliga a pasar 'model_name' como parámetro de consulta y se limita a los valores permitidos.
):
    """Predecir la especie usando el modelo seleccionado"""
    # Validación: comprobar que el modelo solicitado esté cargado en memoria.
    if model_name not in models:
        # Si no está disponible, devolver un error 400 con explicación.
        raise HTTPException(status_code=400, detail=f"Modelo '{model_name}' no disponible")
    
    # Extraer el objeto modelo y el scaler asociados al nombre solicitado.
    model = models[model_name]["model"]
    scaler = models[model_name]["scaler"]

    try:
        # Construir la matriz de características en el mismo orden en que se entrenó el modelo.
        # Se construye como una lista de listas para crear un array 2D con forma (1, n_features).
        x = np.array([[features.bill_length_mm,
                       features.bill_depth_mm,
                       features.flipper_length_mm,
                       features.body_mass_g,
                       features.year,
                       features.sex_Female,
                       features.sex_Male,
                       features.island_Biscoe,
                       features.island_Dream,
                       features.island_Torgersen]])
        
        # Aplicar el scaler correspondiente para llevar las variables a la misma escala usada en entrenamiento.
        # Se usa scaler.transform, no fit_transform, para mantener la coherencia con los parámetros aprendidos.
        x_scaled = scaler.transform(x)
        # Obtener la predicción (id de la especie). [0] para extraer el valor de la primera (única) fila.
        prediction = model.predict(x_scaled)[0]
        # Obtener las probabilidades por clase para el registro de entrada. [0] para la fila única.
        probabilities = model.predict_proba(x_scaled)[0]

        # Construir un diccionario {especie: probabilidad} legible para la salida.
        # Se usa enumerate(probabilities) y se suma 1 a i porque el mapeo de especies fue 1..3.
        # Nota: esto asume que el orden de probabilidades corresponde a [1,2,3]. Si se quisiese mayor robustez,
        # se podría mapear usando model.classes_ en lugar de asumir el orden.
        prob_dict = {species_mapping[i+1]: float(prob) for i, prob in enumerate(probabilities)}

        # Devolver un JSON con información del modelo usado, id y nombre de la especie y probabilidades.
        return {
            "model_used": model_name,
            "species_id": int(prediction),                  # Asegurar que sea un int serializable.
            "species_name": species_mapping[prediction],    # Traducir id a nombre legible.
            "probability": prob_dict
        }

    except Exception as e:
        # Registrar el error para debugging y devolver un HTTPException con detalle del error.
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
