# Taller: Predicción de Especies de Pingüinos con FastAPI y Docker

En este taller se desarrolló un proyecto completo que abarca desde la obtención y procesamiento de datos hasta la creación y despliegue de una API para realizar inferencias con modelos de machine learning entrenados. Se usaron varias tecnologías y buenas prácticas para lograr un producto funcional, modular y escalable.

## 1. Descarga y Procesamiento de Datos

* Se utilizó la librería **palmerpenguins** para descargar los datos originales de las especies de pingüinos, un dataset clásico para clasificación.
* Se cargaron los datos en un **DataFrame de pandas** para su manipulación y exploración inicial.
* Se verificó la existencia de datos nulos y se eliminaron las filas que contenían valores faltantes.
* Se identificaron las variables categóricas y se transformaron en variables dummy (one-hot encoding) para su uso en modelos de ML.
* La variable objetivo fue convertida de valores categóricos a valores numéricos para facilitar el modelado.

## 2. Entrenamiento de Modelos de Machine Learning

* Se definieron múltiples modelos para clasificación:
 + Regresión logística
 + Árbol de decisión
 + K-Nearest Neighbors (KNN)
* Se dividieron los datos en conjuntos de entrenamiento y prueba.
* Se estandarizaron las características numéricas para mejorar el rendimiento de los modelos.
* Cada modelo fue entrenado con los datos escalados y se calcularon métricas de desempeño para evaluar su calidad.
* Los modelos y escaladores fueron guardados en archivos `.pkl` para su uso posterior en producción.

## 3. Creación de la API con FastAPI

* Se construyó una API REST utilizando **FastAPI** que permite:
 + Recibir datos de entrada con las características de un pingüino.
 + Seleccionar dinámicamente qué modelo usar para la predicción.
 + Realizar la inferencia escalando los datos de entrada y aplicando el modelo seleccionado.
 + Retornar la especie predicha junto con las probabilidades para cada clase.
* La API fue diseñada para cargar los modelos y escaladores durante el evento de inicio para optimizar la performance.
* Se implementó manejo de errores y logging para mejorar la trazabilidad y robustez del servicio.

## 4. Contenerización con Docker

* Se creó un **Dockerfile** basado en una imagen liviana de Python 3.11 slim para contenerizar la aplicación.
* Se instalaron solo las dependencias necesarias para ejecutar la API.
* Se montaron los modelos en el contenedor usando un volumen con solo lectura.
* Se expuso el puerto 8000 internamente en el contenedor y se configuró para mapearlo al puerto 8989 del host.
* Se configuró el contenedor para reiniciarse automáticamente en caso de fallos y se añadió un healthcheck.

## 5. Orquestación con Docker Compose

* Se utilizó un archivo **docker-compose.yml** para definir el servicio del API.
* Se montaron los modelos desde el host para mantenerlos actualizables sin reconstrucción.
* Se establecieron variables de entorno para garantizar la correcta ejecución de la aplicación.
* Se configuró el puerto y el healthcheck para mejorar la confiabilidad en entornos productivos o de desarrollo.

## 6. Bonus: Selección Dinámica de Modelo

* Se incorporó en la API la funcionalidad para que el usuario pueda seleccionar qué modelo usar en cada petición de inferencia.
* Esto permite comparar resultados y seleccionar el modelo que mejor se ajuste a diferentes casos de uso.

## Resultado Final

* Una API REST robusta y escalable que permite predecir la especie de pingüino dado un conjunto de características.
* Contenerización del servicio para facilitar despliegue y distribución.
* Código modular y bien documentado para facilitar mantenibilidad y extensibilidad.
* Posibilidad de comparar múltiples modelos de machine learning a través de un parámetro de selección en la petición.
